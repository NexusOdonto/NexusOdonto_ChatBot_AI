import asyncio
import os
import logging
import time
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger
from apscheduler.triggers.interval import IntervalTrigger
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, HTMLResponse
import httpx
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

from app.agents.tools.qdrant_tool import initialize_qdrant
from app.core.config import settings
from app.services.appointment_reminders import (
    enviar_recordatorios_citas,
    enviar_recordatorios_30_minutos,
)
from app.graph.builder import create_graph
from app.session.postgres_checkpointer import PostgresCheckpointer

# Cargar variables de entorno desde el archivo .env
load_dotenv()

# Configuración básica de logs
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger(__name__)

# Importar los routers de la API
from app.api.routes.webhook import router as webhook_router
from app.api.routes.agent_handoff import router as handoff_router
from app.api.routes.reminders import router as reminders_router

# Inicialización de la aplicación FastAPI
app = FastAPI(
    title="NexusOdonto ChatBot AI",
    description="Servidor de integración con Evolution API y LangGraph para atención odontológica",
    version="1.0.0"
)

scheduler = AsyncIOScheduler(timezone=settings.reminder_timezone)
# APScheduler ejecutará el job dentro del ciclo de vida de FastAPI.
checkpoint_store: PostgresCheckpointer | None = None
# Serializa lectura/conexión del QR con el logout para que un poll del
# navegador que disparó el unlink no vuelva a llamar /instance/connect
# y reactive la sesión Baileys que acaba de cerrarse.
_qr_session_lock = asyncio.Lock()
_qr_unlinking = False
# Tras purgar Baileys, connectionState puede seguir diciendo "open" por la
# caché local de Evolution. En esa ventana no se acepta "open" sin revalidar.
_qr_fresh_until = 0.0

_QR_NO_CACHE_HEADERS = {
    "Cache-Control": "no-store, no-cache, must-revalidate",
    "Pragma": "no-cache",
    "Expires": "0",
}


@app.middleware("http")
async def validate_evolution_webhook(request: Request, call_next):
    """Bloquea webhooks falsos antes de que alcancen el router o LangGraph."""
    if request.url.path == "/webhook/whatsapp" and request.method == "POST":
        # Evolution API no soporta HMAC nativamente, así que validamos un token estático
        auth_header = request.headers.get("Authorization", "") or ""
        apikey_header = request.headers.get("apikey", "") or ""
        host = (request.headers.get("host") or "").lower()
        secret = (settings.webhook_secret or "").strip()

        logger.info(f"Headers recibidos: {request.headers}")

        if secret:
            auth_ok = (
                secret in auth_header
                or secret == apikey_header
                or (settings.evolution_api_key and apikey_header == settings.evolution_api_key)
            )
            # Evolution 2.3 global webhook a veces omite Authorization aunque esté configurado.
            # Aceptamos llamadas internas Docker (host agente-python) sin token.
            internal_host = host.startswith("agente-python") or host.startswith("agente_python")
            if not auth_ok and not (internal_host and not auth_header and not apikey_header):
                logger.warning("Webhook rechazado: token ausente o inválido")
                return JSONResponse(
                    status_code=403,
                    content={"status": "forbidden", "message": "Token de webhook inválido."},
                )

    return await call_next(request)

# Configuración de CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Registro de rutas
app.include_router(webhook_router, tags=["WhatsApp Webhook"])
app.include_router(handoff_router, tags=["Agent Handoff & Messaging"])
app.include_router(handoff_router, prefix="/api/v1", tags=["Agent Handoff & Messaging (v1)"])
app.include_router(reminders_router, tags=["Recordatorios de Citas"])
app.include_router(reminders_router, prefix="/api/v1", tags=["Recordatorios de Citas (v1)"])

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Error interno no controlado en la ruta {request.url.path}: {str(exc)}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"status": "error", "message": "Error interno del servidor."}
    )


@app.on_event("startup")
async def startup() -> None:
    global checkpoint_store
    # PostgreSQL se prepara antes de aceptar mensajes para recuperar threads existentes.
    checkpoint_store = PostgresCheckpointer(settings.postgres_checkpoint_url)
    await checkpoint_store.start()
    create_graph(checkpoint_store.saver)

    # Comprueba la conexión y prepara la colección antes de atender solicitudes.
    initialize_qdrant()
    scheduler.add_job(
        # El job consulta las citas de mañana y envía los recordatorios.
        enviar_recordatorios_citas,
        CronTrigger(
            hour=settings.reminder_schedule_hour,
            minute=settings.reminder_schedule_minute,
            timezone=settings.reminder_timezone,
        ),
        id="appointment-reminders",
        replace_existing=True,
        max_instances=1,
        coalesce=True,
    )
    scheduler.add_job(
        # El job monitorea citas de hoy y envía recordatorio 30 minutos antes.
        enviar_recordatorios_30_minutos,
        IntervalTrigger(minutes=2, timezone=settings.reminder_timezone),
        id="appointment-reminders-30m",
        replace_existing=True,
        max_instances=1,
        coalesce=True,
    )
    scheduler.start()
    logger.info(
        "Recordatorios programados: Diario a las %02d:%02d (%s) y en tiempo real cada 2 min (30 min antes de la cita)",
        settings.reminder_schedule_hour,
        settings.reminder_schedule_minute,
        settings.reminder_timezone,
    )


@app.on_event("shutdown")
async def shutdown() -> None:
    # Cerrar el scheduler evita tareas huérfanas al detener el servidor.
    if scheduler.running:
        scheduler.shutdown(wait=False)
    if checkpoint_store is not None:
        await checkpoint_store.stop()

@app.get("/health", tags=["Health Check"])
async def health_check():
    return {"status": "healthy", "service": "NexusOdonto ChatBot AI"}

def _qr_instance_name() -> str:
    return os.getenv("EVOLUTION_INSTANCE_NAME", getattr(settings, "instance_name", "Nexus_Odonto"))


def _evolution_headers() -> dict[str, str]:
    return {"apikey": settings.evolution_api_key, "Content-Type": "application/json"}


def _qr_json(payload: dict, status_code: int = 200) -> JSONResponse:
    return JSONResponse(content=payload, status_code=status_code, headers=_QR_NO_CACHE_HEADERS)


def _extract_qr_base64(payload: dict) -> str:
    if not isinstance(payload, dict):
        return ""
    direct = payload.get("base64") or ""
    if isinstance(direct, str) and direct:
        return direct
    nested = payload.get("qrcode")
    if isinstance(nested, dict):
        return nested.get("base64") or ""
    return ""


def _disconnected_payload(instance_name: str, state: str = "close") -> dict:
    return {
        "connected": False,
        "state": state,
        "base64": "",
        "instanceName": instance_name,
    }


async def _ensure_instance(
    client: httpx.AsyncClient,
    instance_name: str,
    webhook: dict | None = None,
) -> None:
    """Crea la instancia en Evolution si aún no existe (404)."""
    url_create = f"{settings.evolution_api_url}/instance/create"
    payload: dict = {
        "instanceName": instance_name,
        "qrcode": True,
        "integration": "WHATSAPP-BAILEYS",
    }
    if webhook and webhook.get("url"):
        payload["webhook"] = {
            "enabled": webhook.get("enabled", True),
            "url": webhook.get("url"),
            "events": webhook.get("events") or [
                "MESSAGES_UPSERT",
                "MESSAGES_UPDATE",
                "CONNECTION_UPDATE",
            ],
        }
    try:
        resp = await client.post(url_create, headers=_evolution_headers(), json=payload)
        if resp.status_code in (200, 201):
            logger.info("[QR] Instancia Evolution creada: %s", instance_name)
        elif resp.status_code not in (403, 409):
            logger.warning(
                "[QR] No se pudo crear instancia %s (HTTP %s): %s",
                instance_name,
                resp.status_code,
                resp.text[:300],
            )
    except Exception as exc:
        logger.warning("[QR] Error creando instancia Evolution: %s", exc)


async def _connection_state(client: httpx.AsyncClient, instance_name: str) -> str:
    url_state = f"{settings.evolution_api_url}/instance/connectionState/{instance_name}"
    resp_state = await client.get(url_state, headers=_evolution_headers())
    if resp_state.status_code == 404:
        return "missing"
    if resp_state.status_code != 200:
        return "unknown"
    data_st = resp_state.json()
    instance = data_st.get("instance") if isinstance(data_st, dict) else None
    if isinstance(instance, dict):
        return instance.get("state") or "disconnected"
    return "disconnected"


async def _snapshot_webhook(client: httpx.AsyncClient, instance_name: str) -> dict | None:
    url = f"{settings.evolution_api_url}/webhook/find/{instance_name}"
    try:
        resp = await client.get(url, headers=_evolution_headers())
        if resp.status_code != 200:
            return None
        data = resp.json()
        if not isinstance(data, dict):
            return None
        webhook = data.get("webhook") if isinstance(data.get("webhook"), dict) else data
        if isinstance(webhook, dict) and webhook.get("url"):
            return webhook
    except Exception as exc:
        logger.warning("[QR] No se pudo leer el webhook de %s: %s", instance_name, exc)
    return None


async def _restore_webhook(client: httpx.AsyncClient, instance_name: str, webhook: dict | None) -> None:
    if not webhook or not webhook.get("url"):
        return
    url = f"{settings.evolution_api_url}/webhook/set/{instance_name}"
    events = webhook.get("events") or [
        "MESSAGES_UPSERT",
        "MESSAGES_UPDATE",
        "CONNECTION_UPDATE",
    ]
    nested = {
        "webhook": {
            "enabled": webhook.get("enabled", True),
            "url": webhook.get("url"),
            "headers": webhook.get("headers") or {},
            "byEvents": webhook.get("byEvents") or webhook.get("webhookByEvents") or False,
            "base64": webhook.get("base64") or webhook.get("webhookBase64") or False,
            "events": events,
        }
    }
    flat = {
        "enabled": webhook.get("enabled", True),
        "url": webhook.get("url"),
        "webhookByEvents": webhook.get("byEvents") or webhook.get("webhookByEvents") or False,
        "webhookBase64": webhook.get("base64") or webhook.get("webhookBase64") or False,
        "events": events,
    }
    try:
        for payload in (nested, flat):
            resp = await client.post(url, headers=_evolution_headers(), json=payload)
            if resp.status_code in (200, 201):
                logger.info("[QR] Webhook restaurado en %s", instance_name)
                return
        logger.warning(
            "[QR] Webhook no restaurado en %s (HTTP %s): %s",
            instance_name,
            resp.status_code,
            resp.text[:300],
        )
    except Exception as exc:
        logger.warning("[QR] Error restaurando webhook de %s: %s", instance_name, exc)


async def _logout_evolution(client: httpx.AsyncClient, instance_name: str) -> int:
    """Evolution v2 desvincula con DELETE; algunas builds aceptan POST."""
    url = f"{settings.evolution_api_url}/instance/logout/{instance_name}"
    headers = _evolution_headers()
    resp = await client.delete(url, headers=headers)
    if resp.status_code in (404, 405):
        resp = await client.post(url, headers=headers)
    logger.info("[QR] Logout Evolution %s HTTP %s", instance_name, resp.status_code)
    return resp.status_code


async def _purge_baileys_session(client: httpx.AsyncClient, instance_name: str) -> None:
    """Borra la instancia (creds Baileys en Postgres/volumen) y deja una vacía.

    Logout a veces deja el auth folder y el siguiente /instance/connect
    vuelve a abrir la misma sesión en el host que posee el socket.
    """
    webhook = await _snapshot_webhook(client, instance_name)
    if not webhook:
        webhook = {
            "enabled": True,
            "url": os.getenv(
                "EVOLUTION_WEBHOOK_URL",
                "http://agente-python:8000/webhook/whatsapp",
            ),
            "events": ["MESSAGES_UPSERT", "MESSAGES_UPDATE", "CONNECTION_UPDATE"],
        }
    url_delete = f"{settings.evolution_api_url}/instance/delete/{instance_name}"
    resp = await client.delete(url_delete, headers=_evolution_headers())
    logger.info("[QR] Delete instancia %s HTTP %s", instance_name, resp.status_code)
    await _ensure_instance(client, instance_name, webhook)
    await _restore_webhook(client, instance_name, webhook)


@app.get("/qr/data", tags=["WhatsApp QR"])
async def get_qr_data():
    """Retorna los datos del QR y estado de conexión en formato JSON."""
    instance_name = _qr_instance_name()
    headers = _evolution_headers()

    async with _qr_session_lock:
        # Mientras se desvincula, no consultar ni reconectar: Evolution puede
        # devolver "open" en caché local y /instance/connect restaura Baileys.
        if _qr_unlinking:
            return _qr_json(_disconnected_payload(instance_name))

        state = "disconnected"
        try:
            async with httpx.AsyncClient(timeout=12.0) as client:
                state = await _connection_state(client, instance_name)
                if state == "missing":
                    await _ensure_instance(client, instance_name)
                    state = await _connection_state(client, instance_name)
        except Exception:
            pass

        stale_open = state == "open" and time.monotonic() < _qr_fresh_until
        if state == "open" and not stale_open:
            return _qr_json(
                {"connected": True, "state": state, "base64": "", "instanceName": instance_name}
            )

        qr_base64 = ""
        try:
            url_connect = f"{settings.evolution_api_url}/instance/connect/{instance_name}"
            async with httpx.AsyncClient(timeout=12.0) as client:
                if _qr_unlinking:
                    return _qr_json(_disconnected_payload(instance_name))
                resp_qr = await client.get(url_connect, headers=headers)
                if resp_qr.status_code == 404:
                    await _ensure_instance(client, instance_name)
                    resp_qr = await client.get(url_connect, headers=headers)
                if resp_qr.status_code == 200:
                    qr_base64 = _extract_qr_base64(resp_qr.json())
                    # connect puede reabrir la sesión si el logout no borró creds.
                    state = await _connection_state(client, instance_name)
        except Exception:
            pass

        if state == "open":
            return _qr_json(
                {"connected": True, "state": state, "base64": "", "instanceName": instance_name}
            )

        return _qr_json(
            {
                "connected": False,
                "state": state if state not in ("disconnected", "missing", "unknown") or not qr_base64 else "connecting",
                "base64": qr_base64,
                "instanceName": instance_name,
            }
        )

@app.post("/qr/restart", tags=["WhatsApp QR"])
async def restart_qr_instance():
    """Reinicia la instancia en Evolution API para generar un QR limpio desde cero."""
    instance_name = _qr_instance_name()
    headers = {"apikey": settings.evolution_api_key}
    try:
        url = f"{settings.evolution_api_url}/instance/restart/{instance_name}"
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.post(url, headers=headers)
            return _qr_json({"status": "restarted", "http_code": resp.status_code})
    except Exception as e:
        return _qr_json({"status": "error", "message": str(e)}, status_code=502)


@app.post("/qr/logout", tags=["WhatsApp QR"])
async def logout_qr_instance():
    """Desvincula WhatsApp en Evolution y borra la sesión Baileys si queda viva."""
    global _qr_unlinking, _qr_fresh_until
    instance_name = _qr_instance_name()
    async with _qr_session_lock:
        _qr_unlinking = True
        try:
            async with httpx.AsyncClient(timeout=45.0) as client:
                logout_code = await _logout_evolution(client, instance_name)
                if logout_code in (401, 403):
                    return _qr_json(
                        {
                            "status": "error",
                            "unlinked": False,
                            "message": "Evolution API rechazó el logout.",
                            "instanceName": instance_name,
                        },
                        status_code=502,
                    )

                # Siempre purgar: logout de Evolution a menudo deja el auth folder
                # de Baileys y el poll del navegador que desvinculó llama a
                # /instance/connect, que reabre esa sesión en este host.
                await _purge_baileys_session(client, instance_name)
                _qr_fresh_until = time.monotonic() + 20.0
                state = await _connection_state(client, instance_name)
                # "connecting" aquí es el QR nuevo, no la sesión anterior.
                if state == "open":
                    return _qr_json(
                        {
                            "status": "error",
                            "unlinked": False,
                            "state": state,
                            "connected": True,
                            "instanceName": instance_name,
                            "message": "La sesión de WhatsApp sigue vinculada.",
                        },
                        status_code=502,
                    )

                return _qr_json(
                    {
                        "status": "logged_out",
                        "unlinked": True,
                        "connected": False,
                        "state": "close" if state in ("missing", "unknown", "disconnected") else state,
                        "instanceName": instance_name,
                    }
                )
        except Exception as exc:
            logger.exception("[QR] Error desvinculando WhatsApp")
            return _qr_json(
                {
                    "status": "error",
                    "unlinked": False,
                    "message": str(exc),
                    "instanceName": instance_name,
                },
                status_code=502,
            )
        finally:
            _qr_unlinking = False

@app.get("/qr", response_class=HTMLResponse, tags=["WhatsApp QR"])
async def get_whatsapp_qr():
    """Interfaz visual en tiempo real para vincular WhatsApp sin recargas bruscas."""
    html_content = """
    <!DOCTYPE html>
    <html lang="es">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Vincular WhatsApp - Nexus Odonto</title>
        <style>
            * { box-sizing: border-box; }
            body {
                font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;
                display: flex;
                flex-direction: column;
                align-items: center;
                justify-content: center;
                min-height: 100vh;
                margin: 0;
                background: radial-gradient(circle at top, #1e293b, #0f172a);
                color: white;
            }
            .card {
                background: rgba(30, 41, 59, 0.95);
                backdrop-filter: blur(10px);
                padding: 32px 28px;
                border-radius: 24px;
                box-shadow: 0 20px 40px -15px rgba(0,0,0,0.6);
                text-align: center;
                border: 1px solid rgba(255,255,255,0.1);
                max-width: 420px;
                width: 92%;
                transition: all 0.3s ease;
            }
            h1 { margin: 0 0 8px 0; color: #14b8a6; font-size: 24px; font-weight: 700; }
            p { color: #94a3b8; font-size: 13px; line-height: 1.5; margin-bottom: 24px; }
            .qr-wrapper {
                position: relative;
                display: inline-block;
                border-radius: 16px;
                padding: 14px;
                background: white;
                box-shadow: 0 8px 25px rgba(0,0,0,0.4);
                min-width: 260px;
                min-height: 260px;
            }
            #qrImg {
                width: 250px;
                height: 250px;
                display: block;
                border-radius: 8px;
            }
            .loader {
                display: flex;
                align-items: center;
                justify-content: center;
                height: 250px;
                width: 250px;
                color: #0f172a;
                font-weight: 600;
                font-size: 14px;
            }
            .status-badge {
                display: inline-flex;
                align-items: center;
                gap: 6px;
                background: #064e3b;
                color: #34d399;
                padding: 6px 14px;
                border-radius: 9999px;
                font-weight: 600;
                font-size: 12px;
                margin-top: 20px;
            }
            .pulse-dot {
                width: 8px;
                height: 8px;
                background: #10b981;
                border-radius: 50%;
                animation: pulse 1.5s infinite;
            }
            @keyframes pulse {
                0% { transform: scale(0.95); opacity: 0.8; }
                50% { transform: scale(1.3); opacity: 1; }
                100% { transform: scale(0.95); opacity: 0.8; }
            }
            .actions { margin-top: 22px; display: flex; gap: 10px; justify-content: center; }
            button {
                background: #334155;
                color: #e2e8f0;
                border: 1px solid #475569;
                padding: 9px 16px;
                border-radius: 12px;
                font-size: 12px;
                font-weight: 600;
                cursor: pointer;
                transition: all 0.2s;
            }
            button:hover { background: #475569; color: white; }
            .manager-link {
                margin-top: 16px;
                font-size: 11px;
                color: #64748b;
            }
            .manager-link a { color: #2dd4bf; text-decoration: none; font-weight: 600; }
            .manager-link a:hover { text-decoration: underline; }
        </style>
    </head>
    <body>
        <div class="card">
            <h1>📱 Vincular WhatsApp</h1>
            <p>Abre WhatsApp en tu teléfono &gt; <b>Dispositivos vinculados</b> &gt; <b>Vincular un dispositivo</b> y escanea este código:</p>
            
            <div class="qr-wrapper">
                <div id="loader" class="loader">Cargando código QR...</div>
                <img id="qrImg" style="display:none;" alt="QR Code WhatsApp" />
            </div>

            <br />
            <div class="status-badge">
                <span class="pulse-dot"></span>
                <span id="statusText">Esperando escaneo...</span>
            </div>

            <div class="actions">
                <button onclick="restartQR()">🔄 Regenerar QR Nuevo</button>
            </div>

            <div class="manager-link">
                Panel Oficial: <a href="http://localhost:8085/manager/" target="_blank">Evolution Manager</a> (Clave: <code>CLAVE_SECRETA_ODONTO_2026</code>)
            </div>
        </div>

        <script>
            let isConnected = false;

            async function updateQR() {
                try {
                    const res = await fetch('/qr/data?t=' + Date.now(), { cache: 'no-store' });
                    const data = await res.json();
                    
                    if (data.connected || data.state === 'open') {
                        if (isConnected) return;
                        isConnected = true;
                        document.querySelector('.card').innerHTML = `
                            <div style="padding: 20px;">
                                <div style="font-size: 60px; margin-bottom: 12px;">✅</div>
                                <h1 style="color: #34d399;">¡WhatsApp Conectado!</h1>
                                <p style="color: #94a3b8; font-size: 14px;">El bot de Nexus Odonto ya está activo y listo para atender a tus pacientes.</p>
                                <div style="background: rgba(16,185,129,0.1); border: 1px solid #10b981; padding: 12px; border-radius: 12px; font-size: 12px; color: #a7f3d0; margin-top: 16px;">
                                    Instancia: <b>${data.instanceName}</b> (ONLINE)
                                </div>
                            </div>
                        `;
                        return;
                    }

                    if (isConnected) {
                        location.reload();
                        return;
                    }

                    if (data.base64) {
                        const img = document.getElementById('qrImg');
                        const loader = document.getElementById('loader');
                        if (img.src !== data.base64) {
                            img.src = data.base64;
                        }
                        img.style.display = 'block';
                        loader.style.display = 'none';
                        document.getElementById('statusText').innerText = 'Código QR listo para escanear';
                    }
                } catch (e) {
                    const status = document.getElementById('statusText');
                    if (status) status.innerText = 'Reconectando con Evolution API...';
                }
            }

            async function restartQR() {
                document.getElementById('statusText').innerText = 'Regenerando QR limpio...';
                document.getElementById('qrImg').style.display = 'none';
                document.getElementById('loader').style.display = 'flex';
                document.getElementById('loader').innerText = 'Generando nuevo código...';
                try {
                    await fetch('/qr/restart', { method: 'POST' });
                    setTimeout(updateQR, 1200);
                } catch (e) {
                    console.error(e);
                }
            }

            updateQR();
            setInterval(updateQR, 3000);
        </script>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)


