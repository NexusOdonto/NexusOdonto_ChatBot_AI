import asyncio
import os
import logging
import time
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger
from apscheduler.triggers.interval import IntervalTrigger
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
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
from app.services.inactivity_service import inactivity_service
from app.clients.evolution_client import evolution_client

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
from app.api.routes.qr import router as qr_router
from app.api.routes.health import router as health_router

# Inicialización de la aplicación FastAPI
app = FastAPI(
    title="NexusOdonto ChatBot AI",
    description="Servidor de integración con Evolution API y LangGraph para atención odontológica",
    version="1.0.0"
)

scheduler = AsyncIOScheduler(timezone=settings.reminder_timezone)
checkpoint_store: PostgresCheckpointer | None = None


@app.middleware("http")
async def validate_evolution_webhook(request: Request, call_next):
    """Bloquea webhooks falsos antes de que alcancen el router o LangGraph."""
    if request.url.path == "/webhook/whatsapp" and request.method == "POST":
        auth_header = request.headers.get("Authorization", "") or ""
        apikey_header = request.headers.get("apikey", "") or ""
        host = (request.headers.get("host") or "").lower()
        secret = (settings.webhook_secret or "").strip()

        if secret:
            auth_ok = (
                secret in auth_header
                or secret == apikey_header
                or (settings.evolution_api_key and apikey_header == settings.evolution_api_key)
            )
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
app.include_router(qr_router)
app.include_router(health_router)


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
    checkpoint_store = PostgresCheckpointer(settings.postgres_checkpoint_url)
    await checkpoint_store.start()
    create_graph(checkpoint_store.saver)

    initialize_qdrant()
    scheduler.add_job(
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
        enviar_recordatorios_30_minutos,
        IntervalTrigger(minutes=2, timezone=settings.reminder_timezone),
        id="appointment-reminders-30m",
        replace_existing=True,
        max_instances=1,
        coalesce=True,
    )
    scheduler.add_job(
        inactivity_service.procesar_expiraciones_pendientes,
        IntervalTrigger(seconds=60, timezone=settings.reminder_timezone),
        id="inactivity-sweeper",
        replace_existing=True,
        max_instances=1,
        coalesce=True,
    )
    scheduler.start()
    asyncio.create_task(inactivity_service.procesar_expiraciones_pendientes(settings.session_ttl_seconds))
    asyncio.create_task(evolution_client.ensure_webhook_configured())
    logger.info(
        "Scheduler activo: Recordatorios (diario 08:00 y cada 2 min) + Barrendero de inactividad (cada 60 s).",
    )


@app.on_event("shutdown")
async def shutdown() -> None:
    if scheduler.running:
        scheduler.shutdown(wait=False)
    await inactivity_service.stop_all()
    if checkpoint_store is not None:
        await checkpoint_store.stop()
