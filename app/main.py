import logging
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
from app.security.webhook_signature import is_valid_webhook_signature, restore_request_body
from app.agents.tools.qdrant_tool import initialize_qdrant
from app.core.config import settings
from app.services.appointment_reminders import enviar_recordatorios_citas
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

# Importar el router del webhook
from app.api.routes.webhook import router as webhook_router

# Inicialización de la aplicación FastAPI
app = FastAPI(
    title="NexusOdonto ChatBot AI",
    description="Servidor de integración con Evolution API y LangGraph para atención odontológica",
    version="1.0.0"
)

scheduler = AsyncIOScheduler(timezone=settings.reminder_timezone)
# APScheduler ejecutará el job dentro del ciclo de vida de FastAPI.
checkpoint_store: PostgresCheckpointer | None = None


@app.middleware("http")
async def validate_evolution_webhook(request: Request, call_next):
    """Bloquea webhooks falsos antes de que alcancen el router o LangGraph."""
    if request.url.path == "/webhook/whatsapp" and request.method == "POST":
        # Evolution API no soporta HMAC nativamente, así que validamos un token estático
        auth_header = request.headers.get("Authorization", "")
        apikey_header = request.headers.get("apikey", "")
        
        logger.info(f"Headers recibidos: {request.headers}")
        
        # Validar si el secreto está presente en los headers de autenticación
        if settings.webhook_secret not in auth_header and settings.webhook_secret != apikey_header:
            logger.warning("Webhook rechazado: token ausente o inválido")
            return JSONResponse(
                status_code=403,
                content={"status": "forbidden", "message": "Token de webhook inválido."},
            )
        
        # Ya no necesitamos leer el body anticipadamente ni sobreescribir _receive

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
    scheduler.start()
    logger.info(
        "Recordatorios programados diariamente a las %02d:%02d (%s)",
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
