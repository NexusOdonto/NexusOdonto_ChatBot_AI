import logging
from fastapi import APIRouter, Request, status
from app.schemas.chat import EvolutionWebhookPayload
from app.clients.evolution_client import evolution_client

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/webhook", tags=["WhatsApp Webhook"])

# Mensaje amigable si falla la API de .NET, base de datos o el bot
MENSAJE_FALLBACK_PACIENTE = (
    "En este momento nuestro sistema de agenda está en mantenimiento o presentando intermitencias. "
    "Por favor, intenta nuevamente en unos minutos. ¡Disculpa las molestias!"
)

@router.post("/whatsapp")
async def receive_whatsapp_message(request: Request):
    numero_paciente = None
    try:
        raw_json = await request.json()
        
        # 1. Validar la estructura con Pydantic
        payload = EvolutionWebhookPayload(**raw_json)
        
        if payload.data and payload.data.message:
            data = payload.data
            
            # Extraer número del paciente
            if data.key and data.key.remoteJid:
                numero_paciente = data.key.remoteJid.split("@")[0]
            
            # Extraer texto del mensaje
            mensaje_texto = ""
            if data.message.conversation:
                mensaje_texto = data.message.conversation
            elif data.message.extendedTextMessage and data.message.extendedTextMessage.text:
                mensaje_texto = data.message.extendedTextMessage.text

            # Ignorar mensajes emitidos por el bot
            if data.key and data.key.fromMe:
                return {"status": "ignored", "reason": "self_message"}

            if numero_paciente and mensaje_texto:
                logger.info(f"[Webhook] Mensaje de {numero_paciente}: {mensaje_texto}")
                
                try:
                    # TODO: Conexión con LangGraph y llamadas a .NET
                    pass
                except Exception as service_err:
                    logger.error(f"[Error de Servicio] Fallo procesando mensaje de {numero_paciente}: {str(service_err)}")
                    # Notificar al paciente por WhatsApp
                    await evolution_client.enviar_mensaje(numero_paciente, MENSAJE_FALLBACK_PACIENTE)
                    return {"status": "fallback_sent"}

        return {"status": "success"}

    except Exception as e:
        logger.error(f"[Webhook Error] Fallo al procesar el webhook: {str(e)}")
        if numero_paciente:
            try:
                await evolution_client.enviar_mensaje(numero_paciente, MENSAJE_FALLBACK_PACIENTE)
            except Exception:
                pass
        return {"status": "error", "message": "Error procesando el payload"}