import logging
import re
import asyncio
from fastapi import APIRouter, Request
from langchain_core.messages import AIMessage, HumanMessage
from app.schemas.chat import EvolutionWebhookPayload
from app.clients.evolution_client import evolution_client
from app.clients.dotnet_client import dotnet_client
from app.core.config import settings
from app.graph.builder import graph
from app.session.memory_store import get_thread_config

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/webhook", tags=["WhatsApp Webhook"])

# Mensaje amigable si falla la API de .NET, base de datos o el bot
MENSAJE_FALLBACK_PACIENTE = (
    "En este momento nuestro sistema de agenda está en mantenimiento o presentando intermitencias. "
    "Por favor, intenta nuevamente en unos minutos. ¡Disculpa las molestias!"
)

MENSAJE_ESCALAMIENTO = (
    "Entiendo. Un asesor de la clínica revisará tu solicitud y te contactará pronto."
)

ESCALAMIENTO_RE = re.compile(
    r"\b(hablar|comunicarme|contactar|atenderme)\b.*\b(persona|humano|asesor|recepcionista)\b|"
    r"\b(persona|humano|asesor|recepcionista)\b.*\b(hablar|comunicarme|contactar|atenderme)\b",
    re.IGNORECASE,
)


def _is_escalation_request(message: str) -> bool:
    # La detección local garantiza que una petición explícita no dependa del LLM.
    normalized = " ".join(message.lower().split())
    return bool(ESCALAMIENTO_RE.search(normalized)) or "hablar con una persona" in normalized


def _is_escalated(thread_id: str) -> bool:
    # El estado persistido evita que el bot responda mientras recepción atiende.
    state = graph.get_state(get_thread_config(thread_id))
    return state.values.get("conversation_status") == "ESCALADA"


async def _escalate_conversation(thread_id: str, phone_number: str, message: str) -> bool:
    # Primero se crea el ticket; solo después se bloquea la conversación.
    ticket = await dotnet_client.crear_ticket_soporte(
        {
            "titulo": "Solicitud de atención humana por WhatsApp",
            "descripcion": message,
            "motivo": "SOLICITUD_USUARIO" if _is_escalation_request(message) else "BAJA_CONFIANZA_RAG",
            "prioridad": "MEDIA",
            "conversacionChatbotId": thread_id,
        }
    )
    if ticket is None:
        return False

    config = get_thread_config(thread_id)
    graph.update_state(config, {"conversation_status": "ESCALADA"})
    await evolution_client.enviar_mensaje(phone_number, MENSAJE_ESCALAMIENTO)
    return True

@router.post("/whatsapp")
async def receive_whatsapp_message(request: Request):
    numero_paciente = None
    try:
        raw_json = await request.json()
        
        # 1. Validar la estructura con Pydantic
        payload = EvolutionWebhookPayload(**raw_json)
        
        if payload.data and payload.data.message:
            data = payload.data
            
            # Extraer número del paciente completo (con el sufijo de whatsapp)
            if data.key and data.key.remoteJid:
                numero_paciente = data.key.remoteJid
            
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

                if _is_escalated(numero_paciente):
					# Una conversación escalada queda bajo control exclusivo del humano.
                    return {"status": "ignored", "reason": "conversation_escalated"}

                if _is_escalation_request(mensaje_texto):
					# La solicitud explícita se atiende sin pasarla por el LLM.
                    if await _escalate_conversation(numero_paciente, numero_paciente, mensaje_texto):
                        return {"status": "escalated"}
                    await evolution_client.enviar_mensaje(numero_paciente, MENSAJE_FALLBACK_PACIENTE)
                    return {"status": "error", "reason": "ticket_not_created"}
                
                try:
                    config = get_thread_config(numero_paciente)
                    result = await asyncio.to_thread(
						# LangGraph es síncrono; moverlo a otro hilo mantiene libre el event loop.
                        graph.invoke,
                        {
                            "messages": [HumanMessage(content=mensaje_texto)],
                            "conversation_status": "ACTIVA",
                            "rag_confidence": 1.0,
                        },
                        config,
                    )
                    if result.get("rag_confidence", 1.0) < settings.rag_min_confidence:
						# No enviamos una respuesta posiblemente incorrecta: escalamos a recepción.
                        if await _escalate_conversation(numero_paciente, numero_paciente, mensaje_texto):
                            return {"status": "escalated", "reason": "low_rag_confidence"}
                        await evolution_client.enviar_mensaje(numero_paciente, MENSAJE_FALLBACK_PACIENTE)
                        return {"status": "error", "reason": "ticket_not_created"}
                    # El envío de la respuesta ya fue gestionado internamente por el nodo del grafo (chatbot_node)
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