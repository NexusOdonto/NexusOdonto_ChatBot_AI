import logging
import re
import asyncio
from fastapi import APIRouter, Request
from langchain_core.messages import AIMessage, HumanMessage
from app.schemas.chat import EvolutionWebhookPayload
from app.clients.evolution_client import evolution_client
from app.clients.dotnet_client import dotnet_client
from app.core.config import settings
from app.graph.builder import get_graph
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


async def _is_escalated(thread_id: str) -> bool:
    # El estado persistido evita que el bot responda mientras recepción atiende.
    state = await get_graph().aget_state(get_thread_config(thread_id))
    return state.values.get("conversation_status") == "ESCALADA"


async def _escalate_conversation(thread_id: str, phone_number: str, message: str) -> bool:
    # Primero se intenta crear el ticket en el backend .NET
    ticket = await dotnet_client.crear_ticket_soporte(
        telefono=phone_number,
        motivo="SOLICITUD_USUARIO" if _is_escalation_request(message) else "BAJA_CONFIANZA_RAG",
        prioridad="MEDIA"
    )
    if ticket is None:
        logger.warning(
            f"[BG] No se pudo crear el ticket de soporte en el backend .NET para {thread_id} (backend caído). "
            f"Procediendo con escalamiento local de la conversación."
        )

    # De todas formas actualizamos el estado de la conversación local a ESCALADA en PostgreSQL
    config = get_thread_config(thread_id)
    await get_graph().aupdate_state(config, {"conversation_status": "ESCALADA"})
    # Y enviamos el mensaje amigable de escalamiento al paciente
    await evolution_client.enviar_mensaje(phone_number, MENSAJE_ESCALAMIENTO)
    return True


async def _process_whatsapp_message(numero_paciente: str, mensaje_texto: str) -> None:
    """Procesa el mensaje del paciente en segundo plano.

    Esta corutina se ejecuta desacoplada del ciclo request/response para que
    Evolution API reciba el HTTP 200 de inmediato y no genere un error de timeout.
    """
    try:
        if await _is_escalated(numero_paciente):
            # Una conversación escalada queda bajo control exclusivo del usuario.
            logger.info(f"[BG] Mensaje ignorado – conversación escalada: {numero_paciente}")
            return

        if _is_escalation_request(mensaje_texto):
            # La solicitud explícita se atiende sin pasarla por el LLM.
            await _escalate_conversation(numero_paciente, numero_paciente, mensaje_texto)
            return

        config = get_thread_config(numero_paciente)
        result = await get_graph().ainvoke(
            {
                "messages": [HumanMessage(content=mensaje_texto)],
                "conversation_status": "ACTIVA",
                "rag_confidence": 1.0,
            },
            config,
        )

        if result.get("conversation_status") == "ESCALADA":
            # Escalado inmediato (ej: triage de urgencia detectado en security_check_node)
            messages = result.get("messages", [])
            if messages:
                last_message = messages[-1]
                if isinstance(last_message, AIMessage) and last_message.content:
                    await evolution_client.enviar_mensaje(numero_paciente, str(last_message.content))
            await _escalate_conversation(numero_paciente, numero_paciente, mensaje_texto)
            return

        if result.get("rag_confidence", 1.0) < settings.rag_min_confidence:
            # No enviamos una respuesta posiblemente incorrecta: escalamos a recepción.
            if not await _escalate_conversation(numero_paciente, numero_paciente, mensaje_texto):
                await evolution_client.enviar_mensaje(numero_paciente, MENSAJE_FALLBACK_PACIENTE)
            return

        # Enviar la respuesta del bot al paciente por WhatsApp si no está bloqueada
        # (ya se envió en el nodo de seguridad cuando conversation_status == "BLOQUEADA").
        if result.get("conversation_status") != "BLOQUEADA":
            messages = result.get("messages", [])
            if messages:
                last_message = messages[-1]
                if isinstance(last_message, AIMessage) and last_message.content:
                    await evolution_client.enviar_mensaje(numero_paciente, str(last_message.content))
                else:
                    logger.warning(
                        f"[BG] La última respuesta no es de tipo AIMessage o está vacía: {last_message}"
                    )
            else:
                logger.warning("[BG] No se encontraron mensajes en el resultado del grafo.")

    except Exception as service_err:
        logger.error(
            f"[Error de Servicio] Fallo procesando mensaje de {numero_paciente}: {str(service_err)}",
            exc_info=True,
        )
        # Solo si el grafo falló por completo y no hay respuesta, se genera una salida amigable
        try:
            from langchain_core.messages import SystemMessage
            from langchain_openai import ChatOpenAI
            emergency_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7, api_key=settings.openai_api_key)
            resp = await emergency_llm.ainvoke([
                SystemMessage(content="Eres el asistente virtual de Nexus Odonto. Responde cordialmente y ofrece asistencia básica o pide que nos contacte al +57 324 6030217."),
                HumanMessage(content=mensaje_texto)
            ])
            await evolution_client.enviar_mensaje(numero_paciente, str(resp.content))
        except Exception as e:
            logger.error(f"Error crítico en fallback: {e}")
            pass


@router.post("/whatsapp")
async def receive_whatsapp_message(request: Request):
    """Endpoint de webhook para Evolution API.

    Retorna HTTP 200 de inmediato para evitar timeouts del proveedor.
    El procesamiento del grafo y el envío de la respuesta ocurren en segundo plano
    mediante asyncio.create_task, desacoplados del ciclo request/response de FastAPI.
    """
    try:
        raw_json = await request.json()

        # 1. Validar la estructura con Pydantic
        payload = EvolutionWebhookPayload(**raw_json)

        if payload.data and payload.data.message:
            data = payload.data

            # Ignorar mensajes emitidos por el bot antes de cualquier procesamiento
            if data.key and data.key.fromMe:
                return {"status": "ignored", "reason": "self_message"}

            # Extraer número del paciente completo (con el sufijo de whatsapp)
            numero_paciente = data.key.remoteJid if (data.key and data.key.remoteJid) else None

            # Extraer texto del mensaje
            mensaje_texto = ""
            if data.message.conversation:
                mensaje_texto = data.message.conversation
            elif data.message.extendedTextMessage and data.message.extendedTextMessage.text:
                mensaje_texto = data.message.extendedTextMessage.text

            if numero_paciente and mensaje_texto:
                logger.info(f"[Webhook] Mensaje recibido de {numero_paciente}: {mensaje_texto}")
                # Encolar el procesamiento pesado en segundo plano.
                # asyncio.create_task garantiza que la corutina vive en el event-loop
                # de FastAPI más allá del ciclo de vida de este request.
                asyncio.create_task(
                    _process_whatsapp_message(numero_paciente, mensaje_texto)
                )

        # Evolution API recibe HTTP 200 de inmediato, sin esperar el grafo.
        return {"status": "queued"}

    except Exception as e:
        logger.error(f"[Webhook Error] Fallo al procesar el webhook: {e}", exc_info=True)
        return {"status": "error", "message": "Error procesando el payload"}