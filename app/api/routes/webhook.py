import logging
import re
import asyncio
from fastapi import APIRouter, Request
from langchain_core.messages import AIMessage, HumanMessage
from app.schemas.chat import EvolutionWebhookPayload, unwrap_message_dict
from app.clients.evolution_client import evolution_client
from app.clients.dotnet_client import dotnet_client
from app.core.config import settings
from app.graph.builder import get_graph
from app.session.memory_store import get_thread_config
from app.session.postgres_checkpointer import get_checkpointer_instance
from app.services.audio_service import (
    extraer_bytes_audio,
    transcribir_audio,
    MENSAJE_AUDIO_NO_ENTENDIDO,
    MENSAJE_ERROR_PROCESANDO_AUDIO,
)

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

MENSAJE_MEDIOS_NO_SOPORTADOS = (
    "Por el momento no puedo procesar ni visualizar fotos, videos, documentos ni archivos directamente 📎📷.\n\n"
    "Por favor, descríbeme detalladamente por texto o mediante una nota de voz lo que necesitas o lo que contiene tu archivo "
    "(por ejemplo, el síntoma que presentas, el tratamiento o la orden médica) para poder ayudarte con mucho gusto."
)

ESCALAMIENTO_RE = re.compile(
    r"\b(hablar|comunicarme|contactar|atenderme)\b.*\b(persona|humano|asesor|recepcionista)\b|"
    r"\b(persona|humano|asesor|recepcionista)\b.*\b(hablar|comunicarme|contactar|atenderme)\b",
    re.IGNORECASE,
)

COMMANDS_RESET = {"/clear", "/reset", "/reiniciar", "/limpiar", "/start", "/inicio"}


def _is_reset_request(message: str) -> bool:
    return message.strip().lower() in COMMANDS_RESET


async def _reset_conversation(phone_number: str) -> None:
    # 1. Purgar checkpoints de PostgreSQL directamente para ese thread_id
    checkpointer = get_checkpointer_instance()
    if checkpointer:
        await checkpointer.clear_thread(phone_number)
    else:
        logger.warning(f"[Reset] No se encontró la instancia del checkpointer para {phone_number}")

    logger.info(f"[Reset] Memoria e historial reiniciados para {phone_number}")
    await evolution_client.enviar_mensaje(
        phone_number,
        "🔄 Memoria reiniciada con éxito. ¡Hola! Soy el asistente virtual de Nexus Odonto. ¿En qué puedo colaborarte hoy?"
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


async def _process_whatsapp_unsupported_media(numero_paciente: str, caption: str = "") -> None:
    """Gestiona la recepción de fotos, videos, documentos o archivos no procesables directamente."""
    try:
        if await _is_escalated(numero_paciente):
            logger.info(f"[BG] Medio ignorado – conversación escalada: {numero_paciente}")
            return

        if caption:
            if _is_reset_request(caption):
                await _reset_conversation(numero_paciente)
                return
            if _is_escalation_request(caption):
                await _escalate_conversation(numero_paciente, numero_paciente, caption)
                return

        await evolution_client.enviar_mensaje(numero_paciente, MENSAJE_MEDIOS_NO_SOPORTADOS)
    except Exception as e:
        logger.error(f"[BG] Error al responder medio no soportado a {numero_paciente}: {e}", exc_info=True)


async def _process_whatsapp_audio(numero_paciente: str, raw_payload_data: dict, raw_message: dict) -> None:
    """Descarga, transcribe con Whisper y procesa un mensaje de audio o nota de voz."""
    try:
        if await _is_escalated(numero_paciente):
            logger.info(f"[BG] Audio ignorado – conversación escalada: {numero_paciente}")
            return

        # 1. Extraer los bytes del audio
        audio_info = await extraer_bytes_audio(raw_payload_data, raw_message)
        if not audio_info:
            logger.warning(f"[BG] No se pudieron extraer los bytes de audio para {numero_paciente}")
            await evolution_client.enviar_mensaje(numero_paciente, MENSAJE_ERROR_PROCESANDO_AUDIO)
            return

        audio_bytes, mimetype = audio_info

        # 2. Transcribir con OpenAI Whisper
        texto_transcrito = await transcribir_audio(audio_bytes, mimetype)
        if not texto_transcrito:
            logger.warning(f"[BG] La transcripción de audio resultó vacía para {numero_paciente}")
            await evolution_client.enviar_mensaje(numero_paciente, MENSAJE_AUDIO_NO_ENTENDIDO)
            return

        logger.info(f"[BG] Audio de {numero_paciente} transcrito: '{texto_transcrito}'")

        # 3. Procesar el texto transcrito con el agente conversacional
        await _process_whatsapp_message(numero_paciente, texto_transcrito)

    except Exception as e:
        logger.error(f"[BG] Error procesando audio de {numero_paciente}: {e}", exc_info=True)
        await evolution_client.enviar_mensaje(numero_paciente, MENSAJE_ERROR_PROCESANDO_AUDIO)


async def _process_whatsapp_message(numero_paciente: str, mensaje_texto: str) -> None:
    """Procesa el mensaje del paciente en segundo plano.

    Esta corutina se ejecuta desacoplada del ciclo request/response para que
    Evolution API reciba el HTTP 200 de inmediato y no genere un error de timeout.
    """
    try:
        # 1. Comandos de reinicio de conversación
        if _is_reset_request(mensaje_texto):
            await _reset_conversation(numero_paciente)
            return

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

        if payload.data:
            data = payload.data

            # Ignorar mensajes emitidos por el bot antes de cualquier procesamiento
            if data.key and data.key.fromMe:
                return {"status": "ignored", "reason": "self_message"}

            # Extraer número del paciente completo (con el sufijo de whatsapp)
            numero_paciente = data.key.remoteJid if (data.key and data.key.remoteJid) else None
            if not numero_paciente:
                return {"status": "ignored", "reason": "no_remote_jid"}

            raw_message = data.message or {}
            unwrapped_message = unwrap_message_dict(raw_message)
            message_type = data.messageType or ""

            # 1. Detectar si es audio o nota de voz
            is_audio = (
                message_type == "audioMessage"
                or "audioMessage" in unwrapped_message
            )

            # 2. Detectar si es medio no soportado (imagen, video, documento, sticker, contacto, ubicación)
            unsupported_keys = {
                "imageMessage",
                "videoMessage",
                "ptvMessage",
                "documentMessage",
                "documentWithCaptionMessage",
                "stickerMessage",
                "contactMessage",
                "contactsArrayMessage",
                "locationMessage",
                "liveLocationMessage",
            }
            is_unsupported_media = (
                message_type in unsupported_keys
                or any(k in unwrapped_message for k in unsupported_keys)
            )

            if is_audio:
                logger.info(f"[Webhook] Audio recibido de {numero_paciente}. Encolando transcripción y procesamiento...")
                raw_payload_data = raw_json.get("data", {})
                asyncio.create_task(
                    _process_whatsapp_audio(numero_paciente, raw_payload_data, unwrapped_message)
                )
            elif is_unsupported_media:
                caption = ""
                for k in ["imageMessage", "videoMessage", "documentMessage"]:
                    if k in unwrapped_message and isinstance(unwrapped_message[k], dict):
                        caption = unwrapped_message[k].get("caption", "") or caption

                logger.info(f"[Webhook] Medio no soportado recibido de {numero_paciente} (caption: '{caption}'). Encolando aviso...")
                asyncio.create_task(
                    _process_whatsapp_unsupported_media(numero_paciente, caption)
                )
            else:
                # Extraer texto del mensaje
                mensaje_texto = ""
                if "conversation" in unwrapped_message and unwrapped_message["conversation"]:
                    mensaje_texto = unwrapped_message["conversation"]
                elif "extendedTextMessage" in unwrapped_message:
                    ext = unwrapped_message["extendedTextMessage"]
                    if isinstance(ext, dict):
                        mensaje_texto = ext.get("text", "")
                    elif hasattr(ext, "text"):
                        mensaje_texto = ext.text or ""

                if mensaje_texto:
                    logger.info(f"[Webhook] Mensaje recibido de {numero_paciente}: {mensaje_texto}")
                    asyncio.create_task(
                        _process_whatsapp_message(numero_paciente, mensaje_texto)
                    )
                else:
                    logger.warning(f"[Webhook] Mensaje no reconocido o vacío de {numero_paciente}: {unwrapped_message}")

        # Evolution API recibe HTTP 200 de inmediato, sin esperar el procesamiento pesado
        return {"status": "queued"}

    except Exception as e:
        logger.error(f"[Webhook Error] Fallo al procesar el webhook: {e}", exc_info=True)
        return {"status": "error", "message": "Error procesando el payload"}