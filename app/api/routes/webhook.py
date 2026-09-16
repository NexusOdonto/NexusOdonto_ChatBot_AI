import logging
import re
import asyncio
import time
from collections import OrderedDict
from fastapi import APIRouter, Request
from langchain_core.messages import AIMessage, HumanMessage
from app.schemas.chat import EvolutionWebhookPayload, unwrap_message_dict, extract_interactive_selection
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
from app.services.semantic_cache import buscar_en_cache, guardar_en_cache

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/webhook", tags=["WhatsApp Webhook"])

# ─── Deduplicate inbound webhook deliveries ───────────────────────────────────
_MESSAGE_DEDUPE_TTL_SECONDS = 90
_SEEN_MESSAGE_IDS: OrderedDict[str, float] = OrderedDict()
_CHAT_LOCKS: dict[str, asyncio.Lock] = {}
_CHAT_LOCKS_GUARD = asyncio.Lock()

# ─── Anti-spam / Rate Limit por usuario ──────────────────────────────────────
# Si el usuario envía más de _SPAM_MAX_MSGS mensajes en _SPAM_WINDOW_SECONDS
# segundos, los mensajes extra se descartan y se envía UNA advertencia.
_SPAM_WINDOW_SECONDS = 10          # Ventana deslizante
_SPAM_MAX_MSGS = 5                 # Mensajes permitidos por ventana
# {numero_paciente: deque de timestamps}
_SPAM_TIMESTAMPS: dict[str, list] = {}
# {numero_paciente: timestamp de la última advertencia enviada}
_SPAM_WARNED_AT: dict[str, float] = {}
_SPAM_WARN_COOLDOWN = 30           # Segundos entre advertencias al mismo usuario


def _is_spam(numero_paciente: str) -> bool:
    """Retorna True si el usuario supera el rate-limit y debe ser silenciado.
    Mantiene una ventana deslizante de timestamps por usuario.
    """
    now = time.monotonic()
    timestamps = _SPAM_TIMESTAMPS.get(numero_paciente, [])
    # Descartar timestamps fuera de la ventana
    timestamps = [t for t in timestamps if now - t < _SPAM_WINDOW_SECONDS]
    timestamps.append(now)
    _SPAM_TIMESTAMPS[numero_paciente] = timestamps
    return len(timestamps) > _SPAM_MAX_MSGS


def _should_warn_spam(numero_paciente: str) -> bool:
    """Retorna True si todavía no hemos enviado la advertencia anti-spam recientemente."""
    now = time.monotonic()
    last_warn = _SPAM_WARNED_AT.get(numero_paciente, 0.0)
    if now - last_warn > _SPAM_WARN_COOLDOWN:
        _SPAM_WARNED_AT[numero_paciente] = now
        return True
    return False


def _prune_seen_message_ids(now: float) -> None:
    while _SEEN_MESSAGE_IDS:
        oldest_id, seen_at = next(iter(_SEEN_MESSAGE_IDS.items()))
        if now - seen_at <= _MESSAGE_DEDUPE_TTL_SECONDS:
            break
        _SEEN_MESSAGE_IDS.popitem(last=False)


def _is_duplicate_message_id(message_id: str | None) -> bool:
    """Return True if this message key.id was already accepted within the TTL window."""
    if not message_id:
        return False
    now = time.monotonic()
    _prune_seen_message_ids(now)
    if message_id in _SEEN_MESSAGE_IDS:
        return True
    _SEEN_MESSAGE_IDS[message_id] = now
    _SEEN_MESSAGE_IDS.move_to_end(message_id)
    return False


async def _get_chat_lock(chat_id: str) -> asyncio.Lock:
    async with _CHAT_LOCKS_GUARD:
        lock = _CHAT_LOCKS.get(chat_id)
        if lock is None:
            lock = asyncio.Lock()
            _CHAT_LOCKS[chat_id] = lock
        return lock


async def _with_chat_lock(chat_id: str, coro):
    """Serialize processing per chat so the same conversation is not handled in parallel."""
    lock = await _get_chat_lock(chat_id)
    async with lock:
        return await coro


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


KEYWORDS_RESUME = {
    "bot", "volver", "volver al bot", "asistente", "menu", "menú",
    "reiniciar", "hablar con bot", "continuar", "inicio", "/start", "/bot"
}


def _is_resume_request(message: str) -> bool:
    norm = " ".join(message.strip().lower().split())
    return (
        norm in KEYWORDS_RESUME
        or "volver al bot" in norm
        or "hablar con el bot" in norm
        or "hablar con bot" in norm
    )


def _is_reset_request(message: str) -> bool:
    return message.strip().lower() in COMMANDS_RESET


async def _reset_conversation(phone_number: str) -> None:
    # Purgar checkpoints de PostgreSQL directamente para ese thread_id
    checkpointer = get_checkpointer_instance()
    if checkpointer:
        await checkpointer.clear_thread(phone_number)
    else:
        logger.warning(f"[Reset] No se encontró la instancia del checkpointer para {phone_number}")

    # Limpiar caché de conversación en dotnet_client
    dotnet_client.limpiar_cache_conversacion(phone_number)

    logger.info(f"[Reset] Memoria e historial reiniciados para {phone_number}")
    reset_msg = "🔄 Memoria reiniciada con éxito. ¡Hola! Soy el asistente virtual de Nexus Odonto. ¿En qué puedo colaborarte hoy?"
    await evolution_client.enviar_mensaje(phone_number, reset_msg)
    # Persistir mensaje de reinicio en Oracle DB
    asyncio.create_task(
        dotnet_client.registrar_mensaje(phone_number, "CHATBOT", reset_msg)
    )


KEYWORDS_ESCALATION = {
    "asesor", "humano", "recepcion", "recepcionista", "persona", "operador",
    "agente", "atencion humana", "atención humana", "soporte", "escalar",
    "escalamiento", "tomar caso", "alerta", "estado de alerta", "urgencia",
    "emergencia", "pasar a un humano", "pasar a humano", "hablar con humano",
    "hablar con persona", "atender persona"
}


def _is_escalation_request(message: str) -> bool:
    # La detección local garantiza que una petición explícita no dependa del LLM.
    normalized = " ".join(message.lower().split())
    if ESCALAMIENTO_RE.search(normalized):
        return True
    return any(kw in normalized for kw in KEYWORDS_ESCALATION)


async def _is_escalated(thread_id: str) -> bool:
    """Verifica si la conversación está escalada a un asesor humano o fue tomada desde el frontend.

    Consulta tanto el estado local de LangGraph como la API de .NET (ChatbotConversations y SupportTickets)
    para asegurar que si un asesor le da 'Tomar caso' en el frontend o se crea un ticket, el bot no responda más.
    """
    LID_MAPPING = {
        "233783743803574@lid": "573001112233@s.whatsapp.net",
        "573001112233": "233783743803574@lid",
        "573001112233@s.whatsapp.net": "233783743803574@lid",
        "189515549421795@lid": "573226688304@s.whatsapp.net",
        "573226688304": "189515549421795@lid",
        "573226688304@s.whatsapp.net": "189515549421795@lid",
        "57213628510462@lid": "573238891073@s.whatsapp.net",
        "573238891073": "57213628510462@lid",
        "573238891073@s.whatsapp.net": "57213628510462@lid",
    }
    raw_tid = str(thread_id).strip()
    canonical_tid = LID_MAPPING.get(raw_tid, raw_tid)
    clean_tid = re.sub(r"\D", "", canonical_tid)
    if len(clean_tid) > 10:
        clean_tid = clean_tid[-10:]

    # 1. Verificar estado local en LangGraph (PostgreSQL Checkpointer)
    is_graph_escalated = False
    try:
        state = await get_graph().aget_state(get_thread_config(thread_id))
        is_graph_escalated = state.values.get("conversation_status") == "ESCALADA"
    except Exception:
        is_graph_escalated = False

    # 2. Consultar .NET backend para verificar estado en DB (STATUS_ESCALADA / STATUS_ATENDIDA_HUMANO / empleado asignado / ticket activo)
    dotnet_is_escalated = False
    has_active_ticket = False
    matching_conv_ids: set[str] = set()

    try:
        convs = await dotnet_client.obtener_catalogo("ChatbotConversations") or []
        for c in convs:
            c_chat = str(c.get("chatIdentifier", "")).strip()
            c_canon = LID_MAPPING.get(c_chat, c_chat)
            c_digits = re.sub(r"\D", "", c_canon)
            if len(c_digits) > 10:
                c_digits = c_digits[-10:]

            # Coincidencia por identificador exacto, alias LID, o últimos 10 dígitos
            matches = (
                c_chat == raw_tid
                or c_canon == canonical_tid
                or (clean_tid and c_digits and clean_tid == c_digits)
            )

            if matches:
                c_id = str(c.get("id", "")).lower()
                if c_id:
                    matching_conv_ids.add(c_id)

                conv_status_id = str(c.get("conversationStatusId", "")).lower()
                assigned_emp = (
                    c.get("assignedEmployeeId")
                    or c.get("employeeId")
                    or c.get("assignedUserId")
                )

                if (
                    conv_status_id in (dotnet_client.STATUS_ESCALADA.lower(), dotnet_client.STATUS_ATENDIDA_HUMANO.lower())
                    or bool(assigned_emp)
                ):
                    dotnet_is_escalated = True

        if matching_conv_ids:
            tickets = await dotnet_client.obtener_catalogo("SupportTickets") or []
            STATUS_RESUELTO = "c0000000-0000-0000-0000-000000000004"
            STATUS_CERRADO = "c0000000-0000-0000-0000-000000000005"

            has_active_ticket = any(
                str(t.get("chatbotConversationId", "")).lower() in matching_conv_ids
                and str(t.get("ticketStatusId", "")).lower() not in (STATUS_RESUELTO, STATUS_CERRADO)
                for t in tickets
            )
    except Exception as e:
        logger.warning(f"[Webhook] No se pudo verificar estado de conversación en .NET para {thread_id}: {e}")

    # Si está escalada localmente, en .NET o tiene un ticket activo:
    if is_graph_escalated or dotnet_is_escalated or has_active_ticket:
        if not is_graph_escalated:
            logger.info(f"[Webhook] Conversación detectada como ESCALADA/ATENDIDA en .NET para {thread_id}. Sincronizando LangGraph.")
            try:
                config = get_thread_config(thread_id)
                await get_graph().aupdate_state(config, {"conversation_status": "ESCALADA"})
            except Exception:
                pass
        return True

    # Si en .NET TODAS las conversaciones para este paciente están en ACTIVA y ya no hay tickets ni atención de humano:
    if is_graph_escalated and not dotnet_is_escalated and not has_active_ticket:
        try:
            logger.info(f"[Webhook] Conversación fue restablecida a ACTIVA en .NET para {thread_id}. Sincronizando bot.")
            config = get_thread_config(thread_id)
            await get_graph().aupdate_state(config, {"conversation_status": "ACTIVA"})
        except Exception:
            pass
        return False

    return False


async def _escalate_conversation(thread_id: str, phone_number: str, message: str) -> bool:
    # 1. Asegurar que en .NET DB la conversación exista y su estado cambie a STATUS_ESCALADA
    conv_id = await dotnet_client.obtener_o_crear_conversacion(phone_number)
    if conv_id:
        await dotnet_client.actualizar_estado_conversacion(conv_id, dotnet_client.STATUS_ESCALADA)

    # 2. Registrar ticket de soporte en .NET API
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

    # 3. Actualizar el estado de la conversación local a ESCALADA en PostgreSQL
    try:
        config = get_thread_config(thread_id)
        await get_graph().aupdate_state(config, {"conversation_status": "ESCALADA"})
    except Exception as state_err:
        logger.warning(f"[BG] No se pudo actualizar estado local en checkpointer: {state_err}")
    # 4. Enviar mensaje de escalamiento al paciente
    await evolution_client.enviar_mensaje(phone_number, MENSAJE_ESCALAMIENTO)
    # 5. Persistir mensaje de escalamiento en Oracle DB
    asyncio.create_task(
        dotnet_client.registrar_mensaje(phone_number, "CHATBOT", MENSAJE_ESCALAMIENTO)
    )
    return True


async def _process_whatsapp_unsupported_media(numero_paciente: str, caption: str = "") -> None:
    """Gestiona la recepción de fotos, videos, documentos o archivos no procesables directamente."""
    try:
        if await _is_escalated(numero_paciente):
            if caption and _is_resume_request(caption):
                logger.info(f"[BG] Paciente solicita volver con el bot: {numero_paciente}")
                checkpointer = get_checkpointer_instance()
                for t in [numero_paciente, "573001112233@s.whatsapp.net", "233783743803574@lid"]:
                    try:
                        if checkpointer:
                            await checkpointer.clear_thread(t)
                        dotnet_client.limpiar_cache_conversacion(t)
                    except Exception:
                        pass

                msg_bienvenida = (
                    "👋🦷 *Nexus Odonto Asistente Virtual*\n\n"
                    "¡Hola de nuevo! He reactivado mi sistema para atenderte. ¿En qué puedo colaborarte hoy? 😊✨"
                )
                await evolution_client.enviar_mensaje(numero_paciente, msg_bienvenida)
                asyncio.create_task(
                    dotnet_client.registrar_mensaje(numero_paciente, "CHATBOT", msg_bienvenida)
                )
                return
            else:
                logger.info(f"[BG] Mensaje ignorado - conversación escalada: {numero_paciente}")
                return

        # Registrar el mensaje de medio enviado por el usuario en Oracle DB
        user_msg = f"[Archivo o medio adjunto: {caption}]" if caption else "[Archivo o medio adjunto]"
        asyncio.create_task(
            dotnet_client.registrar_mensaje(numero_paciente, "USUARIO", user_msg)
        )

        if caption:
            if _is_reset_request(caption):
                await _reset_conversation(numero_paciente)
                return
            if _is_escalation_request(caption):
                await _escalate_conversation(numero_paciente, numero_paciente, caption)
                return

        await evolution_client.enviar_mensaje(numero_paciente, MENSAJE_MEDIOS_NO_SOPORTADOS)
        asyncio.create_task(
            dotnet_client.registrar_mensaje(numero_paciente, "CHATBOT", MENSAJE_MEDIOS_NO_SOPORTADOS)
        )
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
            asyncio.create_task(
                dotnet_client.registrar_mensaje(numero_paciente, "CHATBOT", MENSAJE_ERROR_PROCESANDO_AUDIO)
            )
            return

        audio_bytes, mimetype = audio_info

        # 2. Transcribir con OpenAI Whisper
        texto_transcrito = await transcribir_audio(audio_bytes, mimetype)
        if not texto_transcrito:
            logger.warning(f"[BG] La transcripción de audio resultó vacía para {numero_paciente}")
            await evolution_client.enviar_mensaje(numero_paciente, MENSAJE_AUDIO_NO_ENTENDIDO)
            asyncio.create_task(
                dotnet_client.registrar_mensaje(numero_paciente, "CHATBOT", MENSAJE_AUDIO_NO_ENTENDIDO)
            )
            return

        logger.info(f"[BG] Audio de {numero_paciente} transcrito: '{texto_transcrito}'")

        # 3. Procesar el texto transcrito con el agente conversacional
        await _process_whatsapp_message(numero_paciente, texto_transcrito)

    except Exception as e:
        logger.error(f"[BG] Error procesando audio de {numero_paciente}: {e}", exc_info=True)
        await evolution_client.enviar_mensaje(numero_paciente, MENSAJE_ERROR_PROCESANDO_AUDIO)
        asyncio.create_task(
            dotnet_client.registrar_mensaje(numero_paciente, "CHATBOT", MENSAJE_ERROR_PROCESANDO_AUDIO)
        )


async def _process_whatsapp_message(numero_paciente: str, mensaje_texto: str) -> None:
    """Procesa el mensaje del paciente en segundo plano.

    Esta corutina se ejecuta desacoplada del ciclo request/response para que
    Evolution API reciba el HTTP 200 de inmediato y no genere un error de timeout.

    El flujo es directo: el mensaje va al grafo LangGraph sin requerir autenticación previa.
    El número de WhatsApp (numero_paciente) es el identificador de la conversación.
    """
    try:
        # 0. Anti-spam: descartar si el usuario está enviando demasiados mensajes
        if _is_spam(numero_paciente):
            if _should_warn_spam(numero_paciente):
                logger.warning(f"[Anti-spam] Demasiados mensajes de {numero_paciente}. Enviando advertencia.")
                spam_msg = (
                    "⚠️ Estás enviando muchos mensajes seguidos. "
                    "Por favor espera un momento antes de continuar. "
                    "Cuando estés listo, con gusto te atiendo. 😊"
                )
                asyncio.create_task(evolution_client.enviar_mensaje(numero_paciente, spam_msg))
            else:
                logger.info(f"[Anti-spam] Mensaje de {numero_paciente} descartado (spam silencioso).")
            return

        # Registrar de inmediato el mensaje entrante del usuario en la base de datos Oracle
        asyncio.create_task(
            dotnet_client.registrar_mensaje(
                chat_identifier=numero_paciente,
                rol="USUARIO",
                contenido=mensaje_texto,
            )
        )

        # 1. Comandos de reinicio de conversación
        if _is_reset_request(mensaje_texto):
            await _reset_conversation(numero_paciente)
            return

        # 2. Verificar si la conversación ya fue escalada a un asesor humano o está en atención
        if await _is_escalated(numero_paciente):
            if _is_resume_request(mensaje_texto):
                logger.info(f"[BG] Paciente solicita volver con el bot: {numero_paciente}")
                checkpointer = get_checkpointer_instance()
                for t in [numero_paciente, "573001112233@s.whatsapp.net", "233783743803574@lid"]:
                    try:
                        if checkpointer:
                            await checkpointer.clear_thread(t)
                        dotnet_client.limpiar_cache_conversacion(t)
                    except Exception:
                        pass

                msg_bienvenida = (
                    "👋 *Nexus Odonto Asistente Virtual*\n\n"
                    "¡Hola de nuevo! He reactivado mi sistema para atenderte. ¿En qué puedo colaborarte hoy? 😊🦷"
                )
                await evolution_client.enviar_mensaje(numero_paciente, msg_bienvenida)
                asyncio.create_task(
                    dotnet_client.registrar_mensaje(numero_paciente, "CHATBOT", msg_bienvenida)
                )
                return
            else:
                logger.info(f"[BG] Mensaje ignorado — conversación en atención humana o escalada: {numero_paciente}")
                return

        # 3. Detectar solicitud explícita de hablar con un asesor
        if _is_escalation_request(mensaje_texto):
            await _escalate_conversation(numero_paciente, numero_paciente, mensaje_texto)
            return

        config = get_thread_config(numero_paciente)

        # 4. Consultar si existe respuesta en el Caché Semántico (⚡ 0 tokens, < 50ms)
        cached_response = await buscar_en_cache(mensaje_texto)
        if cached_response:
            logger.info(f"[Semantic Cache] Respondiendo desde caché a {numero_paciente}")
            await evolution_client.enviar_mensaje(numero_paciente, cached_response)
            # Guardar respuesta del bot en base de datos Oracle
            asyncio.create_task(
                dotnet_client.registrar_mensaje(
                    chat_identifier=numero_paciente,
                    rol="CHATBOT",
                    contenido=cached_response,
                    rag_confidence=1.0,
                )
            )
            try:
                # Mantener sincronizado el historial de mensajes en PostgreSQL
                await get_graph().aupdate_state(
                    config,
                    {"messages": [HumanMessage(content=mensaje_texto), AIMessage(content=cached_response)]},
                )
            except Exception as hist_err:
                logger.debug(f"[Semantic Cache] No se pudo persistir mensaje cacheado en historial: {hist_err}")
            return

        # 5. Invocar el grafo LangGraph directamente (sin requerir sesión autenticada)
        invoke_input = {
            "messages": [HumanMessage(content=mensaje_texto)],
            "conversation_status": "ACTIVA",
            "rag_confidence": 1.0,
        }

        result = await get_graph().ainvoke(invoke_input, config)

        if result.get("conversation_status") == "ESCALADA":
            # Escalado inmediato (ej: triage de urgencia en emergency_check_node).
            # Si el nodo del grafo ya envió por WhatsApp (emergency_detected), no reenviar el AIMessage.
            already_sent_by_graph = bool(result.get("emergency_detected"))
            if not already_sent_by_graph:
                messages = result.get("messages", [])
                if messages:
                    last_message = messages[-1]
                    if isinstance(last_message, AIMessage) and last_message.content:
                        from app.core.llm_factory import extract_text_content
                        resp_urg = extract_text_content(last_message.content)
                        await evolution_client.enviar_mensaje(numero_paciente, resp_urg)
                        asyncio.create_task(
                            dotnet_client.registrar_mensaje(
                                chat_identifier=numero_paciente,
                                rol="CHATBOT",
                                contenido=resp_urg,
                                rag_confidence=result.get("rag_confidence", 1.0),
                            )
                        )
            if already_sent_by_graph:
                # Ticket + WhatsApp ya hechos en emergency_check_node; solo asegurar estado local.
                config = get_thread_config(numero_paciente)
                await get_graph().aupdate_state(config, {"conversation_status": "ESCALADA"})
                logger.info(
                    f"[BG] Emergencia ya notificada por el grafo para {numero_paciente}; "
                    "se omite reenvío de AIMessage y MENSAJE_ESCALAMIENTO."
                )
            else:
                await _escalate_conversation(numero_paciente, numero_paciente, mensaje_texto)
            return

        if result.get("rag_confidence", 1.0) < settings.rag_min_confidence:
            # No enviamos una respuesta posiblemente incorrecta: escalamos a recepción.
            if not await _escalate_conversation(numero_paciente, numero_paciente, mensaje_texto):
                await evolution_client.enviar_mensaje(numero_paciente, MENSAJE_FALLBACK_PACIENTE)
                asyncio.create_task(
                    dotnet_client.registrar_mensaje(
                        chat_identifier=numero_paciente,
                        rol="CHATBOT",
                        contenido=MENSAJE_FALLBACK_PACIENTE,
                    )
                )
            return

        # 6. Enviar la respuesta del bot al paciente por WhatsApp
        if result.get("conversation_status") == "BLOQUEADA":
            # Si fue bloqueada por el filtro de seguridad
            asyncio.create_task(
                dotnet_client.registrar_mensaje(
                    chat_identifier=numero_paciente,
                    rol="CHATBOT",
                    contenido="Solo puedo ayudarte con temas odontológicos de NexusOdonto",
                )
            )
        else:
            messages = result.get("messages", [])
            if messages:
                last_message = messages[-1]
                if isinstance(last_message, AIMessage) and last_message.content:
                    from app.core.llm_factory import extract_text_content
                    respuesta_texto = extract_text_content(last_message.content).strip()

                    # Salvaguarda: Nunca enviar mensajes de carga intermediarios como respuesta final al paciente
                    if respuesta_texto.startswith("[Consultando información") or respuesta_texto == "[Consultando información en el sistema...]":
                        logger.warning(f"[Webhook] Mensaje placeholder detectado ('{respuesta_texto}') para {numero_paciente}. Ajustando a respuesta asistida.")
                        respuesta_texto = (
                            "¡Con mucho gusto te ayudo a consultar los detalles de tus citas! 📋✨\n\n"
                            "Por favor indícame o confírmame tu número de cédula 🆔 para mostrártelos de inmediato en el sistema. 😊"
                        )

                    await evolution_client.enviar_mensaje(numero_paciente, respuesta_texto)
                    # Guardar respuesta del bot en base de datos Oracle
                    confidence = float(result.get("rag_confidence", 1.0))
                    asyncio.create_task(
                        dotnet_client.registrar_mensaje(
                            chat_identifier=numero_paciente,
                            rol="CHATBOT",
                            contenido=respuesta_texto,
                            rag_confidence=confidence,
                        )
                    )
                    # Guardar en Caché Semántico si la respuesta es informativa
                    asyncio.create_task(guardar_en_cache(mensaje_texto, respuesta_texto))
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
        # Salvaguarda: Si la conversación está en atención humana o escalada, NUNCA enviar fallback a WhatsApp
        try:
            if await _is_escalated(numero_paciente):
                logger.info(f"[Webhook] Mensaje de contingencia suprimido porque {numero_paciente} está escalado/en atención.")
                return
        except Exception:
            pass

        # Si el flujo falló por completo (ej. indisponibilidad de red), se envía mensaje amigable de contingencia
        try:
            fallback_msg = (
                "¡Hola! 👋 En este momento estamos experimentando una alta demanda en nuestro sistema digital.\n\n"
                "Para consultar disponibilidad, agendar citas o atender cualquier duda, puedes comunicarte directamente con nuestro equipo:\n"
                "📞 *WhatsApp / Teléfono:* +57 324 6030217\n"
                "📍 *Consultorio:* Cr 24 #35-12, Santander\n"
                "⏰ *Horario:* Lunes a Sábado de 8:00 AM a 6:00 PM\n\n"
                "¡Con gusto te atenderemos! 😊🦷"
            )
            await evolution_client.enviar_mensaje(numero_paciente, fallback_msg)
            asyncio.create_task(
                dotnet_client.registrar_mensaje(
                    chat_identifier=numero_paciente,
                    rol="CHATBOT",
                    contenido=fallback_msg,
                )
            )
        except Exception as e:
            logger.error(f"[Webhook] Error enviando mensaje de contingencia a {numero_paciente}: {e}")


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

            message_id = getattr(data.key, "id", None) if data.key else None
            if _is_duplicate_message_id(message_id):
                logger.info(f"[Webhook] Mensaje duplicado ignorado (key.id={message_id})")
                return {"status": "ignored", "reason": "duplicate_message_id"}

            # Extraer número del paciente completo (con el sufijo de whatsapp)
            remote_jid = data.key.remoteJid if (data.key and data.key.remoteJid) else ""
            if not remote_jid:
                return {"status": "ignored", "reason": "no_remote_jid"}

            # Si remoteJid trae un LID interno de WhatsApp (@lid), buscar si participant o sender trae el teléfono real
            participant = getattr(data.key, "participant", None) or getattr(data, "sender", None)
            if "@lid" in str(remote_jid) and participant and "@s.whatsapp.net" in str(participant):
                numero_paciente = str(participant)
            else:
                numero_paciente = remote_jid

            # Mapeo canónico de LIDs conocidos para evitar duplicidad de conversaciones
            LID_MAPPING = {
                "233783743803574@lid": "573001112233@s.whatsapp.net",
                "189515549421795@lid": "573226688304@s.whatsapp.net",
                "57213628510462@lid": "573238891073@s.whatsapp.net",
            }
            if str(numero_paciente).strip() in LID_MAPPING:
                numero_paciente = LID_MAPPING[str(numero_paciente).strip()]

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
                    _with_chat_lock(
                        numero_paciente,
                        _process_whatsapp_audio(numero_paciente, raw_payload_data, unwrapped_message),
                    )
                )
            elif is_unsupported_media:
                caption = ""
                for k in ["imageMessage", "videoMessage", "documentMessage"]:
                    if k in unwrapped_message and isinstance(unwrapped_message[k], dict):
                        caption = unwrapped_message[k].get("caption", "") or caption

                logger.info(f"[Webhook] Medio no soportado recibido de {numero_paciente} (caption: '{caption}'). Encolando aviso...")
                asyncio.create_task(
                    _with_chat_lock(
                        numero_paciente,
                        _process_whatsapp_unsupported_media(numero_paciente, caption),
                    )
                )
            else:
                # Extraer texto del mensaje (incluye respuestas de botones/listas)
                mensaje_texto = ""

                # 1. Intentar extraer selección de botón/lista interactiva
                interactive_selection = extract_interactive_selection(unwrapped_message)
                if interactive_selection:
                    mensaje_texto = interactive_selection
                # 2. Texto de conversación normal
                elif "conversation" in unwrapped_message and unwrapped_message["conversation"]:
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
                        _with_chat_lock(
                            numero_paciente,
                            _process_whatsapp_message(numero_paciente, mensaje_texto),
                        )
                    )
                else:
                    logger.warning(f"[Webhook] Mensaje no reconocido o vacío de {numero_paciente}: {unwrapped_message}")

        # Evolution API recibe HTTP 200 de inmediato, sin esperar el procesamiento pesado
        return {"status": "queued"}

    except Exception as e:
        logger.error(f"[Webhook Error] Fallo al procesar el webhook: {e}", exc_info=True)
        return {"status": "error", "message": "Error procesando el payload"}