"""Procesador central de mensajes de chat en segundo plano para Nexus Odonto.
Ejecuta la orquestación del grafo LangGraph, control de TTL de sesión (15 min),
caché semántico, escalamiento a asesores humanos y auditoría a .NET / Oracle.
"""

import os
import time
import re
import asyncio
import logging
from typing import Any, Optional
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from app.clients.evolution_client import evolution_client
from app.clients.dotnet_client import dotnet_client
from app.core.config import settings
from app.core.llm_concurrency import (
    GraphDeadlineExceeded,
    LLMCallTimeoutError,
    graph_queue_wait_total,
    run_with_graph_deadline,
)
from app.graph.builder import get_graph
from app.session.memory_store import get_thread_config
from app.session.postgres_checkpointer import get_checkpointer_instance
from app.services.audio_service import (
    extraer_bytes_audio,
    transcribir_audio,
    MENSAJE_AUDIO_NO_ENTENDIDO,
    MENSAJE_ERROR_PROCESANDO_AUDIO,
)
from app.services.semantic_cache import buscar_en_cache
from app.services.inactivity_service import inactivity_service
from app.core.llm_factory import extract_text_content

logger = logging.getLogger(__name__)

# Mensajes institucionales de contingencia y medios
MENSAJE_FALLBACK_PACIENTE = (
    "En este momento presentamos intermitencias temporales en el servicio. "
    "Por favor, intenta nuevamente en unos momentos. ¡Disculpa las molestias!"
)

_DEFAULT_WEB_PORTAL_URL = "https://nexusodonto.chatcampuslands.com/login"
_PRIMERA_VEZ_PORTAL_FLAG = "[primera_vez_portal]"
_PORTAL_HINT_MARKERS = (
    "plataforma virtual",
    "portal del paciente",
    "plataforma web",
    "nexusodonto.chatcampuslands.com",
)
_CREDENTIALS_HINT_MARKERS = (
    "usuario es tu número de cédula",
    "contraseña también es tu número de cédula",
)
# Portal ONLY after these tools succeed THIS turn — never mid-booking / ask-cédula / slots.
_PORTAL_ELIGIBLE_TOOLS = frozenset(
    {
        "agendar_cita_tool",
        "modificar_cita_tool",
        "cancelar_cita_tool",
        "consultar_cita_por_cedula_tool",
    }
)
_MUTATION_SUCCESS_MARKERS = (
    "cita confirmada con éxito",
    "confirmada con éxito",
    "reprogramada exitosamente",
    "cancelada exitosamente",
)
# Consult list with real appointments — not "no tienes citas" empty results.
_CONSULT_LIST_MARKERS = (
    "próximas citas programadas",
    "citas registradas para la cédula",
    "historial reciente",
)


def _tool_messages_this_turn(messages: list) -> list:
    """Only ToolMessages after the latest HumanMessage (current turn)."""
    if not messages:
        return []
    start = 0
    for i, msg in enumerate(messages):
        if isinstance(msg, HumanMessage):
            start = i + 1
    return [m for m in messages[start:] if isinstance(m, ToolMessage)]


def _portal_eligibility_from_tools(tool_msgs: list) -> tuple[bool, bool]:
    """
    Returns (should_remind_portal, is_primera_vez).
    Primera vez = flag from agendar success (new account / first booking).
    """
    should_remind = False
    is_primera_vez = False
    for msg in tool_msgs:
        tool_name = (getattr(msg, "name", None) or "").strip()
        if tool_name not in _PORTAL_ELIGIBLE_TOOLS:
            continue
        content = str(msg.content or "")
        content_low = content.lower()
        if tool_name == "consultar_cita_por_cedula_tool":
            if any(m in content_low for m in _CONSULT_LIST_MARKERS):
                should_remind = True
            continue
        if any(m in content_low for m in _MUTATION_SUCCESS_MARKERS):
            should_remind = True
            if _PRIMERA_VEZ_PORTAL_FLAG in content_low or "usuario es tu número de cédula" in content_low:
                is_primera_vez = True
    return should_remind, is_primera_vez


def _ensure_portal_reminder_after_booking(respuesta: str, messages: list) -> str:
    """
    Reinject portal ONLY if this turn had a successful agendar/modificar/cancelar
    or a consult that listed appointments. Never append from older turns.
    First-time credentials tip (rule only, no digits) when flagged by agendar.
    """
    text = (respuesta or "").strip()
    if not text:
        return respuesta

    # Never leak the machine flag to WhatsApp
    text = text.replace(_PRIMERA_VEZ_PORTAL_FLAG, "").replace(_PRIMERA_VEZ_PORTAL_FLAG.upper(), "")
    text = text.strip()

    should_remind, is_primera_vez = _portal_eligibility_from_tools(
        _tool_messages_this_turn(messages)
    )
    if not should_remind:
        return text

    low = text.lower()
    has_portal = any(marker in low for marker in _PORTAL_HINT_MARKERS)
    has_creds = any(marker in low for marker in _CREDENTIALS_HINT_MARKERS)
    web_url = os.getenv("WEB_PORTAL_URL", _DEFAULT_WEB_PORTAL_URL).strip() or _DEFAULT_WEB_PORTAL_URL

    if is_primera_vez and (not has_portal or not has_creds):
        # Drop a bare portal block if present without credentials tip, then append full tip.
        tip = (
            "\n\nTambién puedes ver tu cita en la *plataforma virtual*:\n"
            f"{web_url}\n"
            "Tu *usuario* es tu número de cédula y la *contraseña* también es tu número de cédula "
            "(acceso temporal). Al entrar, cámbiala por tu seguridad; "
            "desde aquí no podemos modificar contraseñas."
        )
        if has_portal and not has_creds:
            # Append credentials rule only
            tip = (
                "\n\nTu *usuario* es tu número de cédula y la *contraseña* también es tu número de cédula "
                "(acceso temporal). Al entrar, cámbiala por tu seguridad; "
                "desde aquí no podemos modificar contraseñas."
            )
        return text.rstrip() + tip

    if has_portal:
        return text

    reminder = (
        "\n\nTambién puedes consultar tu cita en la *plataforma virtual*:\n"
        f"{web_url}"
    )
    return text.rstrip() + reminder

# Kept for logs/docs only — never send this to WhatsApp (users want real replies, not retry spam).
MENSAJE_GRAPH_TIMEOUT = (
    "En este momento hay bastante movimiento y tu consulta está tardando un poco. "
    "¿Me reenvías el mensaje en un momentito? Con gusto te ayudo."
)

MENSAJE_ESCALAMIENTO = (
    "Entiendo. Un asesor de la clínica revisará tu solicitud y te contactará pronto."
)

MENSAJE_MEDIOS_NO_SOPORTADOS = (
    "Por aquí no alcanzo a ver fotos, videos ni documentos.\n\n"
    "¿Me cuentas por texto o nota de voz qué necesitas "
    "(síntoma, tratamiento u orden médica)? Así te ayudo mejor."
)

COMMANDS_RESET = {"/clear", "/reset", "/reiniciar", "/limpiar", "/start", "/inicio"}

KEYWORDS_RESUME = {
    "bot", "volver", "volver al bot", "asistente", "menu", "menú",
    "reiniciar", "hablar con bot", "continuar", "inicio", "/start", "/bot"
}

KEYWORDS_ESCALATION = {
    "asesor", "humano", "recepcion", "recepcionista", "persona", "operador",
    "agente", "atencion humana", "atención humana", "soporte", "escalar",
    "escalamiento", "tomar caso", "alerta", "estado de alerta", "urgencia",
    "emergencia", "pasar a un humano", "pasar a humano", "hablar con humano",
    "hablar con persona", "atender persona"
}

ESCALAMIENTO_RE = re.compile(
    r"\b(hablar|comunicarme|contactar|atenderme)\b.*\b(persona|humano|asesor|recepcionista)\b|"
    r"\b(persona|humano|asesor|recepcionista)\b.*\b(hablar|comunicarme|contactar|atenderme)\b",
    re.IGNORECASE,
)

_CHAT_LOCKS: dict[str, asyncio.Lock] = {}
_CHAT_LOCKS_GUARD = asyncio.Lock()
_USER_LAST_ACTIVE: dict[str, float] = {}


async def get_chat_lock(chat_id: str) -> asyncio.Lock:
    async with _CHAT_LOCKS_GUARD:
        lock = _CHAT_LOCKS.get(chat_id)
        if lock is None:
            lock = asyncio.Lock()
            _CHAT_LOCKS[chat_id] = lock
        return lock


async def with_chat_lock(chat_id: str, coro):
    """Serializa el procesamiento por chat para evitar concurrencias simultáneas en un mismo hilo."""
    lock = await get_chat_lock(chat_id)
    async with lock:
        return await coro


def is_reset_request(message: str) -> bool:
    return message.strip().lower() in COMMANDS_RESET


def is_resume_request(message: str) -> bool:
    norm = " ".join(message.strip().lower().split())
    return (
        norm in KEYWORDS_RESUME
        or "volver al bot" in norm
        or "hablar con el bot" in norm
        or "hablar con bot" in norm
    )


def is_escalation_request(message: str) -> bool:
    normalized = " ".join(message.lower().split())
    if ESCALAMIENTO_RE.search(normalized):
        return True
    return any(kw in normalized for kw in KEYWORDS_ESCALATION)


async def reset_conversation(phone_number: str) -> None:
    """Purga los checkpoints de PostgreSQL y reinicia la memoria conversacional."""
    checkpointer = get_checkpointer_instance()
    if checkpointer:
        await checkpointer.clear_thread(phone_number)
    else:
        logger.warning(f"[Reset] No se encontró checkpointer para {phone_number}")

    dotnet_client.limpiar_cache_conversacion(phone_number)
    try:
        from app.services.booking_flow import clear_awaiting_booking_identity
        clear_awaiting_booking_identity(phone_number)
    except Exception:
        pass
    logger.info(f"[Reset] Memoria e historial reiniciados para {phone_number}")
    # Human persona: never say "asistente virtual" / bot / sistema on WhatsApp.
    reset_msg = (
        "¡Hola de nuevo! 👋 Estoy aquí para ayudarte en *Nexus Odonto*. "
        "¿En qué te puedo colaborar hoy? 😊🦷"
    )
    await evolution_client.enviar_mensaje(phone_number, reset_msg)
    asyncio.create_task(
        dotnet_client.registrar_mensaje(phone_number, "CHATBOT", reset_msg)
    )


_ESCALATION_CACHE: dict[str, tuple[bool, float]] = {}
_ESCALATION_CACHE_TTL = 8.0


async def is_escalated(thread_id: str) -> bool:
    """Verifica si la conversación está escalada a un asesor humano o fue tomada desde el frontend."""
    from app.api.routes.agent_handoff import esta_recien_reactivada
    if esta_recien_reactivada(thread_id):
        logger.info(f"[Processor] Conversación {thread_id} fue reactivada recientemente por asesor. Bot activo.")
        return False

    from app.services.whatsapp_identity import obtener_telefono_canonico
    raw_tid = str(thread_id).strip()
    canonical_tid = obtener_telefono_canonico(raw_tid)
    clean_tid = re.sub(r"\D", "", canonical_tid)
    if len(clean_tid) > 10:
        clean_tid = clean_tid[-10:]

    is_graph_escalated = False
    try:
        state = await get_graph().aget_state(get_thread_config(thread_id))
        is_graph_escalated = state.values.get("conversation_status") == "ESCALADA"
    except Exception:
        is_graph_escalated = False

    from app.infra.external.dotnet.tickets_api import tickets_api

    cache_key = clean_tid or raw_tid
    now_mono = time.monotonic()

    def _conv_is_escalated(c: dict) -> bool:
        conv_status_id = str(c.get("conversationStatusId", "")).lower()
        assigned_emp = (
            c.get("assignedEmployeeId")
            or c.get("employeeId")
            or c.get("assignedUserId")
        )
        return (
            conv_status_id in (dotnet_client.STATUS_ESCALADA.lower(), dotnet_client.STATUS_ATENDIDA_HUMANO.lower())
            or bool(assigned_emp)
        )

    dotnet_is_escalated = False
    cached_esc = _ESCALATION_CACHE.get(cache_key)
    if cached_esc and (now_mono - cached_esc[1]) < _ESCALATION_CACHE_TTL:
        dotnet_is_escalated = cached_esc[0]
    else:
        try:
            cached_id = tickets_api.peek_cached_conversation_id(thread_id)
            if cached_id:
                ctx = await tickets_api.obtener_contexto_conversacion(cached_id)
                if isinstance(ctx, dict):
                    dotnet_is_escalated = _conv_is_escalated(ctx)
            else:
                convs = await tickets_api._list_conversations() or []
                for c in convs:
                    c_chat = str(c.get("chatIdentifier", "")).strip()
                    c_canon = obtener_telefono_canonico(c_chat)
                    c_digits = re.sub(r"\D", "", c_canon)
                    if len(c_digits) > 10:
                        c_digits = c_digits[-10:]
                    matches = (
                        c_chat == raw_tid
                        or c_canon == canonical_tid
                        or (clean_tid and c_digits and clean_tid == c_digits)
                    )
                    if matches and _conv_is_escalated(c):
                        dotnet_is_escalated = True
                        break
            _ESCALATION_CACHE[cache_key] = (dotnet_is_escalated, now_mono)
        except Exception as e:
            logger.warning(f"[Processor] Error verificando estado en .NET para {thread_id}: {e}")

    if is_graph_escalated or dotnet_is_escalated:
        if not is_graph_escalated and dotnet_is_escalated:
            try:
                config = get_thread_config(thread_id)
                await get_graph().aupdate_state(config, {"conversation_status": "ESCALADA"})
            except Exception:
                pass
        return True

    if is_graph_escalated and not dotnet_is_escalated:
        try:
            config = get_thread_config(thread_id)
            await get_graph().aupdate_state(config, {"conversation_status": "ACTIVA"})
        except Exception:
            pass
        return False

    return False


async def escalate_conversation(thread_id: str, phone_number: str, message: str) -> bool:
    """Escala la conversación a atención humana creando ticket y actualizando estado en .NET."""
    from app.api.routes.agent_handoff import desmarcar_reactivada
    desmarcar_reactivada(thread_id)
    desmarcar_reactivada(phone_number)

    conv_id = await dotnet_client.obtener_o_crear_conversacion(phone_number)
    if conv_id:
        await dotnet_client.actualizar_estado_conversacion(conv_id, dotnet_client.STATUS_ESCALADA)

    await dotnet_client.crear_ticket_soporte(
        telefono=phone_number,
        motivo="SOLICITUD_USUARIO" if is_escalation_request(message) else "BAJA_CONFIANZA_RAG",
        prioridad="MEDIA"
    )

    try:
        config = get_thread_config(thread_id)
        await get_graph().aupdate_state(config, {"conversation_status": "ESCALADA"})
    except Exception as state_err:
        logger.warning(f"[Processor] Error actualizando estado en checkpointer: {state_err}")

    inactivity_service.cancel(thread_id)
    inactivity_service.cancel(phone_number)

    await evolution_client.enviar_mensaje(phone_number, MENSAJE_ESCALAMIENTO)
    asyncio.create_task(
        dotnet_client.registrar_mensaje(phone_number, "CHATBOT", MENSAJE_ESCALAMIENTO)
    )
    return True


async def registrar_mensaje_asesor(numero_paciente: str, mensaje_texto: str) -> None:
    """Registra en .NET un mensaje enviado manualmente por el asesor desde el celular o web."""
    try:
        logger.info(f"[fromMe-Humano] Registrando respuesta del asesor para {numero_paciente}: '{mensaje_texto}'")
        await dotnet_client.registrar_mensaje(
            chat_identifier=numero_paciente,
            rol="ASESOR",
            contenido=mensaje_texto,
        )
        # Mantener la conversación en atención humana (Uso Manual) para que el bot no interfiera
        conv_id = await dotnet_client.obtener_o_crear_conversacion(numero_paciente)
        if conv_id:
            await dotnet_client.actualizar_estado_conversacion(conv_id, dotnet_client.STATUS_ATENDIDA_HUMANO)
        try:
            config = get_thread_config(numero_paciente)
            await get_graph().aupdate_state(
                config,
                {
                    "messages": [AIMessage(content=f"[Asesor]: {mensaje_texto}")],
                    "conversation_status": "ESCALADA",
                },
            )
        except Exception:
            pass
    except Exception as e:
        logger.warning(f"[fromMe-Humano] No se pudo registrar mensaje del asesor: {e}")


async def _silent_resume_from_escalation(numero_paciente: str) -> None:
    """Reactivate bot after escalation without WhatsApp/DB bubbles that reveal bot↔human switch."""
    from app.api.routes.agent_handoff import marcar_conversacion_reactivada
    from app.services.whatsapp_identity import obtener_telefono_canonico, obtener_destino_envio

    checkpointer = get_checkpointer_instance()
    canon = obtener_telefono_canonico(numero_paciente)
    targets_clear = {numero_paciente, canon}
    dest = obtener_destino_envio(numero_paciente)
    if dest:
        targets_clear.add(dest)
    for t in targets_clear:
        try:
            marcar_conversacion_reactivada(t)
            if checkpointer:
                await checkpointer.clear_thread(t)
            dotnet_client.limpiar_cache_conversacion(t)
        except Exception:
            pass
    try:
        conv_id = await dotnet_client.obtener_o_crear_conversacion(numero_paciente)
        if conv_id:
            await dotnet_client.actualizar_estado_conversacion(conv_id, dotnet_client.STATUS_ACTIVA)
        config = get_thread_config(numero_paciente)
        await get_graph().aupdate_state(config, {"conversation_status": "ACTIVA"})
    except Exception as e:
        logger.warning(f"[Processor] Silent resume state sync failed for {numero_paciente}: {e}")


async def process_whatsapp_unsupported_media(numero_paciente: str, caption: str = "") -> None:
    """Gestiona la recepción de archivos o fotos no procesables directamente."""
    try:
        if await is_escalated(numero_paciente):
            if caption and is_resume_request(caption):
                logger.info(f"[Processor] Paciente solicita volver con el bot (sin aviso WA): {numero_paciente}")
                await _silent_resume_from_escalation(numero_paciente)
                return
            return

        user_msg = f"[Archivo o medio adjunto: {caption}]" if caption else "[Archivo o medio adjunto]"
        asyncio.create_task(dotnet_client.registrar_mensaje(numero_paciente, "USUARIO", user_msg))

        if caption:
            if is_reset_request(caption):
                await reset_conversation(numero_paciente)
                return
            if is_escalation_request(caption):
                await escalate_conversation(numero_paciente, numero_paciente, caption)
                return

        await evolution_client.enviar_mensaje(numero_paciente, MENSAJE_MEDIOS_NO_SOPORTADOS)
        asyncio.create_task(dotnet_client.registrar_mensaje(numero_paciente, "CHATBOT", MENSAJE_MEDIOS_NO_SOPORTADOS))
    except Exception as e:
        logger.error(f"[Processor] Error respondiendo medio no soportado a {numero_paciente}: {e}", exc_info=True)


async def process_whatsapp_audio(numero_paciente: str, raw_payload_data: dict, raw_message: dict, push_name: str = "") -> None:
    """Descarga, transcribe con Whisper y procesa notas de voz de WhatsApp."""
    try:
        if await is_escalated(numero_paciente):
            return

        # Simular presencia "Grabando audio..." o "Escribiendo..." mientras se transcribe y procesa
        asyncio.create_task(evolution_client.enviar_presencia(numero_paciente, "recording"))

        audio_info = await extraer_bytes_audio(raw_payload_data, raw_message)
        if not audio_info:
            await evolution_client.enviar_mensaje(numero_paciente, MENSAJE_ERROR_PROCESANDO_AUDIO)
            asyncio.create_task(dotnet_client.registrar_mensaje(numero_paciente, "CHATBOT", MENSAJE_ERROR_PROCESANDO_AUDIO))
            return

        audio_bytes, mimetype = audio_info
        texto_transcrito = await transcribir_audio(audio_bytes, mimetype)
        if not texto_transcrito:
            await evolution_client.enviar_mensaje(numero_paciente, MENSAJE_AUDIO_NO_ENTENDIDO)
            asyncio.create_task(dotnet_client.registrar_mensaje(numero_paciente, "CHATBOT", MENSAJE_AUDIO_NO_ENTENDIDO))
            return

        logger.info(f"[Audio] Nota de voz de {numero_paciente} transcrita: '{texto_transcrito}'")
        await process_whatsapp_message(numero_paciente, texto_transcrito, push_name=push_name)
    except Exception as e:
        logger.error(f"[Audio] Error procesando audio de {numero_paciente}: {e}", exc_info=True)
        await evolution_client.enviar_mensaje(numero_paciente, MENSAJE_ERROR_PROCESANDO_AUDIO)
        asyncio.create_task(dotnet_client.registrar_mensaje(numero_paciente, "CHATBOT", MENSAJE_ERROR_PROCESANDO_AUDIO))


async def process_whatsapp_message(
    numero_paciente: str,
    mensaje_texto: str,
    push_name: str = "",
    registrar_usuario_db: bool = False,
) -> None:
    """Procesa el mensaje consolidado del paciente a través del grafo LangGraph."""
    spans: dict[str, float] = {}
    t_total = time.perf_counter()
    try:
        # Notificar en vivo a WhatsApp que el bot está redactando la respuesta
        asyncio.create_task(evolution_client.enviar_presencia(numero_paciente, "composing"))

        if registrar_usuario_db:
            asyncio.create_task(
                dotnet_client.registrar_mensaje(
                    chat_identifier=numero_paciente,
                    rol="USUARIO",
                    contenido=mensaje_texto,
                )
            )

        if is_reset_request(mensaje_texto):
            await reset_conversation(numero_paciente)
            return

        from app.services.chat.chat_orchestrator import ChatOrchestrator
        if ChatOrchestrator.is_nonsense_or_gibberish(mensaje_texto):
            resp_gibberish = (
                "No te entendí bien. ¿Me lo dices otra vez con palabras? "
                "Por ejemplo: agendar, precios o servicios."
            )
            t_send = time.perf_counter()
            await evolution_client.enviar_mensaje(numero_paciente, resp_gibberish)
            spans["send"] = time.perf_counter() - t_send
            asyncio.create_task(
                dotnet_client.registrar_mensaje(
                    chat_identifier=numero_paciente,
                    rol="CHATBOT",
                    contenido=resp_gibberish,
                )
            )
            return

        t_esc = time.perf_counter()
        escalated = await is_escalated(numero_paciente)
        spans["escalate_check"] = time.perf_counter() - t_esc

        if escalated:
            if is_resume_request(mensaje_texto):
                logger.info(f"[Processor] Paciente solicita volver con el bot (sin aviso WA): {numero_paciente}")
                await _silent_resume_from_escalation(numero_paciente)
                return
            else:
                return

        if is_escalation_request(mensaje_texto):
            await escalate_conversation(numero_paciente, numero_paciente, mensaje_texto)
            return

        config = get_thread_config(numero_paciente)

        # Control de expiración de sesión (15 min / session_ttl_seconds)
        now_ts = time.time()
        last_active = _USER_LAST_ACTIVE.get(numero_paciente)
        session_expired = False
        checkpointer = get_checkpointer_instance()

        if last_active is not None and (now_ts - last_active) > settings.session_ttl_seconds:
            session_expired = True
        elif checkpointer:
            try:
                segundos_inactivo = await checkpointer.obtener_segundos_inactividad(numero_paciente)
                if segundos_inactivo is not None and segundos_inactivo > settings.session_ttl_seconds:
                    session_expired = True
                elif segundos_inactivo is None and last_active is None:
                    # Sin fila en conversation_sessions tras restart/sweep: los checkpoints
                    # huérfanos en Postgres seguirían cargando memoria antigua si no purgamos.
                    session_expired = True
            except Exception:
                pass

        if session_expired:
            logger.info(f"[TTL Purge] Sesión de 15m expirada para {numero_paciente}. Purgando memoria...")
            from app.services.whatsapp_identity import obtener_telefono_canonico, obtener_destino_envio
            canon = obtener_telefono_canonico(numero_paciente)
            targets_clear = {numero_paciente, canon}
            dest = obtener_destino_envio(numero_paciente)
            if dest:
                targets_clear.add(dest)
            if checkpointer:
                for t in targets_clear:
                    try:
                        await checkpointer.clear_thread(t)
                    except Exception:
                        pass
            for t in targets_clear:
                try:
                    dotnet_client.limpiar_cache_conversacion(t)
                except Exception:
                    pass
            _USER_LAST_ACTIVE.pop(numero_paciente, None)
            inactivity_service.cancel(numero_paciente)
            try:
                from app.services.booking_flow import clear_awaiting_booking_identity
                clear_awaiting_booking_identity(numero_paciente)
            except Exception:
                pass

        _USER_LAST_ACTIVE[numero_paciente] = now_ts
        if checkpointer:
            try:
                await checkpointer.actualizar_actividad(numero_paciente)
            except Exception:
                pass

        # Booking continuity: reuse cédula+nombre from recent history / pending flag.
        from app.services.booking_flow import (
            build_ask_service_response,
            clear_awaiting_booking_identity,
            extract_identity_from_history_newest_first,
            is_awaiting_booking_identity,
            is_booking_start_intent,
            mark_awaiting_booking_identity,
            parse_identity_from_text,
            prev_asked_for_booking_identity,
        )

        history_msgs: list = []
        prev_ai_text = ""
        try:
            snap = await get_graph().aget_state(config)
            history_msgs = list((snap.values or {}).get("messages") or [])
            for m in reversed(history_msgs):
                if isinstance(m, AIMessage) and m.content:
                    prev_ai_text = str(m.content)
                    break
        except Exception as hist_err:
            logger.debug("[Processor] No se pudo leer historial para booking: %s", hist_err)

        hist_identity = extract_identity_from_history_newest_first(history_msgs)
        msg_identity = parse_identity_from_text(mensaje_texto)
        known_cedula = msg_identity.cedula or hist_identity.cedula
        known_nombre = msg_identity.nombre or hist_identity.nombre
        awaiting_id = is_awaiting_booking_identity(numero_paciente) or prev_asked_for_booking_identity(
            prev_ai_text
        )

        # Caché semántico (⚡ 0 tokens) — booking-aware
        t_cache = time.perf_counter()
        cached_response = await buscar_en_cache(
            mensaje_texto,
            known_cedula=known_cedula,
            known_nombre=known_nombre,
            awaiting_booking_identity=awaiting_id,
        )
        spans["cache_lookup"] = time.perf_counter() - t_cache
        if cached_response:
            # Track booking identity step vs continue-to-service
            asks_identity = (
                "número de cédula" in cached_response.lower()
                or "numero de cedula" in cached_response.lower()
            ) and "nombre completo" in cached_response.lower()
            asks_service = "qué tratamiento o servicio" in cached_response.lower() or (
                "tratamiento o servicio" in cached_response.lower()
            )
            if asks_identity and is_booking_start_intent(mensaje_texto):
                mark_awaiting_booking_identity(numero_paciente)
            elif asks_service or (awaiting_id and msg_identity.cedula):
                clear_awaiting_booking_identity(numero_paciente)

            logger.info(f"[Semantic Cache] Respondiendo a {numero_paciente} desde caché.")
            t_send = time.perf_counter()
            await evolution_client.enviar_mensaje(numero_paciente, cached_response)
            spans["send"] = time.perf_counter() - t_send
            asyncio.create_task(
                dotnet_client.registrar_mensaje(
                    chat_identifier=numero_paciente,
                    rol="CHATBOT",
                    contenido=cached_response,
                )
            )
            try:
                await get_graph().aupdate_state(
                    config,
                    {
                        "messages": [
                            HumanMessage(content=mensaje_texto),
                            AIMessage(content=cached_response),
                        ],
                        "user_context": {
                            "nombre": known_nombre or "",
                            "primer_nombre": (known_nombre or "").split()[0] if known_nombre else "",
                            "cedula": known_cedula,
                            "is_registered": bool(known_cedula),
                            "phone": numero_paciente,
                            "push_name": push_name.strip() if push_name else "",
                        },
                    },
                )
            except Exception:
                pass
            inactivity_service.touch(numero_paciente, settings.session_ttl_seconds)
            spans["total"] = time.perf_counter() - t_total
            logger.info(
                f"[Latency] phone={numero_paciente} path=cache "
                + " ".join(f"{k}={v:.3f}s" for k, v in spans.items())
            )
            return

        # Extra safety: if cache missed but we just got identity mid-booking, do not greet.
        if awaiting_id and msg_identity.cedula and (msg_identity.nombre or known_nombre):
            cached_response = build_ask_service_response(
                nombre=msg_identity.nombre or known_nombre
            )
            clear_awaiting_booking_identity(numero_paciente)
            logger.info(
                "[BookingFlow] Identity mid-booking → ask service (cache miss fallback) phone=%s",
                numero_paciente,
            )
            t_send = time.perf_counter()
            await evolution_client.enviar_mensaje(numero_paciente, cached_response)
            spans["send"] = time.perf_counter() - t_send
            asyncio.create_task(
                dotnet_client.registrar_mensaje(
                    chat_identifier=numero_paciente,
                    rol="CHATBOT",
                    contenido=cached_response,
                )
            )
            try:
                await get_graph().aupdate_state(
                    config,
                    {
                        "messages": [
                            HumanMessage(content=mensaje_texto),
                            AIMessage(content=cached_response),
                        ],
                        "user_context": {
                            "nombre": msg_identity.nombre or known_nombre or "",
                            "primer_nombre": (
                                (msg_identity.nombre or known_nombre or "").split()[0]
                                if (msg_identity.nombre or known_nombre)
                                else ""
                            ),
                            "cedula": msg_identity.cedula or known_cedula,
                            "is_registered": True,
                            "phone": numero_paciente,
                            "push_name": push_name.strip() if push_name else "",
                        },
                    },
                )
            except Exception:
                pass
            inactivity_service.touch(numero_paciente, settings.session_ttl_seconds)
            spans["total"] = time.perf_counter() - t_total
            logger.info(
                f"[Latency] phone={numero_paciente} path=booking_identity "
                + " ".join(f"{k}={v:.3f}s" for k, v in spans.items())
            )
            return

        # Contexto del paciente: only from what the user already wrote (Habeas Data)
        user_context = {
            "nombre": known_nombre or "",
            "primer_nombre": (known_nombre or "").split()[0] if known_nombre else "",
            "cedula": known_cedula,
            "is_registered": bool(known_cedula),
            "phone": numero_paciente,
            "push_name": push_name.strip() if push_name else "",
        }

        invoke_input = {
            "messages": [HumanMessage(content=mensaje_texto)],
            "conversation_status": "ACTIVA",
            "user_context": user_context,
        }

        t_graph = time.perf_counter()
        try:
            timeout_s = float(settings.graph_timeout_seconds or 0)
            if timeout_s > 0:
                # High silent safety net; queue wait excluded from budget.
                result = await run_with_graph_deadline(
                    get_graph().ainvoke(invoke_input, config),
                    timeout_seconds=timeout_s,
                )
            else:
                result = await get_graph().ainvoke(invoke_input, config)
        except (GraphDeadlineExceeded, LLMCallTimeoutError) as timeout_exc:
            # Do NOT WhatsApp a retry/error copy — users need the real reply path,
            # not "reenvía". Log only; safety net should be high enough this is rare.
            spans["graph_total"] = time.perf_counter() - t_graph
            spans["queue_wait_total"] = graph_queue_wait_total()
            spans["total"] = time.perf_counter() - t_total
            kind = (
                "graph_timeout_silent"
                if isinstance(timeout_exc, GraphDeadlineExceeded)
                else "llm_call_timeout_silent"
            )
            logger.warning(
                f"[Latency] {kind} phone={numero_paciente} "
                f"limit={settings.graph_timeout_seconds}s "
                f"llm_call_limit={settings.llm_call_timeout_seconds}s "
                f"queue_wait_excluded={spans['queue_wait_total']:.3f}s "
                f"err={timeout_exc!s} (no WhatsApp timeout message) "
                + " ".join(f"{k}={v:.3f}s" for k, v in spans.items() if k != "queue_wait_total")
            )
            return
        spans["graph_total"] = time.perf_counter() - t_graph
        spans["queue_wait_total"] = graph_queue_wait_total()

        rag_conf = result.get("rag_confidence")

        if result.get("conversation_status") == "ESCALADA":
            already_sent_by_graph = bool(result.get("emergency_detected"))
            if not already_sent_by_graph:
                messages = result.get("messages", [])
                if messages:
                    last_msg = messages[-1]
                    if isinstance(last_msg, AIMessage) and last_msg.content:
                        resp_urg = extract_text_content(last_msg.content)
                        t_send = time.perf_counter()
                        await evolution_client.enviar_mensaje(numero_paciente, resp_urg)
                        spans["send"] = time.perf_counter() - t_send
                        asyncio.create_task(
                            dotnet_client.registrar_mensaje(
                                chat_identifier=numero_paciente,
                                rol="CHATBOT",
                                contenido=resp_urg,
                                rag_confidence=rag_conf,
                            )
                        )
                await escalate_conversation(numero_paciente, numero_paciente, mensaje_texto)
            else:
                config = get_thread_config(numero_paciente)
                await get_graph().aupdate_state(config, {"conversation_status": "ESCALADA"})
            spans["total"] = time.perf_counter() - t_total
            logger.info(
                f"[Latency] phone={numero_paciente} path=escalated "
                + " ".join(f"{k}={v:.3f}s" for k, v in spans.items())
            )
            return

        # Solo escalar por baja confianza RAG cuando hubo score real de tool RAG
        if rag_conf is not None and rag_conf < settings.rag_min_confidence:
            if not await escalate_conversation(numero_paciente, numero_paciente, mensaje_texto):
                await evolution_client.enviar_mensaje(numero_paciente, MENSAJE_FALLBACK_PACIENTE)
                asyncio.create_task(
                    dotnet_client.registrar_mensaje(
                        chat_identifier=numero_paciente,
                        rol="CHATBOT",
                        contenido=MENSAJE_FALLBACK_PACIENTE,
                    )
                )
            return

        # Extraer respuesta final del bot
        mensajes_resultado = result.get("messages", [])
        if not mensajes_resultado:
            await evolution_client.enviar_mensaje(numero_paciente, MENSAJE_FALLBACK_PACIENTE)
            return

        ultimo_mensaje = mensajes_resultado[-1]
        respuesta_texto = extract_text_content(getattr(ultimo_mensaje, "content", "")).strip()

        if not respuesta_texto:
            respuesta_texto = MENSAJE_FALLBACK_PACIENTE
        else:
            # Portal only if THIS turn succeeded agendar/modificar/cancelar or listed citas.
            respuesta_texto = _ensure_portal_reminder_after_booking(
                respuesta_texto, mensajes_resultado
            )

        t_send = time.perf_counter()
        await evolution_client.enviar_mensaje(numero_paciente, respuesta_texto)
        spans["send"] = time.perf_counter() - t_send
        asyncio.create_task(
            dotnet_client.registrar_mensaje(
                chat_identifier=numero_paciente,
                rol="CHATBOT",
                contenido=respuesta_texto,
                rag_confidence=rag_conf,
            )
        )

        inactivity_service.touch(numero_paciente, settings.session_ttl_seconds)
        spans["total"] = time.perf_counter() - t_total
        logger.info(
            f"[Latency] phone={numero_paciente} path=graph "
            + " ".join(f"{k}={v:.3f}s" for k, v in spans.items())
        )

    except (GraphDeadlineExceeded, LLMCallTimeoutError, asyncio.TimeoutError) as timeout_exc:
        # Silent: never WhatsApp timeout/retry spam.
        logger.warning(
            f"[Processor] Timeout silencioso para {numero_paciente}: {timeout_exc!s} "
            "(sin mensaje de reintento al usuario)"
        )
    except asyncio.CancelledError:
        raise
    except Exception as exc:
        logger.error(f"[Processor] Error procesando mensaje para {numero_paciente}: {exc}", exc_info=True)
        try:
            await evolution_client.enviar_mensaje(numero_paciente, MENSAJE_FALLBACK_PACIENTE)
            asyncio.create_task(
                dotnet_client.registrar_mensaje(
                    chat_identifier=numero_paciente,
                    rol="CHATBOT",
                    contenido=MENSAJE_FALLBACK_PACIENTE,
                )
            )
        except Exception:
            pass
