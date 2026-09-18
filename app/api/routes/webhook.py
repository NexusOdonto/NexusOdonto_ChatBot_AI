import logging
import re
import asyncio
import time
from collections import OrderedDict
from typing import Any
from fastapi import APIRouter, Request
from langchain_core.messages import AIMessage, HumanMessage
from app.schemas.chat import EvolutionWebhookPayload, unwrap_message_dict, extract_interactive_selection
from app.clients.evolution_client import evolution_client
from app.clients.evolution_client import is_bot_message_id
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
from app.services.semantic_cache import buscar_en_cache, purgar_cache_semantico

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/webhook", tags=["WhatsApp Webhook"])

# ─── Deduplicate inbound webhook deliveries ───────────────────────────────────
_MESSAGE_DEDUPE_TTL_SECONDS = 90
_SEEN_MESSAGE_IDS: OrderedDict[str, float] = OrderedDict()
_CHAT_LOCKS: dict[str, asyncio.Lock] = {}
_CHAT_LOCKS_GUARD = asyncio.Lock()

# ─── Anti-spam / Rate Limit & Debounce por usuario ────────────────────────────
_SPAM_WINDOW_SECONDS = 5.0          # Ventana para medir ráfagas de mensajes (segundos)
_SPAM_MAX_BURST = 3                 # Máximo de mensajes en esa ventana antes de activar bloqueo
_SPAM_PENALTY_SECONDS = 15.0        # Segundos que el usuario queda silenciado si hace spam
_SPAM_WARN_COOLDOWN = 30.0          # Segundos mínimos entre avisos de advertencia al mismo usuario
_DEBOUNCE_WAIT_SECONDS = 2.0        # Tiempo de espera (segundos) para agrupar mensajes fragmentados

_SPAM_TIMESTAMPS: dict[str, list[float]] = {}
_SPAM_BLOCKED_UNTIL: dict[str, float] = {}
_SPAM_WARNED_AT: dict[str, float] = {}

_USER_MESSAGE_BUFFERS: dict[str, list[str]] = {}
_USER_DEBOUNCE_TASKS: dict[str, asyncio.Task] = {}
_USER_LAST_ACTIVE: dict[str, float] = {}
_USER_PUSH_NAMES: dict[str, str] = {}
_PATIENT_CONTEXT_CACHE: dict[str, tuple[float, dict[str, Any]]] = {}
_PATIENT_CONTEXT_CACHE_TTL = 300.0  # 5 minutos de caché en memoria por teléfono


def _is_user_spam_blocked(numero_paciente: str) -> bool:
    """Retorna True si el usuario se encuentra dentro del período de penalización por spam."""
    return time.monotonic() < _SPAM_BLOCKED_UNTIL.get(numero_paciente, 0.0)


def _should_warn_spam(numero_paciente: str) -> bool:
    now = time.monotonic()
    last_warn = _SPAM_WARNED_AT.get(numero_paciente, 0.0)
    if now - last_warn > _SPAM_WARN_COOLDOWN:
        _SPAM_WARNED_AT[numero_paciente] = now
        return True
    return False


def _check_and_trigger_spam(numero_paciente: str) -> bool:
    """Verifica si los mensajes recientes del usuario constituyen spam en tiempo real.
    Si superan el umbral, cancela buffers pendientes, bloquea al usuario temporalmente
    y envía UNA advertencia.
    """
    now = time.monotonic()
    timestamps = _SPAM_TIMESTAMPS.get(numero_paciente, [])
    timestamps = [t for t in timestamps if now - t < _SPAM_WINDOW_SECONDS]
    timestamps.append(now)
    _SPAM_TIMESTAMPS[numero_paciente] = timestamps

    if len(timestamps) > _SPAM_MAX_BURST:
        # Activar penalización por spam
        _SPAM_BLOCKED_UNTIL[numero_paciente] = now + _SPAM_PENALTY_SECONDS

        # Limpiar buffers acumulados y cancelar tarea de debounce pendiente
        _USER_MESSAGE_BUFFERS.pop(numero_paciente, None)
        existing_task = _USER_DEBOUNCE_TASKS.pop(numero_paciente, None)
        if existing_task and not existing_task.done():
            existing_task.cancel()

        if _should_warn_spam(numero_paciente):
            logger.warning(
                f"[Anti-spam Gate] Ráfaga de spam detectada para {numero_paciente} "
                f"({len(timestamps)} msgs en {_SPAM_WINDOW_SECONDS}s). Enviando aviso."
            )
            spam_msg = (
                "⚠️ Estás enviando muchos mensajes seguidos.\n"
                "Por favor espera un momento y escribe tu consulta en un solo mensaje para poder atenderte bien. 😊"
            )
            asyncio.create_task(evolution_client.enviar_mensaje(numero_paciente, spam_msg))
        else:
            logger.info(f"[Anti-spam Gate] Mensaje extra de spam descartado silenciosamente para {numero_paciente}")

        return True

    return False


def _is_nonsense_or_gibberish(texto: str) -> bool:
    """Detecta si el mensaje es spam de letras/números repetidos, teclado aporreado o sin sentido."""
    raw = texto.strip()
    if not raw:
        return False

    # No filtrar comandos ni solicitudes del sistema
    if _is_reset_request(raw) or _is_resume_request(raw) or _is_escalation_request(raw):
        return False

    # Si contiene palabras clave del ámbito odontológico o cortesía, NUNCA es gibberish
    KEYWORDS_CLINICA = {
        "cita", "citas", "agendar", "agenda", "apartar", "programar", "horario", "horarios",
        "doctor", "doctora", "odontologo", "odontologa", "precio", "precios", "costo",
        "cuanto", "cuánto", "vale", "servicio", "servicios", "limpieza", "diseño",
        "ortodoncia", "bracket", "brackets", "calza", "resina", "extraccion", "extracción",
        "cordal", "cordales", "corona", "implante", "dolor", "urgencia", "emergencia",
        "cedula", "cédula", "documento", "nombre", "hola", "buenos", "buenas", "tardes",
        "dias", "días", "noches", "gracias", "cancelar", "modificar", "reprogramar",
        "confirmar", "asistir", "consulta", "telefono", "teléfono", "direccion", "dirección",
        "si", "sí", "no", "ok", "vale", "listo", "dale", "bien", "perfecto"
    }
    palabras_lower = [p.strip(".,;:!?()[]\"'").lower() for p in raw.split()]
    if any(p in KEYWORDS_CLINICA for p in palabras_lower):
        return False

    # Validar si son únicamente dígitos numéricos
    solo_digitos = "".join(c for c in raw if c.isdigit())
    chars_sin_separadores = raw.replace(" ", "").replace(".", "").replace("-", "")
    if solo_digitos and len(solo_digitos) == len(chars_sin_separadores):
        # Cédulas o teléfonos válidos (7 a 12 dígitos)
        if 7 <= len(solo_digitos) <= 12:
            return False
        # Opciones válidas de menú interactivo
        if solo_digitos in ("1", "2", "3", "4", "5"):
            return False
        # Números aleatorios cortos (< 7 dígitos) o excesivamente largos (> 12) sin contexto
        return True

    # Análisis de repetición y aporreo si el mensaje tiene 3 palabras o menos
    if len(palabras_lower) <= 3:
        # 1. Caracteres repetidos 4 o más veces seguidas (ej: "aaaaa", "11111", "jjjjj", ".......", "????")
        if re.search(r"(.)\1{3,}", raw, re.IGNORECASE):
            return True

        # 2. Patrones repetidos en bucle (ej: "asdasdasd", "12121212", "qweqweqwe")
        if re.search(r"^(.{2,4})\1{2,}$", raw.replace(" ", ""), re.IGNORECASE):
            return True

        # 3. Palabras largas sin vocales o aporreo de teclado (ej: "sdfghjkl", "zxcvbnm", "qwrtyp")
        for p in palabras_lower:
            solo_letras = re.sub(r"[^a-záéíóúñ]", "", p)
            if len(solo_letras) >= 6:
                vocales = len(re.findall(r"[aeiouáéíóú]", solo_letras))
                if vocales == 0:
                    return True
                if len(solo_letras) >= 8 and (vocales / len(solo_letras)) < 0.15:
                    return True

    return False


async def _resolver_contexto_paciente(numero_paciente: str, push_name: str = "") -> dict[str, Any]:
    """Consulta en .NET o en el perfil de WhatsApp para personalizar el saludo cordial al usuario.

    POLÍTICA ESTRICTA DE SEGURIDAD Y PRIVACIDAD DE DATOS (HABEAS DATA / LEY 1581):
    1. Solo se usa el nombre para saludar cordialmente al usuario ('¡Hola, Jeison! 👋✨').
    2. La CÉDULA NUNCA se pre-asume ni se autoriza automáticamente por teléfono.
       El paciente SIEMPRE debe proporcionar su documento en el chat para consultar,
       agendar o modificar citas, garantizando que un número falso o compartido jamás
       filtre información médica ni citas de terceros.
    3. Si el número es un identificador LID de WhatsApp (@lid) o contiene más de 11 dígitos,
       se omite la consulta por teléfono y se usa push_name.
    """
    from app.services.whatsapp_identity import obtener_telefono_canonico, es_identificador_lid, limpiar_digitos
    numero_canonico = obtener_telefono_canonico(numero_paciente)
    cached = _PATIENT_CONTEXT_CACHE.get(numero_canonico) or _PATIENT_CONTEXT_CACHE.get(numero_paciente)
    if cached and (now - cached[0]) < _PATIENT_CONTEXT_CACHE_TTL:
        ctx = dict(cached[1])
        if push_name and not ctx.get("push_name"):
            ctx["push_name"] = push_name
        return ctx

    clean_digits = limpiar_digitos(numero_canonico)
    es_lid = es_identificador_lid(numero_canonico)

    nombre_detectado = ""
    primer_nombre = ""

    # Solo buscar persona en .NET si es un número telefónico móvil real (no LID)
    if not es_lid:
        try:
            persona = await asyncio.wait_for(
                dotnet_client.buscar_persona_por_telefono(numero_canonico),
                timeout=3.0,
            )
            if persona and isinstance(persona, dict):
                first_name = (persona.get("firstName") or "").strip()
                last_name = (persona.get("lastName") or "").strip()
                nombre_detectado = f"{first_name} {last_name}".strip()
                primer_nombre = first_name
        except asyncio.TimeoutError:
            logger.warning(f"[Patient ID] Timeout al buscar persona por teléfono en .NET para {numero_paciente}")
        except Exception as e:
            logger.warning(f"[Patient ID] Error al consultar persona por teléfono en .NET para {numero_paciente}: {e}")

    # Si no se detectó por teléfono o vino de WhatsApp pushName, usar push_name
    if not primer_nombre and push_name:
        primer_nombre = push_name.strip().split()[0]
    if not nombre_detectado and push_name:
        nombre_detectado = push_name.strip()

    ctx = {
        "nombre": nombre_detectado,
        "primer_nombre": primer_nombre,
        "cedula": None,  # NUNCA pre-cargar cédula; requerir validación explícita del paciente
        "is_registered": False,
        "phone": numero_paciente,
        "push_name": push_name.strip() if push_name else "",
    }
    _PATIENT_CONTEXT_CACHE[numero_paciente] = (now, ctx)
    if primer_nombre:
        logger.info(f"[Patient ID] Interlocutor identificado para saludo ({numero_paciente}): {primer_nombre}")
    return ctx


def _enqueue_user_message(numero_paciente: str, mensaje_texto: str, push_name: str = "") -> None:
    """Encola el mensaje para debouncing y valida spam en tiempo real al llegar el webhook."""
    # 1. Descartar si el usuario está en penalización activa por spam
    if _is_user_spam_blocked(numero_paciente):
        logger.info(f"[Anti-spam Gate] Mensaje de {numero_paciente} descartado (penalización activa)")
        return

    # 2. Validar ráfaga de mensajes en tiempo real
    if _check_and_trigger_spam(numero_paciente):
        return

    if push_name:
        _USER_PUSH_NAMES[numero_paciente] = push_name.strip()

    # 3. Registrar el mensaje individual en base de datos para que el frontend lo visualice de inmediato
    asyncio.create_task(
        dotnet_client.registrar_mensaje(
            chat_identifier=numero_paciente,
            rol="USUARIO",
            contenido=mensaje_texto,
        )
    )

    # 4. Acumular en el buffer del usuario para agrupar mensajes continuos
    if numero_paciente not in _USER_MESSAGE_BUFFERS:
        _USER_MESSAGE_BUFFERS[numero_paciente] = []
    _USER_MESSAGE_BUFFERS[numero_paciente].append(mensaje_texto.strip())

    # 5. Si ya había un temporizador esperando, cancelarlo para extender la ventana y agrupar el nuevo mensaje
    existing_task = _USER_DEBOUNCE_TASKS.get(numero_paciente)
    if existing_task and not existing_task.done():
        existing_task.cancel()

    # 6. Lanzar temporizador de agrupación (espera antes de procesar con el bot)
    _USER_DEBOUNCE_TASKS[numero_paciente] = asyncio.create_task(
        _debounce_worker(numero_paciente)
    )


async def _debounce_worker(numero_paciente: str) -> None:
    """Espera la ventana de silencio para agrupar mensajes fragmentados y enviarlos juntos al bot."""
    try:
        await asyncio.sleep(_DEBOUNCE_WAIT_SECONDS)
    except asyncio.CancelledError:
        # Se canceló porque llegó otro mensaje antes de vencer el tiempo
        return

    mensajes = _USER_MESSAGE_BUFFERS.pop(numero_paciente, [])
    _USER_DEBOUNCE_TASKS.pop(numero_paciente, None)
    push_name = _USER_PUSH_NAMES.get(numero_paciente, "")

    if not mensajes:
        return

    # Unir fragmentos en un solo mensaje completo
    if len(mensajes) == 1:
        texto_final = mensajes[0]
    else:
        texto_final = "\n".join(mensajes)
        logger.info(f"[Debounce] Agrupados {len(mensajes)} mensajes para {numero_paciente} en una sola consulta: '{texto_final}'")

    asyncio.create_task(
        _with_chat_lock(
            numero_paciente,
            _process_whatsapp_message(numero_paciente, texto_final, registrar_usuario_db=False, push_name=push_name),
        )
    )


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


async def _registrar_mensaje_asesor(numero_paciente: str, mensaje_texto: str) -> None:
    """Registra en el backend .NET un mensaje enviado manualmente desde el celular vinculado.
    Se usa el rol ASESOR para que el frontend lo muestre como mensaje del operador en tiempo real.
    También sincroniza el mensaje en el estado de LangGraph para mantener el contexto de la conversación.
    """
    try:
        logger.info(f"[fromMe-Humano] Registrando respuesta manual del asesor para {numero_paciente}: '{mensaje_texto}'")
        await dotnet_client.registrar_mensaje(
            chat_identifier=numero_paciente,
            rol="ASESOR",
            contenido=mensaje_texto,
        )
        try:
            # Sincronizar con el historial en LangGraph para que el LLM sepa qué respondió el asesor
            config = get_thread_config(numero_paciente)
            await get_graph().aupdate_state(
                config,
                {"messages": [AIMessage(content=f"[Asesor]: {mensaje_texto}")]},
            )
        except Exception as graph_err:
            logger.debug(f"[fromMe-Humano] No se pudo sincronizar en LangGraph (normal si no hay sesión activa): {graph_err}")
    except Exception as e:
        logger.warning(f"[fromMe-Humano] No se pudo registrar mensaje del asesor: {e}")

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
    # 0. Si el asesor devolvió recientemente la conversación al bot, permitir atención inmediata
    from app.api.routes.agent_handoff import esta_recien_reactivada
    if esta_recien_reactivada(thread_id):
        logger.info(f"[Webhook] Conversación {thread_id} fue reactivada recientemente por asesor. Bot activo.")
        return False

    from app.services.whatsapp_identity import obtener_telefono_canonico
    raw_tid = str(thread_id).strip()
    canonical_tid = obtener_telefono_canonico(raw_tid)
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
            c_canon = obtener_telefono_canonico(c_chat)
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

        if matching_conv_ids and dotnet_is_escalated:
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

    # Si está escalada localmente, o en .NET con asesor o ticket activo:
    if is_graph_escalated or dotnet_is_escalated:
        if not is_graph_escalated and dotnet_is_escalated:
            logger.info(f"[Webhook] Conversación detectada como ESCALADA/ATENDIDA en .NET para {thread_id}. Sincronizando LangGraph.")
            try:
                config = get_thread_config(thread_id)
                await get_graph().aupdate_state(config, {"conversation_status": "ESCALADA"})
            except Exception:
                pass
        return True

    # Si en .NET la conversación está en ACTIVA y ya no hay atención humana:
    if is_graph_escalated and not dotnet_is_escalated:
        try:
            logger.info(f"[Webhook] Conversación fue restablecida a ACTIVA en .NET para {thread_id}. Sincronizando bot.")
            config = get_thread_config(thread_id)
            await get_graph().aupdate_state(config, {"conversation_status": "ACTIVA"})
        except Exception:
            pass
        return False

    return False


async def _escalate_conversation(thread_id: str, phone_number: str, message: str) -> bool:
    from app.api.routes.agent_handoff import desmarcar_reactivada
    desmarcar_reactivada(thread_id)
    desmarcar_reactivada(phone_number)

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
                from app.services.whatsapp_identity import obtener_telefono_canonico, obtener_destino_envio
                canon = obtener_telefono_canonico(numero_paciente)
                targets_clear = {numero_paciente, canon}
                dest_envio = obtener_destino_envio(numero_paciente)
                if dest_envio:
                    targets_clear.add(dest_envio)
                for t in targets_clear:
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


async def _process_whatsapp_audio(numero_paciente: str, raw_payload_data: dict, raw_message: dict, push_name: str = "") -> None:
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
        await _process_whatsapp_message(numero_paciente, texto_transcrito, push_name=push_name)

    except Exception as e:
        logger.error(f"[BG] Error procesando audio de {numero_paciente}: {e}", exc_info=True)
        await evolution_client.enviar_mensaje(numero_paciente, MENSAJE_ERROR_PROCESANDO_AUDIO)
        asyncio.create_task(
            dotnet_client.registrar_mensaje(numero_paciente, "CHATBOT", MENSAJE_ERROR_PROCESANDO_AUDIO)
        )


async def _process_whatsapp_message(
    numero_paciente: str,
    mensaje_texto: str,
    registrar_usuario_db: bool = False,
    push_name: str = "",
) -> None:
    """Procesa el mensaje del paciente en segundo plano.

    Esta corutina se ejecuta desacoplada del ciclo request/response para que
    Evolution API reciba el HTTP 200 de inmediato y no genere un error de timeout.

    El flujo es directo: el mensaje va al grafo LangGraph sin requerir autenticación previa.
    El número de WhatsApp (numero_paciente) es el identificador de la conversación.
    """
    try:
        # Registrar en DB solo si no fue registrado previamente por _enqueue_user_message
        if registrar_usuario_db:
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

        # 2. Filtrar mensajes sin sentido o aporreo de teclado (gibberish / spam de letras o números)
        if _is_nonsense_or_gibberish(mensaje_texto):
            logger.info(f"[Spam Filter] Mensaje sin sentido detectado de {numero_paciente}: '{mensaje_texto}'")
            resp_gibberish = (
                "No logro comprender tu mensaje 🤔. Por favor escribe con palabras claras lo que necesitas "
                "(por ejemplo: agendar una cita, consultar precios o ver servicios y horarios) y con gusto te ayudo. 😊🦷"
            )
            await evolution_client.enviar_mensaje(numero_paciente, resp_gibberish)
            asyncio.create_task(
                dotnet_client.registrar_mensaje(
                    chat_identifier=numero_paciente,
                    rol="CHATBOT",
                    contenido=resp_gibberish,
                    rag_confidence=1.0,
                )
            )
            return

        # 2. Verificar si la conversación ya fue escalada a un asesor humano o está en atención
        if await _is_escalated(numero_paciente):
            if _is_resume_request(mensaje_texto):
                logger.info(f"[BG] Paciente solicita volver con el bot: {numero_paciente}")
                checkpointer = get_checkpointer_instance()
                from app.services.whatsapp_identity import obtener_telefono_canonico, obtener_destino_envio
                canon = obtener_telefono_canonico(numero_paciente)
                targets_clear = {numero_paciente, canon}
                dest_envio = obtener_destino_envio(numero_paciente)
                if dest_envio:
                    targets_clear.add(dest_envio)
                for t in targets_clear:
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

        # 4. Validar expiración de sesión (TTL) para evitar contaminación de contextos viejos
        now_ts = time.time()
        last_active = _USER_LAST_ACTIVE.get(numero_paciente)
        session_expired = False

        if last_active is not None and (now_ts - last_active) > settings.session_ttl_seconds:
            session_expired = True
        elif last_active is None:
            # Si el servidor acaba de iniciar o es la primera interacción del proceso actual,
            # verificar la antigüedad del checkpoint en PostgreSQL
            try:
                state_snapshot = await get_graph().aget_state(config)
                if state_snapshot and state_snapshot.values and state_snapshot.values.get("messages"):
                    created_at_val = getattr(state_snapshot, "created_at", None)
                    if created_at_val:
                        from datetime import datetime, timezone
                        if isinstance(created_at_val, str):
                            dt = datetime.fromisoformat(created_at_val.replace("Z", "+00:00"))
                        elif isinstance(created_at_val, datetime):
                            dt = created_at_val
                        else:
                            dt = None
                        if dt and (datetime.now(timezone.utc) - dt).total_seconds() > settings.session_ttl_seconds:
                            session_expired = True
            except Exception as ttl_err:
                logger.debug(f"[TTL Check] No se pudo verificar antigüedad del hilo en DB: {ttl_err}")

        if session_expired:
            logger.info(f"[TTL Check] Sesión expirada para {numero_paciente} (> {settings.session_ttl_seconds}s). Reiniciando hilo...")
            checkpointer = get_checkpointer_instance()
            if checkpointer:
                await checkpointer.clear_thread(numero_paciente)
            dotnet_client.limpiar_cache_conversacion(numero_paciente)

        _USER_LAST_ACTIVE[numero_paciente] = now_ts

        # 5. Resolver identidad del paciente en la base de datos de .NET o por WhatsApp pushName
        user_context = await _resolver_contexto_paciente(numero_paciente, push_name)

        # 6. Consultar si existe respuesta en el Caché Semántico (⚡ 0 tokens, < 50ms)
        cached_response = await buscar_en_cache(mensaje_texto)
        if cached_response:
            # Personalizar saludo rápido si se conoce el nombre del paciente
            saludo_nombre = user_context.get("primer_nombre") or user_context.get("nombre")
            if saludo_nombre and "¡Hola!" in cached_response:
                cached_response = cached_response.replace("¡Hola!", f"¡Hola, {saludo_nombre}!")

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

        # 7. Invocar el grafo LangGraph directamente con el contexto del paciente
        invoke_input = {
            "messages": [HumanMessage(content=mensaje_texto)],
            "conversation_status": "ACTIVA",
            "rag_confidence": 1.0,
            "user_context": user_context,
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
                    # NOTA DE SEGURIDAD Y PRIVACIDAD (Habeas Data / Ley 1581):
                    # Las respuestas dinámicas del LLM en chats con pacientes NUNCA se guardan en el
                    # caché semántico global para evitar cualquier filtración o reutilización de datos entre usuarios.
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
                "📍 *Consultorio:* Calle 100 # 15-20, Centro Médico Odontológico\n"
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


@router.post("/cache/purge")
async def purge_cache_endpoint():
    """Purga el caché semántico en memoria (L1) y la colección en Qdrant (L2)."""
    exito = purgar_cache_semantico()
    if exito:
        return {
            "status": "ok",
            "message": "Caché semántico purgado y colección recreada exitosamente. Se eliminaron datos transaccionales residuales."
        }
    return {
        "status": "error",
        "message": "No se pudo purgar la colección en Qdrant (verificar si el servicio está activo)."
    }


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

            # Mensajes enviados desde el dispositivo vinculado (fromMe)
            if data.key and data.key.fromMe:
                # Obtener el ID del mensaje para distinguir bot vs humano
                msg_id_from_me = getattr(data.key, "id", None) or ""
                if is_bot_message_id(msg_id_from_me):
                    # Es un eco del propio bot — ya fue registrado al enviarlo
                    return {"status": "ignored", "reason": "self_message_bot"}

                # Es un mensaje escrito manualmente por el operador en el celular vinculado
                remote_jid_me = data.key.remoteJid if (data.key and data.key.remoteJid) else ""
                participant_me = getattr(data.key, "participant", None) or getattr(data, "sender", None)
                if "@lid" in str(remote_jid_me) and participant_me and "@s.whatsapp.net" in str(participant_me):
                    destinatario = str(participant_me)
                else:
                    destinatario = remote_jid_me

                from app.services.whatsapp_identity import obtener_telefono_canonico
                destinatario = obtener_telefono_canonico(destinatario)

                raw_msg_me = data.message or {}
                unwrapped_me = unwrap_message_dict(raw_msg_me)
                texto_asesor = (
                    unwrapped_me.get("conversation")
                    or (unwrapped_me.get("extendedTextMessage") or {}).get("text", "")
                    or ""
                )
                if not texto_asesor:
                    for k in ["imageMessage", "videoMessage", "documentMessage"]:
                        if k in unwrapped_me and isinstance(unwrapped_me[k], dict):
                            cap = unwrapped_me[k].get("caption", "")
                            if cap:
                                texto_asesor = cap
                                break

                if texto_asesor and destinatario:
                    asyncio.create_task(
                        _registrar_mensaje_asesor(destinatario, texto_asesor)
                    )
                    logger.info(f"[fromMe-Humano] Mensaje del asesor registrado hacia {destinatario}: {texto_asesor}")
                return {"status": "ignored", "reason": "self_message_human_registered"}

            message_id = getattr(data.key, "id", None) if data.key else None
            if _is_duplicate_message_id(message_id):
                logger.info(f"[Webhook] Mensaje duplicado ignorado (key.id={message_id})")
                return {"status": "ignored", "reason": "duplicate_message_id"}

            # Extraer número del paciente completo (con el sufijo de whatsapp)
            remote_jid = data.key.remoteJid if (data.key and data.key.remoteJid) else ""
            if not remote_jid:
                return {"status": "ignored", "reason": "no_remote_jid"}

            from app.services.whatsapp_identity import extraer_identidad_webhook
            numero_paciente, _ = extraer_identidad_webhook(raw_json, data)

            # Extraer nombre público del perfil de WhatsApp si está disponible
            push_name = str(getattr(data, "pushName", None) or "").strip()

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
                if _is_user_spam_blocked(numero_paciente) or _check_and_trigger_spam(numero_paciente):
                    return {"status": "ignored", "reason": "spam_blocked"}
                logger.info(f"[Webhook] Audio recibido de {numero_paciente}. Encolando transcripción y procesamiento...")
                raw_payload_data = raw_json.get("data", {})
                asyncio.create_task(
                    _with_chat_lock(
                        numero_paciente,
                        _process_whatsapp_audio(numero_paciente, raw_payload_data, unwrapped_message, push_name=push_name),
                    )
                )
            elif is_unsupported_media:
                if _is_user_spam_blocked(numero_paciente) or _check_and_trigger_spam(numero_paciente):
                    return {"status": "ignored", "reason": "spam_blocked"}
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
                    _enqueue_user_message(numero_paciente, mensaje_texto, push_name=push_name)
                else:
                    logger.warning(f"[Webhook] Mensaje no reconocido o vacío de {numero_paciente}: {unwrapped_message}")

        # Evolution API recibe HTTP 200 de inmediato, sin esperar el procesamiento pesado
        return {"status": "queued"}

    except Exception as e:
        logger.error(f"[Webhook Error] Fallo al procesar el webhook: {e}", exc_info=True)
        return {"status": "error", "message": "Error procesando el payload"}