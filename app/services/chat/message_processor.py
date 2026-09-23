"""Procesador central de mensajes de chat en segundo plano para Nexus Odonto.
Ejecuta la orquestación del grafo LangGraph, control de TTL de sesión (15 min),
caché semántico, escalamiento a asesores humanos y auditoría a .NET / Oracle.
"""

import time
import re
import asyncio
import logging
from typing import Any, Optional
from langchain_core.messages import AIMessage, HumanMessage

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
from app.services.semantic_cache import buscar_en_cache
from app.services.chat.booking_fastpath import (
    try_booking_fastpath,
    clear_booking_state,
    set_booking_state,
    is_agendar_intent,
    get_booking_state,
)
from app.services.inactivity_service import inactivity_service
from app.core.llm_factory import extract_text_content

logger = logging.getLogger(__name__)

# Mensajes institucionales de contingencia y medios
MENSAJE_FALLBACK_PACIENTE = (
    "En este momento presentamos intermitencias temporales en el servicio. "
    "Por favor, intenta nuevamente en unos momentos. ¡Disculpa las molestias!"
)

MENSAJE_ESCALAMIENTO = (
    "Entiendo. Un asesor de la clínica revisará tu solicitud y te contactará pronto."
)

MENSAJE_MEDIOS_NO_SOPORTADOS = (
    "Por el momento no puedo procesar ni visualizar fotos, videos, documentos ni archivos directamente 📎📷.\n\n"
    "Por favor, descríbeme detalladamente por texto o mediante una nota de voz lo que necesitas o lo que contiene tu archivo "
    "(por ejemplo, el síntoma que presentas, el tratamiento o la orden médica) para poder ayudarte con mucho gusto."
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
    clear_booking_state(phone_number)
    logger.info(f"[Reset] Memoria e historial reiniciados para {phone_number}")
    reset_msg = "🔄 Memoria reiniciada con éxito. ¡Hola! Soy el asistente virtual de Nexus Odonto. ¿En qué puedo colaborarte hoy?"
    await evolution_client.enviar_mensaje(phone_number, reset_msg)
    asyncio.create_task(
        dotnet_client.registrar_mensaje(phone_number, "CHATBOT", reset_msg)
    )


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

    dotnet_is_escalated = False
    try:
        convs = await dotnet_client.obtener_catalogo("ChatbotConversations") or []
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

            if matches:
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
                    break
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
    clear_booking_state(thread_id)
    clear_booking_state(phone_number)

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


async def process_whatsapp_unsupported_media(numero_paciente: str, caption: str = "") -> None:
    """Gestiona la recepción de archivos o fotos no procesables directamente."""
    try:
        if await is_escalated(numero_paciente):
            if caption and is_resume_request(caption):
                logger.info(f"[Processor] Paciente solicita volver con el bot: {numero_paciente}")
                checkpointer = get_checkpointer_instance()
                from app.services.whatsapp_identity import obtener_telefono_canonico, obtener_destino_envio
                canon = obtener_telefono_canonico(numero_paciente)
                targets_clear = {numero_paciente, canon}
                dest = obtener_destino_envio(numero_paciente)
                if dest:
                    targets_clear.add(dest)
                for t in targets_clear:
                    try:
                        if checkpointer:
                            await checkpointer.clear_thread(t)
                        dotnet_client.limpiar_cache_conversacion(t)
                    except Exception:
                        pass

                msg_bienvenida = "👋🦷 *Nexus Odonto Asistente Virtual*\n\n¡Hola de nuevo! He reactivado mi sistema para atenderte. ¿En qué puedo colaborarte hoy? 😊✨"
                await evolution_client.enviar_mensaje(numero_paciente, msg_bienvenida)
                asyncio.create_task(dotnet_client.registrar_mensaje(numero_paciente, "CHATBOT", msg_bienvenida))
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
                "No logro comprender tu mensaje 🤔. Por favor escribe con palabras claras lo que necesitas "
                "(por ejemplo: agendar una cita, consultar precios o ver servicios y horarios) y con gusto te ayudo. 😊🦷"
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
                checkpointer = get_checkpointer_instance()
                from app.services.whatsapp_identity import obtener_telefono_canonico, obtener_destino_envio
                canon = obtener_telefono_canonico(numero_paciente)
                targets_clear = {numero_paciente, canon}
                dest = obtener_destino_envio(numero_paciente)
                if dest:
                    targets_clear.add(dest)
                for t in targets_clear:
                    try:
                        if checkpointer:
                            await checkpointer.clear_thread(t)
                        dotnet_client.limpiar_cache_conversacion(t)
                        clear_booking_state(t)
                    except Exception:
                        pass

                msg_bienvenida = "👋 *Nexus Odonto Asistente Virtual*\n\n¡Hola de nuevo! He reactivado mi sistema para atenderte. ¿En qué puedo colaborarte hoy? 😊🦷"
                await evolution_client.enviar_mensaje(numero_paciente, msg_bienvenida)
                asyncio.create_task(dotnet_client.registrar_mensaje(numero_paciente, "CHATBOT", msg_bienvenida))
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
                    clear_booking_state(t)
                except Exception:
                    pass
            _USER_LAST_ACTIVE.pop(numero_paciente, None)
            inactivity_service.cancel(numero_paciente)

        _USER_LAST_ACTIVE[numero_paciente] = now_ts
        if checkpointer:
            try:
                await checkpointer.actualizar_actividad(numero_paciente)
            except Exception:
                pass

        # --- Booking fast-path (sin LLM) ---
        last_bot_text = ""
        try:
            st = await get_graph().aget_state(config)
            msgs = (st.values or {}).get("messages") or []
            for m in reversed(msgs):
                if isinstance(m, AIMessage) and getattr(m, "content", None):
                    last_bot_text = extract_text_content(m.content)
                    break
        except Exception:
            last_bot_text = ""

        t_fp = time.perf_counter()
        fp = try_booking_fastpath(numero_paciente, mensaje_texto, last_bot_text=last_bot_text)
        spans["booking_fastpath"] = time.perf_counter() - t_fp
        if fp:
            respuesta_fp, _st = fp
            t_send = time.perf_counter()
            await evolution_client.enviar_mensaje(numero_paciente, respuesta_fp)
            spans["send"] = time.perf_counter() - t_send
            asyncio.create_task(
                dotnet_client.registrar_mensaje(
                    chat_identifier=numero_paciente,
                    rol="CHATBOT",
                    contenido=respuesta_fp,
                )
            )
            try:
                await get_graph().aupdate_state(
                    config,
                    {
                        "messages": [
                            HumanMessage(content=mensaje_texto),
                            AIMessage(content=respuesta_fp),
                        ]
                    },
                )
            except Exception:
                pass
            inactivity_service.touch(numero_paciente, settings.session_ttl_seconds)
            spans["total"] = time.perf_counter() - t_total
            logger.info(
                f"[Latency] phone={numero_paciente} path=booking_fastpath "
                + " ".join(f"{k}={v:.3f}s" for k, v in spans.items())
            )
            return

        # Caché semántico (⚡ 0 tokens)
        t_cache = time.perf_counter()
        cached_response = await buscar_en_cache(mensaje_texto)
        spans["cache_lookup"] = time.perf_counter() - t_cache
        if cached_response:
            logger.info(f"[Semantic Cache] Respondiendo a {numero_paciente} desde caché.")
            if is_agendar_intent(mensaje_texto):
                set_booking_state(numero_paciente, step="awaiting_cedula", cedula=None, nombre=None)
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
                    {"messages": [HumanMessage(content=mensaje_texto), AIMessage(content=cached_response)]},
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

        # Contexto del paciente: Cero asunción (Habeas Data)
        booking = get_booking_state(numero_paciente) or {}
        nombre_booking = (booking.get("nombre") or "").strip()
        cedula_booking = booking.get("cedula")
        user_context = {
            "nombre": nombre_booking,
            "primer_nombre": nombre_booking.split()[0] if nombre_booking else "",
            "cedula": cedula_booking,
            "is_registered": False,
            "phone": numero_paciente,
            "push_name": push_name.strip() if push_name else "",
            "booking_step": booking.get("step"),
        }

        invoke_input = {
            "messages": [HumanMessage(content=mensaje_texto)],
            "conversation_status": "ACTIVA",
            "user_context": user_context,
        }

        t_graph = time.perf_counter()
        result = await get_graph().ainvoke(invoke_input, config)
        spans["graph_total"] = time.perf_counter() - t_graph

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
