"""Nodo principal del Asistente Virtual (Chatbot).
Configura el System Message, enlaza las herramientas clínicas y de agenda,
e invoca el modelo de lenguaje de forma resiliente con soporte multi-proveedor.
"""

import logging
import re
import time
from functools import lru_cache
from datetime import datetime
from zoneinfo import ZoneInfo
from langchain_core.messages import SystemMessage, ToolMessage, AIMessage, HumanMessage
from app.core.llm_factory import get_chat_llm
from app.core.config import settings
from app.core.llm_concurrency import with_llm_slot
from app.graph.state import AgentState

from app.agents.tools.clinical_rag_tool import clinical_knowledge_tool
from app.agents.tools.catalog_tools import (
    consultar_disponibilidad_tool,
    consultar_doctores_tool,
    consultar_servicios_y_precios_tool,
)
from app.agents.tools.appointment_tools import (
    agendar_cita_tool,
    consultar_cita_por_cedula_tool,
    cancelar_cita_tool,
    modificar_cita_tool,
    confirmar_cita_tool,
)

logger = logging.getLogger(__name__)

# Compact system prompt: keep booking correctness, drop redundant prose that
# duplicated the temporal context block (cuts ~60% input tokens → faster Gemini TTFT).
SYSTEM_MESSAGE = SystemMessage(
    content=(
        "Eres la asesora de atención al paciente de Nexus Odonto (WhatsApp, español Colombia). "
        "Tono cálido, breve y humano. NUNCA suenes a menú robótico ni inventes datos.\n\n"
        "FORMATO WHATSAPP: *negrita* con un solo asterisco (nunca **). Viñetas con •. "
        "1-2 emojis por mensaje. Respuestas cortas.\n\n"
        "ALCANCE: solo odontología / citas / servicios / precios / doctores / cuidados. "
        "Fuera de tema: orienta amable a Nexus Odonto. "
        "NO cambies contraseñas: indica login web https://nexusodonto.chatcampuslands.com/login "
        "(cédula + contraseña temporal).\n\n"
        "CONTACTO: +57 324 6030217 | Calle 100 # 15-20 | Lun-Sáb 8:00 AM–6:00 PM | soporte@nexusodonto.com\n\n"
        "HABEAS DATA: NUNCA asumas nombre ni cédula. Pide SIEMPRE la cédula (mín. 7 dígitos) "
        "antes de ver/agendar/reprogramar/cancelar/confirmar. Si ya la escribió en este chat, no la vuelvas a pedir. "
        "PROHIBIDO inventar nombres (p. ej. 'Paciente Nexus').\n\n"
        "HERRAMIENTAS (usa solo cuando haga falta datos reales):\n"
        "1. consultar_doctores_tool\n"
        "2. consultar_servicios_y_precios_tool — fuente de verdad de precios; NUNCA inventes tarifas. "
        "No vuelques el catálogo entero: destaca 3-4 servicios y pregunta qué necesita.\n"
        "3. consultar_disponibilidad_tool(especialidad/servicio, fecha YYYY-MM-DD)\n"
        "4. agendar_cita_tool — SOLO tras confirmación explícita del paciente y con nombre+cédula+servicio+horario. "
        "Si el paciente pidió un odontólogo por nombre (ej: Ana Sofía), pasa ESE nombre en profesional_id; "
        "PROHIBIDO sustituirlo por Laura Gómez u otro por defecto/cabecera.\n"
        "5. consultar_cita_por_cedula_tool — VER citas. NO usar para agendar ni reprogramar\n"
        "6. modificar_cita_tool(cedula, nueva_fecha_hora opcional) — reprogramar\n"
        "7. cancelar_cita_tool(cedula, cita_id opcional)\n"
        "8. confirmar_cita_tool(cedula)\n"
        "9. clinical_knowledge_tool — dudas clínicas (sin mezclar con agenda)\n\n"
        "AGENDAR — primera respuesta si pide cita y falta cédula: "
        "pide cédula y nombre completo. NO invoques herramientas todavía.\n"
        "Si ya dio cédula+nombre en este chat (aunque sea antes del 'quiero agendar'), "
        "NO vuelvas a pedirlos: confirma y pregunta el servicio/tratamiento.\n"
        "Si responde con cédula+nombre tras pedirlos para agendar: "
        "PROHIBIDO menú de bienvenida ni '¿en qué te ayudamos?'; "
        "confirma y pregunta servicio. PROHIBIDO consultar_cita_por_cedula_tool.\n"
        "Si responde solo con cédula en hilo de agendar: "
        "PROHIBIDO consultar_cita_por_cedula_tool; confirma cédula y pide nombre si falta, luego servicio/horario.\n"
        "Protocolo: (1) cédula+nombre+servicio+horario vía disponibilidad "
        "(2) muestra ficha Propuesta de Cita "
        "(3) solo si confirma ('sí'/'confirmo'), llama agendar_cita_tool.\n"
        "TRAS AGENDAR / REPROGRAMAR / CANCELAR CON ÉXITO, o al LISTAR citas del paciente "
        "(consultar_cita_por_cedula_tool con citas): incluye un recordatorio breve de la "
        "plataforma virtual / portal del paciente con URL "
        "https://nexusodonto.chatcampuslands.com/login. "
        "Si el resultado de la herramienta indica primera vez / cuenta nueva "
        "([primera_vez_portal] o tip de credenciales): explica la REGLA "
        "«tu usuario es tu número de cédula y la contraseña también» "
        "SIN escribir los dígitos de la cédula ni la contraseña. "
        "PROHIBIDO mencionar el portal al pedir cédula, en saludos, disponibilidad, "
        "slots alternos, errores mid-flujo o cuando no hubo éxito de agenda/consulta."
        "\n\n"
        "REPROGRAMAR: con cédula conocida → modificar_cita_tool de inmediato (nunca consultar_cita_por_cedula_tool). "
        "CANCELAR: con cédula → cancelar_cita_tool de inmediato.\n"
        "Citas múltiples: usa ordinales 1/2 en cita_id; nunca pidas UUIDs.\n\n"
        "HORARIOS CLÍNICOS: Lun-Vie 8:00–12:00 y 14:00–17:00; Sáb 8:00–12:00; Dom/festivos cerrado. "
        "Almuerzo 12:00–14:00 sin citas. Slots cada 30 min. No inventes turnos.\n\n"
        "DOLOR / SIN CUPOS: empatía + sobrecupo presencial + línea +57 324 6030217 + paliativos seguros "
        "(compresa fría, enjuague salino; NUNCA aspirina sobre el diente). "
        "Sin diagnósticos invasivos; urgencias severas → centro médico."
    )
)

# Cap history fed to Gemini (tool schemas + system already dominate TTFT).
_MAX_HISTORY_MESSAGES = 16
_MAX_TOOL_MSG_CHARS = 1800
_MAX_AI_MSG_CHARS = 1200

# Chat replies on WhatsApp stay short; lower caps cut decode time.
_CHAT_MAX_OUTPUT_TOKENS = 512

_ALL_TOOLS = (
    clinical_knowledge_tool,
    consultar_disponibilidad_tool,
    agendar_cita_tool,
    consultar_cita_por_cedula_tool,
    cancelar_cita_tool,
    modificar_cita_tool,
    consultar_doctores_tool,
    consultar_servicios_y_precios_tool,
    confirmar_cita_tool,
)


def _select_tools(last_user_msg: str, prev_ai_msg: str, cedula: str | None) -> tuple:
    """Bind only tools likely needed this turn — smaller schemas → faster Gemini TTFT."""
    norm = (last_user_msg or "").lower()
    prev = (prev_ai_msg or "").lower()

    # Pure price / catalog questions
    if any(k in norm for k in ("precio", "precios", "vale", "cuesta", "cuanto", "tarif")) and not any(
        k in norm for k in ("cita", "agendar", "cancelar", "modificar", "reprogramar")
    ):
        return (consultar_servicios_y_precios_tool, consultar_doctores_tool)

    if any(k in norm for k in ("servicio", "servicios", "tratamiento", "tratamientos", "limpieza", "profilaxis", "blanqueamiento")) and not any(
        k in norm for k in ("cita", "agendar", "disponib")
    ):
        return (consultar_servicios_y_precios_tool,)

    if any(k in norm for k in ("doctor", "doctora", "odontologo", "especialista", "especialistas")):
        return (consultar_doctores_tool, consultar_servicios_y_precios_tool)

    # Clinical Q&A without booking language
    if any(k in norm for k in ("dolor", "duele", "brackets", "ortodoncia", "extraccion", "cuidado", "recomienda", "que comer", "preparacion")) and not any(
        k in norm for k in ("cita", "agendar", "cancelar", "modificar")
    ):
        return (clinical_knowledge_tool, consultar_servicios_y_precios_tool)

    # Booking start without cédula: no tools (text-only)
    if any(k in norm for k in ("quiero una cita", "quiero agendar", "necesito una cita", "agendar cita")) and not cedula:
        return tuple()

    # Identity just provided (cédula + name) mid-booking: allow services catalog only
    if re.search(r"\b\d{7,12}\b", norm) and any(
        k in prev for k in ("cédula", "cedula", "nombre completo", "agendar")
    ):
        return (consultar_servicios_y_precios_tool, consultar_doctores_tool)

    # Reschedule / cancel paths
    if any(k in norm for k in ("reprogramar", "modificar")) or any(k in prev for k in ("reprogramar", "modificar")):
        return (modificar_cita_tool, consultar_disponibilidad_tool, consultar_servicios_y_precios_tool)
    if "cancelar" in norm or "cancelar" in prev:
        return (cancelar_cita_tool,)
    if "confirmar" in norm:
        return (confirmar_cita_tool,)

    # Mid-booking / availability
    if any(k in norm for k in ("cita", "agendar", "disponib", "horario", "turno")) or cedula:
        return (
            consultar_disponibilidad_tool,
            agendar_cita_tool,
            consultar_cita_por_cedula_tool,
            cancelar_cita_tool,
            modificar_cita_tool,
            consultar_doctores_tool,
            consultar_servicios_y_precios_tool,
            confirmar_cita_tool,
        )

    return _ALL_TOOLS


@lru_cache(maxsize=16)
def _bound_llm_for_tools(tool_names: tuple[str, ...]):
    name_to_tool = {t.name: t for t in _ALL_TOOLS}
    tools = [name_to_tool[n] for n in tool_names if n in name_to_tool]
    primary_llm = get_chat_llm(temperature=0, max_tokens=_CHAT_MAX_OUTPUT_TOKENS)
    if not tools:
        return primary_llm
    return primary_llm.bind_tools(tools)


def get_llm_with_tools(last_user_msg: str = "", prev_ai_msg: str = "", cedula: str | None = None):
    selected = _select_tools(last_user_msg, prev_ai_msg, cedula)
    tool_names = tuple(t.name for t in selected)
    return _bound_llm_for_tools(tool_names), tool_names


def _trim_history(raw_msgs: list, *, is_gemini: bool) -> list:
    """Keep recent turns; drop noise and truncate bulky tool payloads."""
    chat_messages = []
    for msg in raw_msgs:
        if isinstance(msg, AIMessage) and msg.content:
            content = str(msg.content)
            if "[Consultando información" in content:
                continue
            if any(
                obs in content
                for obs in (
                    "Cr 24 #35-12",
                    "0a00dfec",
                    "Tu Próxima Cita Programada",
                    "Tus Citas en Nexus Odonto",
                )
            ):
                continue
            if len(content) > _MAX_AI_MSG_CHARS:
                # Keep original message object; only shorten text for the prompt view.
                msg = HumanMessage(content=content[:_MAX_AI_MSG_CHARS] + "… [respuesta previa truncada]")
                chat_messages.append(msg)
                continue
        if isinstance(msg, SystemMessage):
            chat_messages.append(
                HumanMessage(content=f"[Contexto / Resumen de conversación previa]:\n{msg.content}")
            )
        elif is_gemini and isinstance(msg, AIMessage) and getattr(msg, "tool_calls", None) and not msg.content:
            continue
        elif is_gemini and isinstance(msg, ToolMessage):
            tool_name = getattr(msg, "name", None) or "herramienta"
            body = str(msg.content or "")
            if len(body) > _MAX_TOOL_MSG_CHARS:
                body = body[:_MAX_TOOL_MSG_CHARS] + "…"
            chat_messages.append(HumanMessage(content=f"[Información del sistema ({tool_name})]:\n{body}"))
        else:
            chat_messages.append(msg)

    if len(chat_messages) > _MAX_HISTORY_MESSAGES:
        chat_messages = chat_messages[-_MAX_HISTORY_MESSAGES:]
    return chat_messages


async def chatbot_node(state: AgentState) -> dict[str, list]:
    """Procesa el historial actual y agrega la respuesta del asistente."""
    try:
        tz = ZoneInfo(settings.reminder_timezone)
    except Exception:
        tz = ZoneInfo("America/Bogota")
    now = datetime.now(tz)
    fecha_str = now.strftime("%Y-%m-%d")
    hora_str = now.strftime("%I:%M %p")
    dias_semana = ["Lunes", "Martes", "Miércoles", "Jueves", "Viernes", "Sábado", "Domingo"]
    dia_nombre = dias_semana[now.weekday()]

    # Compact temporal block (rules already in SYSTEM_MESSAGE).
    context_str = (
        f"HOY={fecha_str} ({dia_nombre}) HORA_CO={hora_str}. "
        "Usa esta fecha para 'hoy'/'mañana'. No ofrezcas horarios pasados. "
        "Disponibilidad solo con consultar_disponibilidad_tool."
    )

    raw_msgs = state.get("messages", [])
    last_user_msg = ""
    prev_ai_msg = ""
    cedula_detectada = None
    nombre_detectado = None

    from app.services.booking_flow import (
        extract_identity_from_history_newest_first,
        is_booking_start_intent,
        parse_identity_from_text,
        prev_asked_for_booking_identity,
    )

    hist_id = extract_identity_from_history_newest_first(raw_msgs)
    cedula_detectada = hist_id.cedula
    nombre_detectado = hist_id.nombre

    for m in reversed(raw_msgs):
        if isinstance(m, HumanMessage) and not last_user_msg and m.content:
            last_user_msg = str(m.content).strip()
        elif isinstance(m, AIMessage) and not prev_ai_msg and m.content:
            prev_ai_msg = str(m.content).strip().lower()

    # Prefer identity on the current turn if present
    if last_user_msg:
        turn_id = parse_identity_from_text(last_user_msg)
        if turn_id.cedula:
            cedula_detectada = turn_id.cedula
        if turn_id.nombre:
            nombre_detectado = turn_id.nombre

    uc = state.get("user_context") or {}
    if not cedula_detectada and uc.get("cedula"):
        cedula_detectada = str(uc.get("cedula")).strip() or None
    if not nombre_detectado and uc.get("nombre"):
        nombre_detectado = str(uc.get("nombre")).strip() or None

    # Inyección contextual de acción inmediata para evitar desvíos o alucinaciones
    if last_user_msg:
        norm_user = last_user_msg.lower()
        turn_id = parse_identity_from_text(last_user_msg)
        if re.match(r"^\d{7,12}$", last_user_msg) and any(
            w in prev_ai_msg for w in ["reprogramar", "modificar", "cambiar", "cambio"]
        ):
            context_str += (
                f"\n[ACCIÓN] Cédula '{last_user_msg}' para reprogramar → "
                f"modificar_cita_tool(cedula='{last_user_msg}') YA. Sin inventar citas."
            )
        elif any(w in norm_user for w in ["modificar", "reprogramar", "cambiar mi cita", "cambiar la cita"]) and cedula_detectada:
            context_str += (
                f"\n[ACCIÓN] Reprogramar con cédula conocida {cedula_detectada} → "
                f"modificar_cita_tool(cedula='{cedula_detectada}') YA. No pedir cédula."
            )
        elif any(w in norm_user for w in ["cancelar", "anular"]) and "cita" in norm_user and cedula_detectada:
            context_str += (
                f"\n[ACCIÓN] Cancelar con cédula {cedula_detectada} → "
                f"cancelar_cita_tool(cedula='{cedula_detectada}') YA."
            )
        elif prev_asked_for_booking_identity(prev_ai_msg) and turn_id.cedula:
            # Bug B: after collecting identity for agendar, continue — never welcome menu.
            nombre_txt = turn_id.nombre or nombre_detectado or ""
            context_str += (
                f"\n[ACCIÓN] Paciente entregó identidad para AGENDAR "
                f"(cédula={turn_id.cedula}"
                + (f", nombre={nombre_txt}" if nombre_txt else "")
                + "). "
                "PROHIBIDO menú de bienvenida / '¿En qué te podemos ayudar?'. "
                "Confirma brevemente y pregunta qué tratamiento o servicio desea agendar. "
                "PROHIBIDO consultar_cita_por_cedula_tool ni inventar horarios en este turno."
            )
        elif is_booking_start_intent(last_user_msg) and cedula_detectada and nombre_detectado:
            # Bug A: booking start with identity already in history.
            context_str += (
                f"\n[ACCIÓN] Inicio de agendamiento CON identidad ya conocida "
                f"(cédula={cedula_detectada}, nombre={nombre_detectado}). "
                "NO pidas cédula ni nombre otra vez. Confirma y pregunta el servicio/tratamiento. "
                "PROHIBIDO menú de bienvenida. Sin inventar citas ni horarios."
            )
        elif is_booking_start_intent(last_user_msg) and not cedula_detectada:
            # One LLM round, no catalog tools — correct booking step 1.
            context_str += (
                "\n[ACCIÓN] Inicio de agendamiento sin cédula: responde en texto pidiendo "
                "número de cédula y nombre completo. PROHIBIDO invocar herramientas en este turno."
            )

    combined_system_message = SystemMessage(
        content=f"{SYSTEM_MESSAGE.content}\n\n[CONTEXTO]\n{context_str}"
    )

    is_gemini = (settings.llm_provider or "openai").lower().strip() == "gemini"
    chat_messages = _trim_history(raw_msgs, is_gemini=is_gemini)
    messages = [combined_system_message, *chat_messages]

    prompt_chars = sum(len(str(getattr(m, "content", "") or "")) for m in messages)
    llm, tool_names = get_llm_with_tools(last_user_msg, prev_ai_msg, cedula_detectada)
    t_llm = time.perf_counter()
    response = await with_llm_slot(
        llm.ainvoke(messages),
        label="chatbot_ainvoke",
    )
    llm_elapsed = time.perf_counter() - t_llm
    n_tools = len(getattr(response, "tool_calls", None) or [])
    logger.info(
        f"[Latency] llm_turn={llm_elapsed:.3f}s prompt_chars={prompt_chars} "
        f"history_msgs={len(chat_messages)} tool_calls={n_tools} "
        f"bound_tools={list(tool_names)} "
        f"model={settings.gemini_model if is_gemini else settings.openai_model} "
        f"max_out={_CHAT_MAX_OUTPUT_TOKENS}"
    )

    # Solo propagar confianza si una tool RAG escribió [RAG_SCORE:...]; nunca default 1.0
    confidence = state.get("rag_confidence")
    for message in reversed(state.get("messages", [])):
        if isinstance(message, ToolMessage) and isinstance(message.content, str):
            match = re.search(r"\[RAG_SCORE:([0-9.]+)\]", message.content)
            if match:
                confidence = float(match.group(1))
                break

    out = {"messages": [response]}
    if confidence is not None:
        out["rag_confidence"] = confidence
    return out
