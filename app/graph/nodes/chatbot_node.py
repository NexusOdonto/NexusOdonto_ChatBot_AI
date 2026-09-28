"""Nodo principal del Asistente Virtual (Chatbot).
Configura el System Message, enlaza las herramientas clínicas y de agenda,
e invoca el modelo de lenguaje de forma resiliente con soporte multi-proveedor.
"""

import logging
import re
import time
from functools import lru_cache
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
from langchain_core.messages import SystemMessage, ToolMessage, AIMessage, HumanMessage
from app.core.llm_factory import get_chat_llm
from app.core.config import settings
from app.core.llm_concurrency import with_llm_slot
from app.core.llm_runtime import active_chat_model, active_provider, is_request_rejected, usage_summary
from app.graph.state import AgentState
from app.security.content_guard import detect_off_topic_non_dental
from app.services.booking_flow import is_unbacked_confirmation

from app.agents.tools.clinical_rag_tool import clinical_knowledge_tool
from app.agents.tools.agenda_helpers import fecha_legible, motivo_dia_cerrado
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

# Fixed system prompt. It must stay byte-identical across turns and come before any
# per-turn context so OpenAI's automatic prefix caching (>1024 tokens) keeps hitting.
SYSTEM_MESSAGE = SystemMessage(
    content=(
        "Eres una persona de recepción de *Nexus Odonto* escribiendo por WhatsApp "
        "(español de Colombia, natural y cercano). El paciente debe sentir que habla "
        "con alguien de la clínica, NO con un bot ni con un menú automático.\n\n"
        "PERSONA Y TONO:\n"
        "- Habla en primera persona como recepción: cálida, breve, conversacional.\n"
        "- Turnos cortos (2-5 líneas cuando se pueda). Una idea por mensaje.\n"
        "- Evita listas largas y numeradas si basta una pregunta natural.\n"
        "- Máximo 0-1 emoji por mensaje (a veces ninguno). Nunca spam de emojis.\n"
        "- Varía la redacción como lo haría una persona: NUNCA repitas literal un mensaje tuyo "
        "anterior de este chat (saludos, pedir cédula, despedidas); di lo mismo con otras palabras. "
        "Si conoces el primer nombre del paciente, úsalo de vez en cuando. "
        "Evita fórmulas rígidas como «Bienvenido/a», «Soy de recepción» o formas «o/a».\n"
        "- PROHIBIDO identificarte como asistente virtual, IA, bot, NexusBot o sistema. "
        "Nunca digas «soy un asistente», «he reactivado mi sistema» ni encabezados "
        "tipo «Asistente Virtual».\n"
        "- PROHIBIDO inventar datos, citas, horarios, precios o nombres.\n"
        "- NUNCA hables de fallas, sistema caído, errores ni pidas «intenta más tarde»: si una herramienta "
        "no trae datos, sigue la conversación (ofrece otra opción, pide un dato o sugiere llamar al +57 324 6030217).\n\n"
        "FORMATO WHATSAPP: *negrita* con un solo asterisco (nunca **). "
        "Si usas viñetas, máximo 3-4 y con •.\n\n"
        "ALCANCE: solo odontología / citas / servicios / precios / doctores / cuidados. "
        "Si el dolor es de otra parte del cuerpo (rodilla, espalda, etc.): explica con amabilidad "
        "que Nexus Odonto solo atiende salud oral; NO inventes urgencia dental ni agendes esa cita. "
        "Si hay vulgaridad, broma ofensiva o falta de respeto (ej. 'muela del ano'): pide respeto "
        "en corto y natural (no un sermón largo ni el mismo párrafo cada vez). "
        "Si se disculpa pero sigue con la broma: reconoce el perdón y pide un caso dental real. "
        "NO trates eso como dolor dental ni avances a agendar. "
        "Fuera de tema no odontológico: orienta amable al alcance de la clínica. "
        "NO cambies contraseñas: indica login web https://nexusodonto.chatcampuslands.com/login "
        "(cédula + contraseña temporal).\n\n"
        "CONTACTO: +57 324 6030217 | Calle 100 # 15-20 | soporte@nexusodonto.com | "
        "Lun-Vie 8:00 AM–12:00 PM y 2:00–5:00 PM; Sáb 8:00 AM–12:00 PM; Dom/festivos cerrado\n\n"
        "HABEAS DATA: NUNCA asumas nombre ni cédula. Pide SIEMPRE la cédula (mín. 7 dígitos) "
        "antes de ver/agendar/reprogramar/cancelar/confirmar. Si ya la escribió en este chat, no la vuelvas a pedir. "
        "PROHIBIDO inventar nombres (p. ej. 'Paciente Nexus').\n\n"
        "HERRAMIENTAS (usa solo cuando haga falta datos reales):\n"
        "1. consultar_doctores_tool\n"
        "2. consultar_servicios_y_precios_tool — fuente de verdad de precios; NUNCA inventes tarifas. "
        "No vuelques el catálogo entero: destaca 3-4 servicios y pregunta qué necesita.\n"
        "3. consultar_disponibilidad_tool(servicio, fecha YYYY-MM-DD opcional) — sin fecha trae los próximos "
        "días con turnos desde hoy (úsalo si pregunta qué días hay)\n"
        "4. agendar_cita_tool — SOLO tras confirmación explícita del paciente y con nombre+cédula+servicio+horario. "
        "Si el paciente pidió un odontólogo por nombre (ej: Ana Sofía), pasa ESE nombre en profesional_id; "
        "PROHIBIDO sustituirlo por Laura Gómez u otro por defecto/cabecera.\n"
        "5. consultar_cita_por_cedula_tool — VER citas. NO usar para agendar ni reprogramar\n"
        "6. modificar_cita_tool(cedula, nueva_fecha_hora opcional) — reprogramar\n"
        "7. cancelar_cita_tool(cedula, cita_id opcional)\n"
        "8. confirmar_cita_tool(cedula)\n"
        "9. buscar_conocimiento_clinico — dudas clínicas (sin mezclar con agenda)\n\n"
        "AGENDAR — primera respuesta si pide cita y falta cédula: "
        "pide cédula y nombre completo en tono natural (sin menú ni checklist robótico). "
        "NO invoques herramientas todavía.\n"
        "Si ya dio cédula+nombre en este chat (aunque sea antes del 'quiero agendar'), "
        "NO vuelvas a pedirlos: confirma y pregunta el servicio/tratamiento.\n"
        "Si responde con cédula+nombre tras pedirlos para agendar: "
        "PROHIBIDO menú de bienvenida ni '¿en qué te ayudamos?'; "
        "confirma y pregunta servicio. PROHIBIDO consultar_cita_por_cedula_tool.\n"
        "Si responde solo con cédula en hilo de agendar: "
        "PROHIBIDO consultar_cita_por_cedula_tool; confirma cédula y pide nombre si falta, luego servicio/horario.\n"
        "Protocolo: (1) cédula+nombre+servicio+horario vía disponibilidad "
        "(2) resume la propuesta de cita de forma clara (sin encabezados de bot) "
        "(3) solo si confirma ('sí'/'confirmo'), llama agendar_cita_tool. "
        "PROHIBIDO agendar un día/hora que el paciente no eligió explícitamente.\n"
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
        "FECHAS: NUNCA calcules tú el día de la semana. Usa el CALENDARIO del contexto y las fechas que "
        "traen las herramientas (ya vienen con su día). «el martes», «el viernes», «el próximo lunes» = "
        "el más cercano hacia adelante en el calendario; «mañana» = hoy + 1. Al proponer o confirmar una "
        "cita di siempre día + fecha tal como aparecen (ej. «jueves 1 de octubre»).\n"
        "DISPONIBILIDAD: ofrece solo 2-4 opciones (mezcla días y horas) y pregunta cuál prefiere; más "
        "horarios solo si los pide. Si el día pedido es domingo o festivo, di con naturalidad que ese día "
        "no atendemos y ofrece los días más cercanos (no digas «no hay turnos con ningún odontólogo»).\n"
        "PRECIOS: formato colombiano $960.000, sin «COP». Al dar precios no pidas cédula; pregunta si "
        "quiere agendar. Pide cédula y nombre UNA sola vez, cuando ya va a apartar.\n"
        "CONFIRMACIONES: PROHIBIDO decir que una cita quedó agendada, reprogramada o cancelada si en "
        "ESTE turno no recibiste el resultado exitoso de la herramienta. Fecha, hora y odontólogo de la "
        "confirmación: cópialos del resultado de la herramienta.\n\n"
        "DOLOR DENTAL / SIN CUPOS: solo si el dolor es de diente, muela, encía o boca — "
        "empatía + sobrecupo presencial + línea +57 324 6030217 + paliativos seguros "
        "(compresa fría, enjuague salino; NUNCA aspirina sobre el diente). "
        "Sin diagnósticos invasivos. Urgencias dentales severas reales (sangrado oral que no para, "
        "trauma, hinchazón con dificultad respiratoria) → orientar a centro médico; "
        "NO dispares alerta de urgencia por cualquier 'duele' ni por zonas no dentales. "
        "Máximo 0-1 emoji; sin spam de sirenas ni banners."
    )
)

# Cap history fed to the LLM (tool schemas + system already dominate input tokens).
_MAX_HISTORY_MESSAGES = 16
_MAX_TOOL_MSG_CHARS = 1800
_MAX_AI_MSG_CHARS = 1200

# Chat replies on WhatsApp stay short; lower caps cut decode time and output tokens.
_CHAT_MAX_OUTPUT_TOKENS = 400
_CHAT_TEMPERATURE = 0.4
_MAX_TOOL_RESULTS_PER_TURN = 4

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


def _same_day_request(last_user_msg: str, now: datetime) -> tuple[datetime, bool] | None:
    """(requested datetime, already_passed) for 'hoy a las X' requests; 15-min prep margin counts as passed."""
    from app.services.booking_flow import parse_same_day_time

    norm = (last_user_msg or "").lower()
    if any(k in norm for k in ("cancelar", "anular", "reprogramar", "modificar")):
        return None
    hm = parse_same_day_time(last_user_msg)
    if not hm:
        return None
    req = now.replace(hour=hm[0], minute=hm[1], second=0, microsecond=0)
    return req, req < now + timedelta(minutes=15)


_CLINICAL_KEYWORDS = (
    "dolor", "duele", "brackets", "ortodoncia", "extraccion", "extracción", "cuidado",
    "recomienda", "que comer", "qué comer", "preparacion", "preparación", "sensibilidad",
    "sangra", "encia", "encía", "caries",
)
_WEEKDAY_WORDS = {
    "lunes": 0, "martes": 1, "miercoles": 2, "miércoles": 2, "jueves": 3,
    "viernes": 4, "sabado": 5, "sábado": 5, "domingo": 6,
}


def _is_clinical_question(norm: str) -> bool:
    return any(k in norm for k in _CLINICAL_KEYWORDS) and not any(
        k in norm for k in ("cita", "agendar", "cancelar", "modificar", "reprogramar")
    )


def _requested_date(last_user_msg: str, now: datetime) -> tuple[str, datetime] | None:
    """('sábado', date) when the patient names a weekday or 'mañana'; weekdays mean the next one ahead."""
    norm = (last_user_msg or "").lower()
    if "pasado mañana" in norm or "pasado manana" in norm:
        return "pasado mañana", now + timedelta(days=2)
    if re.search(r"\bma(ñ|n)ana\b", norm) and not re.search(r"\b(en|por|de) la ma(ñ|n)ana\b", norm):
        return "mañana", now + timedelta(days=1)
    for word, weekday in _WEEKDAY_WORDS.items():
        if re.search(rf"\b{word}\b", norm):
            ahead = (weekday - now.weekday()) % 7 or 7
            return word, now + timedelta(days=ahead)
    return None


def _calendario(now: datetime, dias: int = 10) -> str:
    lineas = []
    for i in range(dias):
        d = now + timedelta(days=i)
        etiqueta = " (hoy)" if i == 0 else " (mañana)" if i == 1 else ""
        cerrado = motivo_dia_cerrado(d.replace(tzinfo=None))
        nota = " — cerrado" if cerrado else ""
        lineas.append(f"{d.strftime('%Y-%m-%d')} = {fecha_legible(d, con_anio=False)}{etiqueta}{nota}")
    return "\n".join(lineas)


def _cedula_from_summaries(raw_msgs: list) -> str | None:
    for msg in reversed(raw_msgs):
        if isinstance(msg, SystemMessage):
            m = re.search(r"\b(\d{7,12})\b", str(msg.content or ""))
            if m:
                return m.group(1)
    return None


_BOOKING_WORDS = ("cita", "turno", "horario", "disponib", "agend", "a las", "hay", "puedo ir", "sirve")
_CONFIRM_RE = re.compile(
    r"\b(s[ií]|confirmo|dale|listo|ok|okay|perfecto|de acuerdo|correcto|h[aá]gale|ag[eé]nd[ae]la)\b"
)
_HORA_PEDIDA_RE = re.compile(
    r"\ba\s+las?\s+(\d{1,2})(?:[:.](\d{2}))?\s*(a\.?\s*m\.?|p\.?\s*m\.?|de la ma[ñn]ana|de la tarde)?"
)


def _hora_pedida(last_user_msg: str) -> str | None:
    """'el martes a las 9' → '09:00'; bare 1-6 means afternoon (clinic hours)."""
    m = _HORA_PEDIDA_RE.search((last_user_msg or "").lower())
    if not m:
        return None
    h, mi, suf = int(m.group(1)), int(m.group(2) or 0), (m.group(3) or "").replace(" ", "").replace(".", "")
    if suf in ("pm", "delatarde") and h < 12:
        h += 12
    elif not suf and 1 <= h <= 6:
        h += 12
    return f"{h:02d}:{mi:02d}" if h <= 23 and mi <= 59 else None

_UNBACKED_CONFIRMATION_NUDGE = (
    "[SISTEMA] Tu respuesta afirma que la cita quedó agendada/reprogramada/cancelada, pero en este turno "
    "NO hay un resultado exitoso de agendar_cita_tool, modificar_cita_tool ni cancelar_cita_tool: la agenda "
    "no cambió. Si el paciente ya confirmó y tienes cédula, nombre, servicio, odontólogo y fecha/hora, "
    "llama la herramienta ahora. Si falta algo o no ha confirmado, resume la propuesta y pregunta; "
    "no digas que quedó hecha."
)


def _tool_messages_this_turn(raw_msgs: list) -> list:
    out = []
    for msg in reversed(raw_msgs):
        if isinstance(msg, HumanMessage):
            break
        if isinstance(msg, ToolMessage):
            out.append(msg)
    return out


def _tool_results_this_turn(raw_msgs: list) -> int:
    count = 0
    for msg in reversed(raw_msgs):
        if isinstance(msg, HumanMessage):
            break
        if isinstance(msg, ToolMessage):
            count += 1
    return count


_BOOKING_TOOLS = (
    consultar_disponibilidad_tool,
    agendar_cita_tool,
    modificar_cita_tool,
    cancelar_cita_tool,
    consultar_servicios_y_precios_tool,
    consultar_doctores_tool,
)


def _select_tools(
    last_user_msg: str,
    prev_ai_msg: str,
    cedula: str | None,
    same_day: tuple[datetime, bool] | None = None,
    booking_active: bool = False,
) -> tuple:
    """Bind only tools likely needed this turn — smaller schemas → fewer input tokens.

    While a booking is in progress the booking tools stay bound: without agendar_cita_tool the
    model can only *claim* the appointment was created.
    """
    norm = (last_user_msg or "").lower()
    prev = (prev_ai_msg or "").lower()

    if same_day and not cedula:
        # Identity step first: the hint already carries today's remaining hours, no lookup needed.
        return tuple()

    # Pure price / catalog questions
    if any(k in norm for k in ("precio", "precios", "vale", "cuesta", "cuanto", "tarif")) and not any(
        k in norm for k in ("cita", "agendar", "cancelar", "modificar", "reprogramar")
    ):
        if booking_active:
            return _BOOKING_TOOLS
        return (consultar_servicios_y_precios_tool, consultar_doctores_tool)

    # Clinical Q&A without booking language
    if _is_clinical_question(norm):
        return (clinical_knowledge_tool, consultar_servicios_y_precios_tool)

    if any(k in norm for k in ("servicio", "servicios", "tratamiento", "tratamientos", "limpieza", "profilaxis", "blanqueamiento")) and not any(
        k in norm for k in ("cita", "agendar", "disponib")
    ):
        if booking_active:
            return _BOOKING_TOOLS
        return (consultar_servicios_y_precios_tool,)

    if any(k in norm for k in ("doctor", "doctora", "odontologo", "especialista", "especialistas")):
        return (consultar_doctores_tool, consultar_servicios_y_precios_tool)

    # Booking start without cédula: no tools (text-only)
    if any(k in norm for k in ("quiero una cita", "quiero agendar", "necesito una cita", "agendar cita")) and not cedula:
        return tuple()

    # Reschedule / cancel paths (before the identity rule: a cédula sent to cancel is not a new booking)
    if any(k in norm for k in ("reprogramar", "modificar")) or any(k in prev for k in ("reprogramar", "modificar")):
        return (modificar_cita_tool, consultar_disponibilidad_tool, consultar_servicios_y_precios_tool)
    if "cancelar" in norm or "cancelar" in prev:
        return (cancelar_cita_tool,)
    if "confirmar" in norm:
        return (confirmar_cita_tool,)

    # Identity provided mid-booking (never consultar_cita_por_cedula here). Sending cédula+nombre is
    # not a "yes": without an explicit confirmation in the same message the bot must summarize and
    # ask first, so agendar is only bound when the message also confirms.
    if re.search(r"\b\d{7,12}\b", norm) and (
        booking_active or any(k in prev for k in ("cédula", "cedula", "nombre completo", "agendar"))
    ):
        if _CONFIRM_RE.search(norm):
            return _BOOKING_TOOLS
        return tuple(t for t in _BOOKING_TOOLS if t is not agendar_cita_tool)

    if booking_active:
        return _BOOKING_TOOLS

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


@lru_cache(maxsize=32)
def _bound_llm_for_tools(tool_names: tuple[str, ...], tool_choice: str | None = None):
    name_to_tool = {t.name: t for t in _ALL_TOOLS}
    tools = [name_to_tool[n] for n in tool_names if n in name_to_tool]
    primary_llm = get_chat_llm(temperature=_CHAT_TEMPERATURE, max_tokens=_CHAT_MAX_OUTPUT_TOKENS)
    if not tools:
        return primary_llm
    if tool_choice:
        return primary_llm.bind_tools(tools, tool_choice=tool_choice)
    return primary_llm.bind_tools(tools)


def get_llm_with_tools(
    last_user_msg: str = "",
    prev_ai_msg: str = "",
    cedula: str | None = None,
    same_day: tuple[datetime, bool] | None = None,
    booking_active: bool = False,
):
    selected = _select_tools(last_user_msg, prev_ai_msg, cedula, same_day, booking_active)
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
            if len(content) > _MAX_AI_MSG_CHARS and not getattr(msg, "tool_calls", None):
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
            chat_messages.append(_tool_result_as_context(msg))
        elif isinstance(msg, ToolMessage):
            body = str(msg.content or "")
            if len(body) > _MAX_TOOL_MSG_CHARS:
                msg = ToolMessage(
                    content=body[:_MAX_TOOL_MSG_CHARS] + "…",
                    tool_call_id=msg.tool_call_id,
                    name=getattr(msg, "name", None),
                )
            chat_messages.append(msg)
        else:
            chat_messages.append(msg)

    if len(chat_messages) > _MAX_HISTORY_MESSAGES:
        chat_messages = chat_messages[-_MAX_HISTORY_MESSAGES:]
    if not is_gemini:
        chat_messages = _pair_tool_messages(chat_messages)
    return chat_messages


def _tool_result_as_context(msg: ToolMessage) -> HumanMessage:
    tool_name = getattr(msg, "name", None) or "herramienta"
    body = str(msg.content or "")
    if len(body) > _MAX_TOOL_MSG_CHARS:
        body = body[:_MAX_TOOL_MSG_CHARS] + "…"
    return HumanMessage(content=f"[Información del sistema ({tool_name})]:\n{body}")


def _pair_tool_messages(msgs: list) -> list:
    """OpenAI rejects tool calls without their results and results without their call.

    Keeps each AI tool-call message only together with the tool results that follow it;
    orphaned results (e.g. cut by the history window) become plain context.
    """
    out: list = []
    i = 0
    while i < len(msgs):
        msg = msgs[i]
        if isinstance(msg, AIMessage) and getattr(msg, "tool_calls", None):
            j = i + 1
            results = []
            while j < len(msgs) and isinstance(msgs[j], ToolMessage):
                results.append(msgs[j])
                j += 1
            call_ids = {tc.get("id") for tc in msg.tool_calls}
            if call_ids and call_ids <= {r.tool_call_id for r in results}:
                out.append(msg)
                out.extend(r for r in results if r.tool_call_id in call_ids)
            else:
                if msg.content:
                    out.append(AIMessage(content=msg.content))
                out.extend(_tool_result_as_context(r) for r in results)
            i = j
            continue
        if isinstance(msg, ToolMessage):
            out.append(_tool_result_as_context(msg))
        else:
            out.append(msg)
        i += 1
    return out


def _same_day_hint(same_day: tuple[datetime, bool], now: datetime, cedula_conocida: bool) -> str:
    from app.agents.tools.agenda_helpers import (
        _formatear_hora_ampm,
        _validar_horario_cita,
        franjas_restantes_hoy,
    )

    req, passed = same_day
    hora_req = _formatear_hora_ampm(req.strftime("%H:%M"))
    pedir_id = (
        "" if cedula_conocida
        else " Para apartarla pide su número de cédula y nombre completo (y el tratamiento si no lo ha dicho)."
    )
    if passed:
        restantes = franjas_restantes_hoy(now)
        if restantes:
            return (
                f"\n[ACCIÓN] Pidió hoy a las {hora_req}, pero esa hora ya pasó (son las "
                f"{_formatear_hora_ampm(now.strftime('%H:%M'))}). Díselo con naturalidad, sin tono de error, "
                f"y ofrécele lo que queda de hoy: atendemos {restantes}."
                + (
                    f"{pedir_id} Los turnos exactos se confirman cuando tengamos esos datos."
                    if pedir_id
                    else " Si ya sabes el tratamiento, usa consultar_disponibilidad_tool con la fecha de HOY "
                    "para darle turnos exactos."
                )
            )
        return (
            f"\n[ACCIÓN] Pidió hoy a las {hora_req}, pero por hoy ya cerramos la jornada. Díselo con "
            f"naturalidad y ofrécele el próximo día hábil.{pedir_id}"
        )
    ok, motivo = _validar_horario_cita(req.replace(tzinfo=None), 30)
    if not ok and motivo:
        return (
            f"\n[ACCIÓN] Pidió hoy a las {hora_req}, pero no está dentro de la jornada: {motivo} "
            f"Explícalo en corto y natural.{pedir_id}"
        )
    return (
        f"\n[ACCIÓN] Pidió hoy a las {hora_req}: es una hora válida dentro de la jornada, continúa el "
        "agendamiento con normalidad (no digas que ya pasó ni que no hay cupo sin consultar)."
        + (pedir_id or " Confirma el tratamiento y consulta disponibilidad de hoy antes de proponer la cita.")
    )


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

    # Compact temporal block (rules already in SYSTEM_MESSAGE). The explicit calendar keeps the
    # model from miscounting weekdays ("el sábado") into the wrong date.
    context_str = (
        f"HOY={fecha_legible(now)} ({fecha_str}) HORA_CO={hora_str}.\n"
        f"CALENDARIO (fecha = día de la semana; úsalo tal cual):\n{_calendario(now)}\n"
        "No ofrezcas horarios pasados. Disponibilidad solo con consultar_disponibilidad_tool."
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
    asked_identity = prev_asked_for_booking_identity(prev_ai_msg)
    if last_user_msg:
        turn_id = parse_identity_from_text(last_user_msg, allow_name_only=asked_identity)
        if turn_id.cedula:
            cedula_detectada = turn_id.cedula
        if turn_id.nombre:
            nombre_detectado = turn_id.nombre

    uc = state.get("user_context") or {}
    if not cedula_detectada and uc.get("cedula"):
        cedula_detectada = str(uc.get("cedula")).strip() or None
    if not nombre_detectado and uc.get("nombre"):
        nombre_detectado = str(uc.get("nombre")).strip() or None
    if not cedula_detectada:
        cedula_detectada = _cedula_from_summaries(raw_msgs)

    same_day = _same_day_request(last_user_msg, now) if last_user_msg else None
    tool_results = _tool_results_this_turn(raw_msgs)
    requested = None

    # Inyección contextual de acción inmediata para evitar desvíos o alucinaciones
    if last_user_msg:
        norm_user = last_user_msg.lower()
        turn_id = parse_identity_from_text(last_user_msg, allow_name_only=asked_identity)
        if same_day and not turn_id.cedula:
            context_str += _same_day_hint(same_day, now, bool(cedula_detectada))
        elif re.match(r"^\d{7,12}$", last_user_msg) and any(
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
                f"cancelar_cita_tool(cedula='{cedula_detectada}') YA. No pidas la cédula otra vez."
            )
        elif turn_id.cedula and any(w in prev_ai_msg for w in ["cancelar", "anular"]):
            context_str += (
                f"\n[ACCIÓN] Cédula {turn_id.cedula} para CANCELAR → "
                f"cancelar_cita_tool(cedula='{turn_id.cedula}') YA. No es un agendamiento nuevo."
            )
        elif turn_id.cedula and (prev_asked_for_booking_identity(prev_ai_msg) or uc.get("booking_flow")):
            # Bug B: after collecting identity for agendar, continue — never welcome menu.
            nombre_txt = turn_id.nombre or nombre_detectado or ""
            context_str += (
                f"\n[ACCIÓN] Paciente entregó identidad para AGENDAR "
                f"(cédula={turn_id.cedula}"
                + (f", nombre={nombre_txt}" if nombre_txt else "")
                + "). "
                "PROHIBIDO menú de bienvenida / '¿En qué te podemos ayudar?' y consultar_cita_por_cedula_tool. "
                "Si en el chat ya eligió servicio y día/hora: resume la cita (servicio, odontólogo, día de la "
                "semana + fecha del calendario, hora) y pide que confirme; si ya la había confirmado de forma "
                "explícita, llama agendar_cita_tool ahora. Si falta el servicio o el horario, pregúntalo. "
                "No digas que quedó agendada sin el resultado exitoso de agendar_cita_tool."
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

        requested = _requested_date(last_user_msg, now) if not same_day else None
        if requested:
            word, day = requested
            context_str += (
                f"\n[FECHA PEDIDA] «{word}» = {dias_semana[day.weekday()]} {day.strftime('%Y-%m-%d')}. "
                "Llama consultar_disponibilidad_tool con EXACTAMENTE esa fecha antes de responder y "
                "no digas que un horario está libre sin verlo en su resultado; usa esa misma fecha en "
                "agendar_cita_tool. No reutilices fechas de mensajes anteriores."
            )
        hora_pedida = _hora_pedida(last_user_msg) if requested else None
        if hora_pedida:
            context_str += (
                f"\n[HORA PEDIDA] {hora_pedida}: pásala como hora='{hora_pedida}' en "
                "consultar_disponibilidad_tool y responde según la línea [HORA PEDIDA] del resultado."
            )
        if _is_clinical_question(norm_user) and not tool_results:
            context_str += (
                f"\n[ACCIÓN] Pregunta clínica: consulta {clinical_knowledge_tool.name} antes de responder "
                "y basa tu respuesta en lo que devuelva."
            )

    off_topic = bool(last_user_msg) and detect_off_topic_non_dental(last_user_msg)
    if off_topic:
        context_str += (
            "\n[ACCIÓN] El paciente habla de un dolor o cita de una zona NO dental. En corto y con "
            "amabilidad: Nexus Odonto solo atiende salud oral, sugiérele un médico general y ofrece "
            "ayuda con algo dental. NO agendes ni lo trates como urgencia dental."
        )

    booking_flow = str(uc.get("booking_flow") or "").strip()
    if booking_flow:
        context_str += (
            f"\n[CITA EN CURSO] {booking_flow}. El último mensaje del paciente sigue esa cita: "
            "continúa desde ahí aunque tenga errores de tipeo. PROHIBIDO reiniciar, "
            "volver a pedir datos ya dados o preguntar «¿en qué te ayudo?» / «¿es para agendar...?». "
            "Horarios reales con consultar_disponibilidad_tool (servicio + fecha YYYY-MM-DD); "
            "si el día está cerrado u ocupado, dilo natural y ofrece 2-4 turnos cercanos. "
            "Solo tras un «sí» explícito crea la cita con agendar_cita_tool."
        )

    combined_system_message = SystemMessage(
        content=f"{SYSTEM_MESSAGE.content}\n\n[CONTEXTO]\n{context_str}"
    )

    is_gemini = active_provider() == "gemini"
    chat_messages = _trim_history(raw_msgs, is_gemini=is_gemini)
    messages = [combined_system_message, *chat_messages]

    prompt_chars = sum(len(str(getattr(m, "content", "") or "")) for m in messages)
    if off_topic:
        llm, tool_names = _bound_llm_for_tools(()), ()
    else:
        llm, tool_names = get_llm_with_tools(
            last_user_msg, prev_ai_msg, cedula_detectada, same_day, booking_active=bool(booking_flow)
        )
        if tool_names and not is_gemini and tool_results >= _MAX_TOOL_RESULTS_PER_TURN:
            # Stops tool loops (same lookup repeated) — answer with the data already gathered.
            llm = _bound_llm_for_tools(tool_names, tool_choice="none")
        elif (
            requested
            and not is_gemini
            and tool_results == 0
            and consultar_disponibilidad_tool.name in tool_names
            and (booking_flow or any(k in last_user_msg.lower() for k in _BOOKING_WORDS))
        ):
            # The patient named a day: answer from a fresh lookup of that date, never from memory
            # of earlier suggestions (that is how a free 9:00 AM was reported as taken).
            llm = _bound_llm_for_tools(tool_names, tool_choice=consultar_disponibilidad_tool.name)
    t_llm = time.perf_counter()
    try:
        response = await with_llm_slot(
            llm.ainvoke(messages),
            label="chatbot_ainvoke",
        )
    except Exception as llm_err:
        if not is_request_rejected(llm_err) or not tool_names:
            raise
        # A rejected tool schema / call must not surface as a service outage; answer text-only.
        logger.warning(f"[Chatbot] LLM call failed with tools={list(tool_names)} ({llm_err!r}); retrying without tools")
        llm = _bound_llm_for_tools(())
        response = await with_llm_slot(llm.ainvoke(messages), label="chatbot_ainvoke_notools")
    if not str(response.content or "").strip() and not getattr(response, "tool_calls", None):
        logger.warning("[Chatbot] Empty LLM reply; retrying once")
        response = await with_llm_slot(llm.ainvoke(messages), label="chatbot_ainvoke_retry")
    if not getattr(response, "tool_calls", None) and is_unbacked_confirmation(
        str(response.content or ""), _tool_messages_this_turn(raw_msgs)
    ):
        logger.warning(
            f"[Chatbot] Reply claims an agenda change without a successful tool result "
            f"(bound_tools={list(tool_names)}); retrying with booking tools"
        )
        retry_llm = _bound_llm_for_tools(tuple(
            t.name for t in _BOOKING_TOOLS if t is not agendar_cita_tool or t.name in tool_names
        ))
        response = await with_llm_slot(
            retry_llm.ainvoke([*messages, HumanMessage(content=_UNBACKED_CONFIRMATION_NUDGE)]),
            label="chatbot_ainvoke_unbacked",
        )
    llm_elapsed = time.perf_counter() - t_llm
    n_tools = len(getattr(response, "tool_calls", None) or [])
    logger.info(
        f"[Latency] llm_turn={llm_elapsed:.3f}s prompt_chars={prompt_chars} "
        f"history_msgs={len(chat_messages)} tool_calls={n_tools} "
        f"bound_tools={list(tool_names)} "
        f"provider={active_provider()} model={active_chat_model()} "
        f"max_out={_CHAT_MAX_OUTPUT_TOKENS} {usage_summary(response)}"
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
