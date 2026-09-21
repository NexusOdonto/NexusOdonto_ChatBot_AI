"""Nodo de compresión de historial de conversación (Summarizer).
Comprime los mensajes antiguos cuando superan el umbral para mantener la memoria
eficiente y reducir drásticamente el consumo de tokens.
"""

import logging
from langchain_core.messages import SystemMessage, ToolMessage, AIMessage, HumanMessage, RemoveMessage
from app.core.llm_factory import get_evaluator_llm, extract_text_content
from app.graph.state import AgentState

logger = logging.getLogger(__name__)

SUMMARY_THRESHOLD: int = 25
RECENT_MESSAGES_KEEP: int = 6


async def summarize_conversation_node(state: AgentState) -> dict:
    """Comprime el historial antiguo cuando supera SUMMARY_THRESHOLD mensajes.
    Conserva los RECENT_MESSAGES_KEEP más recientes intactos y genera un resumen conciso.
    """
    messages = state.get("messages", [])
    total = len(messages)

    cut_idx = max(0, total - RECENT_MESSAGES_KEEP)
    while cut_idx > 0 and isinstance(messages[cut_idx], ToolMessage):
        cut_idx -= 1
    if cut_idx < total and cut_idx > 0 and isinstance(messages[cut_idx - 1], AIMessage) and getattr(messages[cut_idx - 1], "tool_calls", None):
        while cut_idx < total and isinstance(messages[cut_idx], ToolMessage):
            cut_idx += 1

    historic_messages = messages[:cut_idx]

    lines: list[str] = []
    for msg in historic_messages:
        if isinstance(msg, HumanMessage):
            lines.append(f"Paciente: {msg.content}")
        elif isinstance(msg, AIMessage) and msg.content and not msg.content.startswith("[Consultando"):
            lines.append(f"Asistente: {msg.content}")

    previous_summary = state.get("conversation_summary") or ""
    transcript = "\n".join(lines)

    summarizer_llm = get_evaluator_llm()

    sys_prompt = (
        "Eres un asistente que genera resúmenes concisos de conversaciones de WhatsApp "
        "entre un paciente y el chatbot del consultorio odontológico Nexus Odonto. "
        "Genera un único párrafo corto (máximo 120 palabras) en español que capture:"
        " el nombre del paciente (si fue mencionado), su número de cédula (si fue mencionado),"
        " los servicios o especialidades que consultó, las citas agendadas o canceladas, y cualquier información relevante "
        "para continuar la atención. No incluyas saludos ni explicaciones extra."
    )

    human_text = f"Transcripción a resumir:\n{transcript}"
    if previous_summary:
        human_text = f"Resumen previo (ya comprimido anteriormente):\n{previous_summary}\n\n{human_text}"

    try:
        response = await summarizer_llm.ainvoke([
            SystemMessage(content=sys_prompt),
            HumanMessage(content=human_text)
        ])
        new_summary: str = extract_text_content(response.content).strip()
        logger.info(
            f"[Summarizer] Historial comprimido: {len(historic_messages)} mensajes → resumen de {len(new_summary)} chars"
        )
    except Exception as exc:
        logger.warning(f"[Summarizer] Error al resumir historial: {exc}")
        new_summary = previous_summary or "Conversación previa sin resumen disponible."

    summary_message = SystemMessage(
        content=f"[RESUMEN DE CONVERSACIÓN PREVIA]\n{new_summary}"
    )

    removals = [RemoveMessage(id=m.id) for m in historic_messages if getattr(m, "id", None)]

    return {
        "messages": [*removals, summary_message],
        "conversation_summary": new_summary,
    }
