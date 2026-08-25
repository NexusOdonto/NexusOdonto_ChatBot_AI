from functools import lru_cache

from langchain_core.messages import SystemMessage
from langchain_openai import ChatOpenAI

from app.agents.tools.qdrant_tool import clinical_knowledge_tool
from app.core.config import settings
from app.graph.state import AgentState


# Define el rol, tono y límites de seguridad del asistente.
SYSTEM_MESSAGE = SystemMessage(
	content=(
		"Eres el asistente virtual de Nexus Odonto, un consultorio odontologico. "
		"Responde en espanol con un tono amable, profesional, claro y breve. "
		"Usa la herramienta buscar_conocimiento_clinico antes de responder sobre "
		"horarios, precios, servicios o preparaciones clinicas. Nunca inventes "
		"precios, horarios, politicas ni disponibilidad. Para agendar, cancelar "
		"o modificar citas debes consultar la API correspondiente; si no esta "
		"disponible, dilo claramente y no confirmes ninguna cita. Nunca des "
		"diagnosticos medicos ni reemplaces la evaluacion de un odontologo. "
		"Ante sintomas o una urgencia, recomienda contactar directamente al "
		"consultorio o acudir a un servicio de urgencias."
	)
)


@lru_cache(maxsize=1)
def get_llm_with_tools():
	# Crea GPT-4o solo cuando llega una solicitud que necesita el LLM.
	if not settings.openai_api_key:
		raise RuntimeError("OPENAI_API_KEY es necesaria para usar el chatbot")

	llm = ChatOpenAI(
		model=settings.openai_model,
		temperature=0,
		api_key=settings.openai_api_key,
	)
	return llm.bind_tools([clinical_knowledge_tool])


def chatbot_node(state: AgentState) -> dict[str, list]:
	"""Procesa el historial actual y agrega la respuesta del asistente."""
	messages = [SYSTEM_MESSAGE, *state["messages"]]
	response = get_llm_with_tools().invoke(messages)
	return {"messages": [response]}
