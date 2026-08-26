from functools import lru_cache
import re
import asyncio
from langchain_core.messages import SystemMessage, ToolMessage
from langchain_core.runnables import RunnableConfig
from langchain_openai import ChatOpenAI

from app.agents.tools.qdrant_tool import clinical_knowledge_tool
from app.clients.evolution_client import evolution_client
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


def chatbot_node(state: AgentState, config: RunnableConfig) -> dict[str, list]:
	"""Procesa el historial actual y agrega la respuesta del asistente."""
	messages = [SYSTEM_MESSAGE, *state["messages"]]
	response = get_llm_with_tools().invoke(messages)
	confidence = state.get("rag_confidence", 1.0)
	for message in reversed(state["messages"]):
		# Recuperamos el último score producido por la herramienta clínica.
		if isinstance(message, ToolMessage) and isinstance(message.content, str):
			match = re.search(r"\[RAG_SCORE:([0-9.]+)\]", message.content)
			if match:
				# El score se guarda en el estado para decidir si hay que escalar.
				confidence = float(match.group(1))
				break
	
	# Enviar el mensaje generado directamente a WhatsApp si tiene texto
	if isinstance(response.content, str) and response.content.strip():
		thread_id = config["configurable"]["thread_id"]
		asyncio.run(evolution_client.enviar_mensaje(thread_id, response.content))
		
	return {"messages": [response], "rag_confidence": confidence}
