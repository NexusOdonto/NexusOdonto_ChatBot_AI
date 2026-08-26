from functools import lru_cache
import re
import asyncio
from langchain_core.messages import SystemMessage, ToolMessage
from langchain_core.runnables import RunnableConfig
from langchain_openai import ChatOpenAI

from app.agents.tools.qdrant_tool import clinical_knowledge_tool
from app.agents.tools.agenda_tools import consultar_disponibilidad_tool, agendar_cita_tool
from app.clients.evolution_client import evolution_client
from app.core.config import settings
from app.graph.state import AgentState


# Define el rol, tono y límites de seguridad del asistente.
SYSTEM_MESSAGE = SystemMessage(
	content=(
		"Eres el asistente virtual de Nexus Odonto, un consultorio odontológico. "
		"Responde en español con un tono amable, profesional, claro y breve. "
		"Usa la herramienta buscar_conocimiento_clinico antes de responder sobre "
		"horarios generales, precios, servicios o preparaciones clínicas. Nunca inventes "
		"precios, horarios, políticas ni disponibilidad. "
		"Para verificar disponibilidad real de citas en una fecha, debes llamar a consultar_disponibilidad_tool con la especialidad y fecha (formato YYYY-MM-DD).\n"
		"IMPORTANTE PARA AGENDAR CITAS: Antes de llamar a la herramienta agendar_cita_tool, debes proponer obligatoriamente los detalles específicos de la cita (Doctor, Especialidad/Servicio, Fecha y Hora) al paciente y solicitarle su confirmación explícita (ej. 'Por favor confirma si estás de acuerdo con esta cita...'). "
		"SOLO si el paciente confirma de manera afirmativa y explícita, debes invocar la herramienta agendar_cita_tool. Nunca la invoques de forma anticipada sin confirmación.\n"
		"Nunca des diagnósticos médicos ni reemplaces la evaluación de un odontólogo. "
		"Ante síntomas o una urgencia, recomienda contactar directamente al consultorio o acudir a un servicio de urgencias."
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
	return llm.bind_tools([clinical_knowledge_tool, consultar_disponibilidad_tool, agendar_cita_tool])


from datetime import datetime
from zoneinfo import ZoneInfo

def chatbot_node(state: AgentState, config: RunnableConfig) -> dict[str, list]:
	"""Procesa el historial actual y agrega la respuesta del asistente."""
	# Determinar la fecha y hora actual local
	try:
		tz = ZoneInfo(settings.reminder_timezone)
	except Exception:
		tz = ZoneInfo("America/Bogota")
	now = datetime.now(tz)
	fecha_str = now.strftime("%Y-%m-%d")
	dias_semana = ["Lunes", "Martes", "Miércoles", "Jueves", "Viernes", "Sábado", "Domingo"]
	dia_nombre = dias_semana[now.weekday()]
	
	context_message = SystemMessage(
		content=f"Fecha y hora actual: {fecha_str} ({dia_nombre}, {now.strftime('%H:%M')}). Usa esta referencia para deducir fechas relativas (ej. 'el viernes' se refiere al próximo viernes respecto a hoy)."
	)
	
	messages = [SYSTEM_MESSAGE, context_message, *state["messages"]]
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
