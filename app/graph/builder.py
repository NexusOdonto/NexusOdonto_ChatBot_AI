from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition

from app.agents.tools.qdrant_tool import clinical_knowledge_tool
from app.agents.tools.agenda_tools import consultar_disponibilidad_tool, agendar_cita_tool
from app.graph.nodes import chatbot_node, security_check_node
from app.graph.state import AgentState
from langgraph.checkpoint.base import BaseCheckpointSaver


graph = None


def security_router(state: AgentState) -> str:
	"""Enruta al final si el estado está BLOQUEADA, o al chatbot si está limpia."""
	if state.get("conversation_status") == "BLOQUEADA":
		return END
	return "chatbot"


def create_graph(checkpointer: BaseCheckpointSaver) -> None:
	"""Compila el grafo con el saver PostgreSQL inicializado en FastAPI."""
	global graph

	# PostgreSQL conserva los mensajes y el estado aun después de reiniciar el servidor.
	builder = StateGraph(AgentState)
	builder.add_node("security_check", security_check_node)
	builder.add_node("chatbot", chatbot_node)
	# Ejecuta las llamadas a herramientas que el LLM solicite.
	builder.add_node("tools", ToolNode([clinical_knowledge_tool, consultar_disponibilidad_tool, agendar_cita_tool]))
	
	builder.add_edge(START, "security_check")
	builder.add_conditional_edges("security_check", security_router)
	builder.add_conditional_edges("chatbot", tools_condition)
	builder.add_edge("tools", "chatbot")
	graph = builder.compile(checkpointer=checkpointer)


def get_graph():
	"""Devuelve el grafo listo para atender mensajes."""
	if graph is None:
		raise RuntimeError("El grafo conversacional todavía no está inicializado")
	return graph
