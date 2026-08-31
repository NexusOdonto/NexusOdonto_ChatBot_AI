from typing import Annotated, NotRequired, TypedDict, Optional

from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages


class AgentState(TypedDict):
	"""Estado compartido por los nodos del agente conversacional."""

	# add_messages conserva el historial y agrega los mensajes nuevos.
	messages: Annotated[list[AnyMessage], add_messages] # Lista de mensajes que conforman el historial de la conversación. Cada mensaje puede ser del tipo HumanMessage, AIMessage, SystemMessage, o ToolMessage.
	conversation_status: NotRequired[Optional[str]]
	rag_confidence: NotRequired[float]
	# Resumen comprimido del historial antiguo. Se actualiza automáticamente
	# cada vez que el nodo summarize_conversation_node reduce el historial.
	conversation_summary: NotRequired[Optional[str]]
	emergency_detected: NotRequired[bool]
	emergency_reason: NotRequired[Optional[str]]
