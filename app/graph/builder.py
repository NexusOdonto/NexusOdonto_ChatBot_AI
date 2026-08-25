from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition

from app.agents.tools.qdrant_tool import clinical_knowledge_tool
from app.graph.nodes import chatbot_node
from app.graph.state import AgentState
from app.session.memory_store import get_memory_saver


# Construye el grafo conversacional con el nodo principal del asistente.
builder = StateGraph(AgentState)
builder.add_node("chatbot", chatbot_node)

# Ejecuta las llamadas a herramientas que el LLM solicite.
builder.add_node("tools", ToolNode([clinical_knowledge_tool]))
builder.add_edge(START, "chatbot")
builder.add_conditional_edges("chatbot", tools_condition)
builder.add_edge("tools", "chatbot")

# MemorySaver conserva el historial separado por thread_id.
graph = builder.compile(checkpointer=get_memory_saver())
