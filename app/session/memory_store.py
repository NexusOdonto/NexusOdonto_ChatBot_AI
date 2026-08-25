from typing import Any

from langgraph.checkpoint.memory import MemorySaver


# Un único checkpointer mantiene las conversaciones durante la ejecución.
memory_saver = MemorySaver()


def get_memory_saver() -> MemorySaver:
	"""Devuelve el checkpointer que usará el grafo de LangGraph."""
	return memory_saver


def get_thread_config(phone_number: str) -> dict[str, Any]:
	"""Construye la configuración que aísla el historial por teléfono."""
	normalized_phone = phone_number.strip()
	if not normalized_phone:
		raise ValueError("El número de teléfono no puede estar vacío")

	return {"configurable": {"thread_id": normalized_phone}}
