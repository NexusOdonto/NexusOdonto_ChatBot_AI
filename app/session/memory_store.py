from typing import Any

def get_thread_config(phone_number: str) -> dict[str, Any]:
	"""Construye la configuración que identifica el hilo persistente por teléfono."""
	normalized_phone = phone_number.strip()
	if not normalized_phone:
		raise ValueError("El número de teléfono no puede estar vacío")

	# El mismo thread_id permite recuperar el estado almacenado en PostgreSQL.
	return {"configurable": {"thread_id": normalized_phone}}
