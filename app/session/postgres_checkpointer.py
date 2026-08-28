import logging
from typing import Optional
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from psycopg_pool import AsyncConnectionPool

logger = logging.getLogger(__name__)

checkpointer_instance: Optional["PostgresCheckpointer"] = None


class PostgresCheckpointer:
	"""Administra el pool y el saver async que persisten el grafo."""

	def __init__(self, connection_string: str) -> None:
		global checkpointer_instance
		# El pool reutiliza conexiones y configura psycopg para los checkpoints.
		self.pool = AsyncConnectionPool(
			conninfo=connection_string,
			kwargs={"autocommit": True, "prepare_threshold": 0},
			open=False,
		)
		self.saver = AsyncPostgresSaver(self.pool)
		checkpointer_instance = self

	async def start(self) -> None:
		"""Abre PostgreSQL y crea las tablas internas de LangGraph."""
		await self.pool.open()
		await self.saver.setup()

	async def stop(self) -> None:
		"""Cierra ordenadamente el pool cuando FastAPI se detiene."""
		await self.pool.close()

	async def clear_thread(self, thread_id: str) -> None:
		"""Elimina todos los checkpoints asociados a un thread_id en PostgreSQL."""
		try:
			async with self.pool.connection() as conn:
				await conn.execute("DELETE FROM checkpoints WHERE thread_id = %s", (thread_id,))
				await conn.execute("DELETE FROM checkpoint_blobs WHERE thread_id = %s", (thread_id,))
				await conn.execute("DELETE FROM checkpoint_writes WHERE thread_id = %s", (thread_id,))
			logger.info(f"[PostgresCheckpointer] Checkpoints eliminados exitosamente para thread_id: {thread_id}")
		except Exception as e:
			logger.error(f"[PostgresCheckpointer] Error al limpiar thread_id {thread_id}: {e}", exc_info=True)


def get_checkpointer_instance() -> Optional[PostgresCheckpointer]:
	return checkpointer_instance