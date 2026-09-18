import logging
from typing import Optional
try:
	from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
	from psycopg_pool import AsyncConnectionPool
except ImportError:
	AsyncPostgresSaver = None
	AsyncConnectionPool = None

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
		"""Abre PostgreSQL y crea las tablas internas de LangGraph con reintentos."""
		import asyncio
		max_retries = 10
		for attempt in range(1, max_retries + 1):
			try:
				await self.pool.open()
				await self.saver.setup()
				async with self.pool.connection() as conn:
					await conn.execute("""
						CREATE TABLE IF NOT EXISTS conversation_sessions (
							thread_id TEXT PRIMARY KEY,
							last_activity_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
						);
					""")
				logger.info("[PostgresCheckpointer] Conectado exitosamente a PostgreSQL y tablas inicializadas.")
				return
			except Exception as exc:
				if attempt == max_retries:
					logger.error(f"[PostgresCheckpointer] No se pudo conectar a PostgreSQL tras {max_retries} intentos: {exc}")
					raise
				logger.warning(f"[PostgresCheckpointer] Intento {attempt}/{max_retries} falló ({exc}). Reintentando en 2s...")
				await asyncio.sleep(2)

	async def stop(self) -> None:
		"""Cierra ordenadamente el pool cuando FastAPI se detiene."""
		await self.pool.close()

	async def actualizar_actividad(self, thread_id: str) -> None:
		"""Actualiza o registra el timestamp de última actividad de una sesión."""
		if not thread_id:
			return
		try:
			async with self.pool.connection() as conn:
				await conn.execute("""
					INSERT INTO conversation_sessions (thread_id, last_activity_at)
					VALUES (%s, NOW())
					ON CONFLICT (thread_id) DO UPDATE SET last_activity_at = NOW();
				""", (thread_id,))
		except Exception as e:
			logger.debug(f"[PostgresCheckpointer] Error actualizando actividad para {thread_id}: {e}")

	async def obtener_segundos_inactividad(self, thread_id: str) -> Optional[float]:
		"""Retorna cuántos segundos han transcurrido desde la última actividad registrada."""
		if not thread_id:
			return None
		try:
			async with self.pool.connection() as conn:
				async with conn.cursor() as cur:
					await cur.execute("""
						SELECT EXTRACT(EPOCH FROM (NOW() - last_activity_at))
						FROM conversation_sessions
						WHERE thread_id = %s;
					""", (thread_id,))
					row = await cur.fetchone()
					if row and row[0] is not None:
						return float(row[0])
		except Exception as e:
			logger.debug(f"[PostgresCheckpointer] Error consultando inactividad para {thread_id}: {e}")
		return None

	async def obtener_sesiones_expiradas(self, ttl_seconds: int) -> list[str]:
		"""Retorna la lista de thread_id cuya inactividad supere ttl_seconds."""
		if not self.pool:
			return []
		try:
			async with self.pool.connection() as conn:
				async with conn.cursor() as cur:
					await cur.execute("""
						SELECT thread_id
						FROM conversation_sessions
						WHERE EXTRACT(EPOCH FROM (NOW() - last_activity_at)) > %s;
					""", (ttl_seconds,))
					rows = await cur.fetchall()
					return [str(row[0]) for row in rows if row and row[0]]
		except Exception as e:
			logger.error(f"[PostgresCheckpointer] Error consultando sesiones expiradas: {e}", exc_info=True)
			return []

	async def clear_thread(self, thread_id: str) -> None:
		"""Elimina todos los checkpoints y estado de sesión asociados a un thread_id en PostgreSQL."""
		try:
			async with self.pool.connection() as conn:
				await conn.execute("DELETE FROM checkpoints WHERE thread_id = %s", (thread_id,))
				await conn.execute("DELETE FROM checkpoint_blobs WHERE thread_id = %s", (thread_id,))
				await conn.execute("DELETE FROM checkpoint_writes WHERE thread_id = %s", (thread_id,))
				await conn.execute("DELETE FROM conversation_sessions WHERE thread_id = %s", (thread_id,))
			logger.info(f"[PostgresCheckpointer] Checkpoints eliminados exitosamente para thread_id: {thread_id}")
		except Exception as e:
			logger.error(f"[PostgresCheckpointer] Error al limpiar thread_id {thread_id}: {e}", exc_info=True)


def get_checkpointer_instance() -> Optional[PostgresCheckpointer]:
	return checkpointer_instance