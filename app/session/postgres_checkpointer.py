from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from psycopg_pool import AsyncConnectionPool


class PostgresCheckpointer:
	"""Administra el pool y el saver async que persisten el grafo."""

	def __init__(self, connection_string: str) -> None:
		# El pool reutiliza conexiones y configura psycopg para los checkpoints.
		self.pool = AsyncConnectionPool(
			conninfo=connection_string,
			kwargs={"autocommit": True, "prepare_threshold": 0},
			open=False,
		)
		self.saver = AsyncPostgresSaver(self.pool)

	async def start(self) -> None:
		"""Abre PostgreSQL y crea las tablas internas de LangGraph."""
		await self.pool.open()
		await self.saver.setup()

	async def stop(self) -> None:
		"""Cierra ordenadamente el pool cuando FastAPI se detiene."""
		await self.pool.close()