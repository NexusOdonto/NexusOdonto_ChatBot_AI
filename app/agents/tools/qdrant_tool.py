import logging
from functools import lru_cache

from langchain_core.embeddings import Embeddings
from langchain_core.tools import Tool
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.http import models
try:
	from sentence_transformers import CrossEncoder
except ImportError:
	CrossEncoder = None

from app.core.config import settings
from app.core.llm_factory import get_embeddings_model

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def get_qdrant_client() -> QdrantClient:
	# Crea un único cliente y verifica que Qdrant esté disponible.
	client = QdrantClient(
		url=settings.qdrant_url,
		api_key=settings.qdrant_api_key or None,
	)
	client.get_collections()
	return client


@lru_cache(maxsize=1)
def get_embeddings() -> Embeddings:
	# Crea el modelo que transforma preguntas y documentos en vectores.
	return get_embeddings_model()


@lru_cache(maxsize=1)
def get_vector_store() -> QdrantVectorStore:
	# Conecta LangChain con la colección clínica de Qdrant.
	client = get_qdrant_client()
	_ensure_collection(client)

	return QdrantVectorStore(
		client=client,
		collection_name=settings.qdrant_collection_name,
		embedding=get_embeddings(),
	)


def _ensure_collection(client: QdrantClient) -> None:
	# Crea la colección solo si todavía no existe en Qdrant o si cambió la dimensión del modelo
	if client.collection_exists(settings.qdrant_collection_name):
		try:
			info = client.get_collection(settings.qdrant_collection_name)
			vectors_config = info.config.params.vectors
			existing_size = getattr(vectors_config, "size", None)
			if existing_size is None and isinstance(vectors_config, dict):
				first_val = next(iter(vectors_config.values()), None)
				existing_size = getattr(first_val, "size", None)

			if existing_size is not None and existing_size != settings.embedding_dimension:
				logger.warning(
					f"[Qdrant] Dimensión incompatible detectada ({existing_size} != {settings.embedding_dimension}). "
					f"Recreando colección '{settings.qdrant_collection_name}'..."
				)
				client.delete_collection(settings.qdrant_collection_name)
				client.create_collection(
					collection_name=settings.qdrant_collection_name,
					vectors_config=models.VectorParams(
						size=settings.embedding_dimension,
						distance=models.Distance.COSINE,
					),
				)
		except Exception as exc:
			logger.warning(f"[Qdrant] Error validando dimensiones de colección: {exc}")
	else:
		client.create_collection(
			collection_name=settings.qdrant_collection_name,
			vectors_config=models.VectorParams(
				size=settings.embedding_dimension,
				distance=models.Distance.COSINE,
			),
		)


def get_clinical_retriever():
	# Devuelve el retriever base que obtiene los candidatos iniciales desde Qdrant.
	return get_vector_store().as_retriever(search_kwargs={"k": settings.rag_candidate_count})


@lru_cache(maxsize=1)
def get_reranker():
	# El modelo se carga una sola vez y se reutiliza en todas las preguntas si está disponible.
	if CrossEncoder is not None:
		try:
			return CrossEncoder(settings.reranker_model)
		except Exception as e:
			logger.warning(f"No se pudo cargar CrossEncoder ({e}), usando similitud coseno nativa.")
	return None


def _score_to_confidence(score: float) -> float:
	# Convierte el logit del Cross-Encoder a un rango 0..1 para el escalamiento.
	if score >= 0:
		return 1.0 / (1.0 + pow(2.718281828, -score))
	return pow(2.718281828, score) / (1.0 + pow(2.718281828, score))


def retrieve_clinical_knowledge(query: str) -> str:
	try:
		from app.infra.persistence.clinical_retriever import recuperar_conocimiento_clinico_robusto
		return recuperar_conocimiento_clinico_robusto(query)
	except Exception as exc:
		logger.error(f"Error al invocar recuperador clínico robusto: {exc}", exc_info=True)
		# Fallback directo a búsqueda vectorial básica si el módulo falla
		try:
			documents_with_scores = get_vector_store().similarity_search_with_score(
				query,
				k=settings.rag_candidate_count,
			)
			if not documents_with_scores:
				return "[RAG_SCORE:0.0]\nNo se encontro informacion clinica relevante."
			content = "\n\n".join(document.page_content for document, _ in documents_with_scores)
			return f"[RAG_SCORE:0.7500]\n{content}"
		except Exception as inner_exc:
			logger.error(f"Error en fallback de recuperación clínica: {inner_exc}")
			return "[RAG_SCORE:0.0]\nNo fue posible acceder a la base de conocimiento clinico en este momento."


from langchain_core.tools import tool

# Herramienta que LangGraph puede incluir junto con sus demás herramientas.
@tool("buscar_conocimiento_clinico")
def clinical_knowledge_tool(query: str) -> str:
	"""Consulta protocolos clínicos, tratamientos, cuidados bucales y preparaciones de Nexus Odonto."""
	return retrieve_clinical_knowledge(query)


search_clinical_knowledge = retrieve_clinical_knowledge


def initialize_qdrant() -> None:
	# Se ejecuta al iniciar FastAPI para comprobar y preparar Qdrant.
	import time
	client = None
	max_retries = 10
	for attempt in range(1, max_retries + 1):
		try:
			client = get_qdrant_client()
			_ensure_collection(client)
			logger.info("[Qdrant] Conexión y colección inicializadas exitosamente.")
			break
		except Exception as exc:
			if attempt == max_retries:
				logger.error(f"[Qdrant] No se pudo conectar a Qdrant tras {max_retries} intentos: {exc}")
				raise
			logger.warning(f"[Qdrant] Intento {attempt}/{max_retries} falló ({exc}). Reintentando en 2s...")
			time.sleep(2)
	try:
		from app.services.semantic_cache import ensure_cache_collection
		ensure_cache_collection(client)
	except Exception as exc:
		logger.warning(f"No se pudo inicializar la coleccion de cache semantico: {exc}")

	try:
		from app.services.knowledge_ingestion import poblar_conocimiento_clinico
		poblar_conocimiento_clinico()
	except Exception as exc:
		logger.warning(f"No se pudo poblar la base de conocimiento clinico automaticamente: {exc}")
