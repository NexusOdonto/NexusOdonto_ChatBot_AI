import logging
from functools import lru_cache

from langchain_core.tools import Tool
from langchain_openai import OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.http import models
try:
	from sentence_transformers import CrossEncoder
except ImportError:
	CrossEncoder = None

from app.core.config import settings

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
def get_embeddings() -> OpenAIEmbeddings:
	# Crea el modelo que transforma preguntas y documentos en vectores.
	return OpenAIEmbeddings(
		model=settings.embedding_model,
		api_key=settings.openai_api_key or None,
	)


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
	# Crea la colección solo si todavía no existe en Qdrant.
	if not client.collection_exists(settings.qdrant_collection_name):
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
def get_reranker() -> CrossEncoder:
	# El modelo se carga una sola vez y se reutiliza en todas las preguntas.
	return CrossEncoder(settings.reranker_model)


def _score_to_confidence(score: float) -> float:
	# Convierte el logit del Cross-Encoder a un rango 0..1 para el escalamiento.
	if score >= 0:
		return 1.0 / (1.0 + pow(2.718281828, -score))
	return pow(2.718281828, score) / (1.0 + pow(2.718281828, score))


def retrieve_clinical_knowledge(query: str) -> str:
	try:
		# Busca información clínica y la convierte en texto para el agente.
		# Primero Qdrant recupera candidatos por similitud vectorial.
		documents_with_scores = get_vector_store().similarity_search_with_score(
			query,
			k=settings.rag_candidate_count,
		)
		if not documents_with_scores:
			return "[RAG_SCORE:0.0]\nNo se encontro informacion clinica relevante."

		# Después el Cross-Encoder compara la pregunta con cada documento completo.
		pairs = [(query, document.page_content) for document, _ in documents_with_scores]
		reranker_scores = get_reranker().predict(pairs)
		ranked_documents = sorted(
			zip((document for document, _ in documents_with_scores), reranker_scores),
			key=lambda item: float(item[1]),
			reverse=True,
		)
		best_confidence = _score_to_confidence(float(ranked_documents[0][1]))
		content = "\n\n".join(document.page_content for document, _ in ranked_documents)
		return f"[RAG_SCORE:{best_confidence}]\n{content}"
	except Exception as exc:
		logger.error(f"Error al recuperar conocimiento clinico: {exc}", exc_info=True)
		return "[RAG_SCORE:0.0]\nNo fue posible acceder a la base de conocimiento clinico en este momento debido a problemas de conexion."


# Herramienta que LangGraph puede incluir junto con sus demás herramientas.
clinical_knowledge_tool = Tool.from_function(
	func=retrieve_clinical_knowledge,
	name="buscar_conocimiento_clinico",
	description=(
		"Busca en la base de conocimiento clinico los horarios, precios y "
		"preparaciones de la clinica. Usa esta herramienta antes de responder "
		"preguntas sobre esos datos."
	),
)


def initialize_qdrant() -> None:
	# Se ejecuta al iniciar FastAPI para comprobar y preparar Qdrant.
	client = get_qdrant_client()
	_ensure_collection(client)
