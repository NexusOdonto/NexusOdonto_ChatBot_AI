from functools import lru_cache

from langchain_core.tools import Tool
from langchain_openai import OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.http import models

from app.core.config import settings


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
	# Devuelve el retriever que busca los cuatro resultados más relevantes.
	return get_vector_store().as_retriever(search_kwargs={"k": 4})


def retrieve_clinical_knowledge(query: str) -> str:
	# Busca información clínica y la convierte en texto para el agente.
	# El score se incluye para que el webhook pueda detectar baja confianza del RAG.
	documents_with_scores = get_vector_store().similarity_search_with_score(query, k=4)
	if not documents_with_scores:
		return "[RAG_SCORE:0.0]\nNo se encontro informacion clinica relevante."

	best_score = max(float(score) for _, score in documents_with_scores)
	content = "\n\n".join(document.page_content for document, _ in documents_with_scores)
	return f"[RAG_SCORE:{best_score}]\n{content}"


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
