"""Servicio de Caché Semántico (Semantic Caching) con Qdrant y OpenAI Embeddings.

Permite interceptar consultas frecuentes o equivalentes y responder en < 50ms
sin consumir tokens ni realizar llamadas adicionales al LLM.
"""

import datetime
import logging
import uuid
from typing import Optional

from qdrant_client import QdrantClient
from qdrant_client.http import models

from app.agents.tools.qdrant_tool import get_embeddings, get_qdrant_client
from app.core.config import settings

logger = logging.getLogger(__name__)


def ensure_cache_collection(client: Optional[QdrantClient] = None) -> None:
    """Crea la colección de caché semántico en Qdrant si no existe."""
    if client is None:
        client = get_qdrant_client()

    col_name = settings.semantic_cache_collection
    if not client.collection_exists(col_name):
        client.create_collection(
            collection_name=col_name,
            vectors_config=models.VectorParams(
                size=settings.embedding_dimension,
                distance=models.Distance.COSINE,
            ),
        )
        logger.info(f"[Semantic Cache] Colección '{col_name}' creada exitosamente en Qdrant.")


def _es_contenido_cacheable(pregunta: str, respuesta: str) -> bool:
    """Valida si un par pregunta-respuesta es apto para almacenarse en el caché semántico.
    
    Excluye respuestas transaccionales (citas agendadas, disponibilidad en tiempo real,
    emergencias médicas o mensajes de error/escalamiento).
    """
    p_lower = pregunta.lower().strip()
    r_lower = respuesta.lower().strip()

    # Descartar preguntas demasiado cortas o de saludo básico
    if len(p_lower) < 6 or len(r_lower) < 15:
        return False

    # Exclusiones por palabras clave transaccionales o dinámicas
    palabras_no_cacheables = [
        "cita agendada",
        "confirmada para el",
        "tu cita ha sido",
        "asesor de la clínica revisará",
        "escalada",
        "código de cita",
        "urgencia médica",
        "acude de inmediato",
        "servicio de urgencias",
        "lo siento, ocurrió un error",
        "no logré escuchar",
        "inconveniente técnico",
    ]

    for palabra in palabras_no_cacheables:
        if palabra in r_lower:
            return False

    # Excluir comandos directos
    if p_lower.startswith("/") or p_lower in ["si", "no", "1", "2", "3", "4", "cancelar"]:
        return False

    return True


async def buscar_en_cache(pregunta: str) -> Optional[str]:
    """Busca una respuesta semánticamente equivalente en la colección de caché de Qdrant.
    
    Retorna la respuesta cacheada si la similitud coseno es mayor o igual a semantic_cache_threshold.
    """
    if not settings.semantic_cache_enabled:
        return None

    query = pregunta.strip()
    if len(query) < 5:
        return None

    try:
        client = get_qdrant_client()
        ensure_cache_collection(client)

        # Generar embedding de la pregunta entrante
        embeddings = get_embeddings()
        vector_pregunta = await embeddings.aembed_query(query)

        # Búsqueda de vecino más cercano en la colección de caché
        col_name = settings.semantic_cache_collection
        search_results = client.search(
            collection_name=col_name,
            query_vector=vector_pregunta,
            limit=1,
            with_payload=True,
        )

        if search_results:
            top_match = search_results[0]
            score = float(top_match.score)

            if score >= settings.semantic_cache_threshold:
                payload = top_match.payload or {}
                respuesta = payload.get("respuesta")
                if respuesta:
                    logger.info(
                        f"[Semantic Cache] ⚡ Cache HIT (score: {score:.4f} >= {settings.semantic_cache_threshold}) "
                        f"para query: '{query[:50]}...' | Guardada: '{payload.get('pregunta', '')[:40]}...'"
                    )
                    return str(respuesta)

            logger.debug(f"[Semantic Cache] Cache MISS (mejor score: {score:.4f} < {settings.semantic_cache_threshold}) para query: '{query[:50]}...'")

        return None
    except Exception as exc:
        logger.warning(f"[Semantic Cache] Error al consultar caché semántico: {exc}")
        return None


async def guardar_en_cache(pregunta: str, respuesta: str, categoria: str = "general") -> bool:
    """Almacena el par (pregunta, respuesta) en la colección de caché semántico de Qdrant."""
    if not settings.semantic_cache_enabled:
        return False

    if not _es_contenido_cacheable(pregunta, respuesta):
        logger.debug(f"[Semantic Cache] Contenido no cacheable omitido: '{pregunta[:40]}...'")
        return False

    try:
        client = get_qdrant_client()
        ensure_cache_collection(client)

        embeddings = get_embeddings()
        vector_pregunta = await embeddings.aembed_query(pregunta.strip())

        point_id = str(uuid.uuid4())
        col_name = settings.semantic_cache_collection

        client.upsert(
            collection_name=col_name,
            points=[
                models.PointStruct(
                    id=point_id,
                    vector=vector_pregunta,
                    payload={
                        "pregunta": pregunta.strip(),
                        "respuesta": respuesta.strip(),
                        "categoria": categoria,
                        "created_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                    },
                )
            ],
        )

        logger.info(f"[Semantic Cache] 💾 Guardada nueva entrada en caché para: '{pregunta[:50]}...'")
        return True
    except Exception as exc:
        logger.warning(f"[Semantic Cache] Error al guardar en caché semántico: {exc}")
        return False
