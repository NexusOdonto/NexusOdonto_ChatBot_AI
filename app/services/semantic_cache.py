"""Qdrant collection maintenance for the (retired) semantic reply cache.

Replies are never replayed from a cache: every answer is written by the LLM. This module only
keeps the collection in a valid state and lets the admin purge leftovers.
"""

import logging
from typing import Optional

from qdrant_client import QdrantClient
from qdrant_client.http import models

from app.agents.tools.qdrant_tool import get_qdrant_client
from app.core.config import settings

logger = logging.getLogger(__name__)


def ensure_cache_collection(client: Optional[QdrantClient] = None) -> None:
    """Crea la colección de caché semántico en Qdrant si no existe o si cambió la dimensión."""
    if client is None:
        client = get_qdrant_client()

    col_name = settings.semantic_cache_collection
    if client.collection_exists(col_name):
        try:
            info = client.get_collection(col_name)
            vectors_config = info.config.params.vectors
            existing_size = getattr(vectors_config, "size", None)
            if existing_size is None and isinstance(vectors_config, dict):
                first_val = next(iter(vectors_config.values()), None)
                existing_size = getattr(first_val, "size", None)

            if existing_size is not None and existing_size != settings.embedding_dimension:
                logger.warning(
                    f"[Semantic Cache] Dimensión incompatible detectada ({existing_size} != {settings.embedding_dimension}). "
                    f"Recreando colección '{col_name}'..."
                )
                client.delete_collection(col_name)
                client.create_collection(
                    collection_name=col_name,
                    vectors_config=models.VectorParams(
                        size=settings.embedding_dimension,
                        distance=models.Distance.COSINE,
                    ),
                )
        except Exception as exc:
            logger.warning(f"[Semantic Cache] Error validando dimensiones de colección: {exc}")
    else:
        client.create_collection(
            collection_name=col_name,
            vectors_config=models.VectorParams(
                size=settings.embedding_dimension,
                distance=models.Distance.COSINE,
            ),
        )
        logger.info(f"[Semantic Cache] Colección '{col_name}' creada exitosamente en Qdrant.")


def purgar_cache_semantico(client: Optional[QdrantClient] = None) -> bool:
    """Purga la colección en Qdrant para que no queden rastros de respuestas antiguas."""
    try:
        if client is None:
            client = get_qdrant_client()
        col_name = settings.semantic_cache_collection
        if client.collection_exists(col_name):
            client.delete_collection(col_name)
            logger.info(f"[Semantic Cache] Colección '{col_name}' eliminada de Qdrant.")

        client.create_collection(
            collection_name=col_name,
            vectors_config=models.VectorParams(
                size=settings.embedding_dimension,
                distance=models.Distance.COSINE,
            ),
        )
        logger.info(f"[Semantic Cache] Colección '{col_name}' recreada vacía en Qdrant exitosamente.")
        return True
    except Exception as exc:
        logger.error(f"[Semantic Cache] Error purgando colección en Qdrant: {exc}", exc_info=True)
        return False
