"""Módulo de ingestión y sincronización de conocimiento clínico en Qdrant y BM25 para Nexus Odonto.

Indexa documentos estructurados sobre preparación de procedimientos, cuidados
postoperatorios, preguntas frecuentes clínicas, políticas de atención y sincroniza
dinámicamente el catálogo de tratamientos y tarifas desde el backend .NET.
"""

import logging
from typing import List, Dict, Any, Optional
try:
    from langchain_core.documents import Document
except ImportError:
    class Document:  # type: ignore
        def __init__(self, page_content: str = "", metadata: Optional[Dict[str, Any]] = None):
            self.page_content = page_content
            self.metadata = metadata or {}

        def __repr__(self) -> str:
            return f"Document(page_content={self.page_content[:30]!r}, metadata={self.metadata})"
try:
    from qdrant_client import QdrantClient
except ImportError:
    QdrantClient = None

from app.domain.clinical.knowledge_base import CONOCIMIENTO_CLINICO_COMPLETO
from app.domain.clinical.bm25_retriever import recargar_bm25_clinico, obtener_bm25_clinico
from app.agents.tools.agenda_helpers import DESCRIPCIONES_SERVICIO_ES, _etiqueta_servicio, _obtener_valor
from app.core.config import settings

logger = logging.getLogger(__name__)

CONOCIMIENTO_CLINICO_DOCUMENTOS: List[dict] = CONOCIMIENTO_CLINICO_COMPLETO


def construir_documentos_base() -> List[Document]:
    """Convierte la base de conocimiento clínico estructurada en Documentos de LangChain."""
    docs: List[Document] = []
    for item in CONOCIMIENTO_CLINICO_DOCUMENTOS:
        doc = Document(
            page_content=item["contenido"],
            metadata={
                "titulo": item["titulo"],
                "categoria": item["categoria"],
                "especialidad": item["especialidad"],
            },
        )
        docs.append(doc)
    return docs


def poblar_conocimiento_clinico(forzar_reindexacion: bool = False) -> int:
    """Ingesta los documentos de conocimiento clínico en la colección Qdrant y actualiza el índice BM25.
    Si forzar_reindexacion es False, solo indexa si la colección tiene menos documentos que la base oficial.
    Retorna la cantidad de documentos indexados.
    """
    docs_base = construir_documentos_base()
    # Actualizar siempre el índice léxico BM25 en memoria
    recargar_bm25_clinico(docs_base)

    try:
        from app.agents.tools.qdrant_tool import get_qdrant_client, get_vector_store, _ensure_collection
        client = get_qdrant_client()
        _ensure_collection(client)

        count_info = client.count(collection_name=settings.qdrant_collection_name)
        puntos_actuales = count_info.count
        logger.info(f"[Qdrant Ingest] Puntos actuales en '{settings.qdrant_collection_name}': {puntos_actuales}")

        if puntos_actuales >= len(CONOCIMIENTO_CLINICO_DOCUMENTOS) and not forzar_reindexacion:
            logger.info(f"[Qdrant Ingest] La colección ya contiene {puntos_actuales} documentos. No se requiere reindexar.")
            return puntos_actuales

        logger.info(f"[Qdrant Ingest] Indexando {len(docs_base)} documentos enriquecidos en Qdrant...")
        vector_store = get_vector_store()
        vector_store.add_documents(docs_base)

        count_after = client.count(collection_name=settings.qdrant_collection_name).count
        logger.info(f"[Qdrant Ingest] Ingestión completada exitosamente. Total de puntos: {count_after}")
        return count_after
    except Exception as exc:
        logger.warning(f"[Qdrant Ingest] Qdrant no disponible ({exc}), índice BM25 permanece activo.")
        return len(docs_base)


async def sincronizar_catalogo_dotnet_a_qdrant() -> int:
    """Consulta los servicios y especialidades activos en la API .NET y genera
    fichas clínicas enriquecidas indexadas en Qdrant y BM25.
    
    Retorna la cantidad de servicios del catálogo sincronizados.
    """
    from app.infra.external.dotnet.catalog_api import DotNetCatalogApi

    try:
        catalog_api = DotNetCatalogApi()
        servicios = await catalog_api.obtener_servicios() or []
        especialidades = await catalog_api.obtener_especialidades() or []

        if not servicios:
            logger.info("[Qdrant Ingest] No se obtuvieron servicios desde la API .NET para sincronizar.")
            return 0

        esp_map = {}
        for esp in especialidades:
            e_id = _obtener_valor(esp, "id", "specialtyId", "especialidadId")
            e_nombre = _obtener_valor(esp, "name", "nombre")
            if e_id and e_nombre:
                esp_map[str(e_id)] = e_nombre

        docs_catalogo: List[Document] = []

        for s in servicios:
            nombre = _etiqueta_servicio(s)
            nombre_orig = str(_obtener_valor(s, "name", "nombre") or "").strip()
            precio = float(_obtener_valor(s, "price", "precio", "cost") or 0)
            duracion = int(_obtener_valor(s, "durationMinutes", "duracionMinutos") or 45)
            esp_id = str(_obtener_valor(s, "specialtyId", "especialidadId") or "")
            esp_nombre = esp_map.get(esp_id, "Odontología General")
            desc_oficial = DESCRIPCIONES_SERVICIO_ES.get(
                _obtener_valor(s, "name", "nombre", "").strip(),
                "Tratamiento odontológico profesional de alta calidad realizado en Nexus Odonto.",
            )

            alias_texto = f" ({nombre_orig})" if nombre_orig and nombre_orig != nombre else ""
            contenido = (
                f"Ficha Técnica de Tratamiento: {nombre}{alias_texto}\n"
                f"- Especialidad encargada: {esp_nombre}\n"
                f"- Duración estimada en clínica: {duracion} minutos\n"
                f"- Valor oficial de la consulta/tratamiento: ${precio:,.0f} COP\n"
                f"- Descripción clínica: {desc_oficial}\n"
                "- Recomendación de agendamiento: Requiere cita previa o valoración por sobrecupo en caso de urgencia."
            )

            doc = Document(
                page_content=contenido,
                metadata={
                    "titulo": f"Ficha de Servicio: {nombre}",
                    "categoria": "catalogo_servicios",
                    "especialidad": esp_nombre,
                    "servicio_id": str(_obtener_valor(s, "id", "serviceId") or ""),
                    "precio": precio,
                },
            )
            docs_catalogo.append(doc)

        if docs_catalogo:
            # 1. Ingestar en Qdrant
            try:
                from app.agents.tools.qdrant_tool import get_vector_store
                vector_store = get_vector_store()
                vector_store.add_documents(docs_catalogo)
            except Exception as q_err:
                logger.warning(f"[Qdrant Ingest] No se pudo persistir catálogo en Qdrant ({q_err}), manteniendo BM25.")

            # 2. Actualizar índice léxico BM25 combinando base clínica + catálogo nuevo
            todos_los_docs = construir_documentos_base() + docs_catalogo
            recargar_bm25_clinico(todos_los_docs)

            logger.info(f"[Qdrant Ingest] Sincronizados {len(docs_catalogo)} servicios del catálogo .NET exitosamente.")
            return len(docs_catalogo)

    except Exception as exc:
        logger.error(f"[Qdrant Ingest] Error sincronizando catálogo .NET con RAG: {exc}", exc_info=True)

    return 0
