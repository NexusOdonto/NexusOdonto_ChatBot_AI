"""Recuperador clínico avanzado híbrido (Dense Vector + BM25 Okapi) con RRF y Re-ranking.
Proporciona respuestas clínicas exactas, resilientes y con guardrails anti-alucinación.
"""

import logging
from typing import Tuple, List, Optional, Dict, Any
try:
    from langchain_core.documents import Document
except ImportError:
    class Document:  # type: ignore
        def __init__(self, page_content: str = "", metadata: Optional[Dict[str, Any]] = None):
            self.page_content = page_content
            self.metadata = metadata or {}

        def __repr__(self) -> str:
            return f"Document(page_content={self.page_content[:30]!r}, metadata={self.metadata})"

import math
from app.domain.clinical.query_expander import expandir_consulta_clinica, extraer_filtros_metadatos
from app.domain.clinical.bm25_retriever import obtener_bm25_clinico
from app.core.config import settings

logger = logging.getLogger(__name__)

def _score_to_confidence(score: float, base_confidence: float = 0.80) -> float:
    """Calcula confianza calibrada para logits de cross-encoder mmarco.
    En mmarco, logits entre -3.5 y +2 indican relevancia clínica válida.
    """
    if score >= 0.0:
        return min(0.98, base_confidence + 0.10)
    elif score >= -3.5:
        return base_confidence
    elif score >= -5.0:
        return max(0.55, base_confidence - 0.15)
    return max(0.30, base_confidence - 0.35)

# Candidatos de búsqueda amplia para re-ranking profundo
CANDIDATOS_RETRIEVAL = 8
TOP_K_FINAL = 3


def fusionar_rrf(
    candidatos_densos: List[Tuple[Document, float]],
    candidatos_bm25: List[Tuple[Document, float]],
    k_rrf: int = 60,
    peso_denso: float = 0.6,
    peso_bm25: float = 0.4,
) -> List[Tuple[Document, float]]:
    """Combina y ordena candidatos mediante Reciprocal Rank Fusion (RRF).
    
    RRF(d) = peso_denso / (k + rank_denso(d)) + peso_bm25 / (k + rank_bm25(d))
    """
    doc_map: Dict[str, Document] = {}
    rrf_scores: Dict[str, float] = {}

    def _doc_key(doc: Document) -> str:
        titulo = (doc.metadata or {}).get("titulo", "")
        if titulo:
            return f"titulo::{titulo}"
        return f"content::{doc.page_content[:80]}"

    # 1. Puntuar ranking denso
    for rank, (doc, _score) in enumerate(candidatos_densos, start=1):
        key = _doc_key(doc)
        doc_map[key] = doc
        rrf_scores[key] = rrf_scores.get(key, 0.0) + (peso_denso / (k_rrf + rank))

    # 2. Puntuar ranking léxico BM25
    for rank, (doc, _score) in enumerate(candidatos_bm25, start=1):
        key = _doc_key(doc)
        doc_map[key] = doc
        rrf_scores[key] = rrf_scores.get(key, 0.0) + (peso_bm25 / (k_rrf + rank))

    # 3. Ordenar por score RRF descendente
    ordenados = sorted(rrf_scores.items(), key=lambda item: item[1], reverse=True)
    return [(doc_map[key], score) for key, score in ordenados]


def recuperar_conocimiento_clinico_robusto(query: str) -> str:
    """Ejecuta una recuperación clínica híbrida de última generación:
    1. Análisis de intención y extracción de filtros de metadatos (Self-Querying).
    2. Expansión semántica de consulta (modismos -> vocabulario clínico).
    3. Recuperación Híbrida:
       - Búsqueda Densa en Qdrant (semántica conceptual).
       - Búsqueda Léxica BM25 Okapi (precisión para fármacos y términos exactos).
    4. Fusión de rankings con Reciprocal Rank Fusion (RRF).
    5. Re-ranking semántico con Cross-Encoder.
    6. Guardrail clínico de confianza y formato estructurado anti-alucinación.
    """
    if not query or not query.strip():
        return "[RAG_SCORE:0.0]\nNo se especificó ninguna consulta clínica."

    # 1. Extracción de metadatos y expansión semántica
    filtros = extraer_filtros_metadatos(query)
    categoria_filtro = filtros.get("categoria")
    query_expandida = expandir_consulta_clinica(query)

    logger.info(
        f"[RAG Híbrido] Consulta: '{query}' | Expansión: '{query_expandida}' | "
        f"Filtro Detectado: {filtros}"
    )

    candidatos_densos: List[Tuple[Document, float]] = []
    candidatos_bm25: List[Tuple[Document, float]] = []

    # 2. Recuperación Densa (Qdrant)
    try:
        from app.agents.tools.qdrant_tool import get_vector_store
        vector_store = get_vector_store()
        candidatos_densos = vector_store.similarity_search_with_score(
            query_expandida, k=CANDIDATOS_RETRIEVAL
        )
        if not candidatos_densos and query_expandida != query:
            candidatos_densos = vector_store.similarity_search_with_score(
                query, k=CANDIDATOS_RETRIEVAL
            )
    except Exception as qdrant_exc:
        logger.warning(f"[RAG Híbrido] Qdrant no disponible o error ({qdrant_exc}). Usando fallback BM25.")

    # 3. Recuperación Léxica BM25 Okapi (enriquecimiento exacto y alta resiliencia)
    if settings.rag_hybrid_enabled:
        try:
            bm25 = obtener_bm25_clinico()
            candidatos_bm25 = bm25.buscar(
                query,
                k=CANDIDATOS_RETRIEVAL,
                filtro_categoria=categoria_filtro,
            )
            # Si el filtro fue muy restrictivo y no trajo nada, buscar sin filtro
            if not candidatos_bm25 and categoria_filtro:
                candidatos_bm25 = bm25.buscar(query, k=CANDIDATOS_RETRIEVAL)
        except Exception as bm25_exc:
            logger.warning(f"[RAG Híbrido] Error en motor BM25: {bm25_exc}")

    # 4. Fusión de candidatos
    if candidatos_densos and candidatos_bm25:
        candidatos_fusionados = fusionar_rrf(
            candidatos_densos=candidatos_densos,
            candidatos_bm25=candidatos_bm25,
            k_rrf=settings.rag_rrf_k,
            peso_denso=settings.rag_dense_weight,
            peso_bm25=settings.rag_bm25_weight,
        )
        candidatos_finales = [doc for doc, _ in candidatos_fusionados]
    elif candidatos_densos:
        candidatos_finales = [doc for doc, _ in candidatos_densos]
    elif candidatos_bm25:
        candidatos_finales = [doc for doc, _ in candidatos_bm25]
    else:
        return (
            "[RAG_SCORE:0.0]\n"
            "[ALERTA CLÍNICA: No se encontró información clínica específica en los protocolos oficiales]\n"
            "DIRECTIVA: Informa al paciente que no se dispone de un protocolo registrado para esta consulta "
            "y recomiéndale solicitar una cita de valoración con el odontólogo en el consultorio."
        )

    # 5. Re-ranking con Cross-Encoder o cálculo de confianza
    dense_scores = [float(s) for _, s in candidatos_densos if s is not None]
    max_dense = max(dense_scores) if dense_scores else 0.0

    if candidatos_densos and candidatos_bm25:
        base_confidence = min(0.95, max(max_dense, 0.82))
    elif candidatos_densos:
        base_confidence = max(max_dense, 0.72)
    elif candidatos_bm25:
        base_confidence = 0.75
    else:
        base_confidence = 0.50

    reranker = None
    try:
        from app.agents.tools.qdrant_tool import get_reranker
        reranker = get_reranker()
    except Exception:
        reranker = None

    confidence = base_confidence

    if reranker is not None and candidatos_finales:
        try:
            pairs = [(query, doc.page_content) for doc in candidatos_finales[:CANDIDATOS_RETRIEVAL]]
            scores = reranker.predict(pairs)
            ordenados = sorted(
                zip(candidatos_finales[:CANDIDATOS_RETRIEVAL], scores),
                key=lambda item: float(item[1]),
                reverse=True,
            )
            docs_seleccionados = [doc for doc, _ in ordenados[:TOP_K_FINAL]]
            mejor_score = float(ordenados[0][1]) if ordenados else 0.0
            confidence = _score_to_confidence(mejor_score, base_confidence=base_confidence)
        except Exception as r_exc:
            logger.warning(f"[RAG Híbrido] Re-ranking falló ({r_exc}), usando orden RRF.")
            docs_seleccionados = candidatos_finales[:TOP_K_FINAL]
            confidence = base_confidence
    else:
        docs_seleccionados = candidatos_finales[:TOP_K_FINAL]
        confidence = base_confidence

    # 6. Formateo y Guardrails de Seguridad
    secciones: List[str] = []
    for i, doc in enumerate(docs_seleccionados, start=1):
        meta = doc.metadata or {}
        titulo = meta.get("titulo", f"Guía Clínica {i}")
        especialidad = meta.get("especialidad", "General")
        categoria = meta.get("categoria", "Información")

        bloque = (
            f"--- [Documento #{i}: {titulo}] ---\n"
            f"Especialidad: {especialidad} | Tipo: {categoria}\n"
            f"{doc.page_content}"
        )
        secciones.append(bloque)

    contenido_formateado = "\n\n".join(secciones)

    # Inyección de guardrail si la confianza está por debajo del umbral mínimo
    guardrail_aviso = ""
    if confidence < settings.rag_min_confidence:
        guardrail_aviso = (
            f"\n\n[GUARDRAIL CLÍNICO ACTIVO: Nivel de confianza moderado ({confidence:.2f})]\n"
            "DIRECTIVA OBLIGATORIA: Si la duda del paciente no queda 100% resuelta con estos fragmentos, "
            "indícale amablemente que requiere ser examinado presencialmente por el especialista en la clínica "
            "(Calle 100 # 15-20 / +57 324 6030217). NUNCA inventes recetas, dosis ni diagnósticos definitivos."
        )

    logger.info(f"[RAG Híbrido] Éxito: {len(docs_seleccionados)} guías recuperadas (Confianza: {confidence:.2f})")
    return (
        f"[RAG_SCORE:{confidence:.4f}]\n"
        f"[CONOCIMIENTO CLÍNICO OFICIAL NEXUS ODONTO]\n"
        f"{contenido_formateado}"
        f"{guardrail_aviso}"
    )
