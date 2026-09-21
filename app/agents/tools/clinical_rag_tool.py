"""Herramienta LangChain de Base de Conocimiento Clínico Odontológico (RAG).
Conecta la recuperación semántica de Qdrant, expansión de consultas y reordenamiento Cross-Encoder.
"""

from app.agents.tools.qdrant_tool import (
    clinical_knowledge_tool,
    search_clinical_knowledge,
    get_vector_store,
    get_qdrant_client,
)

__all__ = [
    "clinical_knowledge_tool",
    "search_clinical_knowledge",
    "get_vector_store",
    "get_qdrant_client",
]
