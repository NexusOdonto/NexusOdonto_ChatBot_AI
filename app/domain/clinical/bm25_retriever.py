"""Motor de Búsqueda Léxica BM25 Okapi para el Dominio Clínico de Nexus Odonto.

Implementa el algoritmo estándar BM25 Okapi con ponderación IDF no negativa,
tokenización en español normalizada (sin tildes, con filtrado de stopwords) y
soporte para filtrado por categoría clínica. Cero dependencias externas pesadas.
"""

import math
import re
import unicodedata
from typing import Dict, List, Optional, Tuple, Any
try:
    from langchain_core.documents import Document
except ImportError:
    class Document:  # type: ignore
        def __init__(self, page_content: str = "", metadata: Optional[Dict[str, Any]] = None):
            self.page_content = page_content
            self.metadata = metadata or {}

        def __repr__(self) -> str:
            return f"Document(page_content={self.page_content[:30]!r}, metadata={self.metadata})"

# Stopwords habituales en español que no aportan discriminación clínica
STOPWORDS_ES = {
    "de", "la", "que", "el", "en", "y", "a", "los", "del", "se", "las", "por",
    "un", "para", "con", "no", "una", "su", "al", "lo", "como", "mas", "pero",
    "sus", "le", "ya", "o", "este", "si", "porque", "esta", "son", "entre",
    "cuando", "muy", "sin", "sobre", "tambien", "me", "hasta", "hay", "donde",
    "quien", "desde", "todo", "nos", "durante", "todos", "uno", "les", "ni",
    "contra", "otros", "ese", "eso", "ante", "ellos", "e", "esto", "mi", "antes",
    "algunos", "unos", "yo", "otro", "otras", "otra", "el", "ella", "usted",
}


def normalizar_token(token: str) -> str:
    """Normaliza texto eliminando acentos y signos de puntuación."""
    if not token:
        return ""
    texto = token.lower().strip()
    return "".join(
        c for c in unicodedata.normalize("NFD", texto)
        if unicodedata.category(c) != "Mn"
    )


def tokenizar(texto: str) -> List[str]:
    """Extrae tokens normalizados mayores a 1 caracter excluyendo stopwords."""
    if not texto:
        return []
    texto_norm = normalizar_token(texto)
    palabras = re.findall(r"\b[a-z0-9]{2,}\b", texto_norm)
    return [p for p in palabras if p not in STOPWORDS_ES]


class BM25OkapiRetriever:
    """Motor de recuperación léxica BM25 Okapi optimizado en memoria."""

    def __init__(
        self,
        documentos: Optional[List[Document]] = None,
        k1: float = 1.5,
        b: float = 0.75,
    ):
        self.k1 = k1
        self.b = b
        self.documentos: List[Document] = []
        self.corpus_size: int = 0
        self.avgdl: float = 0.0
        self.doc_lengths: List[int] = []
        self.doc_token_freqs: List[Dict[str, int]] = []
        self.idf: Dict[str, float] = {}

        if documentos:
            self.indexar(documentos)

    def indexar(self, documentos: List[Document]) -> None:
        """Indexa una lista de Documentos de LangChain en memoria."""
        self.documentos = list(documentos)
        self.corpus_size = len(self.documentos)

        if self.corpus_size == 0:
            self.avgdl = 0.0
            self.doc_lengths = []
            self.doc_token_freqs = []
            self.idf = {}
            return

        total_length = 0
        self.doc_lengths = []
        self.doc_token_freqs = []
        doc_frequencies: Dict[str, int] = {}

        for doc in self.documentos:
            # Combinamos título y contenido para darle peso léxico al tema principal
            titulo = (doc.metadata or {}).get("titulo", "")
            texto_completo = f"{titulo} {doc.page_content}" if titulo else doc.page_content
            tokens = tokenizar(texto_completo)

            length = len(tokens)
            self.doc_lengths.append(length)
            total_length += length

            freqs: Dict[str, int] = {}
            for t in tokens:
                freqs[t] = freqs.get(t, 0) + 1
            self.doc_token_freqs.append(freqs)

            for t in freqs.keys():
                doc_frequencies[t] = doc_frequencies.get(t, 0) + 1

        self.avgdl = total_length / self.corpus_size if self.corpus_size > 0 else 0.0

        # Cálculo de IDF suavizado y estrictamente no-negativo (estándar Lucene)
        self.idf = {}
        for token, df in doc_frequencies.items():
            # log(1 + (N - df + 0.5) / (df + 0.5)) garantiza valores positivos
            self.idf[token] = math.log(1.0 + (self.corpus_size - df + 0.5) / (df + 0.5))

    def buscar(
        self,
        query: str,
        k: int = 5,
        filtro_categoria: Optional[str] = None,
    ) -> List[Tuple[Document, float]]:
        """Busca y rankea documentos según el score BM25 Okapi.
        
        Retorna una lista de tuplas (Document, score) ordenadas descendentemente.
        """
        if not query or self.corpus_size == 0:
            return []

        tokens_query = tokenizar(query)
        if not tokens_query:
            return []

        scores: List[Tuple[int, float]] = []

        for idx, freqs in enumerate(self.doc_token_freqs):
            doc = self.documentos[idx]

            # Filtrado opcional por categoría
            if filtro_categoria:
                doc_cat = (doc.metadata or {}).get("categoria", "")
                if doc_cat and doc_cat.lower() != filtro_categoria.lower():
                    continue

            score = 0.0
            doc_len = self.doc_lengths[idx]

            # Longitud normalizada
            longitud_normalizada = 1.0 - self.b + self.b * (doc_len / self.avgdl) if self.avgdl > 0 else 1.0

            for token in tokens_query:
                if token not in freqs:
                    continue
                tf = freqs[token]
                idf = self.idf.get(token, 0.0)

                # Fórmula Okapi BM25
                numerador = tf * (self.k1 + 1.0)
                denominador = tf + self.k1 * longitud_normalizada
                score += idf * (numerador / denominador)

            if score > 0.0:
                scores.append((idx, score))

        # Ordenar de mayor a menor score
        scores.sort(key=lambda item: item[1], reverse=True)
        top_indices = scores[:k]

        return [(self.documentos[idx], score) for idx, score in top_indices]


# Instancia singleton perezosa del índice BM25 clínico
_bm25_clinico_global: Optional[BM25OkapiRetriever] = None


def obtener_bm25_clinico() -> BM25OkapiRetriever:
    """Devuelve o construye el retriever BM25 global con la base oficial de Nexus Odonto."""
    global _bm25_clinico_global
    if _bm25_clinico_global is None:
        from app.domain.clinical.knowledge_base import CONOCIMIENTO_CLINICO_COMPLETO

        documentos = [
            Document(
                page_content=item["contenido"],
                metadata={
                    "titulo": item["titulo"],
                    "categoria": item["categoria"],
                    "especialidad": item["especialidad"],
                },
            )
            for item in CONOCIMIENTO_CLINICO_COMPLETO
        ]
        _bm25_clinico_global = BM25OkapiRetriever(documentos)
    return _bm25_clinico_global


def recargar_bm25_clinico(documentos: List[Document]) -> None:
    """Recarga el índice léxico cuando se ingesta o actualiza el catálogo."""
    global _bm25_clinico_global
    _bm25_clinico_global = BM25OkapiRetriever(documentos)
