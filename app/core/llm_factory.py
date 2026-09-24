"""Fábrica de Modelos de Lenguaje y Embeddings (LLM Factory).

Provee instancias desacopladas de modelos de chat y embeddings compatibles
con LangChain y LangGraph, permitiendo alternar entre OpenAI y Google Gemini
mediante configuración sin modificar el código de los agentes o flujos.
"""

import logging
from typing import Any, Optional

from langchain_core.embeddings import Embeddings
from langchain_core.language_models.chat_models import BaseChatModel

from app.core.config import settings

logger = logging.getLogger(__name__)


def extract_text_content(content: Any) -> str:
    """Extrae texto plano del contenido de un mensaje, soportando tanto strings como listas de bloques."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        text_parts = []
        for item in content:
            if isinstance(item, str):
                text_parts.append(item)
            elif isinstance(item, dict) and "text" in item:
                text_parts.append(item["text"])
        return "".join(text_parts) if text_parts else str(content)
    return str(content) if content is not None else ""


def get_chat_llm(
    model: Optional[str] = None,
    temperature: float = 0.0,
    max_tokens: Optional[int] = None,
    provider: Optional[str] = None,
) -> BaseChatModel:
    """Retorna una instancia de chat LLM según el proveedor configurado (OpenAI o Google Gemini).

    Args:
        model: Nombre del modelo específico a usar. Si no se pasa, toma el configurado por defecto.
        temperature: Temperatura de muestreo (0.0 para determinismo en tools/agentes).
        max_tokens: Límite máximo de tokens de salida.
        provider: Proveedor a forzar ("openai" o "gemini"). Si es None, usa settings.llm_provider.

    Returns:
        BaseChatModel (ChatGoogleGenerativeAI o ChatOpenAI).
    """
    active_provider = (provider or settings.llm_provider or "openai").lower().strip()

    if active_provider == "gemini":
        if not settings.gemini_api_key:
            raise RuntimeError(
                "GEMINI_API_KEY no está configurada. Para usar Google Gemini, define "
                "GEMINI_API_KEY en tu archivo .env o cambia LLM_PROVIDER='openai'."
            )
        from langchain_google_genai import ChatGoogleGenerativeAI

        target_model = model or settings.gemini_model
        logger.debug(f"[LLM Factory] Instanciando ChatGoogleGenerativeAI (modelo: {target_model})")
        return ChatGoogleGenerativeAI(
            model=target_model,
            google_api_key=settings.gemini_api_key,
            temperature=temperature,
            max_output_tokens=max_tokens,
            max_retries=0,
        )

    # Proveedor por defecto: OpenAI
    if not settings.openai_api_key:
        raise RuntimeError(
            "OPENAI_API_KEY no está configurada. Define OPENAI_API_KEY en tu archivo .env "
            "o cambia LLM_PROVIDER='gemini' junto con GEMINI_API_KEY."
        )
    from langchain_openai import ChatOpenAI

    target_model = model or settings.openai_model
    logger.debug(f"[LLM Factory] Instanciando ChatOpenAI (modelo: {target_model})")
    return ChatOpenAI(
        model=target_model,
        api_key=settings.openai_api_key,
        temperature=temperature,
        max_tokens=max_tokens,
    )


def get_evaluator_llm(provider: Optional[str] = None) -> BaseChatModel:
    """Retorna un modelo ligero y rápido para tareas de clasificación, triage y compresión."""
    active_provider = (provider or settings.llm_provider or "openai").lower().strip()
    if active_provider == "gemini":
        target = settings.gemini_model if "flash" in settings.gemini_model else "gemini-1.5-flash"
        return get_chat_llm(model=target, temperature=0.0, provider="gemini")
    else:
        return get_chat_llm(model="gpt-4o-mini", temperature=0.0, provider="openai")


def get_embeddings_model(provider: Optional[str] = None) -> Embeddings:
    """Retorna la instancia de Embeddings adecuada según el proveedor configurado."""
    active_provider = (provider or settings.embedding_provider or "openai").lower().strip()

    if active_provider == "gemini":
        if not settings.gemini_api_key:
            raise RuntimeError(
                "GEMINI_API_KEY no está configurada para generar embeddings con Gemini."
            )
        from langchain_google_genai import GoogleGenerativeAIEmbeddings

        target_model = settings.gemini_embedding_model
        logger.debug(f"[LLM Factory] Instanciando GoogleGenerativeAIEmbeddings (modelo: {target_model})")
        return GoogleGenerativeAIEmbeddings(
            model=target_model,
            google_api_key=settings.gemini_api_key,
        )

    # Proveedor por defecto: OpenAI
    from langchain_openai import OpenAIEmbeddings

    target_model = settings.embedding_model
    logger.debug(f"[LLM Factory] Instanciando OpenAIEmbeddings (modelo: {target_model})")
    return OpenAIEmbeddings(
        model=target_model,
        api_key=settings.openai_api_key or None,
    )
