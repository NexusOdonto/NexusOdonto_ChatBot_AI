"""Provider-neutral runtime for chat LLM calls.

Holds the per-turn time budget shared by every LLM call of a chat turn, the single
"LLM unavailable" error that triggers the outage notice, and the OpenAI chat model
that honours both.
"""

from __future__ import annotations

import asyncio
import contextvars
import logging
import time
from typing import Any, Optional

import httpx
from langchain_openai import ChatOpenAI

from app.core.config import settings

logger = logging.getLogger(__name__)

# Monotonic deadline shared by every LLM call of the current chat turn.
_TURN_DEADLINE: contextvars.ContextVar[Optional[float]] = contextvars.ContextVar(
    "llm_turn_deadline", default=None
)


class LLMUnavailableError(RuntimeError):
    """The LLM could not answer this turn (outage, quota, auth, timeout or budget exhausted)."""


def start_turn_budget(seconds: Optional[float] = None) -> contextvars.Token:
    budget = float(seconds if seconds is not None else settings.llm_turn_budget_seconds or 0)
    return _TURN_DEADLINE.set(time.monotonic() + budget if budget > 0 else None)


def end_turn_budget(token: contextvars.Token) -> None:
    _TURN_DEADLINE.reset(token)


def turn_deadline(default_seconds: float) -> float:
    """Deadline of the current turn, or now + default when called outside a turn."""
    turn = _TURN_DEADLINE.get()
    if turn is not None:
        return turn
    budget = float(settings.llm_turn_budget_seconds or 0)
    return time.monotonic() + (budget if budget > 0 else default_seconds)


def active_provider() -> str:
    return (settings.llm_provider or "openai").lower().strip()


def active_chat_model() -> str:
    return settings.gemini_model if active_provider() == "gemini" else settings.openai_model


def is_request_rejected(exc: BaseException) -> bool:
    """The provider refused this request's shape (e.g. a tool schema); retrying without tools may work."""
    if isinstance(exc, httpx.HTTPStatusError):
        return exc.response.status_code in (400, 403)
    try:
        import openai
    except ImportError:
        return False
    return isinstance(exc, openai.BadRequestError)


def _to_unavailable(exc: BaseException) -> Optional[LLMUnavailableError]:
    import openai

    if isinstance(exc, openai.APITimeoutError):
        return LLMUnavailableError(f"[OpenAI] timeout ({exc!s})")
    if isinstance(exc, openai.APIConnectionError):
        return LLMUnavailableError(f"[OpenAI] connection error ({exc!s})")
    if isinstance(exc, openai.APIStatusError) and not isinstance(exc, openai.BadRequestError):
        return LLMUnavailableError(f"[OpenAI] HTTP {exc.status_code} ({type(exc).__name__})")
    return None


def usage_summary(message: Any) -> str:
    """prompt/completion/cached token counts reported by the provider for one reply."""
    usage = getattr(message, "usage_metadata", None) or {}
    meta = getattr(message, "response_metadata", None) or {}
    raw = meta.get("token_usage") or meta.get("usage") or {}
    prompt = usage.get("input_tokens", raw.get("prompt_tokens", raw.get("promptTokenCount")))
    completion = usage.get("output_tokens", raw.get("completion_tokens", raw.get("candidatesTokenCount")))
    details = raw.get("prompt_tokens_details") or {}
    cached = (usage.get("input_token_details") or {}).get("cache_read", details.get("cached_tokens"))
    if cached is None:
        cached = raw.get("cachedContentTokenCount", 0)
    return f"prompt_tokens={prompt} completion_tokens={completion} cached_tokens={cached}"


class BudgetedChatOpenAI(ChatOpenAI):
    """ChatOpenAI whose calls never outlive the turn budget and fail as LLMUnavailableError."""

    def _remaining(self) -> float:
        remaining = turn_deadline(float(settings.llm_attempt_timeout_seconds or 14.0)) - time.monotonic()
        if remaining <= 0.5:
            raise LLMUnavailableError("[OpenAI] turn budget exhausted")
        return remaining

    def _request_timeout(self, remaining: float) -> float:
        return max(1.0, min(float(settings.llm_attempt_timeout_seconds or remaining), remaining))

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        remaining = self._remaining()
        kwargs.setdefault("timeout", self._request_timeout(remaining))
        try:
            return super()._generate(messages, stop=stop, run_manager=run_manager, **kwargs)
        except Exception as exc:
            mapped = _to_unavailable(exc)
            if mapped is None:
                raise
            logger.warning(f"[OpenAI] {self.model_name} unavailable: {mapped}")
            raise mapped from exc

    async def _agenerate(self, messages, stop=None, run_manager=None, **kwargs):
        remaining = self._remaining()
        kwargs.setdefault("timeout", self._request_timeout(remaining))
        try:
            return await asyncio.wait_for(
                super()._agenerate(messages, stop=stop, run_manager=run_manager, **kwargs),
                timeout=remaining,
            )
        except asyncio.TimeoutError as exc:
            logger.warning(f"[OpenAI] {self.model_name} exceeded the turn budget ({remaining:.1f}s)")
            raise LLMUnavailableError("[OpenAI] turn budget exhausted") from exc
        except Exception as exc:
            mapped = _to_unavailable(exc)
            if mapped is None:
                raise
            logger.warning(f"[OpenAI] {self.model_name} unavailable: {mapped}")
            raise mapped from exc
