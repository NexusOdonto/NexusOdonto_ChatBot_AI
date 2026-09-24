"""Fast Gemini chat+tools client via REST (thinkingLevel=MINIMAL for lower TTFT).

Used by the chatbot node when LLM_PROVIDER=gemini to avoid the old
langchain-google-genai path that cannot pass Gemini 3 thinkingConfig.
"""

from __future__ import annotations

import json
import logging
import time
from typing import Any, Optional, Sequence

import httpx
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.tools import BaseTool
from pydantic import Field, PrivateAttr

from app.core.config import settings

logger = logging.getLogger(__name__)

_GEMINI_BASE = "https://generativelanguage.googleapis.com/v1beta"
_async_client: Optional[httpx.AsyncClient] = None


def _get_async_client(timeout: float) -> httpx.AsyncClient:
    global _async_client
    if _async_client is None or _async_client.is_closed:
        _async_client = httpx.AsyncClient(
            timeout=timeout,
            limits=httpx.Limits(max_keepalive_connections=20, max_connections=40),
        )
    return _async_client


def _part_text(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        bits = []
        for item in content:
            if isinstance(item, str):
                bits.append(item)
            elif isinstance(item, dict) and "text" in item:
                bits.append(str(item["text"]))
        return "".join(bits)
    return str(content)


def _tool_to_declaration(tool: BaseTool) -> dict:
    schema = {}
    try:
        if hasattr(tool, "args_schema") and tool.args_schema is not None:
            schema = tool.args_schema.model_json_schema()
    except Exception:
        schema = {"type": "object", "properties": {}}
    # Gemini wants uppercase types in some versions; keep JSON-schema-ish and let API coerce.
    props = schema.get("properties") or {}
    required = schema.get("required") or []
    return {
        "name": tool.name,
        "description": (tool.description or tool.name)[:1024],
        "parameters": {
            "type": "OBJECT",
            "properties": {
                k: {
                    "type": str(v.get("type", "STRING")).upper()
                    if isinstance(v.get("type"), str)
                    else "STRING",
                    "description": v.get("description", "")[:512],
                }
                for k, v in props.items()
            },
            "required": list(required),
        },
    }


def _messages_to_gemini(messages: Sequence[BaseMessage]) -> tuple[Optional[dict], list]:
    system_parts: list[str] = []
    contents: list[dict] = []
    for msg in messages:
        text = _part_text(getattr(msg, "content", ""))
        if isinstance(msg, SystemMessage):
            if text:
                system_parts.append(text)
            continue
        if isinstance(msg, HumanMessage):
            contents.append({"role": "user", "parts": [{"text": text}]})
            continue
        if isinstance(msg, ToolMessage):
            # Represent tool results as user context (compatible with Gemini chat).
            name = getattr(msg, "name", None) or "tool"
            contents.append(
                {
                    "role": "user",
                    "parts": [{"text": f"[Resultado herramienta {name}]:\n{text}"}],
                }
            )
            continue
        if isinstance(msg, AIMessage):
            parts: list[dict] = []
            if text:
                parts.append({"text": text})
            for tc in getattr(msg, "tool_calls", None) or []:
                parts.append(
                    {
                        "functionCall": {
                            "name": tc.get("name"),
                            "args": tc.get("args") or {},
                        }
                    }
                )
            if parts:
                contents.append({"role": "model", "parts": parts})
            continue
        if text:
            contents.append({"role": "user", "parts": [{"text": text}]})

    system_instruction = (
        {"parts": [{"text": "\n\n".join(system_parts)}]} if system_parts else None
    )
    return system_instruction, contents


class FastGeminiChat(BaseChatModel):
    """Minimal ChatModel that hits Gemini REST with thinkingLevel=MINIMAL."""

    model: str = Field(default="")
    temperature: float = 0.0
    max_output_tokens: int = 512
    google_api_key: str = Field(default="")
    timeout: float = 90.0
    _bound_tools: list = PrivateAttr(default_factory=list)

    @property
    def _llm_type(self) -> str:
        return "fast-gemini-rest"

    def _tools(self) -> list:
        """Safe read of PrivateAttr — avoids ModelPrivateAttr-not-iterable on bare instances."""
        priv = getattr(self, "__pydantic_private__", None)
        if isinstance(priv, dict):
            val = priv.get("_bound_tools")
            if isinstance(val, list):
                return val
        try:
            val = object.__getattribute__(self, "_bound_tools")
            if isinstance(val, list):
                return val
        except Exception:
            pass
        return []

    def bind_tools(self, tools: Sequence[BaseTool], **kwargs: Any) -> "FastGeminiChat":
        clone = FastGeminiChat(
            model=self.model,
            temperature=self.temperature,
            max_output_tokens=self.max_output_tokens,
            google_api_key=self.google_api_key,
            timeout=self.timeout,
        )
        # Write into pydantic private store so _tools() always sees a real list.
        priv = getattr(clone, "__pydantic_private__", None)
        if not isinstance(priv, dict):
            object.__setattr__(clone, "__pydantic_private__", {})
            priv = clone.__pydantic_private__
        priv["_bound_tools"] = list(tools)
        return clone

    def _build_body(self, messages: Sequence[BaseMessage]) -> dict[str, Any]:
        system_instruction, contents = _messages_to_gemini(messages)
        body: dict[str, Any] = {
            "contents": contents or [{"role": "user", "parts": [{"text": ""}]}],
            "generationConfig": {
                "temperature": self.temperature,
                "maxOutputTokens": self.max_output_tokens,
                "thinkingConfig": {"thinkingLevel": "MINIMAL"},
            },
        }
        if system_instruction:
            body["systemInstruction"] = system_instruction
        bound = self._tools()
        if bound:
            body["tools"] = [
                {"functionDeclarations": [_tool_to_declaration(t) for t in bound]}
            ]
        return body

    def _parse_response(self, data: dict, elapsed: float, model: str) -> ChatResult:
        usage = data.get("usageMetadata") or {}
        logger.info(
            f"[FastGemini] t={elapsed:.3f}s model={model} "
            f"tools={len(self._tools())} usage={usage}"
        )
        cands = data.get("candidates") or []
        if not cands:
            return ChatResult(generations=[ChatGeneration(message=AIMessage(content=""))])

        parts = ((cands[0].get("content") or {}).get("parts")) or []
        text_bits: list[str] = []
        tool_calls: list[dict] = []
        for idx, part in enumerate(parts):
            if "text" in part and part["text"]:
                text_bits.append(part["text"])
            fc = part.get("functionCall") or part.get("function_call")
            if fc:
                args = fc.get("args") or {}
                if isinstance(args, str):
                    try:
                        args = json.loads(args)
                    except Exception:
                        args = {"raw": args}
                tool_calls.append(
                    {
                        "id": f"call_{idx}",
                        "name": fc.get("name") or "tool",
                        "args": args,
                        "type": "tool_call",
                    }
                )
        msg = AIMessage(content="".join(text_bits), tool_calls=tool_calls)
        return ChatResult(generations=[ChatGeneration(message=msg)])

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        model = self.model or settings.gemini_model
        api_key = self.google_api_key or settings.gemini_api_key
        body = self._build_body(messages)
        url = f"{_GEMINI_BASE}/models/{model}:generateContent?key={api_key}"
        t0 = time.perf_counter()
        with httpx.Client(timeout=self.timeout) as client:
            resp = client.post(url, json=body)
            if resp.status_code >= 400:
                body["generationConfig"].pop("thinkingConfig", None)
                resp = client.post(url, json=body)
            resp.raise_for_status()
            data = resp.json()
        return self._parse_response(data, time.perf_counter() - t0, model)

    async def _agenerate(self, messages, stop=None, run_manager=None, **kwargs):
        model = self.model or settings.gemini_model
        api_key = self.google_api_key or settings.gemini_api_key
        body = self._build_body(messages)
        url = f"{_GEMINI_BASE}/models/{model}:generateContent?key={api_key}"
        t0 = time.perf_counter()
        client = _get_async_client(self.timeout)
        resp = await client.post(url, json=body)
        if resp.status_code >= 400:
            logger.warning(
                f"[FastGemini] HTTP {resp.status_code} with thinkingConfig; retrying without. "
                f"body={resp.text[:200]}"
            )
            body["generationConfig"].pop("thinkingConfig", None)
            resp = await client.post(url, json=body)
        resp.raise_for_status()
        data = resp.json()
        return self._parse_response(data, time.perf_counter() - t0, model)