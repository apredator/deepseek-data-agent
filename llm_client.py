"""
Provider-agnostic LLM interface for the HR/payroll agent.

The agent never talks to a vendor SDK directly - it only knows
:class:`BaseLLMClient`.  Concrete adapters:

* :class:`~deepseek_client.DeepSeekClient`      - primary backend, native function calling
* :class:`~hf_client.HuggingFaceLLMClient`      - secondary/fallback backend (HF Inference Providers)
* :class:`MockLLMClient`                        - deterministic, offline (tests, demos, CI)

Keeping the orchestrator model-agnostic is exactly what the specification asks
for in section 5.1 ("абстрактный LLMClient, а не жёсткая привязка к одному
провайдеру").
"""

from __future__ import annotations

import json
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Union

from config import Settings, get_settings

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Errors
# --------------------------------------------------------------------------- #

class LLMError(RuntimeError):
    """Base class for LLM transport errors."""


class MissingCredentialsError(LLMError):
    """Raised when no usable provider credentials are configured."""


class ToolsNotSupportedError(LLMError):
    """Raised when the selected model/endpoint cannot do native function calling."""


# --------------------------------------------------------------------------- #
# Message model
# --------------------------------------------------------------------------- #

@dataclass
class ToolCall:
    """A single tool call requested by the model."""

    id: str
    name: str
    arguments: Dict[str, Any] = field(default_factory=dict)
    raw_arguments: str = ""

    @classmethod
    def from_api(cls, payload: Dict[str, Any]) -> "ToolCall":
        function = payload.get("function", {}) or {}
        raw_arguments = function.get("arguments") or "{}"
        if isinstance(raw_arguments, dict):
            arguments = raw_arguments
        else:
            try:
                arguments = json.loads(raw_arguments) if raw_arguments.strip() else {}
            except json.JSONDecodeError:
                arguments = {}
        return cls(
            id=payload.get("id") or f"call_{function.get('name', 'unknown')}",
            name=function.get("name", ""),
            arguments=arguments if isinstance(arguments, dict) else {},
            raw_arguments=raw_arguments if isinstance(raw_arguments, str) else json.dumps(raw_arguments),
        )


@dataclass
class LLMMessage:
    """Normalized assistant message (final text and/or tool calls)."""

    role: str = "assistant"
    content: Optional[str] = None
    tool_calls: List[ToolCall] = field(default_factory=list)
    raw: Dict[str, Any] = field(default_factory=dict)

    @property
    def has_tool_calls(self) -> bool:
        return bool(self.tool_calls)

    @classmethod
    def from_api_message(cls, message: Dict[str, Any]) -> "LLMMessage":
        raw_calls = message.get("tool_calls") or []
        return cls(
            role=message.get("role", "assistant"),
            content=message.get("content"),
            tool_calls=[ToolCall.from_api(call) for call in raw_calls],
            raw=message,
        )

    def to_api_dict(self) -> Dict[str, Any]:
        """Serialize back into the OpenAI/DeepSeek chat format."""
        payload: Dict[str, Any] = {"role": self.role, "content": self.content or ""}
        if self.tool_calls:
            payload["tool_calls"] = [
                {
                    "id": call.id,
                    "type": "function",
                    "function": {"name": call.name, "arguments": json.dumps(call.arguments, ensure_ascii=False)},
                }
                for call in self.tool_calls
            ]
        return payload


def tool_message(tool_call_id: str, name: str, content: str) -> Dict[str, Any]:
    """Build the ``role: tool`` message that answers a tool call."""
    return {"role": "tool", "tool_call_id": tool_call_id, "name": name, "content": content}


# --------------------------------------------------------------------------- #
# Base client
# --------------------------------------------------------------------------- #

class BaseLLMClient(ABC):
    """Minimal chat interface every provider adapter must implement."""

    provider: str = "base"
    supports_tools: bool = True

    def __init__(self, model: str = "", timeout: int = 60) -> None:
        self.model = model
        self.timeout = timeout

    @abstractmethod
    def chat(
        self,
        messages: Sequence[Dict[str, Any]],
        tools: Optional[Sequence[Dict[str, Any]]] = None,
        temperature: float = 0.2,
        max_tokens: int = 1500,
    ) -> LLMMessage:
        """Send a chat-completion request and return a normalized message."""

    def health(self) -> Dict[str, Any]:
        return {
            "provider": self.provider,
            "model": self.model,
            "supports_tools": self.supports_tools,
        }


# --------------------------------------------------------------------------- #
# Mock client (offline / tests / air-gapped demo)
# --------------------------------------------------------------------------- #

class MockLLMClient(BaseLLMClient):
    """
    Deterministic scripted client.

    ``scripted`` is a list of canned responses consumed in order.  A ``str`` is
    returned as a final answer; a ``dict`` may contain ``content`` and/or
    ``tool_calls`` (as ``[{"name": ..., "arguments": {...}}]``).

    Used by the test-suite (no API key required) and by ``--mock`` demo mode.
    """

    provider = "mock"
    supports_tools = True

    def __init__(
        self,
        scripted: Optional[Sequence[Union[str, Dict[str, Any], LLMMessage]]] = None,
        default_final: str = "No scripted response left (mock client exhausted).",
        model: str = "mock-model",
    ) -> None:
        super().__init__(model=model, timeout=1)
        self.scripted: List[Union[str, Dict[str, Any], LLMMessage]] = list(scripted or [])
        self.default_final = default_final
        self.calls: List[Dict[str, Any]] = []

    def queue(self, response: Union[str, Dict[str, Any], LLMMessage]) -> None:
        self.scripted.append(response)

    def chat(
        self,
        messages: Sequence[Dict[str, Any]],
        tools: Optional[Sequence[Dict[str, Any]]] = None,
        temperature: float = 0.2,
        max_tokens: int = 1500,
    ) -> LLMMessage:
        self.calls.append({"messages": list(messages), "tools": list(tools or [])})
        if tools and not self.supports_tools:
            raise ToolsNotSupportedError(
                "MockLLMClient was configured without native tool support (test fixture)."
            )
        if not self.scripted:
            return LLMMessage(content=self.default_final)

        response = self.scripted.pop(0)
        if isinstance(response, LLMMessage):
            return response
        if isinstance(response, str):
            return LLMMessage(content=response)

        calls: List[ToolCall] = []
        for index, item in enumerate(response.get("tool_calls", []) or []):
            calls.append(ToolCall(
                id=item.get("id") or f"call_{index}",
                name=item.get("name", ""),
                arguments=item.get("arguments", {}) or {},
            ))
        return LLMMessage(content=response.get("content"), tool_calls=calls, raw=response)


# --------------------------------------------------------------------------- #
# Factory
# --------------------------------------------------------------------------- #

def create_llm_client(
    provider: Optional[str] = None,
    settings: Optional[Settings] = None,
    *,
    require_credentials: bool = True,
    **kwargs: Any,
) -> BaseLLMClient:
    """
    Build the configured LLM client.

    Falls back from DeepSeek to Hugging Face when ``ENABLE_HF_FALLBACK=true`` and
    a DeepSeek key is missing, which demonstrates the model-agnostic design.
    """
    settings = settings or get_settings()
    provider = (provider or settings.llm_provider or "deepseek").lower()

    def _deepseek() -> BaseLLMClient:
        from deepseek_client import DeepSeekClient  # lazy import avoids a cycle

        return DeepSeekClient(
            api_key=settings.deepseek_api_key,
            model=settings.deepseek_model,
            api_base=settings.deepseek_api_base,
            timeout=settings.request_timeout_seconds,
            **kwargs,
        )

    def _hf() -> BaseLLMClient:
        from hf_client import HuggingFaceLLMClient  # lazy import

        return HuggingFaceLLMClient(
            api_token=settings.hf_api_token,
            model=settings.hf_model,
            api_base=settings.hf_api_base,
            timeout=settings.request_timeout_seconds,
            **kwargs,
        )

    if provider in {"mock", "offline"}:
        return MockLLMClient(**kwargs)

    if provider == "hf":
        if not settings.hf_api_token and require_credentials:
            raise MissingCredentialsError(
                "LLM_PROVIDER=hf but HF_TOKEN is not set. "
                "Export HF_TOKEN or run with --audit/--mock."
            )
        return _hf()

    # default: DeepSeek with optional HF fallback
    if not settings.deepseek_api_key:
        if settings.enable_hf_fallback and settings.hf_api_token:
            logger.warning("DEEPSEEK_API_KEY missing - falling back to Hugging Face inference")
            return _hf()
        if require_credentials:
            raise MissingCredentialsError(
                "DEEPSEEK_API_KEY is not set. Create a .env file (see .env.example), "
                "or use '--audit' for the deterministic scan and '--mock' for an offline demo."
            )
    return _deepseek()


__all__ = [
    "BaseLLMClient",
    "LLMError",
    "LLMMessage",
    "MissingCredentialsError",
    "MockLLMClient",
    "ToolCall",
    "ToolsNotSupportedError",
    "create_llm_client",
    "tool_message",
]
