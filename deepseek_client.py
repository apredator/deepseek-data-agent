"""
DeepSeek API client with native function-calling support.

Backwards compatible with the original data-analysis agent (``call``/``analyze``
keep working), but now also implements the provider-agnostic
:class:`~llm_client.BaseLLMClient` interface via :meth:`DeepSeekClient.chat`.

If the configured model/endpoint rejects the ``tools`` parameter, the client
flips :attr:`supports_tools` to ``False`` and raises
:class:`~llm_client.ToolsNotSupportedError`; the orchestrator then falls back to
a text-based ReAct loop (specification section 10, "Риски").
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any, Dict, List, Optional, Sequence

import requests
from dotenv import load_dotenv

from config import KNOWN_DEEPSEEK_MODELS, get_settings
from llm_client import BaseLLMClient, LLMMessage, ToolsNotSupportedError

load_dotenv()

logger = logging.getLogger(__name__)

#: Fragments in an error body that indicate the endpoint cannot do tool calling.
_TOOL_REJECTION_HINTS = (
    "tool", "function", "not supported", "unsupported", "unknown parameter",
    "does not support", "invalid_request_error",
)


class DeepSeekClient(BaseLLMClient):
    """Client for the DeepSeek chat-completions API."""

    provider = "deepseek"

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        api_base: Optional[str] = None,
        timeout: Optional[int] = None,
        debug: Optional[bool] = None,
        settings: Optional[Any] = None,
    ) -> None:
        settings = settings or get_settings()
        self.api_key = (
            api_key
            or settings.deepseek_api_key
            or os.getenv("DEEPSEEK_API_KEY", "")
        ).strip()
        self.model = model or settings.deepseek_model
        self.api_base = (api_base or settings.deepseek_api_base).rstrip("/")
        super().__init__(model=self.model, timeout=int(timeout or settings.request_timeout_seconds))
        self.debug = settings.debug if debug is None else debug
        self.messages: List[Dict[str, Any]] = []

        if not self.api_key:
            raise ValueError(
                "DeepSeek API key not found. Set DEEPSEEK_API_KEY in your .env file, "
                "or run the agent with '--audit' (no LLM required)."
            )

        if self.model not in KNOWN_DEEPSEEK_MODELS:
            logger.warning(
                "Model '%s' is not in the known DeepSeek model list %s - proceeding anyway.",
                self.model, KNOWN_DEEPSEEK_MODELS,
            )

        if self.debug:
            logger.debug("DeepSeekClient initialised: model=%s api_base=%s", self.model, self.api_base)

    # ------------------------------------------------------------------ #
    # Legacy conversation helpers
    # ------------------------------------------------------------------ #
    def add_message(self, role: str, content: str) -> None:
        """Append a message to the internal (legacy) conversation history."""
        self.messages.append({"role": role, "content": content})

    def clear_history(self) -> None:
        """Clear the legacy conversation history."""
        self.messages = []

    def get_conversation_history(self) -> List[Dict[str, Any]]:
        """Return a copy of the legacy conversation history."""
        return list(self.messages)

    # ------------------------------------------------------------------ #
    # Transport
    # ------------------------------------------------------------------ #
    def _headers(self) -> Dict[str, str]:
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

    @staticmethod
    def _looks_like_tool_rejection(status_code: int, body: str) -> bool:
        if status_code not in (400, 404, 405, 422, 500, 501):
            return False
        lowered = body.lower()
        return any(hint in lowered for hint in _TOOL_REJECTION_HINTS)

    def _post(self, payload: Dict[str, Any], *, used_tools: bool = False) -> Dict[str, Any]:
        """POST to /chat/completions and return the decoded JSON body."""
        url = f"{self.api_base}/chat/completions"
        if self.debug:
            logger.debug("POST %s payload=%s", url, json.dumps(payload)[:2000])

        try:
            response = requests.post(url, headers=self._headers(), json=payload, timeout=self.timeout)
        except requests.exceptions.Timeout:
            logger.error("DeepSeek API timeout after %ss", self.timeout)
            raise
        except requests.exceptions.ConnectionError as error:
            logger.error("DeepSeek API connection error: %s", error)
            raise

        if response.status_code != 200:
            body = response.text
            logger.error("DeepSeek API error %s: %s", response.status_code, body[:500])
            if used_tools and self._looks_like_tool_rejection(response.status_code, body):
                self.supports_tools = False
                raise ToolsNotSupportedError(
                    "DeepSeek endpoint rejected the 'tools' parameter "
                    f"(HTTP {response.status_code}). Falling back to text-based ReAct. "
                    f"Response: {body[:300]}"
                )
            raise requests.exceptions.RequestException(
                f"DeepSeek API status {response.status_code}: {body[:500]}"
            )

        try:
            data = response.json()
        except ValueError as error:
            raise requests.exceptions.RequestException(
                f"Invalid JSON from DeepSeek API: {response.text[:300]}"
            ) from error

        if self.debug:
            logger.debug("DeepSeek response: %s", json.dumps(data)[:2000])

        if not data.get("choices"):
            raise requests.exceptions.RequestException(
                f"Unexpected DeepSeek response without 'choices': {str(data)[:300]}"
            )
        return data

    # ------------------------------------------------------------------ #
    # BaseLLMClient interface
    # ------------------------------------------------------------------ #
    def chat(
        self,
        messages: Sequence[Dict[str, Any]],
        tools: Optional[Sequence[Dict[str, Any]]] = None,
        temperature: float = 0.2,
        max_tokens: int = 1500,
    ) -> LLMMessage:
        """Send a stateless chat request, optionally advertising tools."""
        if tools and not self.supports_tools:
            raise ToolsNotSupportedError(
                "This DeepSeek model/endpoint does not support function calling."
            )

        payload: Dict[str, Any] = {
            "model": self.model,
            "messages": list(messages),
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": False,
        }
        if tools:
            payload["tools"] = list(tools)
            payload["tool_choice"] = "auto"

        data = self._post(payload, used_tools=bool(tools))
        message = data["choices"][0]["message"]
        return LLMMessage.from_api_message(message)

    # ------------------------------------------------------------------ #
    # Legacy single-shot helpers
    # ------------------------------------------------------------------ #
    def call(
        self,
        user_message: str,
        temperature: float = 0.7,
        max_tokens: int = 2000,
    ) -> str:
        """Legacy: append a user message and return the assistant's text answer."""
        self.add_message("user", user_message)
        payload = {
            "model": self.model,
            "messages": self.messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": False,
        }
        data = self._post(payload)
        content = data["choices"][0]["message"].get("content") or ""
        self.add_message("assistant", content)
        return content

    def analyze(self, context: str, question: str, temperature: float = 0.5) -> str:
        """Legacy: answer a question given a textual context block."""
        prompt = (
            f"Context about the data:\n{context}\n\n"
            f"Question: {question}\n\n"
            "Please provide a detailed analysis based on the context provided. "
            "Answer in the same language as the question."
        )
        return self.call(prompt, temperature=temperature)

    def test_connection(self) -> bool:
        """Ping the API with a trivial request."""
        try:
            logger.info("Testing DeepSeek API connection...")
            data = self._post({
                "model": self.model,
                "messages": [{"role": "user", "content": "Reply with the single word: ok"}],
                "temperature": 0.0,
                "max_tokens": 8,
                "stream": False,
            })
            content = (data["choices"][0]["message"].get("content") or "").strip()
            logger.info("DeepSeek API reachable: %s", content)
            return True
        except Exception as error:  # noqa: BLE001 - surface any failure to the caller
            logger.error("DeepSeek connection test failed: %s", error)
            return False


__all__ = ["DeepSeekClient"]
