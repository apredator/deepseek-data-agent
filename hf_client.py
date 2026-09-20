"""
Hugging Face Inference Providers adapter - the secondary LLM backend.

This module exists to prove that the orchestrator is provider-agnostic
(specification section 5.1): the agent's tool-calling loop is written against
:class:`~llm_client.BaseLLMClient`, so swapping DeepSeek for a model served by
Hugging Face requires no change in :mod:`hr_agent`.

Because most HF-served models do not offer reliable OpenAI-style function
calling, this client advertises ``supports_tools = False`` by default and the
agent transparently falls back to the text-based ReAct loop.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, Optional, Sequence

import requests

from llm_client import (
    BaseLLMClient,
    LLMMessage,
    MissingCredentialsError,
    ToolsNotSupportedError,
)

logger = logging.getLogger(__name__)


class HuggingFaceLLMClient(BaseLLMClient):
    """Chat client for Hugging Face Inference Providers (OpenAI-compatible router)."""

    provider = "hf"

    def __init__(
        self,
        api_token: Optional[str] = None,
        model: str = "Qwen/Qwen2.5-7B-Instruct",
        api_base: str = "https://router.huggingface.co/v1",
        timeout: int = 60,
        supports_tools: bool = False,
    ) -> None:
        super().__init__(model=model, timeout=int(timeout))
        self.api_token = (api_token or "").strip()
        self.api_base = api_base.rstrip("/")
        self.supports_tools = supports_tools

        if not self.api_token:
            raise MissingCredentialsError(
                "Hugging Face token not found. Set HF_TOKEN to use LLM_PROVIDER=hf."
            )

    def _headers(self) -> Dict[str, str]:
        return {
            "Authorization": f"Bearer {self.api_token}",
            "Content-Type": "application/json",
        }

    def chat(
        self,
        messages: Sequence[Dict[str, Any]],
        tools: Optional[Sequence[Dict[str, Any]]] = None,
        temperature: float = 0.2,
        max_tokens: int = 1500,
    ) -> LLMMessage:
        if tools and not self.supports_tools:
            raise ToolsNotSupportedError(
                "HuggingFaceLLMClient does not advertise native function calling; "
                "the agent will use the ReAct fallback."
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

        url = f"{self.api_base}/chat/completions"
        logger.debug("HF request to %s (model=%s)", url, self.model)

        response = requests.post(url, headers=self._headers(), json=payload, timeout=self.timeout)
        if response.status_code != 200:
            logger.error("Hugging Face API error %s: %s", response.status_code, response.text[:500])
            raise requests.exceptions.RequestException(
                f"Hugging Face API status {response.status_code}: {response.text[:500]}"
            )

        try:
            data = response.json()
        except ValueError as error:
            raise requests.exceptions.RequestException(
                f"Invalid JSON from Hugging Face API: {response.text[:300]}"
            ) from error

        if not data.get("choices"):
            raise requests.exceptions.RequestException(
                f"Unexpected Hugging Face response: {json.dumps(data)[:300]}"
            )
        return LLMMessage.from_api_message(data["choices"][0]["message"])

    def health(self) -> Dict[str, Any]:
        info = super().health()
        info["api_base"] = self.api_base
        return info


__all__ = ["HuggingFaceLLMClient"]
