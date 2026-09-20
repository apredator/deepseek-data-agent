"""
Tests for the provider layer: factory, DeepSeek function-calling parsing and the
graceful fallback when an endpoint does not support the ``tools`` parameter.

No network access is used - ``requests.post`` is monkeypatched.
"""

from __future__ import annotations

import json

import pytest

from config import Settings
from llm_client import (
    LLMMessage,
    MissingCredentialsError,
    MockLLMClient,
    ToolCall,
    ToolsNotSupportedError,
    create_llm_client,
    tool_message,
)


class _FakeResponse:
    def __init__(self, status_code: int, payload: dict | None = None, text: str = "") -> None:
        self.status_code = status_code
        self._payload = payload
        self.text = text or json.dumps(payload or {})

    def json(self) -> dict:
        if self._payload is None:
            raise ValueError("no json")
        return self._payload


def _settings(**overrides) -> Settings:
    base = Settings.from_env()
    return Settings(**{**base.__dict__, **overrides})


# --------------------------------------------------------------------------- #
# Message model
# --------------------------------------------------------------------------- #

def test_tool_call_parses_json_arguments() -> None:
    call = ToolCall.from_api({
        "id": "call_1",
        "function": {"name": "check_negative_net_pay", "arguments": '{"period": "2024P10"}'},
    })
    assert call.name == "check_negative_net_pay"
    assert call.arguments == {"period": "2024P10"}


def test_tool_call_tolerates_broken_json() -> None:
    call = ToolCall.from_api({"function": {"name": "x", "arguments": "{not json"}})
    assert call.arguments == {}


def test_llm_message_roundtrip() -> None:
    message = LLMMessage(content=None, tool_calls=[ToolCall(id="a", name="t", arguments={"x": 1})])
    payload = message.to_api_dict()
    assert payload["tool_calls"][0]["function"]["name"] == "t"
    assert json.loads(payload["tool_calls"][0]["function"]["arguments"]) == {"x": 1}


def test_tool_message_shape() -> None:
    message = tool_message("id1", "get_data_overview", "payload")
    assert message == {"role": "tool", "tool_call_id": "id1", "name": "get_data_overview", "content": "payload"}


# --------------------------------------------------------------------------- #
# Factory
# --------------------------------------------------------------------------- #

def test_create_mock_client() -> None:
    client = create_llm_client(provider="mock", settings=_settings())
    assert isinstance(client, MockLLMClient)
    assert client.provider == "mock"


def test_factory_raises_without_credentials() -> None:
    settings = _settings(deepseek_api_key="", hf_api_token="", enable_hf_fallback=False, llm_provider="deepseek")
    with pytest.raises(MissingCredentialsError):
        create_llm_client(settings=settings)


def test_factory_falls_back_to_huggingface() -> None:
    settings = _settings(
        deepseek_api_key="", hf_api_token="hf_test", enable_hf_fallback=True, llm_provider="deepseek",
    )
    client = create_llm_client(settings=settings)
    assert client.provider == "hf"
    assert client.supports_tools is False


def test_hf_requires_token() -> None:
    from hf_client import HuggingFaceLLMClient

    with pytest.raises(MissingCredentialsError):
        HuggingFaceLLMClient(api_token="")


def test_mock_client_raises_when_tools_unsupported() -> None:
    client = MockLLMClient(["x"])
    client.supports_tools = False
    with pytest.raises(ToolsNotSupportedError):
        client.chat([{"role": "user", "content": "hi"}], tools=[{"type": "function"}])


# --------------------------------------------------------------------------- #
# DeepSeek client
# --------------------------------------------------------------------------- #

def test_deepseek_client_requires_api_key() -> None:
    from deepseek_client import DeepSeekClient

    with pytest.raises(ValueError):
        DeepSeekClient(api_key="", settings=_settings(deepseek_api_key=""))


def test_deepseek_chat_parses_function_call(monkeypatch) -> None:
    from deepseek_client import DeepSeekClient

    captured: dict = {}

    def fake_post(url, headers=None, json=None, timeout=None):  # noqa: A002
        captured["url"] = url
        captured["payload"] = json
        return _FakeResponse(200, {
            "choices": [{"message": {
                "role": "assistant",
                "content": None,
                "tool_calls": [{
                    "id": "call_9",
                    "function": {"name": "check_retro_without_flag", "arguments": "{}"},
                }],
            }}]
        })

    monkeypatch.setattr("deepseek_client.requests.post", fake_post)
    client = DeepSeekClient(api_key="test-key", model="deepseek-chat", api_base="https://api.deepseek.com/v1")
    response = client.chat(
        [{"role": "user", "content": "check retro"}],
        tools=[{"type": "function", "function": {"name": "check_retro_without_flag", "parameters": {}}}],
    )

    assert response.has_tool_calls
    assert response.tool_calls[0].name == "check_retro_without_flag"
    assert captured["payload"]["tools"]
    assert captured["payload"]["tool_choice"] == "auto"
    assert captured["url"].endswith("/chat/completions")


def test_deepseek_flips_off_tools_on_rejection(monkeypatch) -> None:
    from deepseek_client import DeepSeekClient

    def fake_post(url, headers=None, json=None, timeout=None):  # noqa: A002
        return _FakeResponse(400, {"error": {"message": "tools are not supported by this model"}})

    monkeypatch.setattr("deepseek_client.requests.post", fake_post)
    client = DeepSeekClient(api_key="test-key", model="deepseek-chat")
    with pytest.raises(ToolsNotSupportedError):
        client.chat([{"role": "user", "content": "hi"}], tools=[{"type": "function"}])

    assert client.supports_tools is False
    # once disabled, further tool requests fail fast without a network round-trip
    with pytest.raises(ToolsNotSupportedError):
        client.chat([{"role": "user", "content": "hi"}], tools=[{"type": "function"}])


def test_deepseek_legacy_call_still_works(monkeypatch) -> None:
    from deepseek_client import DeepSeekClient

    def fake_post(url, headers=None, json=None, timeout=None):  # noqa: A002
        return _FakeResponse(200, {"choices": [{"message": {"role": "assistant", "content": "hello"}}]})

    monkeypatch.setattr("deepseek_client.requests.post", fake_post)
    client = DeepSeekClient(api_key="test-key")
    assert client.call("hi") == "hello"
    assert client.analyze("ctx", "question") == "hello"
    assert len(client.get_conversation_history()) == 4
    client.clear_history()
    assert client.get_conversation_history() == []
