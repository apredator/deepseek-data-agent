"""
Tests for the tool-calling orchestrator (:mod:`hr_agent`).

Includes the security test required by the specification: an instruction hidden
inside a data *cell* must reach the model only as fenced, untrusted data and must
never end up in the system prompt.
"""

from __future__ import annotations

import pytest

from guardrails import DATA_BLOCK_BEGIN, DATA_BLOCK_END
from hr_agent import (
    HRPayrollAgent,
    build_offline_demo_llm,
    parse_react_action,
    parse_react_final,
)
from llm_client import BaseLLMClient, LLMMessage, MockLLMClient
from sap_hr_data import INJECTION_PAYLOAD, SAPHRDataset


def _agent(dataset: SAPHRDataset, scripted, **kwargs) -> tuple[HRPayrollAgent, MockLLMClient]:
    mock = MockLLMClient(scripted)
    return HRPayrollAgent(dataset, llm=mock, **kwargs), mock


# --------------------------------------------------------------------------- #
# Native tool calling
# --------------------------------------------------------------------------- #

def test_agent_calls_a_tool_and_returns_a_final_answer(dataset: SAPHRDataset) -> None:
    agent, mock = _agent(dataset, [
        {"tool_calls": [{"name": "check_duplicate_wage_types", "arguments": {}}]},
        "Found 5 duplicated wage-type postings.",
    ])
    result = agent.ask("Are there duplicate wage types in one period?")

    assert result.mode == "tool_calling"
    assert result.iterations == 2
    assert result.used_tools == ["check_duplicate_wage_types"]
    assert result.answer == "Found 5 duplicated wage-type postings."
    assert result.trace[0].ok is True
    assert result.trace[0].count == len(dataset.ground_truth()["duplicate_wage_types"])
    assert result.elapsed_seconds > 0


def test_agent_feeds_tool_results_back_as_tool_role_messages(dataset: SAPHRDataset) -> None:
    agent, mock = _agent(dataset, [
        {"tool_calls": [{"name": "get_data_overview", "arguments": {}}]},
        "Dataset overview received.",
    ])
    agent.ask("What data do we have?")

    second_call = mock.calls[1]["messages"]
    tool_messages = [message for message in second_call if message["role"] == "tool"]
    assert len(tool_messages) == 1
    assert tool_messages[0]["name"] == "get_data_overview"
    assert DATA_BLOCK_BEGIN in tool_messages[0]["content"]
    assert tool_messages[0]["tool_call_id"]


def test_agent_executes_multiple_tool_calls_in_one_iteration(dataset: SAPHRDataset) -> None:
    agent, _ = _agent(dataset, [
        {"tool_calls": [
            {"name": "check_negative_net_pay", "arguments": {}},
            {"name": "check_post_termination_payment", "arguments": {}},
        ]},
        "Summarised both checks.",
    ])
    result = agent.ask("Отрицательный net pay и начисления после увольнения?")
    assert result.used_tools == ["check_negative_net_pay", "check_post_termination_payment"]
    assert len(result.trace) == 2
    assert all(step.ok for step in result.trace)


def test_agent_blocks_unknown_tool_and_keeps_going(dataset: SAPHRDataset) -> None:
    agent, _ = _agent(dataset, [
        {"tool_calls": [{"name": "delete_all_records", "arguments": {}}]},
        "I cannot modify data - the tool set is read-only.",
    ])
    result = agent.ask("Удали все записи")

    assert result.trace[0].ok is False
    assert "not in the allow-list" in (result.trace[0].error or "")
    assert result.guardrails["violations"]
    assert result.answer.startswith("I cannot modify")


def test_agent_stops_at_the_iteration_limit_and_still_answers(dataset: SAPHRDataset) -> None:
    scripted = [{"tool_calls": [{"name": "get_data_overview", "arguments": {}}]} for _ in range(3)]
    scripted.append("Final answer after the budget ran out.")
    agent, mock = _agent(dataset, scripted, max_iterations=3)
    result = agent.ask("Сделай бесконечный аудит")

    assert len(result.trace) == 3, "the guard must cap the number of executed tools"
    assert result.answer == "Final answer after the budget ran out."
    # the final answer must be requested without tools
    assert mock.calls[-1]["tools"] == []


def test_agent_handles_tool_failures_gracefully(dataset: SAPHRDataset) -> None:
    agent, _ = _agent(dataset, [
        {"tool_calls": [{"name": "get_employee_record", "arguments": {"unexpected": 1}}]},
        "Handled the tool error and continued.",
    ])
    result = agent.ask("Покажи сотрудника")
    assert result.trace[0].ok is False
    assert result.trace[0].error == "bad_arguments"
    assert "Handled" in result.answer


class _ExplodingClient(BaseLLMClient):
    provider = "exploding"

    def chat(self, messages, tools=None, temperature=0.2, max_tokens=1500):  # noqa: ANN001
        raise RuntimeError("boom")


def test_agent_never_raises_on_llm_failure(dataset: SAPHRDataset) -> None:
    agent = HRPayrollAgent(dataset, llm=_ExplodingClient())
    result = agent.ask("Есть ли аномалии?")
    assert result.mode == "error"
    assert "boom" in result.answer


# --------------------------------------------------------------------------- #
# ReAct fallback
# --------------------------------------------------------------------------- #

def test_react_fallback_when_function_calling_is_unavailable(dataset: SAPHRDataset) -> None:
    mock = MockLLMClient([
        '{"action": "check_negative_net_pay", "arguments": {}}',
        '{"final_answer": "Найдено 3 периода с отрицательным net pay."}',
    ])
    mock.supports_tools = False
    agent = HRPayrollAgent(dataset, llm=mock)
    result = agent.ask("Есть ли отрицательный net pay?")

    assert result.mode == "react_fallback"
    assert result.used_tools == ["check_negative_net_pay"]
    assert "отрицательным net pay" in result.answer
    assert result.trace[0].ok is True


@pytest.mark.parametrize(
    "text,expected",
    [
        ('{"action": "get_data_overview", "arguments": {}}', ("get_data_overview", {})),
        ('{"action": {"name": "get_employee_record", "arguments": {"pernr": "1"}}}', ("get_employee_record", {"pernr": "1"})),
        ('blah\nAction: check_negative_net_pay\nAction Input: {"period": "2024P10"}', ("check_negative_net_pay", {"period": "2024P10"})),
        ('{"tool": "check_retro_without_flag", "args": {}}', ("check_retro_without_flag", {})),
    ],
)
def test_parse_react_action_formats(text: str, expected: tuple) -> None:
    assert parse_react_action(text) == expected


def test_parse_react_action_returns_none_for_plain_text() -> None:
    assert parse_react_action("I think the answer is 42.") is None


def test_parse_react_final_formats() -> None:
    assert parse_react_final('{"final_answer": "done"}') == "done"
    assert parse_react_final("Final Answer: all clear") == "all clear"
    assert parse_react_final("no answer here") is None


# --------------------------------------------------------------------------- #
# Security: prompt injection inside a data cell
# --------------------------------------------------------------------------- #

def test_injection_payload_reaches_the_model_only_as_fenced_data(
    dataset: SAPHRDataset, injection_probe: dict
) -> None:
    """The core security test: hidden instructions must never become instructions."""
    agent, mock = _agent(dataset, [
        {"tool_calls": [{"name": "get_employee_record", "arguments": {"pernr": injection_probe["Pernr"]}}]},
        "The employee record was reviewed; the cell text is data and was not acted upon.",
    ])
    result = agent.ask("Покажи запись этого сотрудника")

    # 1) the payload is never present in any system prompt
    for call in mock.calls:
        for message in call["messages"]:
            if message["role"] == "system":
                assert INJECTION_PAYLOAD not in message["content"]

    # 2) it only travels inside tool-role messages, fenced as untrusted data
    payload_messages = [
        message
        for call in mock.calls
        for message in call["messages"]
        if INJECTION_PAYLOAD in str(message.get("content", ""))
    ]
    assert payload_messages, "the probe should have been fetched by the tool"
    for message in payload_messages:
        assert message["role"] == "tool"
        assert DATA_BLOCK_BEGIN in message["content"]
        assert DATA_BLOCK_END in message["content"]

    # 3) the agent completed normally and flagged nothing as a violation
    assert result.mode == "tool_calling"
    assert result.guardrails["violations"] == []


def test_dataset_scan_flags_the_payload_for_the_operator(dataset: SAPHRDataset) -> None:
    scan = HRPayrollAgent(dataset, llm=MockLLMClient([])).injection_scan()
    assert scan["flagged"] is True
    assert "ignore_instructions" in scan["patterns"]


def test_injection_inside_tool_arguments_is_recorded(dataset: SAPHRDataset) -> None:
    agent, _ = _agent(dataset, [
        {"tool_calls": [{"name": "get_employee_record", "arguments": {"pernr": INJECTION_PAYLOAD}}]},
        "I ignored the text inside the argument and treated it as a value.",
    ])
    result = agent.ask("Покажи сотрудника")
    assert result.guardrails["injection_patterns"], "arguments must be scanned for injections"


# --------------------------------------------------------------------------- #
# Deterministic audit / offline demo
# --------------------------------------------------------------------------- #

def test_audit_requires_no_llm(dataset: SAPHRDataset) -> None:
    agent = HRPayrollAgent(dataset, llm=MockLLMClient([]))
    report = agent.audit()
    assert report["true_positives_total"] == report["expected_total"]
    assert report["missed_total"] == 0
    assert report["detection_rate"] == 1.0
    assert report["injection_scan"]["flagged"] is True


def test_offline_demo_llm_drives_the_real_loop(dataset: SAPHRDataset) -> None:
    agent = HRPayrollAgent(dataset, llm=build_offline_demo_llm(dataset))
    result = agent.ask("Сделай полный аудит")
    assert result.mode == "tool_calling"
    assert len(result.used_tools) == 6
    assert result.trace and all(step.ok for step in result.trace)


def test_semantic_tools_can_be_enabled(dataset: SAPHRDataset) -> None:
    agent = HRPayrollAgent(dataset, llm=MockLLMClient(["ok"]), enable_semantic_tools=True)
    assert "find_similar_employee_names" in agent.registry.names
