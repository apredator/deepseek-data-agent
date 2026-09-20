"""
HR/payroll audit agent - the tool-calling orchestrator.

This is the piece the specification calls the core difference from the original
single-shot agent: instead of pushing a text summary into the LLM, the agent
advertises a set of **read-only tools** and lets the model decide what to call.

Flow (``ask``):

1. send the question + tool schemas to the LLM;
2. if the model returns ``tool_calls`` - execute them through the guardrails,
   feed the (fenced, untrusted) results back with ``role: tool``;
3. repeat until the model answers in plain text;
4. hard cap of ``MAX_TOOL_ITERATIONS`` (default 5) iterations.

Every step is logged and captured in an :class:`AgentResult` trace, which is what
makes the agent demonstrably transparent (and is what the API and Streamlit demo
visualise).

If the backend cannot do native function calling, the loop falls back to a
text-based ReAct protocol (:meth:`HRPayrollAgent._run_react`).

``HRPayrollAgent.audit()`` runs the same checks deterministically without any
LLM - handy for CI, for an air-gapped demo and for the "found N of M" scorecard.
"""

from __future__ import annotations

import json
import logging
import re
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from config import Settings, get_settings
from guardrails import (
    GUARDRAIL_SYSTEM_CLAUSE,
    GuardrailViolation,
    ToolCallGuard,
    scan_dataset,
    wrap_tool_result,
)
from hr_tools import HRToolbox, ToolRegistry
from llm_client import (
    BaseLLMClient,
    MissingCredentialsError,
    MockLLMClient,
    ToolsNotSupportedError,
    create_llm_client,
    tool_message,
)
from sap_hr_data import SAPHRDataset, load_or_generate

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Result objects
# --------------------------------------------------------------------------- #

@dataclass
class ToolStep:
    """One executed (or refused) tool call, as shown in the trace."""

    iteration: int
    name: str
    arguments: Dict[str, Any]
    ok: bool
    summary: str = ""
    count: int = 0
    duration_ms: float = 0.0
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class AgentResult:
    """Full outcome of one agent question."""

    question: str
    answer: str
    trace: List[ToolStep] = field(default_factory=list)
    iterations: int = 0
    used_tools: List[str] = field(default_factory=list)
    mode: str = "tool_calling"
    provider: str = ""
    elapsed_seconds: float = 0.0
    guardrails: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "question": self.question,
            "answer": self.answer,
            "mode": self.mode,
            "provider": self.provider,
            "iterations": self.iterations,
            "used_tools": self.used_tools,
            "elapsed_seconds": self.elapsed_seconds,
            "trace": [step.to_dict() for step in self.trace],
            "guardrails": self.guardrails,
        }

    def render_trace(self) -> str:
        """Human-readable trace for the CLI."""
        lines = []
        for step in self.trace:
            status = "ok" if step.ok else "!!"
            lines.append(
                f"  [{status}] #{step.iteration} {step.name}"
                f"({json.dumps(step.arguments, ensure_ascii=False)})"
                f" -> {step.count} record(s) in {step.duration_ms:.0f} ms"
                + (f" | {step.error}" if step.error else "")
            )
        return "\n".join(lines) if lines else "  (no tools called)"


# --------------------------------------------------------------------------- #
# Prompts
# --------------------------------------------------------------------------- #

SYSTEM_PROMPT_TEMPLATE = """\
You are an SAP HCM payroll audit agent. You work with synthetic (fictional) SAP HCM
data: infotypes PA0001 (organisational assignment), PA0008 (basic pay),
PA0014 (recurring payments) and payroll results RT.

Your users are HR/payroll analysts and SAP HCM consultants. They ask audit
questions such as "are there employees with a retro calculation into a closed
period without a reason?" or "show everyone whose salary jumped more than 40%
without a promotion record".

HOW TO WORK
- NEVER invent numbers, Pernr values or periods. Every claim must come from a tool result.
- Choose the most specific check tool for the question. Use `get_data_overview` only
  when you need orientation, and `get_employee_record` to inspect a specific Pernr
  that a check has flagged.
- A question may need more than one tool call; you have a hard budget of
  {max_iterations} tool-calling iterations, so be economical.
- When you are done, answer in plain text: state how many cases you found, list the
  concrete Pernr / period / amount evidence and name the affected org units or wage
  types. If a check returns nothing, say so explicitly instead of guessing.
- Answer in the same language as the user's question.

{guardrails}"""

REACT_INSTRUCTIONS = """
YOUR BACKEND HAS NO NATIVE FUNCTION CALLING - USE THIS TEXT PROTOCOL
Respond with exactly one JSON object and nothing else:
  to call a tool:  {{"action": "tool_name", "arguments": {{"arg": "value"}}}}
  to finish:       {{"final_answer": "your answer text"}}
You will receive the tool output as an "Observation:" message. Never write plain
prose until you are ready to give the final answer."""

REACT_FINAL_NUDGE = (
    "Iteration budget exhausted. Give your final answer now as JSON: "
    '{"final_answer": "..."} using only evidence you already collected.'
)


# --------------------------------------------------------------------------- #
# Agent
# --------------------------------------------------------------------------- #

class HRPayrollAgent:
    """Tool-calling orchestrator over a synthetic SAP HCM dataset."""

    def __init__(
        self,
        dataset: SAPHRDataset,
        llm: Optional[BaseLLMClient] = None,
        settings: Optional[Settings] = None,
        toolbox: Optional[HRToolbox] = None,
        max_iterations: Optional[int] = None,
        enable_semantic_tools: Optional[bool] = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.dataset = dataset
        self.toolbox = toolbox or HRToolbox(dataset, settings=self.settings)
        self.registry = ToolRegistry(self.toolbox)

        if enable_semantic_tools is None:
            enable_semantic_tools = self.settings.enable_semantic_tools
        self.semantic_enabled = False
        if enable_semantic_tools:
            self._enable_semantic_tools()

        self.max_iterations = int(max_iterations or self.settings.max_tool_iterations)
        self.guard = ToolCallGuard(
            self.registry.names,
            max_iterations=self.max_iterations,
            read_only=self.settings.read_only,
        )
        self._llm = llm
        self._injection_scan = None

    # ------------------------------------------------------------------ #
    # Setup helpers
    # ------------------------------------------------------------------ #
    def _enable_semantic_tools(self) -> None:
        try:
            from semantic_tools import build_semantic_registry_extras

            specs, functions = build_semantic_registry_extras(self.dataset)
            for spec in specs:
                self.registry.register(spec, functions[spec["name"]])
            self.semantic_enabled = True
            logger.info("Semantic tools enabled: %s", [spec["name"] for spec in specs])
        except Exception as error:  # noqa: BLE001 - optional feature must never break the agent
            logger.warning("Could not enable semantic tools: %s", error)

    @property
    def llm(self) -> BaseLLMClient:
        if self._llm is None:
            self._llm = create_llm_client(settings=self.settings)
        return self._llm

    def injection_scan(self) -> Dict[str, Any]:
        """Scan the dataset once for prompt-injection payloads (cached)."""
        if self._injection_scan is None:
            report = scan_dataset(self.dataset)
            self._injection_scan = report.to_dict()
        return self._injection_scan

    def reset(self) -> None:
        """Reset the per-question guardrail counters."""
        self.guard = ToolCallGuard(
            self.registry.names,
            max_iterations=self.max_iterations,
            read_only=self.settings.read_only,
        )

    def _system_prompt(self) -> str:
        return SYSTEM_PROMPT_TEMPLATE.format(
            max_iterations=self.max_iterations,
            guardrails=GUARDRAIL_SYSTEM_CLAUSE,
        )

    def _tool_catalogue(self) -> str:
        return "\n".join(
            f"- {item['name']}: {item['description'].split('.')[0]}."
            for item in self.registry.describe()
        )

    # ------------------------------------------------------------------ #
    # Tool execution
    # ------------------------------------------------------------------ #
    def _execute_tool(
        self,
        name: str,
        arguments: Dict[str, Any],
        iteration: int,
        trace: List[ToolStep],
    ) -> str:
        """Validate, execute and fence one tool call. Always returns a tool message body."""
        started = time.perf_counter()

        try:
            self.guard.check_tool(name)
            self.guard.check_arguments(name, arguments)
        except GuardrailViolation as violation:
            self.guard.violations.append(str(violation))
            logger.warning("Guardrail blocked tool '%s': %s", name, violation)
            trace.append(ToolStep(
                iteration=iteration, name=name, arguments=arguments, ok=False,
                summary="blocked by guardrails", error=str(violation),
                duration_ms=(time.perf_counter() - started) * 1000,
            ))
            return wrap_tool_result(name, {"error": "guardrail_violation", "detail": str(violation)})

        try:
            result = self.registry.call(name, arguments)
        except Exception as error:  # noqa: BLE001 - never let a tool kill the request
            logger.exception("Tool %s crashed", name)
            result = {
                "tool": name, "ok": False, "count": 0, "records": [], "truncated": False,
                "summary": f"Tool '{name}' failed: {error}", "error": "tool_failure",
            }

        duration_ms = (time.perf_counter() - started) * 1000
        records = result.get("records", []) if isinstance(result, dict) else []
        trace.append(ToolStep(
            iteration=iteration,
            name=name,
            arguments=arguments,
            ok=bool(result.get("ok")) if isinstance(result, dict) else False,
            summary=str(result.get("summary", ""))[:400] if isinstance(result, dict) else "",
            count=len(records) if isinstance(records, list) else 0,
            duration_ms=duration_ms,
            error=result.get("error") if isinstance(result, dict) else None,
        ))
        logger.info(
            "iteration=%d tool=%s args=%s -> %s record(s) in %.0f ms",
            iteration, name, json.dumps(arguments, ensure_ascii=False),
            len(records) if isinstance(records, list) else 0, duration_ms,
        )
        return wrap_tool_result(name, result)

    # ------------------------------------------------------------------ #
    # Main entry point
    # ------------------------------------------------------------------ #
    def ask(self, question: str, reset: bool = True) -> AgentResult:
        """Answer one question, using tools as needed. Never raises on guardrail issues."""
        if reset:
            self.reset()
        started = time.perf_counter()
        try:
            result = self._run_tool_calling(question)
        except ToolsNotSupportedError as error:
            if not self.settings.allow_react_fallback:
                raise
            logger.warning("Native function calling unavailable (%s) - switching to ReAct", error)
            violations = list(self.guard.violations)
            injection_report = self.guard.injection_report
            self.reset()
            self.guard.violations.extend(violations)
            self.guard.injection_report = injection_report
            result = self._run_react(question, note=str(error))
        except MissingCredentialsError:
            raise
        except Exception as error:  # noqa: BLE001 - degrade gracefully, report in the answer
            logger.exception("Agent failure")
            result = AgentResult(
                question=question,
                answer=f"The agent could not complete the request: {error}",
                mode="error",
                provider=getattr(self._llm, "provider", "unknown"),
                guardrails=self.guard.summary(),
            )
        result.elapsed_seconds = round(time.perf_counter() - started, 3)
        result.guardrails = self.guard.summary()
        return result

    # ------------------------------------------------------------------ #
    # Native function-calling loop
    # ------------------------------------------------------------------ #
    def _run_tool_calling(self, question: str) -> AgentResult:
        messages: List[Dict[str, Any]] = [
            {"role": "system", "content": self._system_prompt() + "\n\nAVAILABLE TOOLS\n" + self._tool_catalogue()},
            {"role": "user", "content": question},
        ]
        schemas = self.registry.schemas()
        trace: List[ToolStep] = []
        used_tools: List[str] = []

        while True:
            try:
                iteration = self.guard.start_iteration()
            except GuardrailViolation as limit:
                logger.warning("Tool loop limit hit: %s", limit)
                final = self._finalize(messages, note=str(limit))
                return AgentResult(
                    question=question, answer=final, trace=trace,
                    iterations=self.guard.iterations, used_tools=used_tools,
                    mode="tool_calling",
                    provider=getattr(self.llm, "provider", "unknown"),
                )

            response = self.llm.chat(messages, tools=schemas, temperature=0.1, max_tokens=1200)

            if not response.has_tool_calls:
                return AgentResult(
                    question=question,
                    answer=(response.content or "").strip() or "(the model returned an empty answer)",
                    trace=trace,
                    iterations=iteration,
                    used_tools=used_tools,
                    mode="tool_calling",
                    provider=getattr(self.llm, "provider", "unknown"),
                )

            messages.append(response.to_api_dict())
            for call in response.tool_calls:
                used_tools.append(call.name)
                content = self._execute_tool(call.name, call.arguments, iteration, trace)
                messages.append(tool_message(call.id, call.name, content))

    def _finalize(self, messages: List[Dict[str, Any]], note: str = "") -> str:
        """Ask for a final text answer after the tool budget is exhausted."""
        messages = list(messages) + [{
            "role": "user",
            "content": (
                (note + " ") if note else ""
            ) + (
                "You have no more tool calls available. Answer the original question now, "
                "in plain text, using only the evidence collected so far."
            ),
        }]
        try:
            response = self.llm.chat(messages, tools=None, temperature=0.2, max_tokens=1200)
            content = (response.content or "").strip()
            if content:
                return content
            return (
                f"Tool-call budget exhausted ({self.max_iterations} iterations). "
                "The evidence collected so far is listed in the trace; ask a narrower "
                "follow-up question to get a full answer."
            )
        except Exception as error:  # noqa: BLE001
            logger.error("Could not obtain a final answer: %s", error)
            return f"Tool budget exhausted and the final answer could not be generated: {error}"

    # ------------------------------------------------------------------ #
    # ReAct fallback (no native tool calling)
    # ------------------------------------------------------------------ #
    def _run_react(self, question: str, note: str = "") -> AgentResult:
        system = (
            self._system_prompt()
            + "\n\nAVAILABLE TOOLS\n"
            + self._tool_catalogue()
            + "\n"
            + REACT_INSTRUCTIONS
        )
        messages: List[Dict[str, Any]] = [
            {"role": "system", "content": system},
            {"role": "user", "content": question},
        ]
        trace: List[ToolStep] = []
        used_tools: List[str] = []

        while True:
            try:
                iteration = self.guard.start_iteration()
            except GuardrailViolation:
                final = self._react_finalize(messages)
                return AgentResult(
                    question=question, answer=final, trace=trace,
                    iterations=self.guard.iterations, used_tools=used_tools,
                    mode="react_fallback",
                    provider=getattr(self.llm, "provider", "unknown"),
                )

            response = self.llm.chat(messages, tools=None, temperature=0.1, max_tokens=1200)
            text = (response.content or "").strip()
            messages.append({"role": "assistant", "content": text})

            action = parse_react_action(text)
            if action is None:
                final = parse_react_final(text)
                return AgentResult(
                    question=question, answer=final or text, trace=trace,
                    iterations=iteration, used_tools=used_tools,
                    mode="react_fallback",
                    provider=getattr(self.llm, "provider", "unknown"),
                )

            name, arguments = action
            used_tools.append(name)
            content = self._execute_tool(name, arguments, iteration, trace)
            messages.append({"role": "user", "content": f"Observation:\n{content}"})

    def _react_finalize(self, messages: List[Dict[str, Any]]) -> str:
        messages = list(messages) + [{"role": "user", "content": REACT_FINAL_NUDGE}]
        try:
            response = self.llm.chat(messages, tools=None, temperature=0.2, max_tokens=1200)
        except Exception as error:  # noqa: BLE001
            return f"Tool budget exhausted and the final answer could not be generated: {error}"
        return parse_react_final(response.content or "") or (response.content or "").strip()

    # ------------------------------------------------------------------ #
    # Deterministic audit (no LLM)
    # ------------------------------------------------------------------ #
    def audit(self) -> Dict[str, Any]:
        """Run every anomaly check directly - the "found N of M" scorecard."""
        report = self.toolbox.audit()
        report["injection_scan"] = self.injection_scan()
        return report


# --------------------------------------------------------------------------- #
# ReAct parsing helpers
# --------------------------------------------------------------------------- #

def _iter_json_objects(text: str) -> List[str]:
    """Yield balanced ``{...}`` substrings found in ``text``."""
    objects: List[str] = []
    depth = 0
    start = -1
    in_string = False
    escape = False
    for index, char in enumerate(text):
        if in_string:
            if escape:
                escape = False
            elif char == "\\":
                escape = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == "{":
            if depth == 0:
                start = index
            depth += 1
        elif char == "}":
            if depth > 0:
                depth -= 1
                if depth == 0 and start >= 0:
                    objects.append(text[start:index + 1])
    return objects


def parse_react_action(text: str) -> Optional[Tuple[str, Dict[str, Any]]]:
    """Extract a tool call from a ReAct-style model response."""
    if not text:
        return None

    for candidate in _iter_json_objects(text):
        try:
            payload = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if not isinstance(payload, dict):
            continue
        action = payload.get("action") or payload.get("tool") or payload.get("tool_name")
        if isinstance(action, dict):
            name = action.get("name") or action.get("tool")
            arguments = action.get("arguments") or action.get("args") or {}
        else:
            name = action
            arguments = payload.get("arguments") or payload.get("args") or {}
        if isinstance(name, str) and name:
            return name, arguments if isinstance(arguments, dict) else {}

    # classic "Action: name / Action Input: {...}" format
    action_match = re.search(r"Action\s*:\s*([A-Za-z_][A-Za-z0-9_]*)", text)
    if action_match:
        name = action_match.group(1)
        input_match = re.search(r"Action\s*Input\s*:\s*(\{.*\})", text, flags=re.DOTALL)
        arguments: Dict[str, Any] = {}
        if input_match:
            for candidate in _iter_json_objects(input_match.group(1)):
                try:
                    parsed = json.loads(candidate)
                    if isinstance(parsed, dict):
                        arguments = parsed
                        break
                except json.JSONDecodeError:
                    continue
        return name, arguments
    return None


def parse_react_final(text: str) -> Optional[str]:
    """Extract a ``final_answer`` payload from a ReAct-style response."""
    if not text:
        return None
    for candidate in _iter_json_objects(text):
        try:
            payload = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict) and payload.get("final_answer"):
            return str(payload["final_answer"])
    match = re.search(r"Final\s*Answer\s*:\s*(.+)", text, flags=re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()
    return None


# --------------------------------------------------------------------------- #
# Convenience builder
# --------------------------------------------------------------------------- #

def build_agent(
    data_dir: Optional[Path | str] = None,
    llm: Optional[BaseLLMClient] = None,
    settings: Optional[Settings] = None,
    generate_if_missing: bool = True,
    enable_semantic_tools: Optional[bool] = None,
    **kwargs: Any,
) -> HRPayrollAgent:
    """Load (or generate) a dataset and wrap it in an :class:`HRPayrollAgent`."""
    settings = settings or get_settings()
    data_dir = Path(data_dir) if data_dir else settings.data_dir
    if generate_if_missing:
        dataset = load_or_generate(data_dir, settings=settings)
    else:
        dataset = SAPHRDataset.load(data_dir)
    return HRPayrollAgent(
        dataset, llm=llm, settings=settings,
        enable_semantic_tools=enable_semantic_tools, **kwargs,
    )


def build_offline_demo_llm(dataset: SAPHRDataset, max_iterations: Optional[int] = None) -> MockLLMClient:
    """
    Scripted LLM that exercises the real tool-calling loop without any API key.

    The first (and only) tool iteration fires every anomaly check at once, then the
    final answer summarises the deterministic audit - so an air-gapped demo still
    shows genuine multi-tool orchestration, a full trace and the found/expected
    scorecard.
    """
    toolbox = HRToolbox(dataset)
    audit = toolbox.audit()
    checks = [
        "check_duplicate_wage_types",
        "check_salary_jump_anomalies",
        "check_retro_without_flag",
        "check_cost_center_mismatch",
        "check_negative_net_pay",
        "check_post_termination_payment",
    ]
    lines = [
        "**Offline demo (scripted LLM).** Deterministic audit: "
        f"{audit['true_positives_total']} of {audit['expected_total']} injected anomalies "
        f"found, {audit['false_positives_total']} false positive(s).",
        "",
    ]
    for anomaly_type, info in audit["per_type"].items():
        lines.append(f"- `{anomaly_type}` -> {info['found']} finding(s): {info['summary']}")

    return MockLLMClient([
        {"tool_calls": [{"name": name, "arguments": {}} for name in checks]},
        "\n".join(lines),
    ])


__all__ = [
    "AgentResult",
    "HRPayrollAgent",
    "ToolStep",
    "build_agent",
    "build_offline_demo_llm",
    "parse_react_action",
    "parse_react_final",
]
