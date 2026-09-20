"""
Streamlit demo UI for the HR/payroll audit agent.

This is the "show it to a client in five minutes" surface required by the
specification (section 4.4): load a synthetic SAP HCM dataset, ask questions in
natural language, and watch the tool-call trace that produced the answer.

Run it with::

    streamlit run demo_app.py

Works without an API key: switch on "Offline demo mode" to run the very same
tool-calling loop against a scripted mock LLM.
"""

from __future__ import annotations

import io
import json
import zipfile
from typing import Any, Dict, List

import pandas as pd
import streamlit as st

from config import get_settings
from hr_agent import HRPayrollAgent, build_offline_demo_llm
from hr_tools import HRToolbox, TOOL_SPECS
from sap_hr_data import SAPHRDataset, SyntheticDatasetGenerator

st.set_page_config(page_title="HR/Payroll Audit Agent (SAP HCM, synthetic)", page_icon="🧾", layout="wide")

EXAMPLE_QUESTIONS = [
    "Проверь на дублирующиеся начисления по одному Lohnart в одном периоде",
    "Покажи всех, у кого оклад вырос более чем на 40% без записи о повышении в оргназначении",
    "Есть ли сотрудники с ретро-расчётом за закрытый период без основания?",
    "Сведи расхождение между Kostl в оргназначении и в результатах расчёта",
    "Есть ли начисления после даты увольнения и отрицательный net pay?",
    "Сделай сводку gross pay по орг-единицам за 2024P10",
]


# --------------------------------------------------------------------------- #
# Session state helpers
# --------------------------------------------------------------------------- #

def _init_state() -> None:
    st.session_state.setdefault("dataset", None)
    st.session_state.setdefault("agent", None)
    st.session_state.setdefault("messages", [])
    st.session_state.setdefault("tool_usage", {})


def _ensure_dataset(num_employees: int, seed: int, periods: int, force: bool = False) -> SAPHRDataset:
    settings = get_settings()
    if force or st.session_state["dataset"] is None:
        with st.spinner("Генерирую синтетический датасет SAP HCM…"):
            dataset = SyntheticDatasetGenerator(
                num_employees=num_employees,
                num_periods=periods,
                closed_periods=settings.closed_periods,
                seed=seed,
            ).generate()
            dataset.save(settings.data_dir)
        st.session_state["dataset"] = dataset
        st.session_state["agent"] = None
        st.session_state["messages"] = []
        st.session_state["tool_usage"] = {}
    return st.session_state["dataset"]


def _dataset_from_zip(payload: bytes) -> SAPHRDataset:
    """Load a dataset from an uploaded ZIP that mirrors the data/ directory layout."""
    settings = get_settings()
    target = settings.data_dir
    target.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        archive.extractall(target)
    return SAPHRDataset.load(target)


def _get_agent(dataset: SAPHRDataset, offline: bool) -> HRPayrollAgent:
    if st.session_state["agent"] is None:
        llm = build_offline_demo_llm(dataset) if offline else None
        st.session_state["agent"] = HRPayrollAgent(dataset, llm=llm)
    return st.session_state["agent"]


def _ask(question: str, offline: bool) -> None:
    dataset = st.session_state["dataset"]
    agent = _get_agent(dataset, offline)
    with st.spinner("Агент вызывает инструменты…"):
        result = agent.ask(question)

    for step in result.trace:
        usage = st.session_state["tool_usage"]
        usage[step.name] = usage.get(step.name, 0) + 1

    st.session_state["messages"].append({
        "question": question,
        "answer": result.answer,
        "trace": [step.to_dict() for step in result.trace],
        "iterations": result.iterations,
        "mode": result.mode,
        "elapsed": result.elapsed_seconds,
        "guardrails": result.guardrails,
    })


# --------------------------------------------------------------------------- #
# Sidebar
# --------------------------------------------------------------------------- #

def _render_sidebar() -> Dict[str, Any]:
    settings = get_settings()
    st.sidebar.title("🧾 HR/Payroll Audit Agent")
    st.sidebar.caption("SAP HCM · синтетические данные · read-only")

    st.sidebar.subheader("Датасет")
    num_employees = st.sidebar.slider("Сотрудников", 50, 2000, settings.num_employees, step=50)
    periods = st.sidebar.slider("Периодов", 6, 12, settings.num_periods)
    seed = st.sidebar.number_input("Seed", value=settings.seed, step=1)
    if st.sidebar.button("Сгенерировать / обновить датасет", use_container_width=True):
        _ensure_dataset(int(num_employees), int(seed), int(periods), force=True)

    uploaded = st.sidebar.file_uploader("…или загрузите ZIP с CSV из data/", type=["zip"])
    if uploaded is not None:
        try:
            st.session_state["dataset"] = _dataset_from_zip(uploaded.getvalue())
            st.session_state["agent"] = None
            st.sidebar.success("Датасет загружен из ZIP")
        except Exception as error:  # noqa: BLE001
            st.sidebar.error(f"Не удалось загрузить ZIP: {error}")

    st.sidebar.divider()
    st.sidebar.subheader("Режим LLM")
    offline = st.sidebar.toggle(
        "Offline demo mode (без API-ключа)",
        value=not bool(settings.deepseek_api_key),
        help="Тот же tool-calling цикл, но ответы выдаёт скриптованный mock-LLM.",
    )
    if offline:
        st.sidebar.info("Offline: используется MockLLMClient, реальный API не вызывается.")
    else:
        if settings.deepseek_api_key:
            st.sidebar.success(f"DeepSeek: `{settings.deepseek_model}`")
        else:
            st.sidebar.warning("DEEPSEEK_API_KEY не задан — включите offline-режим.")

    st.sidebar.caption(
        f"Лимит шагов: {settings.max_tool_iterations} · порог оклада: "
        f"{settings.salary_jump_threshold_pct:.0f}%"
    )

    st.sidebar.divider()
    if st.sidebar.button("🔍 Детерминированный аудит (без LLM)", use_container_width=True):
        st.session_state["audit"] = HRToolbox(st.session_state["dataset"]).audit()
    if st.sidebar.button("🧨 Проверить данные на prompt injection", use_container_width=True):
        agent = _get_agent(st.session_state["dataset"], offline)
        st.session_state["injection"] = agent.injection_scan()

    return {"offline": offline}


# --------------------------------------------------------------------------- #
# Panels
# --------------------------------------------------------------------------- #

def _render_dataset_panel(dataset: SAPHRDataset) -> None:
    meta = dataset.manifest.get("meta", {})
    columns = st.columns(4)
    columns[0].metric("Сотрудников", dataset.num_employees)
    columns[1].metric("Периодов", len(dataset.periods()))
    columns[2].metric("Строк в RT", f"{dataset.rt.shape[0]:,}".replace(",", " "))
    columns[3].metric("Закрытых периодов", meta.get("closed_periods", "—"))

    with st.expander("Схема таблиц и внедрённые аномалии"):
        st.json(dataset.summary())
        st.caption("Ground truth (сколько аномалий заложено генератором):")
        st.json(dataset.manifest.get("totals", {}))


def _render_trace(trace: List[Dict[str, Any]]) -> None:
    if not trace:
        st.caption("Инструменты не вызывались.")
        return
    frame = pd.DataFrame([
        {
            "шаг": step["iteration"],
            "инструмент": step["name"],
            "аргументы": json.dumps(step["arguments"], ensure_ascii=False),
            "записей": step["count"],
            "мс": round(step["duration_ms"], 1),
            "статус": "ok" if step["ok"] else "заблокировано",
        }
        for step in trace
    ])
    st.dataframe(frame, use_container_width=True, hide_index=True)


def _render_audit(audit: Dict[str, Any]) -> None:
    st.subheader("Результат детерминированного аудита")
    top = st.columns(4)
    top[0].metric("Найдено / всего", f"{audit['true_positives_total']} / {audit['expected_total']}")
    top[1].metric("Пропущено", audit["missed_total"])
    top[2].metric("False positives", audit["false_positives_total"])
    top[3].metric("Detection rate", f"{audit['detection_rate'] * 100:.0f}%")

    rows = []
    for anomaly_type, info in audit["per_type"].items():
        rows.append({
            "проверка": anomaly_type,
            "инструмент": info["tool"],
            "найдено": info["found"],
            "ожидалось": info["expected"],
            "пропущено": len(info["missed"]),
            "false positives": len(info["false_positives"]),
        })
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    scan = audit.get("injection_scan") or {}
    if scan.get("flagged"):
        st.warning(
            "В данных найдены строки, похожие на prompt injection "
            f"({', '.join(scan.get('patterns', []))}). Агент обрабатывает их как данные."
        )
        st.json(scan)
    else:
        st.success("Prompt-injection payload в данных не обнаружен.")


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def main() -> None:
    _init_state()
    mode = _render_sidebar()
    dataset = _ensure_dataset(
        get_settings().num_employees, get_settings().seed, get_settings().num_periods,
    )

    st.title("AI-агент для аудита HR/payroll-данных SAP HCM")
    st.caption(
        "Демо на **синтетических** данных: инфотипы PA0001 / PA0008 / PA0014 и результаты "
        "расчёта RT. Агент сам решает, какие read-only проверки вызвать, и показывает трассу."
    )

    _render_dataset_panel(dataset)

    left, right = st.columns([2, 1])
    with left:
        st.subheader("Чат с агентом")
        for message in st.session_state["messages"]:
            with st.chat_message("user"):
                st.markdown(message["question"])
            with st.chat_message("assistant"):
                st.markdown(message["answer"])
                st.caption(
                    f"режим: {message['mode']} · шагов: {message['iterations']} · "
                    f"{message['elapsed']:.2f} с"
                )
                with st.expander("Трасса вызовов инструментов"):
                    _render_trace(message["trace"])
                guardrails = message.get("guardrails", {})
                if guardrails.get("violations"):
                    st.error("Guardrails заблокировали вызовы: " + "; ".join(guardrails["violations"]))

        question = st.chat_input("Спросите про аномалии, оклады, МВЗ, ретро-расчёты…")
        if question:
            _ask(question, mode["offline"])
            st.rerun()

        st.caption("Примеры вопросов:")
        st.write(" · ".join(f"`{q}`" for q in EXAMPLE_QUESTIONS[:3]))

    with right:
        st.subheader("Инструменты")
        st.dataframe(
            pd.DataFrame([{"инструмент": spec["name"], "назначение": spec["description"][:90] + "…"}
                          for spec in TOOL_SPECS]),
            use_container_width=True, hide_index=True,
        )
        usage = st.session_state["tool_usage"]
        if usage:
            st.subheader("Частота вызовов")
            st.bar_chart(pd.Series(usage, name="вызовы"))

    if "audit" in st.session_state:
        st.divider()
        _render_audit(st.session_state["audit"])
    if "injection" in st.session_state:
        st.divider()
        st.subheader("Prompt-injection скан")
        st.json(st.session_state["injection"])


if __name__ == "__main__":
    main()
