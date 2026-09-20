"""
Examples for both agents in this repository.

Offline examples (no API key needed):
    python example_usage.py 5   # synthetic SAP HCM dataset + audit scorecard
    python example_usage.py 6   # full tool-calling loop with a mock LLM
    python example_usage.py 7   # prompt-injection security test

API examples (require DEEPSEEK_API_KEY):
    python example_usage.py 1   # basic generic data analysis
    python example_usage.py 2   # DataAnalyzer features
    python example_usage.py 3   # batch processing
    python example_usage.py 4   # export results / conversation
    python example_usage.py 8   # ask the HR/payroll agent a real question
"""

from __future__ import annotations

import json

import pandas as pd


# --------------------------------------------------------------------------- #
# Original generic examples (require a DeepSeek API key)
# --------------------------------------------------------------------------- #

def create_sample_data():
    """Create a sample dataset for testing."""
    import numpy as np

    np.random.seed(42)
    data = {
        "Date": pd.date_range("2024-01-01", periods=100),
        "Product": np.random.choice(["A", "B", "C"], 100),
        "Sales": np.random.randint(100, 1000, 100),
        "Quantity": np.random.randint(1, 50, 100),
        "Region": np.random.choice(["North", "South", "East", "West"], 100),
        "Customer_Rating": np.random.uniform(1, 5, 100),
    }
    frame = pd.DataFrame(data)
    frame.to_csv("sample_data.csv", index=False)
    print("✓ Sample data created: sample_data.csv")
    return "sample_data.csv"


def example_1_basic_analysis():
    from main import DataAnalysisAgent

    print("\n" + "=" * 70)
    print("EXAMPLE 1: Basic Data Analysis")
    print("=" * 70)
    data_file = create_sample_data()
    agent = DataAnalysisAgent()
    agent.load_data(data_file)
    for query in [
        "What are the top products by sales?",
        "Which region has the highest average rating?",
        "What is the trend in sales over time?",
    ]:
        print(f"\n🤔 Query: {query}")
        print(f"🤖 Response: {agent.analyze_query(query)}\n")


def example_2_data_analyzer():
    from data_analyzer import DataAnalyzer

    print("\n" + "=" * 70)
    print("EXAMPLE 2: Direct Data Analyzer Usage")
    print("=" * 70)
    analyzer = DataAnalyzer(create_sample_data())
    print(analyzer.get_summary())
    print("\n📊 Filtering: Sales > 500")
    print(f"Found {len(analyzer.filter_data('Sales', '>', 500))} records\n")
    print("📊 Group by Region - Average Sales:")
    print(analyzer.group_and_aggregate("Region", "Sales", "mean"))


def example_3_batch_processing():
    from main import DataAnalysisAgent

    print("\n" + "=" * 70)
    print("EXAMPLE 3: Batch Query Processing")
    print("=" * 70)
    agent = DataAnalysisAgent()
    agent.load_data(create_sample_data())
    results = agent.batch_analysis([
        "Summarize the dataset",
        "What are the key statistics?",
        "Identify any patterns or trends",
    ])
    for query, response in results.items():
        print(f"\n❓ {query}\n✓ {response}\n")


def example_4_export_results():
    from main import DataAnalysisAgent

    print("\n" + "=" * 70)
    print("EXAMPLE 4: Export Results and Conversation")
    print("=" * 70)
    agent = DataAnalysisAgent()
    agent.load_data(create_sample_data())
    agent.analyze_query("What is the average sales per region?")
    agent.export_conversation("conversation_history.txt")


# --------------------------------------------------------------------------- #
# HR/payroll examples (offline, no API key)
# --------------------------------------------------------------------------- #

def _load_hr_dataset(employees: int = 300):
    from sap_hr_data import generate_dataset

    return generate_dataset(num_employees=employees, seed=42)


def example_5_synthetic_dataset_and_audit():
    """Generate the synthetic SAP HCM data and print the audit scorecard."""
    print("\n" + "=" * 70)
    print("EXAMPLE 5: Synthetic SAP HCM dataset + deterministic audit")
    print("=" * 70)
    dataset = _load_hr_dataset()
    for name, frame in dataset.tables.items():
        print(f"  {name:22s} {frame.shape[0]:>6d} rows x {frame.shape[1]} columns")

    print("\nInjected anomalies (ground truth):")
    print(json.dumps(dataset.manifest["totals"], indent=2))

    from hr_tools import HRToolbox

    report = HRToolbox(dataset).audit()
    print(f"\nDetection: {report['true_positives_total']}/{report['expected_total']} anomalies, "
          f"{report['false_positives_total']} false positives, "
          f"rate={report['detection_rate'] * 100:.0f}%")
    for anomaly_type, info in report["per_type"].items():
        print(f"  {anomaly_type:32s} {info['found']:>3d} findings  | {info['summary']}")


def example_6_tool_calling_loop_offline():
    """Run the real tool-calling loop against a scripted mock LLM."""
    print("\n" + "=" * 70)
    print("EXAMPLE 6: Tool-calling loop (mock LLM, no API key)")
    print("=" * 70)
    from hr_agent import HRPayrollAgent
    from llm_client import MockLLMClient

    dataset = _load_hr_dataset()
    mock = MockLLMClient([
        {"tool_calls": [
            {"name": "check_duplicate_wage_types", "arguments": {}},
            {"name": "check_retro_without_flag", "arguments": {}},
        ]},
        "Найдены дублирующиеся начисления и ретро-расчёты без основания — см. трассу.",
    ])
    agent = HRPayrollAgent(dataset, llm=mock)
    result = agent.ask("Проверь дубли Lohnart и ретро-расчёты за закрытый период")
    print(f"\nmode={result.mode} iterations={result.iterations} tools={result.used_tools}")
    print(f"answer: {result.answer}")
    print("trace:")
    print(result.render_trace())


def example_7_security_prompt_injection():
    """Show the prompt-injection defence: data is fenced, and mutating tools are refused."""
    print("\n" + "=" * 70)
    print("EXAMPLE 7: Prompt-injection security test")
    print("=" * 70)
    from guardrails import ToolCallGuard, GuardrailViolation, scan_dataset, wrap_tool_result
    from hr_agent import HRPayrollAgent
    from llm_client import MockLLMClient

    dataset = _load_hr_dataset()
    report = scan_dataset(dataset)
    print(f"Injection scan flagged={report.flagged} patterns={sorted(report.patterns)}")
    for finding in report.findings[:3]:
        print(f"  - {finding.location}: [{finding.pattern}] {finding.snippet[:80]}")

    probe = dataset.manifest["injection_probes"][0]
    print(f"\nPayload planted in {probe['field']} of Pernr {probe['Pernr']}")

    # the data reaches the model only inside an explicitly untrusted block
    fenced = wrap_tool_result("get_employee_record", {"records": [{"Vorna": probe["payload"]}]})
    print("\nTool result seen by the model (first 3 lines):")
    print("\n".join(fenced.splitlines()[:3]))

    # a model that tries to modify data is blocked
    guard = ToolCallGuard(["get_employee_record"], max_iterations=5, read_only=True)
    try:
        guard.check_tool("delete_employee_record")
    except GuardrailViolation as violation:
        print(f"\nGuardrail blocked a mutating call: {violation}")

    # and the agent keeps treating the payload as data
    mock = MockLLMClient(["Оклад сотрудника — обычные данные; инструкций из ячейки я не выполняю."])
    result = HRPayrollAgent(dataset, llm=mock).ask("Покажи данные сотрудника")
    print(f"\nAgent answer: {result.answer}")


def example_8_hr_agent_with_api():
    """Ask the HR/payroll agent a real question (requires DEEPSEEK_API_KEY)."""
    print("\n" + "=" * 70)
    print("EXAMPLE 8: HR/payroll audit agent (live LLM)")
    print("=" * 70)
    from hr_agent import build_agent

    agent = build_agent()
    for question in [
        "Есть ли сотрудники с ретро-расчётом за закрытый период без основания?",
        "Покажи всех, у кого оклад вырос более чем на 40% без записи о повышении в оргназначении",
        "Сведи расхождение между Kostl (МВЗ) в оргназначении и в результатах расчёта",
    ]:
        print(f"\n❓ {question}")
        result = agent.ask(question)
        print(f"🤖 {result.answer}")
        print(result.render_trace())


EXAMPLES = {
    "1": example_1_basic_analysis,
    "2": example_2_data_analyzer,
    "3": example_3_batch_processing,
    "4": example_4_export_results,
    "5": example_5_synthetic_dataset_and_audit,
    "6": example_6_tool_calling_loop_offline,
    "7": example_7_security_prompt_injection,
    "8": example_8_hr_agent_with_api,
}


if __name__ == "__main__":
    import sys

    from config import configure_logging

    configure_logging()
    if len(sys.argv) > 1 and sys.argv[1] in EXAMPLES:
        EXAMPLES[sys.argv[1]]()
    else:
        print("Available examples:")
        print("  python example_usage.py 1  - Generic: basic analysis        (needs API key)")
        print("  python example_usage.py 2  - Generic: DataAnalyzer features")
        print("  python example_usage.py 3  - Generic: batch processing      (needs API key)")
        print("  python example_usage.py 4  - Generic: export results        (needs API key)")
        print("  python example_usage.py 5  - HR: synthetic data + audit     (offline)")
        print("  python example_usage.py 6  - HR: tool-calling loop          (offline)")
        print("  python example_usage.py 7  - HR: prompt-injection test      (offline)")
        print("  python example_usage.py 8  - HR: live agent questions       (needs API key)")
