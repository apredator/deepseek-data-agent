"""
Entry point for both agents in this repository.

1. **HR/payroll audit agent (new, SAP HCM domain specific)** - the tool-calling
   agent that fulfils the specification::

       python main.py hr --audit
       python main.py hr --question "Есть ли ретро-расчёт за закрытый период без основания?"
       python main.py hr                 # interactive session
       python main.py hr --mock          # offline tool-calling demo, no API key

2. **Generic data-analysis agent (original project, kept for compatibility)**::

       python main.py                       # interactive, asks for a file
       python main.py sample_data.csv       # load a file and start the session
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

from config import configure_logging, get_settings

# NOTE: data_analyzer (matplotlib/seaborn) and deepseek_client are imported lazily
# so that the HR/payroll agent and its tests do not require the plotting stack.

HR_COMMANDS = {"hr", "hr-agent", "payroll", "sap"}


# =========================================================================== #
# 1. Original generic data-analysis agent (unchanged behaviour)
# =========================================================================== #

class DataAnalysisAgent:
    """Main AI Agent for data analysis (original, single-shot behaviour)."""

    def __init__(self, api_key: Optional[str] = None):
        from deepseek_client import DeepSeekClient

        self.deepseek = DeepSeekClient(api_key=api_key)
        self.analyzer: Optional[Any] = None
        self.data_loaded = False

    def load_data(self, file_path: str) -> bool:
        """Load data file."""
        from data_analyzer import DataAnalyzer

        if not os.path.exists(file_path):
            print(f"✗ File not found: {file_path}")
            return False
        self.analyzer = DataAnalyzer(file_path)
        self.data_loaded = True
        return True

    def analyze_query(self, query: str) -> str:
        """Analyze a user query using DeepSeek with a textual data summary."""
        if not self.data_loaded or self.analyzer is None:
            return "Please load data first using load_data()"

        data_summary = self.analyzer.get_summary()
        context = f"""
I have loaded a dataset with the following characteristics:
{data_summary}

User's question: {query}

Please provide a helpful analysis based on the data summary.
"""
        try:
            return self.deepseek.call(context, temperature=0.7, max_tokens=1500)
        except Exception as error:  # noqa: BLE001
            return f"Error during analysis: {error}"

    def interactive_session(self) -> None:
        """Run the interactive analysis session."""
        print("\n" + "=" * 60)
        print("🤖 AI Data Analysis Agent (DeepSeek)")
        print("=" * 60)

        if not self.data_loaded:
            print("\n📁 Step 1: Load your data file")
            while True:
                file_path = input("Enter path to data file (CSV/Excel/JSON): ").strip()
                if self.load_data(file_path):
                    print("✓ Data loaded successfully!")
                    break
                print("✗ Failed to load data. Try again.")

        print("\n📊 Data Summary:")
        print(self.analyzer.get_summary())

        print("\n💬 Step 2: Ask questions about your data")
        print("(Type 'quit' to exit, 'summary' for data summary)")
        print("-" * 60)

        while True:
            user_input = input("\n🤔 Your question: ").strip()
            if user_input.lower() == "quit":
                print("\n👋 Thank you for using the Data Analysis Agent!")
                break
            if user_input.lower() == "summary":
                print(self.analyzer.get_summary())
            elif user_input:
                print("\n🔄 Analyzing...")
                print(f"\n🤖 Analysis:\n{self.analyze_query(user_input)}")
            else:
                print("Please enter a question.")

    def batch_analysis(self, queries: list) -> dict:
        """Analyze multiple queries at once."""
        results = {}
        for index, query in enumerate(queries, 1):
            print(f"Analyzing query {index}/{len(queries)}...")
            results[query] = self.analyze_query(query)
        return results

    def clear_conversation(self) -> None:
        """Clear conversation history."""
        self.deepseek.clear_history()
        print("✓ Conversation history cleared")

    def export_conversation(self, file_path: str) -> bool:
        """Export conversation history to a text file."""
        try:
            history = self.deepseek.get_conversation_history()
            with open(file_path, "w", encoding="utf-8") as handle:
                for index, message in enumerate(history, 1):
                    handle.write(f"\n{'=' * 60}\n")
                    handle.write(f"Message {index} - Role: {message['role'].upper()}\n")
                    handle.write(f"{'=' * 60}\n")
                    handle.write(str(message.get("content", "")) + "\n")
            print(f"✓ Conversation exported to {file_path}")
            return True
        except Exception as error:  # noqa: BLE001
            print(f"✗ Error exporting conversation: {error}")
            return False


# =========================================================================== #
# 2. HR/payroll audit agent CLI
# =========================================================================== #

HR_HELP = """\
Интерактивные команды:
  <вопрос>        задать вопрос агенту (естественный язык)
  audit           детерминированный аудит всех проверок + scorecard
  tools           показать список read-only инструментов
  security        проверить данные на prompt injection
  trace on|off    показывать/скрывать трассу вызовов инструментов
  clear           очистить историю диалога
  help            эта справка
  quit            выход
"""


def _print_audit(report: Dict[str, Any]) -> None:
    print("\n" + "=" * 72)
    print("ДЕТЕРМИНИРОВАННЫЙ АУДИТ (без LLM)")
    print("=" * 72)
    for anomaly_type, info in report["per_type"].items():
        print(f"  {anomaly_type:32s} найдено={info['found']:>3d}  "
              f"ожидалось={info['expected']:>3d}  FP={len(info['false_positives'])}")
    print("-" * 72)
    print(f"  ИТОГО: {report['true_positives_total']} из {report['expected_total']} "
          f"внедрённых аномалий, false positives: {report['false_positives_total']}, "
          f"detection rate: {report['detection_rate'] * 100:.0f}%")
    scan = report.get("injection_scan") or {}
    if scan.get("flagged"):
        print(f"  ⚠ prompt-injection payload в данных: {', '.join(scan.get('patterns', []))}")
    print("=" * 72 + "\n")


def _build_hr_arg_parser() -> argparse.ArgumentParser:
    settings = get_settings()
    parser = argparse.ArgumentParser(
        prog="python main.py hr",
        description="AI-агент для аудита и аналитики HR/payroll-данных SAP HCM (синтетика).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--data", type=Path, default=settings.data_dir,
                        help="Каталог с CSV-таблицами датасета.")
    parser.add_argument("--question", "-q", action="append", default=[],
                        help="Вопрос агенту (можно указать несколько раз).")
    parser.add_argument("--audit", action="store_true",
                        help="Выполнить детерминированный аудит и выйти (LLM не нужен).")
    parser.add_argument("--security", action="store_true",
                        help="Просканировать датасет на prompt-injection и выйти.")
    parser.add_argument("--tools", action="store_true", help="Показать список инструментов и выйти.")
    parser.add_argument("--mock", action="store_true",
                        help="Оффлайн-демо: скриптованный LLM, реальный API не вызывается.")
    parser.add_argument("--provider", choices=["deepseek", "hf", "mock"], default=None,
                        help="Переопределить LLM-провайдера из .env.")
    parser.add_argument("--no-trace", action="store_true", help="Не печатать трассу вызовов.")
    parser.add_argument("--regenerate", action="store_true",
                        help="Перегенерировать синтетический датасет перед запуском.")
    return parser


def hr_main(argv: Optional[List[str]] = None) -> int:
    """CLI for the HR/payroll audit agent."""
    args = _build_hr_arg_parser().parse_args(argv)
    configure_logging()
    settings = get_settings()

    from hr_agent import build_agent, build_offline_demo_llm
    from llm_client import MissingCredentialsError, create_llm_client
    from sap_hr_data import generate_dataset, load_or_generate

    data_dir = Path(args.data)
    if args.regenerate:
        dataset = generate_dataset(settings=settings)
        dataset.save(data_dir)
    else:
        dataset = load_or_generate(data_dir, settings=settings)

    llm = None
    if args.mock or args.provider == "mock":
        llm = build_offline_demo_llm(dataset)
    elif args.provider:
        llm = create_llm_client(provider=args.provider, settings=settings)
    elif not args.audit and not args.security and not args.tools:
        try:
            llm = create_llm_client(settings=settings)
        except MissingCredentialsError as error:
            print(f"\n⚠ {error}\n")
            print("Показываю детерминированный аудит вместо диалога.\n")
            args.audit = True

    agent = build_agent(data_dir, llm=llm, settings=settings, generate_if_missing=False)

    if args.tools:
        print("\nRead-only инструменты агента:\n")
        for spec in agent.registry.describe():
            print(f"  • {spec['name']}\n      {spec['description']}\n")
        return 0

    if args.security:
        print(json.dumps(agent.injection_scan(), ensure_ascii=False, indent=2))
        return 0

    if args.audit:
        _print_audit(agent.audit())
        return 0

    show_trace = not args.no_trace

    def _ask(question: str) -> None:
        result = agent.ask(question)
        print(f"\n🤖 Ответ ({result.mode}, {result.iterations} шаг(ов), "
              f"{result.elapsed_seconds:.2f} с):\n")
        print(result.answer)
        if show_trace:
            print("\n🔧 Трасса инструментов:")
            print(result.render_trace())
        if result.guardrails.get("violations"):
            print("\n⛔ Guardrails: " + "; ".join(result.guardrails["violations"]))
        print()

    if args.question:
        for question in args.question:
            print(f"\n❓ {question}")
            _ask(question)
        return 0

    print("\n" + "=" * 72)
    print("🧾 AI-агент аудита HR/payroll (SAP HCM, синтетические данные)")
    print("=" * 72)
    print(f"Датасет: {dataset.data_dir or data_dir} · сотрудников: {dataset.num_employees} · "
          f"периодов: {len(dataset.periods())}")
    print(f"LLM: {getattr(llm, 'provider', 'deepseek')}/{getattr(llm, 'model', settings.deepseek_model)}")
    print(HR_HELP)

    while True:
        try:
            user_input = input("❓ Вопрос: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n👋 Выход.")
            break

        if not user_input:
            continue
        lowered = user_input.lower()
        if lowered in {"quit", "exit", "q"}:
            print("👋 Выход.")
            break
        if lowered == "help":
            print(HR_HELP)
        elif lowered == "audit":
            _print_audit(agent.audit())
        elif lowered == "tools":
            for spec in agent.registry.describe():
                print(f"  • {spec['name']}: {spec['description']}")
        elif lowered == "security":
            print(json.dumps(agent.injection_scan(), ensure_ascii=False, indent=2))
        elif lowered.startswith("trace"):
            value = lowered.split()[-1] if " " in lowered else "on"
            show_trace = value not in {"off", "0", "false"}
            print(f"Трасса: {'вкл' if show_trace else 'выкл'}")
        elif lowered == "clear":
            agent.reset()
            print("История очищена.")
        else:
            _ask(user_input)

    return 0


# =========================================================================== #
# Dispatch
# =========================================================================== #

def main() -> int:
    """Dispatch to the HR agent or the legacy generic agent."""
    argv = sys.argv[1:]
    if argv and argv[0].lower() in HR_COMMANDS:
        return hr_main(argv[1:])

    configure_logging()
    agent = DataAnalysisAgent()
    if argv:
        data_file = argv[0]
        if agent.load_data(data_file):
            print(f"✓ Data loaded: {data_file}")
            agent.interactive_session()
        else:
            print(f"✗ Could not load data file: {data_file}")
            return 1
    else:
        agent.interactive_session()
    return 0


if __name__ == "__main__":
    sys.exit(main())
