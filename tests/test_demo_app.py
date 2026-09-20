"""
Headless test of the Streamlit demo UI.

Uses Streamlit's own ``AppTest`` harness, so the real script runs end-to-end
(dataset generation -> offline tool-calling loop -> audit panel) without a
browser or an API key.
"""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("streamlit", reason="Streamlit is required for the demo app")

from streamlit.testing.v1 import AppTest  # noqa: E402

from config import reload_settings  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture()
def app(tmp_path, monkeypatch):
    """A Streamlit app under test, pointed at an isolated synthetic dataset."""
    monkeypatch.setenv("DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("SYNTHETIC_NUM_EMPLOYEES", "120")
    monkeypatch.setenv("SYNTHETIC_SEED", "42")
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    reload_settings()

    instance = AppTest.from_file(str(REPO_ROOT / "demo_app.py"), default_timeout=120)
    instance.run()
    yield instance
    reload_settings()


def test_demo_app_boots_without_errors(app: AppTest) -> None:
    assert not app.exception, app.exception
    assert app.title[0].value.startswith("AI-агент")


def test_offline_mode_is_selected_without_api_key(app: AppTest) -> None:
    toggles = app.sidebar.toggle
    assert toggles and toggles[0].value is True, "offline mode should default on without a key"


def test_deterministic_audit_button_renders_scorecard(app: AppTest) -> None:
    audit_buttons = [button for button in app.sidebar.button if "аудит" in button.label.lower()]
    assert audit_buttons, "the deterministic-audit button should exist"
    audit_buttons[0].click().run()
    assert not app.exception, app.exception
    assert "audit" in app.session_state


def test_chat_question_runs_the_tool_loop_offline(app: AppTest) -> None:
    app.chat_input[0].set_value("Сделай полный аудит payroll-данных").run()
    assert not app.exception, app.exception
    messages = app.session_state["messages"]
    assert len(messages) == 1
    assert messages[0]["mode"] == "tool_calling"
    assert len(messages[0]["trace"]) == 6
