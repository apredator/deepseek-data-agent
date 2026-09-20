"""
Tests for the FastAPI service.

Skipped automatically when the optional API dependencies are not installed
(``pip install -r requirements.txt`` provides fastapi + uvicorn + httpx).
"""

from __future__ import annotations

import pytest

pytest.importorskip("fastapi", reason="FastAPI extras not installed")
pytest.importorskip("httpx", reason="httpx is required by fastapi.testclient")

from fastapi.testclient import TestClient  # noqa: E402

import api  # noqa: E402
from config import reload_settings  # noqa: E402
from sap_hr_data import SyntheticDatasetGenerator  # noqa: E402


@pytest.fixture()
def client(tmp_path, monkeypatch):
    """API client backed by a freshly generated dataset in a temp directory."""
    monkeypatch.setenv("DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("SYNTHETIC_NUM_EMPLOYEES", "120")
    monkeypatch.setenv("DEEPSEEK_API_KEY", "")
    reload_settings()

    dataset = SyntheticDatasetGenerator(num_employees=120, seed=42).generate()
    dataset.save(tmp_path / "data")

    api._datasets.clear()
    api._agents.clear()
    with TestClient(api.app) as test_client:
        yield test_client
    api._datasets.clear()
    api._agents.clear()
    reload_settings()


def test_health(client) -> None:
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["synthetic_data_only"] is True


def test_tools_endpoint_lists_read_only_tools(client) -> None:
    response = client.get("/tools")
    assert response.status_code == 200
    body = response.json()
    assert body["read_only"] is True
    names = {tool["name"] for tool in body["tools"]}
    assert {"check_duplicate_wage_types", "check_retro_without_flag"} <= names


def test_audit_endpoint_needs_no_llm(client) -> None:
    response = client.post("/audit")
    assert response.status_code == 200
    body = response.json()
    assert body["detection_rate"] == 1.0
    assert body["missed_total"] == 0
    assert body["injection_scan"]["flagged"] is True


def test_load_dataset_endpoint(client) -> None:
    response = client.post("/datasets/load", json={"dataset_id": "alt", "num_employees": 60, "seed": 3})
    assert response.status_code == 200
    body = response.json()
    assert body["dataset_id"] == "alt"
    assert body["num_employees"] == 60
    assert body["synthetic"] is True

    listing = client.get("/datasets").json()["datasets"]
    assert any(item["dataset_id"] == "alt" for item in listing)


def test_analyze_without_credentials_returns_503(client) -> None:
    response = client.post("/analyze", json={"question": "Есть ли аномалии?"})
    assert response.status_code == 503
    assert "DEEPSEEK_API_KEY" in response.json()["detail"]
