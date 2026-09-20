"""
FastAPI service exposing the HR/payroll audit agent.

Specification section 4.4 asks for a ``POST /analyze {question, dataset_id}``
endpoint returning the answer **plus the trace of the tools that were called**.
The service deliberately reuses the same :class:`~hr_agent.HRPayrollAgent` as the
CLI, so behaviour is identical across every interface.

Run it with::

    uvicorn api:app --reload --port 8000

then::

    curl -s http://127.0.0.1:8000/analyze \
      -H 'Content-Type: application/json' \
      -d '{"question": "Are there duplicate wage types in one period?"}' | jq
"""

from __future__ import annotations

import logging
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional

try:  # pragma: no cover - import guard for environments without the extra
    from fastapi import FastAPI, HTTPException
    from pydantic import BaseModel, Field
except ImportError as error:  # pragma: no cover
    raise RuntimeError(
        "FastAPI is required for the API service. Install it with: "
        "pip install -r requirements.txt  (fastapi + uvicorn)"
    ) from error

from config import configure_logging, get_settings
from hr_agent import HRPayrollAgent
from llm_client import LLMError, MissingCredentialsError
from sap_hr_data import SAPHRDataset, load_or_generate

configure_logging()
logger = logging.getLogger(__name__)

app = FastAPI(
    title="HR/Payroll Audit Agent API",
    description=(
        "Read-only tool-calling AI agent for auditing synthetic SAP HCM HR/payroll data. "
        "Every response contains the full trace of the tools that were executed."
    ),
    version="2.0.0",
)

_lock = threading.Lock()
_datasets: Dict[str, SAPHRDataset] = {}
_agents: Dict[str, HRPayrollAgent] = {}


# --------------------------------------------------------------------------- #
# Schemas
# --------------------------------------------------------------------------- #

class AnalyzeRequest(BaseModel):
    question: str = Field(..., min_length=1, description="Natural-language audit question.")
    dataset_id: str = Field("default", description="Identifier of the loaded dataset.")
    max_iterations: Optional[int] = Field(None, ge=1, le=10, description="Tool-call budget override.")


class ToolStepModel(BaseModel):
    iteration: int
    name: str
    arguments: Dict[str, Any]
    ok: bool
    summary: str = ""
    count: int = 0
    duration_ms: float = 0.0
    error: Optional[str] = None


class AnalyzeResponse(BaseModel):
    question: str
    answer: str
    mode: str
    provider: str
    iterations: int
    used_tools: List[str]
    elapsed_seconds: float
    trace: List[ToolStepModel]
    guardrails: Dict[str, Any]


class LoadDatasetRequest(BaseModel):
    dataset_id: str = "default"
    data_dir: Optional[str] = Field(None, description="Directory with the synthetic CSVs.")
    num_employees: Optional[int] = Field(None, ge=10, le=20_000)
    seed: Optional[int] = None
    force_regenerate: bool = False


# --------------------------------------------------------------------------- #
# Internal helpers
# --------------------------------------------------------------------------- #

def _get_dataset(dataset_id: str) -> SAPHRDataset:
    dataset = _datasets.get(dataset_id)
    if dataset is None:
        settings = get_settings()
        logger.info("Loading dataset '%s' from %s", dataset_id, settings.data_dir)
        dataset = load_or_generate(settings.data_dir, settings=settings)
        _datasets[dataset_id] = dataset
    return dataset


def _get_agent(dataset_id: str, max_iterations: Optional[int] = None) -> HRPayrollAgent:
    with _lock:
        if max_iterations is not None or dataset_id not in _agents:
            _agents[dataset_id] = HRPayrollAgent(
                _get_dataset(dataset_id), max_iterations=max_iterations,
            )
        return _agents[dataset_id]


# --------------------------------------------------------------------------- #
# Endpoints
# --------------------------------------------------------------------------- #

@app.get("/health", tags=["meta"])
def health() -> Dict[str, Any]:
    """Liveness probe plus a summary of what is configured."""
    settings = get_settings()
    return {
        "status": "ok",
        "synthetic_data_only": True,
        "llm_provider": settings.llm_provider,
        "deepseek_model": settings.deepseek_model,
        "deepseek_key_configured": bool(settings.deepseek_api_key),
        "hf_fallback_enabled": settings.enable_hf_fallback and bool(settings.hf_api_token),
        "max_tool_iterations": settings.max_tool_iterations,
        "salary_jump_threshold_pct": settings.salary_jump_threshold_pct,
        "datasets": sorted(_datasets),
    }


@app.get("/tools", tags=["meta"])
def list_tools(dataset_id: str = "default") -> Dict[str, Any]:
    """Return the read-only tool catalogue advertised to the model."""
    agent = _get_agent(dataset_id)
    return {"read_only": True, "tools": agent.registry.describe()}


@app.get("/datasets", tags=["datasets"])
def list_datasets() -> Dict[str, Any]:
    """List the datasets currently held in memory."""
    return {
        "datasets": [
            {
                "dataset_id": dataset_id,
                "num_employees": dataset.num_employees,
                "periods": dataset.periods(),
                "rows": {name: int(frame.shape[0]) for name, frame in dataset.tables.items()},
            }
            for dataset_id, dataset in _datasets.items()
        ]
    }


@app.post("/datasets/load", tags=["datasets"])
def load_dataset(request: LoadDatasetRequest) -> Dict[str, Any]:
    """Register a dataset (loading it from disk or generating a synthetic one)."""
    settings = get_settings()
    data_dir = Path(request.data_dir) if request.data_dir else settings.data_dir

    if request.force_regenerate or request.num_employees or request.seed is not None:
        from sap_hr_data import generate_dataset

        dataset = generate_dataset(
            num_employees=request.num_employees,
            seed=request.seed,
            settings=settings,
        )
        dataset.save(data_dir)
    else:
        dataset = load_or_generate(data_dir, settings=settings)

    with _lock:
        _datasets[request.dataset_id] = dataset
        _agents.pop(request.dataset_id, None)

    return {
        "dataset_id": request.dataset_id,
        "data_dir": str(data_dir),
        "num_employees": dataset.num_employees,
        "periods": dataset.periods(),
        "synthetic": True,
    }


@app.post("/analyze", response_model=AnalyzeResponse, tags=["agent"])
def analyze(request: AnalyzeRequest) -> AnalyzeResponse:
    """Ask the agent a natural-language audit question."""
    try:
        agent = _get_agent(request.dataset_id, max_iterations=request.max_iterations)
    except MissingCredentialsError as error:
        raise HTTPException(status_code=503, detail=str(error)) from error

    try:
        result = agent.ask(request.question)
    except MissingCredentialsError as error:
        raise HTTPException(status_code=503, detail=str(error)) from error
    except LLMError as error:
        raise HTTPException(status_code=502, detail=f"LLM backend error: {error}") from error

    return AnalyzeResponse(**result.to_dict())


@app.post("/audit", tags=["agent"])
def audit(dataset_id: str = "default") -> Dict[str, Any]:
    """Deterministic scan of every anomaly check - no LLM, no API key required."""
    agent = _get_agent(dataset_id)
    return agent.audit()


@app.post("/analyze/stream-trace", tags=["agent"])
def analyze_trace_only(request: AnalyzeRequest) -> Dict[str, Any]:
    """Convenience endpoint returning only the tool trace (for UI timelines)."""
    result = analyze(request)
    return {"used_tools": result.used_tools, "trace": [step.model_dump() for step in result.trace]}


__all__ = ["app"]
