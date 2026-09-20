"""
Shared pytest fixtures.

The dataset is generated once per session with a fixed seed, so every check has a
*known* answer (the ground-truth manifest) to be compared against.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from hr_tools import HRToolbox  # noqa: E402
from sap_hr_data import SAPHRDataset, SyntheticDatasetGenerator  # noqa: E402

#: Small enough to keep the suite fast, large enough for every injected pool.
NUM_EMPLOYEES = 150
NUM_PERIODS = 12
CLOSED_PERIODS = 9
SEED = 42


@pytest.fixture(scope="session")
def dataset() -> SAPHRDataset:
    """A deterministic synthetic SAP HCM dataset with known injected anomalies."""
    generator = SyntheticDatasetGenerator(
        num_employees=NUM_EMPLOYEES,
        num_periods=NUM_PERIODS,
        closed_periods=CLOSED_PERIODS,
        seed=SEED,
    )
    return generator.generate()


@pytest.fixture()
def toolbox(dataset: SAPHRDataset) -> HRToolbox:
    """A fresh read-only toolbox bound to the session dataset."""
    return HRToolbox(dataset)


@pytest.fixture()
def injection_probe(dataset: SAPHRDataset) -> dict:
    """The planted prompt-injection payload metadata."""
    probes = dataset.manifest.get("injection_probes") or []
    assert probes, "the generator must plant an injection probe by default"
    return probes[0]
