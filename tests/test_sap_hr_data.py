"""Tests for the synthetic SAP HCM data layer (specification section 4.1)."""

from __future__ import annotations

import pandas as pd
import pytest

from conftest import CLOSED_PERIODS, NUM_EMPLOYEES, NUM_PERIODS, SEED
from sap_hr_data import (
    ANOMALY_SPECS,
    INJECTION_PAYLOAD,
    SYNTHETIC_DISCLAIMER,
    SAPHRDataset,
    SyntheticDatasetGenerator,
    build_periods,
    is_closed_period,
    normalize_period,
    period_from_date,
    period_to_date,
)

EXPECTED_COLUMNS = {
    "pa0001": {"Pernr", "Orgeh", "Kostl", "Planstelle", "Begda", "Endda"},
    "pa0008": {"Pernr", "Lohnart", "Betrag", "Waehrung", "Begda", "Endda"},
    "pa0014": {"Pernr", "Lohnart", "Betrag", "Periodizitaet"},
    "rt": {
        "Pernr", "Abrechnungsperiode", "Lohnart", "Betrag",
        "InPeriod", "ForPeriod", "Retro", "Kostl",
    },
}


@pytest.mark.parametrize("table,required", list(EXPECTED_COLUMNS.items()))
def test_tables_have_infotype_structure(dataset: SAPHRDataset, table: str, required: set) -> None:
    """Every generated table exposes at least the fields required by the spec."""
    frame = getattr(dataset, table)
    assert required.issubset(set(frame.columns)), f"{table} is missing {required - set(frame.columns)}"
    assert not frame.empty


def test_dataset_size_matches_configuration(dataset: SAPHRDataset) -> None:
    assert dataset.num_employees == NUM_EMPLOYEES
    assert len(dataset.periods()) == NUM_PERIODS
    assert len(build_periods(NUM_PERIODS)) == NUM_PERIODS


def test_generation_is_deterministic() -> None:
    """The same seed must produce byte-identical tables and manifest totals."""
    first = SyntheticDatasetGenerator(num_employees=40, seed=7).generate()
    second = SyntheticDatasetGenerator(num_employees=40, seed=7).generate()
    pd.testing.assert_frame_equal(first.rt, second.rt)
    assert first.manifest["totals"] == second.manifest["totals"]

    other = SyntheticDatasetGenerator(num_employees=40, seed=8).generate()
    assert not other.rt.equals(first.rt)


def test_all_six_anomaly_types_are_injected(dataset: SAPHRDataset) -> None:
    totals = dataset.manifest["totals"]
    assert set(totals) == set(ANOMALY_SPECS), "manifest must cover every declared anomaly type"
    for anomaly_type, count in totals.items():
        assert count > 0, f"no {anomaly_type} was injected"


def test_ground_truth_manifest_is_scoreable(dataset: SAPHRDataset) -> None:
    truth = dataset.ground_truth()
    assert set(truth) == set(ANOMALY_SPECS)
    for anomaly_type, keys in truth.items():
        assert keys, f"ground truth for {anomaly_type} is empty"
        for key in keys:
            assert all(part is not None for part in key), f"incomplete key {key} for {anomaly_type}"


def test_injection_probe_is_planted(dataset: SAPHRDataset, injection_probe: dict) -> None:
    assert INJECTION_PAYLOAD in injection_probe["payload"]
    rows = dataset.pa0001[dataset.pa0001["Pernr"] == injection_probe["Pernr"]]
    assert not rows.empty
    assert INJECTION_PAYLOAD in set(rows["Vorna"])


def test_data_is_explicitly_marked_synthetic(dataset: SAPHRDataset) -> None:
    assert dataset.manifest["meta"]["synthetic"] is True
    assert "SYNTHETIC" in SYNTHETIC_DISCLAIMER.upper()
    assert dataset.manifest["meta"]["disclaimer"] == SYNTHETIC_DISCLAIMER


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("2024P07", "2024P07"),
        ("2024-07", "2024P07"),
        ("202407", "2024P07"),
        ("P07", "2024P07"),
        ("07", "2024P07"),
        (7, "2024P07"),
        ("2024-P07", "2024P07"),
    ],
)
def test_normalize_period_accepts_model_formats(raw, expected: str) -> None:
    assert normalize_period(raw) == expected


def test_normalize_period_rejects_garbage() -> None:
    with pytest.raises(ValueError):
        normalize_period("not-a-period")


def test_period_date_roundtrip() -> None:
    assert period_to_date("2024P07") == pd.Timestamp("2024-07-01")
    assert period_from_date("2024-07-15") == "2024P07"


@pytest.mark.parametrize("month,closed", [(1, True), (9, True), (10, False), (12, False)])
def test_is_closed_period(month: int, closed: bool) -> None:
    assert is_closed_period(f"2024P{month:02d}", CLOSED_PERIODS) is closed


def test_save_and_load_roundtrip(dataset: SAPHRDataset, tmp_path) -> None:
    dataset.save(tmp_path)
    assert (tmp_path / "anomalies_manifest.json").exists()
    assert (tmp_path / "DISCLAIMER.txt").exists()

    reloaded = SAPHRDataset.load(tmp_path)
    assert reloaded.num_employees == dataset.num_employees
    assert reloaded.rt.shape == dataset.rt.shape
    assert reloaded.ground_truth() == dataset.ground_truth()

    with pytest.raises(FileNotFoundError):
        SAPHRDataset.load(tmp_path / "does-not-exist")
