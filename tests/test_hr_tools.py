"""
Tests for every read-only tool in :mod:`hr_tools`.

Each check is validated against the dataset's ground-truth manifest, so a test
passes only if the tool finds *exactly* the injected anomalies - no misses and no
false positives (specification section 6 and Definition of Done).
"""

from __future__ import annotations

import pandas as pd
import pytest

from hr_tools import HRToolbox, TOOL_SPECS, ToolRegistry
from sap_hr_data import ANOMALY_SPECS, PAYMENT_WAGE_TYPES, PROMOTION_ACTION, SAPHRDataset


def _keys(anomaly_type: str, result: dict) -> set:
    return HRToolbox.result_keys(anomaly_type, result)


# --------------------------------------------------------------------------- #
# Registry / schemas
# --------------------------------------------------------------------------- #

def test_registry_exposes_every_documented_tool(toolbox: HRToolbox) -> None:
    registry = ToolRegistry(toolbox)
    assert set(registry.names) == {spec["name"] for spec in TOOL_SPECS}
    assert registry.read_only is True
    schemas = registry.schemas()
    assert all(schema["type"] == "function" for schema in schemas)
    assert all("parameters" in schema["function"] for schema in schemas)


def test_every_anomaly_type_has_a_registered_check(toolbox: HRToolbox) -> None:
    registry = ToolRegistry(toolbox)
    for _, (tool_name, _) in ANOMALY_SPECS.items():
        assert tool_name in registry.names


def test_registry_coerces_string_arguments(toolbox: HRToolbox) -> None:
    registry = ToolRegistry(toolbox)
    result = registry.call("check_salary_jump_anomalies", {"threshold_pct": "10"})
    assert result["ok"] is True


def test_registry_rejects_unknown_tool(toolbox: HRToolbox) -> None:
    with pytest.raises(KeyError):
        ToolRegistry(toolbox).call("drop_table", {})


def test_registry_reports_bad_arguments(toolbox: HRToolbox) -> None:
    result = ToolRegistry(toolbox).call("get_employee_record", {"unknown_arg": 1})
    assert result["ok"] is False
    assert result["error"] == "bad_arguments"


# --------------------------------------------------------------------------- #
# Individual tools
# --------------------------------------------------------------------------- #

def test_get_data_overview(toolbox: HRToolbox, dataset: SAPHRDataset) -> None:
    result = toolbox.get_data_overview()
    overview = result["records"][0]
    assert result["ok"] is True
    assert overview["num_employees"] == dataset.num_employees
    assert overview["periods"] == dataset.periods()
    assert set(overview["closed_periods"]).isdisjoint(overview["open_periods"])
    assert overview["synthetic"] is True


def test_get_employee_record(toolbox: HRToolbox, dataset: SAPHRDataset) -> None:
    pernr = dataset.pa0001.iloc[0]["Pernr"]
    result = toolbox.get_employee_record(pernr)
    assert result["ok"] is True
    record = result["records"][0]
    assert record["Pernr"] == pernr
    assert record["RT_payroll_results"], "payroll results should be returned"


def test_get_employee_record_unknown_pernr(toolbox: HRToolbox) -> None:
    result = toolbox.get_employee_record("99999999")
    assert result["ok"] is False
    assert result["error"] == "unknown_pernr"


def test_check_duplicate_wage_types(toolbox: HRToolbox, dataset: SAPHRDataset) -> None:
    result = toolbox.check_duplicate_wage_types()
    assert _keys("duplicate_wage_types", result) == dataset.ground_truth()["duplicate_wage_types"]
    assert all(record["occurrences"] > 1 for record in result["records"])
    assert all(record["Lohnart"] in PAYMENT_WAGE_TYPES for record in result["records"])


def test_check_duplicate_wage_types_period_filter(toolbox: HRToolbox, dataset: SAPHRDataset) -> None:
    expected = dataset.ground_truth()["duplicate_wage_types"]
    period = sorted({key[1] for key in expected})[0]
    result = toolbox.check_duplicate_wage_types(period=period)
    assert _keys("duplicate_wage_types", result) == {key for key in expected if key[1] == period}

    # format-insensitive: "2024-10" must behave like "2024P10"
    alt = toolbox.check_duplicate_wage_types(period=period.replace("P", "-"))
    assert _keys("duplicate_wage_types", alt) == _keys("duplicate_wage_types", result)


def test_check_salary_jump_anomalies(toolbox: HRToolbox, dataset: SAPHRDataset) -> None:
    result = toolbox.check_salary_jump_anomalies()
    assert _keys("salary_jump_without_promotion", result) == \
        dataset.ground_truth()["salary_jump_without_promotion"]
    assert all(record["increase_pct"] > 40 for record in result["records"])
    assert all(record["promotion_record_found"] is False for record in result["records"])


def test_salary_jump_respects_custom_threshold(toolbox: HRToolbox, dataset: SAPHRDataset) -> None:
    """A very high threshold must silence the check (no hard-coded 40%)."""
    result = toolbox.check_salary_jump_anomalies(threshold_pct=500)
    assert result["count"] == 0


def test_salary_jump_ignores_raises_with_promotion_record(
    toolbox: HRToolbox, dataset: SAPHRDataset
) -> None:
    """Employees whose big raise has a PA0001 PROMO entry must NOT be reported."""
    flagged = {key[0] for key in _keys("salary_jump_without_promotion", toolbox.check_salary_jump_anomalies())}
    promotions = dataset.pa0001[dataset.pa0001["Massn"] == PROMOTION_ACTION]
    promoted_pernrs = set(promotions["Pernr"])
    assert flagged.isdisjoint(promoted_pernrs), "a documented promotion was treated as an anomaly"


def test_check_retro_without_flag(toolbox: HRToolbox, dataset: SAPHRDataset) -> None:
    result = toolbox.check_retro_without_flag()
    assert _keys("retro_without_flag", result) == dataset.ground_truth()["retro_without_flag"]
    assert all(record["ForPeriod"] < record["InPeriod"] for record in result["records"])


def test_retro_check_ignores_flagged_retro(toolbox: HRToolbox, dataset: SAPHRDataset) -> None:
    """A retro posting that carries flag+reason must not be reported."""
    flagged = _keys("retro_without_flag", toolbox.check_retro_without_flag())
    legit = dataset.rt[dataset.rt["Retro"].astype(str).str.upper() == "R"]
    legit_keys = set(zip(legit["Pernr"], legit["ForPeriod"]))
    assert legit_keys, "the generator should also produce legitimate retro postings"
    assert flagged.isdisjoint(legit_keys)


def test_check_cost_center_mismatch(toolbox: HRToolbox, dataset: SAPHRDataset) -> None:
    result = toolbox.check_cost_center_mismatch()
    assert _keys("cost_center_mismatch", result) == dataset.ground_truth()["cost_center_mismatch"]
    assert all(record["org_kostl"] != record["payroll_kostl"] for record in result["records"])


def test_cost_center_check_matches_correct_assignments(toolbox: HRToolbox, dataset: SAPHRDataset) -> None:
    """Employees with a time-dependent Kostl transfer must not be flagged by mistake."""
    mapping = toolbox.org_assignment_by_period()
    assert not mapping.empty

    flagged = {key[0] for key in _keys("cost_center_mismatch", toolbox.check_cost_center_mismatch())}
    injected = {key[0] for key in dataset.ground_truth()["cost_center_mismatch"]}
    transfers = set(toolbox.pa0001[toolbox.pa0001["Massn"] == "TRANS"]["Pernr"])
    assert transfers, "the generator should create Kostl transfers"
    assert (flagged & transfers) <= injected


def test_check_post_termination_payment(toolbox: HRToolbox, dataset: SAPHRDataset) -> None:
    result = toolbox.check_post_termination_payment()
    assert _keys("post_termination_payment", result) == \
        dataset.ground_truth()["post_termination_payment"]
    for record in result["records"]:
        period = record["Abrechnungsperiode"]
        period_start = pd.Timestamp(f"{period[:4]}-{period[5:7]}-01")
        assert period_start > pd.Timestamp(record["termination_date"])


def test_termination_dates_ignore_transfer_validity_gaps(toolbox: HRToolbox) -> None:
    """A closed PA0001 row caused by a transfer is not a termination."""
    terminations = set(toolbox.termination_dates()["Pernr"])
    last_valid = toolbox.pa0001.sort_values("Begda").drop_duplicates("Pernr", keep="last")
    open_ended = set(last_valid[last_valid["Endda"] >= pd.Timestamp("2199-01-01")]["Pernr"])
    assert terminations.isdisjoint(open_ended)
    assert set(toolbox.pa0001[toolbox.pa0001["Massn"] == "TRANS"]["Pernr"])


def test_check_negative_net_pay(toolbox: HRToolbox, dataset: SAPHRDataset) -> None:
    result = toolbox.check_negative_net_pay()
    assert _keys("negative_net_pay", result) == dataset.ground_truth()["negative_net_pay"]
    assert all(record["net_pay"] < 0 for record in result["records"])


def test_aggregate_by_org_unit_gross_pay(toolbox: HRToolbox) -> None:
    result = toolbox.aggregate_by_org_unit(metric="gross_pay", agg="sum")
    assert result["ok"] is True
    assert result["records"], "org-unit aggregation must return rows"
    assert all(record["value"] > 0 for record in result["records"])
    assert all(record["metric"] == "gross_pay" for record in result["records"])


def test_aggregate_by_org_unit_headcount(toolbox: HRToolbox, dataset: SAPHRDataset) -> None:
    result = toolbox.aggregate_by_org_unit(metric="headcount")
    total = sum(record["employees"] for record in result["records"])
    assert total == dataset.num_employees


def test_aggregate_by_org_unit_rejects_bad_input(toolbox: HRToolbox) -> None:
    assert toolbox.aggregate_by_org_unit(metric="nonsense")["ok"] is False
    assert toolbox.aggregate_by_org_unit(metric="gross_pay", agg="nonsense")["ok"] is False


# --------------------------------------------------------------------------- #
# Whole-suite guarantees
# --------------------------------------------------------------------------- #

def test_audit_finds_all_injected_anomalies(toolbox: HRToolbox) -> None:
    """Definition of Done: 100% of the injected anomalies, zero false positives."""
    report = toolbox.audit()
    assert report["missed_total"] == 0, report["per_type"]
    assert report["false_positives_total"] == 0, report["per_type"]
    assert report["detection_rate"] == 1.0
    assert report["true_positives_total"] == report["expected_total"] > 0


def test_run_all_checks_returns_every_type(toolbox: HRToolbox) -> None:
    results = toolbox.run_all_checks()
    assert set(results) == set(ANOMALY_SPECS)
    assert all(result["ok"] for result in results.values())


def test_audit_survives_a_csv_roundtrip(dataset: SAPHRDataset, tmp_path) -> None:
    """Regression test: coded fields must stay strings after save+load."""
    dataset.save(tmp_path)
    reloaded = SAPHRDataset.load(tmp_path)

    assert reloaded.pa0008["Lohnart"].map(type).eq(str).all()
    assert reloaded.rt["Lohnart"].map(type).eq(str).all()

    report = HRToolbox(reloaded).audit()
    assert report["detection_rate"] == 1.0, report["per_type"]
    assert report["missed_total"] == 0
