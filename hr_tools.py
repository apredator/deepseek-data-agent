"""
Read-only HR/payroll audit tools exposed to the LLM via function calling.

Every tool is a plain Python function over the pandas tables of a
:class:`sap_hr_data.SAPHRDataset` and returns a JSON-serializable ``dict``.

Design rules (specification section 4.2):

* **Read-only.**  No tool mutates the dataset; the guardrail layer enforces this.
* **Deterministic.**  Each check implements a real payroll-audit rule, so the
  agent's answer is grounded in code, not in the model's imagination.
* **Scoreable.**  Result records carry exactly the key columns declared in
  :data:`sap_hr_data.ANOMALY_SPECS`, so findings can be compared against the
  dataset's ground-truth manifest.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import pandas as pd

from config import Settings, get_settings
from guardrails import MAX_RECORDS_PER_TOOL_RESULT
from sap_hr_data import (
    ANOMALY_SPECS,
    BASE_SALARY_WAGE_TYPE,
    INFINITE_DATE,
    NET_PAY_WAGE_TYPE,
    PAYMENT_WAGE_TYPES,
    PROMOTION_ACTION,
    WAGE_TYPES,
    SAPHRDataset,
    coerce_code_columns,
    is_closed_period,
    normalize_period,
    period_from_date,
    period_to_date,
)

logger = logging.getLogger(__name__)

RETRO_FLAG_VALUE = "R"


def _round(value: Any, digits: int = 2) -> float:
    try:
        return round(float(value), digits)
    except (TypeError, ValueError):
        return 0.0


def _period_order(period: str) -> int:
    """Sortable integer key for a ``YYYYPMM`` period string."""
    normalized = normalize_period(period)
    return int(normalized[:4]) * 12 + int(normalized[5:7])


@dataclass
class ToolResult:
    """Uniform envelope returned by every tool."""

    tool: str
    ok: bool
    payload: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return {"tool": self.tool, "ok": self.ok, **self.payload}


class HRToolbox:
    """Container of read-only audit tools bound to one dataset."""

    def __init__(
        self,
        dataset: SAPHRDataset,
        settings: Optional[Settings] = None,
        salary_jump_threshold_pct: Optional[float] = None,
    ) -> None:
        self.dataset = dataset
        self.settings = settings or get_settings()
        # Keep SAP coded fields (Lohnart, Kostl, periods...) as strings so that
        # comparisons never silently fail after a CSV round-trip.
        coerce_code_columns(dataset)
        self.salary_jump_threshold_pct = (
            salary_jump_threshold_pct
            if salary_jump_threshold_pct is not None
            else self.settings.salary_jump_threshold_pct
        )
        self.closed_periods = int(
            dataset.manifest.get("meta", {}).get("closed_periods", self.settings.closed_periods)
        )
        self._org_period_cache: Optional[pd.DataFrame] = None
        self._termination_cache: Optional[pd.DataFrame] = None

    # ------------------------------------------------------------------ #
    # Shared helpers
    # ------------------------------------------------------------------ #
    @property
    def rt(self) -> pd.DataFrame:
        return self.dataset.rt

    @property
    def pa0001(self) -> pd.DataFrame:
        return self.dataset.pa0001

    @property
    def pa0008(self) -> pd.DataFrame:
        return self.dataset.pa0008

    def _periods(self) -> List[str]:
        periods = self.dataset.periods()
        if periods:
            return periods
        return sorted(self.rt["Abrechnungsperiode"].astype(str).unique().tolist())

    def org_assignment_by_period(self) -> pd.DataFrame:
        """
        Time-dependent join of PA0001 onto every payroll period.

        Returns columns ``Pernr``, ``Abrechnungsperiode``, ``Orgeh``, ``org_kostl``.
        """
        if self._org_period_cache is not None:
            return self._org_period_cache

        org = self.pa0001.copy()
        org["Begda"] = pd.to_datetime(org["Begda"], errors="coerce")
        org["Endda"] = pd.to_datetime(org["Endda"], errors="coerce")

        frames: List[pd.DataFrame] = []
        for period in self._periods():
            start = period_to_date(period)
            mask = (org["Begda"] <= start) & (org["Endda"] >= start)
            subset = org.loc[mask, ["Pernr", "Orgeh", "Kostl", "Begda"]].copy()
            if subset.empty:
                continue
            subset = subset.sort_values("Begda").drop_duplicates(subset=["Pernr"], keep="last")
            subset = subset.rename(columns={"Kostl": "org_kostl"})
            subset["Abrechnungsperiode"] = period
            frames.append(subset[["Pernr", "Abrechnungsperiode", "Orgeh", "org_kostl"]])

        result = (
            pd.concat(frames, ignore_index=True)
            if frames
            else pd.DataFrame(columns=["Pernr", "Abrechnungsperiode", "Orgeh", "org_kostl"])
        )
        self._org_period_cache = result
        return result

    def termination_dates(self) -> pd.DataFrame:
        """Return ``Pernr``/``termination_date`` for employees whose PA0001 validity ended."""
        if self._termination_cache is not None:
            return self._termination_cache

        org = self.pa0001.copy()
        org["Begda"] = pd.to_datetime(org["Begda"], errors="coerce")
        org["Endda"] = pd.to_datetime(org["Endda"], errors="coerce")
        # Only the *last* validity record decides whether employment ended; earlier
        # records legitimately end on a transfer / promotion date.
        last_validity = org.sort_values("Begda").drop_duplicates(subset=["Pernr"], keep="last")
        terminated = last_validity[last_validity["Endda"] < INFINITE_DATE]
        if terminated.empty:
            result = pd.DataFrame(columns=["Pernr", "termination_date"])
        else:
            result = (
                terminated[["Pernr", "Endda"]]
                .rename(columns={"Endda": "termination_date"})
                .reset_index(drop=True)
            )
        self._termination_cache = result
        return result

    def _payment_rows(self) -> pd.DataFrame:
        return self.rt[self.rt["Lohnart"].isin(PAYMENT_WAGE_TYPES)]

    def _truncate(self, records: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], bool]:
        if len(records) > MAX_RECORDS_PER_TOOL_RESULT:
            return records[:MAX_RECORDS_PER_TOOL_RESULT], True
        return records, False

    @staticmethod
    def _envelope(
        tool: str,
        records: List[Dict[str, Any]],
        summary: str,
        truncated: bool = False,
        **extra: Any,
    ) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "count": len(records),
            "summary": summary,
            "records": records,
            "truncated": truncated,
        }
        payload.update(extra)
        return {"tool": tool, "ok": True, **payload}

    # ------------------------------------------------------------------ #
    # Tool 1: overview
    # ------------------------------------------------------------------ #
    def get_data_overview(self) -> Dict[str, Any]:
        """Describe the loaded dataset: tables, columns, periods, wage types."""
        periods = self._periods()
        overview = {
            "synthetic": True,
            "num_employees": self.dataset.num_employees,
            "periods": periods,
            "closed_periods": [
                period for period in periods
                if is_closed_period(period, self.closed_periods)
            ],
            "open_periods": [
                period for period in periods
                if not is_closed_period(period, self.closed_periods)
            ],
            "wage_types": WAGE_TYPES,
            "tables": {
                name: {"rows": int(frame.shape[0]), "columns": list(frame.columns)}
                for name, frame in self.dataset.tables.items()
            },
        }
        summary = (
            f"SAP HCM synthetic dataset: {overview['num_employees']} employees, "
            f"{len(periods)} payroll periods ({periods[0] if periods else '?'}.."
            f"{periods[-1] if periods else '?'}), "
            f"{self.closed_periods} closed period(s)."
        )
        return {"tool": "get_data_overview", "ok": True, "count": 1,
                "summary": summary, "records": [overview], "truncated": False}

    # ------------------------------------------------------------------ #
    # Tool 2: employee record
    # ------------------------------------------------------------------ #
    def get_employee_record(self, pernr: Any) -> Dict[str, Any]:
        """Return every stored row for one employee across all four tables."""
        pernr = str(pernr).strip()
        org = self.pa0001[self.pa0001["Pernr"] == pernr]
        pay = self.pa0008[self.pa0008["Pernr"] == pernr]
        recurring = self.dataset.pa0014[self.dataset.pa0014["Pernr"] == pernr]
        results = self.rt[self.rt["Pernr"] == pernr]

        if org.empty and pay.empty and results.empty:
            return {
                "tool": "get_employee_record", "ok": False, "count": 0,
                "summary": f"No record found for Pernr {pernr}.",
                "records": [], "truncated": False, "error": "unknown_pernr",
            }

        def _fmt(frame: pd.DataFrame) -> List[Dict[str, Any]]:
            out = frame.copy()
            for column in ("Begda", "Endda"):
                if column in out.columns:
                    out[column] = pd.to_datetime(out[column], errors="coerce").dt.strftime("%Y-%m-%d")
            return out.replace({float("nan"): None}).to_dict(orient="records")

        term = self.termination_dates()
        term_row = term[term["Pernr"] == pernr]

        records = [{
            "Pernr": pernr,
            "PA0001_org": _fmt(org),
            "PA0008_basic_pay": _fmt(pay),
            "PA0014_recurring": _fmt(recurring),
            "RT_payroll_results": _fmt(results.head(MAX_RECORDS_PER_TOOL_RESULT)),
            "termination_date": (
                pd.to_datetime(term_row.iloc[0]["termination_date"]).strftime("%Y-%m-%d")
                if not term_row.empty else None
            ),
        }]
        summary = (
            f"Pernr {pernr}: {len(org)} org record(s), {len(pay)} pay record(s), "
            f"{len(results)} payroll result row(s)."
        )
        return {"tool": "get_employee_record", "ok": True, "count": len(records),
                "summary": summary, "records": records,
                "truncated": len(results) > MAX_RECORDS_PER_TOOL_RESULT}

    # ------------------------------------------------------------------ #
    # Tool 3: duplicate wage types
    # ------------------------------------------------------------------ #
    def check_duplicate_wage_types(self, period: Optional[Any] = None) -> Dict[str, Any]:
        """Find a wage type paid more than once for the same employee and period."""
        frame = self._payment_rows().copy()
        if period:
            normalized = normalize_period(period)
            frame = frame[frame["Abrechnungsperiode"] == normalized]

        if frame.empty:
            return self._envelope("check_duplicate_wage_types", [],
                                  "No payment rows match the given filter.")

        grouped = (
            frame.groupby(["Pernr", "Abrechnungsperiode", "Lohnart"])
            .agg(occurrences=("Betrag", "size"), total_amount=("Betrag", "sum"))
            .reset_index()
        )
        duplicates = grouped[grouped["occurrences"] > 1].sort_values(
            ["Abrechnungsperiode", "Pernr", "Lohnart"]
        )

        records = [
            {
                "Pernr": row["Pernr"],
                "Abrechnungsperiode": row["Abrechnungsperiode"],
                "Lohnart": row["Lohnart"],
                "wage_type_text": WAGE_TYPES.get(row["Lohnart"], ""),
                "occurrences": int(row["occurrences"]),
                "total_amount": _round(row["total_amount"]),
            }
            for _, row in duplicates.iterrows()
        ]
        records, truncated = self._truncate(records)
        summary = (
            f"Found {len(duplicates)} duplicated wage-type posting(s) across "
            f"{duplicates['Pernr'].nunique() if not duplicates.empty else 0} employee(s)."
        )
        return self._envelope("check_duplicate_wage_types", records, summary, truncated)

    # ------------------------------------------------------------------ #
    # Tool 4: salary jumps without a promotion record
    # ------------------------------------------------------------------ #
    def check_salary_jump_anomalies(self, threshold_pct: Optional[float] = None) -> Dict[str, Any]:
        """Find basic-pay increases above the threshold with no PA0001 promotion entry."""
        threshold = float(threshold_pct) if threshold_pct is not None else self.salary_jump_threshold_pct

        base = self.pa0008[self.pa0008["Lohnart"] == BASE_SALARY_WAGE_TYPE].copy()
        if base.empty:
            return self._envelope("check_salary_jump_anomalies", [],
                                  "No basic-pay (Lohnart 1000) records found.")

        base["Begda"] = pd.to_datetime(base["Begda"], errors="coerce")
        promotions = self.pa0001[self.pa0001["Massn"] == PROMOTION_ACTION].copy()
        promotions["Begda"] = pd.to_datetime(promotions["Begda"], errors="coerce")
        promo_dates: Dict[str, List[pd.Timestamp]] = {}
        for _, row in promotions.iterrows():
            promo_dates.setdefault(row["Pernr"], []).append(row["Begda"])

        records: List[Dict[str, Any]] = []
        for pernr, group in base.sort_values("Begda").groupby("Pernr"):
            rows = group.to_dict(orient="records")
            for previous, current in zip(rows, rows[1:]):
                old_amount = float(previous["Betrag"])
                new_amount = float(current["Betrag"])
                if old_amount <= 0:
                    continue
                increase = (new_amount - old_amount) / old_amount * 100.0
                if increase <= threshold:
                    continue

                approved = any(
                    previous["Begda"] <= promo_date <= current["Begda"]
                    for promo_date in promo_dates.get(pernr, [])
                )
                if approved:
                    continue

                records.append({
                    "Pernr": pernr,
                    "effective_period": period_from_date(current["Begda"]),
                    "effective_date": pd.Timestamp(current["Begda"]).strftime("%Y-%m-%d"),
                    "previous_amount": _round(old_amount),
                    "new_amount": _round(new_amount),
                    "increase_pct": _round(increase),
                    "threshold_pct": _round(threshold),
                    "promotion_record_found": False,
                })

        records.sort(key=lambda item: (-item["increase_pct"], item["Pernr"]))
        records, truncated = self._truncate(records)
        summary = (
            f"Found {len(records)} basic-pay increase(s) above {_round(threshold)}% "
            "without a corresponding PA0001 promotion (Massn=PROMO) record."
        )
        return self._envelope("check_salary_jump_anomalies", records, summary, truncated)

    # ------------------------------------------------------------------ #
    # Tool 5: retro calculation into a closed period without a flag
    # ------------------------------------------------------------------ #
    def check_retro_without_flag(self) -> Dict[str, Any]:
        """Find retro postings for a closed period that carry no retro flag/reason."""
        frame = self.rt.copy()
        frame["ForPeriodNorm"] = frame["ForPeriod"].astype(str).map(
            lambda value: normalize_period(value) if pd.notna(value) else None
        )
        frame["InPeriodNorm"] = frame["Abrechnungsperiode"].astype(str).map(
            lambda value: normalize_period(value) if pd.notna(value) else None
        )
        frame = frame.dropna(subset=["ForPeriodNorm", "InPeriodNorm"])

        is_retro = frame.apply(
            lambda row: _period_order(row["ForPeriodNorm"]) < _period_order(row["InPeriodNorm"]),
            axis=1,
        )
        in_closed = frame["ForPeriodNorm"].map(
            lambda period: is_closed_period(period, self.closed_periods)
        )
        flag_present = frame["Retro"].astype(str).str.strip().str.upper().eq(RETRO_FLAG_VALUE)
        reason_present = (
            frame["RetroReason"].astype(str).str.strip().ne("")
            & frame["RetroReason"].notna()
            & frame["RetroReason"].astype(str).str.lower().ne("nan")
        )

        suspicious = frame[is_retro & in_closed & ~(flag_present & reason_present)]
        grouped = (
            suspicious.groupby(["Pernr", "ForPeriodNorm", "InPeriodNorm"], as_index=False)
            .agg(total_amount=("Betrag", "sum"), rows=("Betrag", "size"))
            .rename(columns={"ForPeriodNorm": "ForPeriod", "InPeriodNorm": "InPeriod"})
            .sort_values(["ForPeriod", "Pernr"])
        )

        records = [
            {
                "Pernr": row["Pernr"],
                "ForPeriod": row["ForPeriod"],
                "InPeriod": row["InPeriod"],
                "retro_flag": "",
                "retro_reason": "",
                "total_amount": _round(row["total_amount"]),
                "affected_rows": int(row["rows"]),
            }
            for _, row in grouped.iterrows()
        ]
        records, truncated = self._truncate(records)
        summary = (
            f"Found {len(records)} retro posting(s) into a closed period "
            "(<= period "
            f"{self.closed_periods:02d}) without a retro flag or reason - "
            f"{grouped['Pernr'].nunique() if not grouped.empty else 0} employee(s)."
        )
        return self._envelope("check_retro_without_flag", records, summary, truncated)

    # ------------------------------------------------------------------ #
    # Tool 6: cost-centre mismatch
    # ------------------------------------------------------------------ #
    def check_cost_center_mismatch(self, period: Optional[Any] = None) -> Dict[str, Any]:
        """Compare Kostl in PA0001 (time-dependent) with Kostl in the payroll results."""
        mapping = self.org_assignment_by_period()
        if mapping.empty:
            return self._envelope("check_cost_center_mismatch", [],
                                  "No organisational assignment records found.")

        frame = self.rt[["Pernr", "Abrechnungsperiode", "Lohnart", "Kostl"]].copy()
        if period:
            frame = frame[frame["Abrechnungsperiode"] == normalize_period(period)]

        merged = frame.merge(
            mapping[["Pernr", "Abrechnungsperiode", "org_kostl"]],
            on=["Pernr", "Abrechnungsperiode"], how="inner",
        )
        mismatches = merged[
            merged["org_kostl"].astype(str) != merged["Kostl"].astype(str)
        ]

        grouped = (
            mismatches.groupby(["Pernr", "Abrechnungsperiode"], as_index=False)
            .agg(
                org_kostl=("org_kostl", "first"),
                payroll_kostl=("Kostl", "first"),
                affected_rows=("Lohnart", "size"),
            )
            .sort_values(["Abrechnungsperiode", "Pernr"])
        )

        records = [
            {
                "Pernr": row["Pernr"],
                "Abrechnungsperiode": row["Abrechnungsperiode"],
                "org_kostl": row["org_kostl"],
                "payroll_kostl": row["payroll_kostl"],
                "affected_rows": int(row["affected_rows"]),
            }
            for _, row in grouped.iterrows()
        ]
        records, truncated = self._truncate(records)
        summary = (
            f"Found {len(records)} employee/period combination(s) where the payroll "
            "cost centre differs from the organisational assignment (PA0001.Kostl)."
        )
        return self._envelope("check_cost_center_mismatch", records, summary, truncated)

    # ------------------------------------------------------------------ #
    # Tool 7: post-termination payments
    # ------------------------------------------------------------------ #
    def check_post_termination_payment(self) -> Dict[str, Any]:
        """Find payroll postings that fall after the employee's PA0001 termination date."""
        terminations = self.termination_dates()
        if terminations.empty:
            return self._envelope("check_post_termination_payment", [],
                                  "No terminated employees (closed PA0001 validity) found.")

        frame = self.rt[["Pernr", "Abrechnungsperiode", "Lohnart", "Betrag"]].copy()
        frame["period_start"] = frame["Abrechnungsperiode"].astype(str).map(period_to_date)

        merged = frame.merge(terminations, on="Pernr", how="inner")
        after = merged[merged["period_start"] > pd.to_datetime(merged["termination_date"])]

        grouped = (
            after.groupby(["Pernr", "Abrechnungsperiode"], as_index=False)
            .agg(
                termination_date=("termination_date", "first"),
                total_amount=("Betrag", "sum"),
                affected_rows=("Lohnart", "size"),
            )
            .sort_values(["Abrechnungsperiode", "Pernr"])
        )

        records = []
        for _, row in grouped.iterrows():
            records.append({
                "Pernr": row["Pernr"],
                "Abrechnungsperiode": row["Abrechnungsperiode"],
                "termination_date": pd.Timestamp(row["termination_date"]).strftime("%Y-%m-%d"),
                "total_amount": _round(row["total_amount"]),
                "affected_rows": int(row["affected_rows"]),
            })
        records, truncated = self._truncate(records)
        summary = (
            f"Found {len(records)} payroll posting(s) after the termination date "
            f"for {grouped['Pernr'].nunique() if not grouped.empty else 0} employee(s)."
        )
        return self._envelope("check_post_termination_payment", records, summary, truncated)

    # ------------------------------------------------------------------ #
    # Tool 8: negative net pay
    # ------------------------------------------------------------------ #
    def check_negative_net_pay(self, period: Optional[Any] = None) -> Dict[str, Any]:
        """Find periods where the net pay (Lohnart /560) is negative."""
        frame = self.rt[self.rt["Lohnart"] == NET_PAY_WAGE_TYPE].copy()
        if period:
            frame = frame[frame["Abrechnungsperiode"] == normalize_period(period)]
        negative = frame[frame["Betrag"] < 0].sort_values(["Abrechnungsperiode", "Pernr"])

        records = [
            {
                "Pernr": row["Pernr"],
                "Abrechnungsperiode": row["Abrechnungsperiode"],
                "net_pay": _round(row["Betrag"]),
                "wage_type_text": WAGE_TYPES.get(NET_PAY_WAGE_TYPE, ""),
            }
            for _, row in negative.iterrows()
        ]
        records, truncated = self._truncate(records)
        summary = (
            f"Found {len(records)} payroll period(s) with a negative net pay "
            f"(Lohnart {NET_PAY_WAGE_TYPE})."
        )
        return self._envelope("check_negative_net_pay", records, summary, truncated)

    # ------------------------------------------------------------------ #
    # Tool 9: aggregation by org unit
    # ------------------------------------------------------------------ #
    def aggregate_by_org_unit(
        self,
        metric: str = "gross_pay",
        agg: str = "sum",
        period: Optional[Any] = None,
    ) -> Dict[str, Any]:
        """Aggregate a payroll metric by organisational unit (Orgeh)."""
        metric_key = str(metric or "gross_pay").strip().lower()
        agg_key = str(agg or "sum").strip().lower()
        allowed_aggs = {"sum", "mean", "count", "min", "max", "median"}
        if agg_key not in allowed_aggs:
            return {
                "tool": "aggregate_by_org_unit", "ok": False, "count": 0,
                "summary": f"Unsupported aggregation '{agg}'. Use one of: {sorted(allowed_aggs)}",
                "records": [], "truncated": False, "error": "bad_aggregation",
            }

        mapping = self.org_assignment_by_period()
        frame = self.rt.copy()
        if period:
            frame = frame[frame["Abrechnungsperiode"] == normalize_period(period)]

        if metric_key in {"gross_pay", "gross", "earnings"}:
            metric_label = "gross_pay"
            frame = frame[frame["Lohnart"].isin(PAYMENT_WAGE_TYPES)]
            value_column = "Betrag"
        elif metric_key in {"net_pay", "net", NET_PAY_WAGE_TYPE}:
            metric_label = "net_pay"
            frame = frame[frame["Lohnart"] == NET_PAY_WAGE_TYPE]
            value_column = "Betrag"
        elif metric_key in {"headcount", "employees", "count"}:
            metric_label = "headcount"
            value_column = "Pernr"
        else:
            return {
                "tool": "aggregate_by_org_unit", "ok": False, "count": 0,
                "summary": (
                    f"Unsupported metric '{metric}'. Use 'gross_pay', 'net_pay' or 'headcount'."
                ),
                "records": [], "truncated": False, "error": "bad_metric",
            }

        merged = frame.merge(
            mapping[["Pernr", "Abrechnungsperiode", "Orgeh"]],
            on=["Pernr", "Abrechnungsperiode"], how="inner",
        )
        if merged.empty:
            return self._envelope("aggregate_by_org_unit", [],
                                  "No payroll rows matched the given filter.")

        if metric_label == "headcount":
            grouped = (
                merged.groupby("Orgeh", as_index=False)["Pernr"].nunique()
                .rename(columns={"Pernr": "value"})
            )
        else:
            grouped = (
                merged.groupby("Orgeh", as_index=False)[value_column].agg(agg_key)
                .rename(columns={value_column: "value"})
            )

        employees = (
            merged.groupby("Orgeh", as_index=False)["Pernr"].nunique()
            .rename(columns={"Pernr": "employees"})
        )
        grouped = grouped.merge(employees, on="Orgeh", how="left")
        grouped = grouped.sort_values("value", ascending=False)

        records = [
            {
                "Orgeh": row["Orgeh"],
                "employees": int(row["employees"]),
                "metric": metric_label,
                "aggregation": agg_key,
                "value": _round(row["value"]),
                "currency": "EUR",
            }
            for _, row in grouped.iterrows()
        ]
        records, truncated = self._truncate(records)
        summary = (
            f"{metric_label} ({agg_key}) by org unit over "
            f"{'all periods' if not period else normalize_period(period)}: "
            f"{len(records)} org unit(s)."
        )
        return self._envelope("aggregate_by_org_unit", records, summary, truncated,
                              metric=metric_label, aggregation=agg_key)

    # ------------------------------------------------------------------ #
    # Deterministic full audit + scorecard (used by CLI --audit and the demo)
    # ------------------------------------------------------------------ #
    def run_all_checks(self) -> Dict[str, Dict[str, Any]]:
        """Run every anomaly check and return results keyed by anomaly type."""
        return {
            "duplicate_wage_types": self.check_duplicate_wage_types(),
            "salary_jump_without_promotion": self.check_salary_jump_anomalies(),
            "retro_without_flag": self.check_retro_without_flag(),
            "cost_center_mismatch": self.check_cost_center_mismatch(),
            "negative_net_pay": self.check_negative_net_pay(),
            "post_termination_payment": self.check_post_termination_payment(),
        }

    @staticmethod
    def result_keys(anomaly_type: str, result: Dict[str, Any]) -> set:
        """Extract the comparable key tuples from a tool result."""
        _, key_columns = ANOMALY_SPECS[anomaly_type]
        keys = set()
        for record in result.get("records", []):
            keys.add(tuple(record.get(column) for column in key_columns))
        return keys

    def audit(self) -> Dict[str, Any]:
        """Deterministic scan of all anomaly checks plus a ground-truth scorecard."""
        per_type: Dict[str, Any] = {}
        found_total = 0
        expected_total = 0

        for anomaly_type, (tool_name, _) in ANOMALY_SPECS.items():
            result = getattr(self, tool_name)()
            found = self.result_keys(anomaly_type, result)
            expected = self.dataset.ground_truth().get(anomaly_type, set())
            true_positive = found & expected
            false_positive = found - expected
            missed = expected - found
            found_total += len(true_positive)
            expected_total += len(expected)
            per_type[anomaly_type] = {
                "tool": tool_name,
                "found": len(found),
                "expected": len(expected),
                "true_positives": sorted(true_positive),
                "false_positives": sorted(false_positive),
                "missed": sorted(missed),
                "summary": result.get("summary", ""),
            }

        return {
            "expected_total": expected_total,
            "true_positives_total": found_total,
            "missed_total": sum(len(v["missed"]) for v in per_type.values()),
            "false_positives_total": sum(len(v["false_positives"]) for v in per_type.values()),
            "detection_rate": round(found_total / expected_total, 3) if expected_total else 0.0,
            "per_type": per_type,
        }


# --------------------------------------------------------------------------- #
# Function-calling schemas
# --------------------------------------------------------------------------- #

TOOL_SPECS: List[Dict[str, Any]] = [
    {
        "name": "get_data_overview",
        "description": (
            "Describe the loaded SAP HCM dataset: number of employees, available payroll "
            "periods, which periods are closed for corrections, the wage type (Lohnart) "
            "catalogue and the table schemas. Call this first if you need orientation."
        ),
        "parameters": {"type": "object", "properties": {}, "required": []},
    },
    {
        "name": "get_employee_record",
        "description": (
            "Fetch the full stored record of one employee (Pernr): organisational "
            "assignment history (PA0001), basic pay history (PA0008), recurring payments "
            "(PA0014) and payroll results (RT). Use it to inspect a specific case."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "pernr": {"type": "string", "description": "Personnel number, e.g. '10000042'."},
            },
            "required": ["pernr"],
        },
    },
    {
        "name": "check_duplicate_wage_types",
        "description": (
            "Detect the same wage type (Lohnart) being posted more than once for the same "
            "employee in the same payroll period - a classic duplicate-payment control."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "period": {
                    "type": "string",
                    "description": "Optional payroll period filter, e.g. '2024P10'. Omit to check all periods.",
                },
            },
            "required": [],
        },
    },
    {
        "name": "check_salary_jump_anomalies",
        "description": (
            "Find basic-pay (Lohnart 1000) increases above a percentage threshold that have "
            "no matching promotion entry (PA0001 Massn='PROMO') in the organisational "
            "assignment. Detects unauthorised or erroneous salary changes."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "threshold_pct": {
                    "type": "number",
                    "description": "Increase threshold in percent. Defaults to the configured value (40).",
                },
            },
            "required": [],
        },
    },
    {
        "name": "check_retro_without_flag",
        "description": (
            "Find retro-calculations (ForPeriod earlier than InPeriod) booked into an "
            "already closed period without a retro flag or a documented reason."
        ),
        "parameters": {"type": "object", "properties": {}, "required": []},
    },
    {
        "name": "check_cost_center_mismatch",
        "description": (
            "Compare the cost centre (Kostl) in the time-dependent organisational "
            "assignment PA0001 with the cost centre stored on the payroll results (RT) "
            "for the same period. Reports posting/cost-allocation discrepancies."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "period": {"type": "string", "description": "Optional period filter, e.g. '2024P08'."},
            },
            "required": [],
        },
    },
    {
        "name": "check_post_termination_payment",
        "description": (
            "Find payroll postings in periods after the employee's termination date "
            "(PA0001 Endda) - payments to people who already left."
        ),
        "parameters": {"type": "object", "properties": {}, "required": []},
    },
    {
        "name": "check_negative_net_pay",
        "description": (
            "Find payroll periods where net pay (Lohnart /560) is negative, which usually "
            "indicates a recovery/offset error or a master-data problem."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "period": {"type": "string", "description": "Optional period filter, e.g. '2024P10'."},
            },
            "required": [],
        },
    },
    {
        "name": "aggregate_by_org_unit",
        "description": (
            "Aggregate a payroll metric by organisational unit (Orgeh): gross pay, net pay "
            "or headcount. Useful for summaries and cost overviews."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "metric": {
                    "type": "string",
                    "enum": ["gross_pay", "net_pay", "headcount"],
                    "description": "Metric to aggregate.",
                },
                "agg": {
                    "type": "string",
                    "enum": ["sum", "mean", "count", "min", "max", "median"],
                    "description": "Aggregation function, default 'sum'.",
                },
                "period": {"type": "string", "description": "Optional period filter, e.g. '2024P10'."},
            },
            "required": ["metric"],
        },
    },
]


def build_deepseek_tools(specs: Optional[Sequence[Dict[str, Any]]] = None) -> List[Dict[str, Any]]:
    """Wrap tool specs in the OpenAI/DeepSeek ``tools`` request format."""
    specs = specs if specs is not None else TOOL_SPECS
    return [
        {
            "type": "function",
            "function": {
                "name": spec["name"],
                "description": spec["description"],
                "parameters": spec["parameters"],
            },
        }
        for spec in specs
    ]


class ToolRegistry:
    """Maps tool names to bound toolbox methods and validates/coerces arguments."""

    def __init__(self, toolbox: HRToolbox, specs: Optional[Sequence[Dict[str, Any]]] = None) -> None:
        self.toolbox = toolbox
        self.specs: List[Dict[str, Any]] = list(specs if specs is not None else TOOL_SPECS)
        self._functions: Dict[str, Callable[..., Dict[str, Any]]] = {}
        self._param_types: Dict[str, Dict[str, str]] = {}
        for spec in self.specs:
            name = spec["name"]
            function = getattr(toolbox, name, None)
            if function is None or not callable(function):
                raise AttributeError(f"Toolbox has no tool implementation for '{name}'")
            self._functions[name] = function
            properties = spec.get("parameters", {}).get("properties", {})
            self._param_types[name] = {
                key: value.get("type", "string") for key, value in properties.items()
            }

    # -- introspection ----------------------------------------------------
    @property
    def names(self) -> List[str]:
        return list(self._functions)

    @property
    def read_only(self) -> bool:
        return True

    def schemas(self) -> List[Dict[str, Any]]:
        return build_deepseek_tools(self.specs)

    def describe(self) -> List[Dict[str, Any]]:
        return [{"name": spec["name"], "description": spec["description"]} for spec in self.specs]

    def register(self, spec: Dict[str, Any], function: Callable[..., Dict[str, Any]]) -> None:
        """Add an extra (still read-only) tool, e.g. a semantic-similarity helper."""
        name = spec["name"]
        if name in self._functions:
            raise ValueError(f"Tool '{name}' is already registered")
        self.specs.append(spec)
        self._functions[name] = function
        properties = spec.get("parameters", {}).get("properties", {})
        self._param_types[name] = {
            key: value.get("type", "string") for key, value in properties.items()
        }

    # -- execution --------------------------------------------------------
    def _coerce(self, name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        types = self._param_types.get(name, {})
        coerced: Dict[str, Any] = {}
        for key, value in (arguments or {}).items():
            expected = types.get(key)
            if value is None:
                continue
            if expected == "number" and not isinstance(value, (int, float)):
                try:
                    value = float(str(value).replace(",", "."))
                except ValueError:
                    pass
            elif expected == "integer" and not isinstance(value, int):
                try:
                    value = int(float(str(value)))
                except ValueError:
                    pass
            elif expected == "string" and not isinstance(value, str):
                value = str(value)
            coerced[key] = value
        return coerced

    def call(self, name: str, arguments: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Execute a tool by name. Raises ``KeyError`` for unknown tools."""
        if name not in self._functions:
            raise KeyError(f"Unknown tool '{name}'. Available: {', '.join(self.names)}")
        arguments = self._coerce(name, arguments or {})
        try:
            return self._functions[name](**arguments)
        except TypeError as error:  # wrong arguments from the model
            logger.warning("Tool %s called with bad arguments %s: %s", name, arguments, error)
            return {
                "tool": name, "ok": False, "count": 0, "records": [], "truncated": False,
                "summary": f"Tool '{name}' could not be executed with arguments {arguments}: {error}",
                "error": "bad_arguments",
            }
        except Exception as error:  # pragma: no cover - defensive
            logger.exception("Tool %s failed", name)
            return {
                "tool": name, "ok": False, "count": 0, "records": [], "truncated": False,
                "summary": f"Tool '{name}' failed: {error}",
                "error": "tool_failure",
            }


__all__ = [
    "HRToolbox",
    "TOOL_SPECS",
    "ToolRegistry",
    "ToolResult",
    "build_deepseek_tools",
]
