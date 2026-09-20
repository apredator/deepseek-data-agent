#!/usr/bin/env python3
"""
Generate the synthetic SAP HCM dataset used by the HR/payroll AI agent.

Usage
-----
    python generate_synthetic_dataset.py                     # defaults from .env
    python generate_synthetic_dataset.py --employees 800 --seed 7
    python generate_synthetic_dataset.py --data-dir data --audit

The script writes the four infotype/result tables, a ``DISCLAIMER.txt`` stating
that the data is synthetic, and ``anomalies_manifest.json`` - the ground truth
of every deliberately injected anomaly, used by the test suite and by the
"found N of M" demonstration.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from config import configure_logging, get_settings
from sap_hr_data import SyntheticDatasetGenerator


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    settings = get_settings()
    parser = argparse.ArgumentParser(
        description="Generate a synthetic SAP HCM HR/payroll dataset with injected anomalies.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--employees", type=int, default=settings.num_employees,
                        help="Number of synthetic employees to generate.")
    parser.add_argument("--periods", type=int, default=settings.num_periods,
                        help="Number of monthly payroll periods.")
    parser.add_argument("--closed-periods", type=int, default=settings.closed_periods,
                        help="How many of the first periods are already closed.")
    parser.add_argument("--seed", type=int, default=settings.seed,
                        help="Random seed (the dataset is fully reproducible).")
    parser.add_argument("--data-dir", type=Path, default=settings.data_dir,
                        help="Output directory for the generated CSV files.")
    parser.add_argument("--no-injection-probe", action="store_true",
                        help="Do not plant the prompt-injection payload in an employee name.")
    parser.add_argument("--quiet", action="store_true", help="Only print the final summary.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    configure_logging()

    generator = SyntheticDatasetGenerator(
        num_employees=args.employees,
        num_periods=args.periods,
        closed_periods=args.closed_periods,
        seed=args.seed,
        injection_probe=not args.no_injection_probe,
    )
    dataset = generator.generate()
    dataset.save(args.data_dir)

    if not args.quiet:
        print("\nSynthetic SAP HCM dataset generated")
        print("=" * 60)
        for name, frame in dataset.tables.items():
            print(f"  {name:22s} {frame.shape[0]:>7d} rows x {frame.shape[1]} columns")
        print(f"\n  Output directory: {args.data_dir}")

    print("\nInjected anomalies (ground truth):")
    for anomaly_type, entries in dataset.manifest["anomalies"].items():
        print(f"  {anomaly_type:32s} {len(entries):>4d}")
    probes = dataset.manifest.get("injection_probes", [])
    if probes:
        print(f"  {'prompt_injection_probe':32s} {len(probes):>4d}")

    print(
        "\nNOTE: this dataset is entirely synthetic/fictional - "
        "no real client, employee or payroll data is involved."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
