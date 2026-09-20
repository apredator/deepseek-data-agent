#!/usr/bin/env python3
"""
Optional: mirror the synthetic SAP HCM dataset to the Hugging Face Datasets Hub.

This is the "публикация датасета" role from the specification (section 5.1): a
public showcase of the synthetic data, discoverable on the Hub independently of
GitHub.  It is entirely optional - nothing in the agent depends on it.

Usage
-----
    export HF_TOKEN=hf_...
    export HF_DATASET_REPO=your-user/synthetic-sap-hcm-payroll
    python publish_to_hf.py --data-dir data

Without ``HF_TOKEN``/``HF_DATASET_REPO`` the script prints exactly which
variables are missing and exits with code 1 - it never uploads anything by
accident.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from config import configure_logging, get_settings
from sap_hr_data import SYNTHETIC_DISCLAIMER, generate_dataset, load_or_generate

DATASET_CARD = """\
---
license: mit
task_categories:
  - tabular-classification
  - text-classification
language:
  - en
tags:
  - sap
  - hcm
  - payroll
  - synthetic
  - hr-analytics
  - anomaly-detection
pretty_name: Synthetic SAP HCM HR/Payroll Dataset (with injected anomalies)
size_categories:
  - 1K<n<10K
---

# Synthetic SAP HCM HR/Payroll Dataset

**This dataset is entirely synthetic and fictional.** It contains no client,
employee or payroll data of any kind. It exists to demonstrate an AI agent that
audits HR/payroll data from SAP HCM.

## Files

| File | Content |
|------|---------|
| `pa0001_org.csv` | Organisational assignment: Pernr, Orgeh, Kostl, Planstelle, Begda, Endda |
| `pa0008_basic_pay.csv` | Basic pay: Pernr, Lohnart, Betrag, Waehrung, Begda, Endda |
| `pa0014_recurring.csv` | Recurring payments: Pernr, Lohnart, Betrag, Periodizitaet |
| `rt_payroll_results.csv` | Payroll results: Pernr, Abrechnungsperiode, Lohnart, Betrag, InPeriod, ForPeriod, Retro, Kostl |
| `anomalies_manifest.json` | Ground truth: the exact location of every deliberately injected anomaly |

## Deliberately injected anomalies

1. Duplicate wage type (Lohnart) in the same period
2. Basic-pay jump > 40% without a promotion record (PA0001 Massn != PROMO)
3. Retro-calculation into a closed period without a retro flag/reason
4. Cost centre (Kostl) mismatch between PA0001 and the payroll results
5. Negative net pay
6. Payment after the termination date

`anomalies_manifest.json` lets you score any detection approach objectively.

## Usage (pandas)

```python
import pandas as pd
rt = pd.read_csv("rt_payroll_results.csv", dtype={"Pernr": str, "Lohnart": str})
```

Keep `Pernr` and `Lohnart` as strings - they are SAP coded fields.

## Related project

Agent that audits this data: see the repository linked from the dataset card on
the Hugging Face Hub.
"""


def main(argv: list[str] | None = None) -> int:
    settings = get_settings()
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", type=Path, default=settings.data_dir)
    parser.add_argument("--repo-id", default=settings.hf_dataset_repo,
                        help="Target dataset repo, e.g. 'user/synthetic-sap-hcm-payroll'.")
    parser.add_argument("--token", default=settings.hf_api_token)
    parser.add_argument("--private", action="store_true", help="Create the repository as private.")
    parser.add_argument("--regenerate", action="store_true")
    args = parser.parse_args(argv)

    configure_logging()

    if not args.token:
        print("HF_TOKEN is not set - nothing to do.\n"
              "  1. create a write token at https://huggingface.co/settings/tokens\n"
              "  2. export HF_TOKEN=hf_...\n"
              "  3. export HF_DATASET_REPO=<user>/<dataset-name>", file=sys.stderr)
        return 1
    if not args.repo_id:
        print("HF_DATASET_REPO (or --repo-id) is not set - refusing to guess a destination.",
              file=sys.stderr)
        return 1

    try:
        from huggingface_hub import HfApi, create_repo
    except ImportError:
        print("huggingface_hub is not installed: pip install -r requirements-optional.txt",
              file=sys.stderr)
        return 1

    if args.regenerate:
        dataset = generate_dataset(settings=settings)
    else:
        dataset = load_or_generate(args.data_dir, settings=settings)
    data_dir = dataset.save(args.data_dir)

    (data_dir / "README.md").write_text(DATASET_CARD, encoding="utf-8")
    (data_dir / "DISCLAIMER.txt").write_text(SYNTHETIC_DISCLAIMER + "\n", encoding="utf-8")

    api = HfApi(token=args.token)
    create_repo(args.repo_id, repo_type="dataset", private=args.private, exist_ok=True, token=args.token)
    api.upload_folder(
        repo_id=args.repo_id,
        repo_type="dataset",
        folder_path=str(data_dir),
        commit_message="Publish synthetic SAP HCM payroll dataset (with anomaly ground truth)",
    )
    print(f"✅ Dataset published: https://huggingface.co/datasets/{args.repo_id}")
    print("   Remember to note in the card that the data is 100% synthetic.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
