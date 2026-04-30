# Data — access policy

## Confidential PMA dataset

The Palestinian Monetary Authority (PMA) supervisory dataset used in the
manuscript (12,952 records, 2023–2025) is **confidential**. It cannot be
redistributed under the contractual confidentiality agreement that governs
its use, and is not contained in this repository.

## What ships with this repository

| Resource | Purpose |
| --- | --- |
| [`synthetic/generate_pma_synthetic.py`](synthetic/generate_pma_synthetic.py) | Generates a schema-preserving synthetic dataset (no real account information) with the same column types and marginal distributions as the preprocessed PMA data. |
| `experiments/05_synthetic_pma_demo.py` | Runs the CFAM pipeline end-to-end on the synthetic data so reviewers can verify the pipeline mechanics. |
| `experiments/01_german_credit_baseline.py` | Independently reproducible baseline on the public German Credit (UCI / OpenML `credit-g`) dataset. |
| `experiments/02_german_credit_reweigh.py` | Reweighing comparator (Kamiran-Calders) on the same public benchmark. |
| `experiments/03_shap_stability.py` | SHAP rank-stability analysis across 10 seeds. |
| `experiments/04_ledger_run.py` | End-to-end CFAM ledger run on German Credit (R-EXP packets, integrity, TTA). |

## Why not just release the PMA data with synthetic generators?

Because supervisory data carry a non-trivial re-identification risk even after
de-identification, the contractual agreement between the corresponding author
and PMA explicitly prohibits redistribution. The repository therefore relies on
**(i)** synthetic data with the same schema for pipeline-mechanics
reproducibility, and **(ii)** the public German Credit benchmark for
externally verifiable quantitative claims.
