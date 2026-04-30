# CFAM — Causal Fairness Auditing & Monitoring Pipeline

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Reproducible](https://img.shields.io/badge/reproducible-German%20Credit-brightgreen.svg)](#reproducing-the-german-credit-results)

Reference implementation accompanying the paper:

> **Salaydi, S., & Çeliktaş, B.** *The CFAM Pipeline: A Causal Framework for Continuous Fairness Auditing and Tamper-Evident Explanation in FinTech Credit Scoring.* SN Computer Science (under review), 2026.

CFAM is a governance-oriented pipeline that integrates four capabilities into a single supervisory workflow:

1. **Causal fairness diagnostics** — Structural Causal Models (SCMs), path-specific effects (PSE), and counterfactual fairness rate (CFR);
2. **Regulator-ready explanations** — standardised **R-EXP** packets built on SHAP attributions;
3. **Tamper-evident audit logging** — a permissioned **hash-chained ledger simulation**;
4. **Time-to-Audit (TTA) metrics** — D2A, R2C, and MTTA for quantifying supervisory responsiveness.

> **Scope note.** The ledger is a hash-chained simulation, not a production blockchain. All security claims are limited to *evidentiary integrity* and *tamper evidence*; decentralised consensus, Byzantine fault tolerance, and adversarial threat modelling are out of scope (see §V-D in the manuscript).

---

## Repository structure

```
cfam-pipeline/
├── README.md                        # this file
├── LICENSE                          # MIT
├── CITATION.cff                     # how to cite
├── requirements.txt                 # pinned dependencies
├── requirements-dev.txt             # extras: pytest, ruff, mypy
├── .gitignore
├── pyproject.toml                   # build metadata
│
├── cfam_core/                       # library code
│   ├── __init__.py
│   ├── r_exp.py                     # RegulatorExplanationPacket dataclass
│   ├── ledger.py                    # HashChainedLedger
│   ├── tta.py                       # TimeToAuditMetrics
│   ├── fairness.py                  # DI, EOD, CFR, bootstrap CIs
│   ├── reweighing.py                # Kamiran-Calders weights
│   └── pse.py                       # path-specific effect estimation
│
├── experiments/
│   ├── 01_german_credit_baseline.py # 10-seed XGBoost baseline (UCI)
│   ├── 02_german_credit_reweigh.py  # Reweighing comparator
│   ├── 03_shap_stability.py         # rank stability across seeds
│   ├── 04_ledger_run.py             # R-EXP packets + ledger + TTA
│   └── 05_synthetic_pma_demo.py     # schema-preserving synthetic demo
│
├── notebooks/
│   ├── colab_phase2_megacell.ipynb  # one-cell Colab reproducer
│   └── figures.ipynb                # paper figures
│
├── data/
│   ├── README.md                    # data access policy
│   └── synthetic/                   # schema-preserving generator
│       └── generate_pma_synthetic.py
│
├── results/                         # populated by experiments
│   ├── baseline_german_credit.csv
│   ├── reweigh_german_credit.csv
│   ├── shap_stability_german_credit.csv
│   └── ledger_summary.json
│
├── tex/
│   └── tables/                      # auto-generated LaTeX snippets
│
└── tests/
    ├── test_ledger_integrity.py
    ├── test_tta_correctness.py
    ├── test_fairness_metrics.py
    └── test_r_exp_schema.py
```

---

## Reproducing the German Credit results

### Option A — Local

```bash
git clone https://github.com/<USERNAME>/cfam-pipeline.git
cd cfam-pipeline
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

python experiments/01_german_credit_baseline.py     # ~2 min
python experiments/02_german_credit_reweigh.py      # ~3 min
python experiments/03_shap_stability.py             # ~2 min
python experiments/04_ledger_run.py                 # ~1 min
```

All experiments use 10 seeds (0–9) and report mean ± 95% percentile bootstrap CIs (2,000 resamples).

### Option B — Google Colab

Open `notebooks/colab_phase2_megacell.ipynb`, run the single cell, and the script will:

1. Install pinned dependencies.
2. Download `credit-g` from OpenML.
3. Train the 10-seed XGBoost baseline.
4. Run the Kamiran–Calders Reweighing comparator.
5. Compute SHAP rank stability across seeds.
6. Generate R-EXP packets, append them to a hash-chained ledger, and report TTA metrics.
7. Print LaTeX-formatted result tables that can be copied directly into the manuscript.

### Expected results

| Metric (10 seeds, mean [95% CI]) | Value |
|---|---|
| AUC–ROC                            | 0.791 [0.789, 0.793] |
| F1                                 | 0.529 [0.517, 0.541] |
| Disparate Impact (sex)             | 0.974 [0.950, 0.999] |
| Equal Opportunity Difference (sex) | -0.023 [-0.039, -0.007] |
| Ledger chain integrity             | ✓ verified |
| TTA — Mean Decision-to-Audit       | reproduced from synthetic timestamps |

Numbers are reproducible bit-for-bit if `random_state` is preserved across all libraries.

---

## PMA confidential dataset

The Palestinian Monetary Authority dataset used in the manuscript is **confidential** and cannot be redistributed under the terms of the contractual agreement that governs its use.

To enable schema-level reproducibility, this repository ships:

- A **synthetic data generator** (`data/synthetic/generate_pma_synthetic.py`) that produces records matching the PMA schema in distribution but with no real account information.
- The full **CFAM code path** that consumes the synthetic data, so the architecture, fairness diagnostics, and audit logic are fully exercisable without the real dataset.
- A **public-benchmark replication** on the UCI German Credit dataset, which serves as the externally verifiable evidence of pipeline behaviour.

---

## Citation

```bibtex
@article{salaydi2026cfam,
  title   = {The CFAM Pipeline: A Causal Framework for Continuous Fairness Auditing and Tamper-Evident Explanation in FinTech Credit Scoring},
  author  = {Salaydi, Samah and Çeliktaş, Barış},
  journal = {SN Computer Science},
  year    = {2026},
  note    = {Under review}
}
```

A `CITATION.cff` file is also provided for GitHub's "Cite this repository" widget.

---

## License

MIT License — see [LICENSE](LICENSE).

The MIT license applies to the code only. Use of the methodology in regulated supervisory contexts is the responsibility of the deploying institution and may require additional governance review.

---

## Acknowledgments

We acknowledge the Palestinian Monetary Authority (PMA) for providing access to confidential supervisory credit data under contractual confidentiality, and colleagues at Birzeit University and Işık University for helpful discussions on regulatory governance and causal fairness modelling.

This work is published with an Article Processing Charge waiver granted under Springer Nature's Research4Life programme, on the basis of the corresponding author's affiliation with Birzeit University in the occupied Palestinian territory (Research4Life Group A).

---

## Contact

- **Samah Salaydi** — Salaydi@birzeit.edu (corresponding author)
- **Barış Çeliktaş** — baris.celiktas@isikun.edu.tr

Issues and pull requests are welcome via the GitHub issue tracker.
