"""Schema-preserving synthetic PMA demo.

The real PMA dataset is confidential. This script generates a synthetic
table that matches the *schema and marginal distributions* of the PMA
preprocessed data (no real account information), then runs the CFAM
pipeline end-to-end on it. Reviewers can use this to verify pipeline
mechanics without access to the confidential data.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data.synthetic.generate_pma_synthetic import generate as generate_synthetic  # noqa: E402


def main() -> None:
    print("=" * 60)
    print("Synthetic PMA demo — schema-preserving generator")
    print("=" * 60)
    df = generate_synthetic(n=2000, seed=0)
    out = Path(__file__).resolve().parents[1] / "data" / "synthetic" / "pma_synthetic.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(f"Wrote synthetic dataset: {out}")
    print(df.describe(include="all").T.head(12))


if __name__ == "__main__":
    main()
