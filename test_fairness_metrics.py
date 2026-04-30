"""Tests for fairness metrics + bootstrap CI utilities."""
from __future__ import annotations

import numpy as np
import pytest

from cfam_core.fairness import (
    bootstrap_ci,
    counterfactual_fairness_rate,
    disparate_impact,
    equal_opportunity_difference,
)


def test_disparate_impact_perfect_parity():
    pred = np.array([1, 1, 0, 0, 1, 1, 0, 0])
    A    = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    assert disparate_impact(pred, A) == pytest.approx(1.0)


def test_eod_zero_when_tprs_equal():
    y    = np.array([1, 1, 0, 0, 1, 1, 0, 0])
    pred = np.array([1, 1, 0, 0, 1, 1, 0, 0])
    A    = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    assert equal_opportunity_difference(y, pred, A) == pytest.approx(0.0)


def test_cfr_no_flips():
    s = np.array([0.1, 0.2, 0.3, 0.7])
    assert counterfactual_fairness_rate(s, s, threshold=0.5) == 0.0


def test_cfr_all_flip():
    a = np.array([0.0, 0.0, 0.0])
    b = np.array([1.0, 1.0, 1.0])
    assert counterfactual_fairness_rate(a, b, threshold=0.5) == 1.0


def test_bootstrap_ci_shape():
    rng = np.random.default_rng(0)
    sample = rng.normal(loc=10.0, scale=1.0, size=200).tolist()
    m, lo, hi = bootstrap_ci(sample, n_boot=500, seed=1)
    assert lo < m < hi
    assert abs(m - 10.0) < 0.3


def test_bootstrap_ci_handles_nans():
    sample = [1.0, 2.0, float("nan"), 3.0]
    m, lo, hi = bootstrap_ci(sample, n_boot=200, seed=1)
    assert m == pytest.approx(2.0, abs=1e-6)
