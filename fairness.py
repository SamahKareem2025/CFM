"""Fairness metrics + bootstrap utilities used throughout CFAM."""
from __future__ import annotations

from typing import Optional, Sequence, Tuple

import numpy as np


def disparate_impact(y_pred: np.ndarray, A: np.ndarray) -> float:
    """DI = P(Y_hat=1 | A=0) / P(Y_hat=1 | A=1).

    Returns ``np.nan`` if the privileged group has zero positive rate or
    if either group is empty.
    """
    y_pred = np.asarray(y_pred).astype(int)
    A = np.asarray(A).astype(int)
    if (A == 0).sum() == 0 or (A == 1).sum() == 0:
        return float("nan")
    p0 = y_pred[A == 0].mean()
    p1 = y_pred[A == 1].mean()
    if p1 == 0:
        return float("nan")
    return float(p0 / p1)


def equal_opportunity_difference(y_true: np.ndarray, y_pred: np.ndarray, A: np.ndarray) -> float:
    """EOD = TPR(A=0) - TPR(A=1) where TPR is the true positive rate."""
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    A = np.asarray(A).astype(int)

    def _tpr(group: int) -> float:
        mask = (A == group) & (y_true == 1)
        n = int(mask.sum())
        if n == 0:
            return float("nan")
        return float(((y_pred == 1) & mask).sum() / n)

    return _tpr(0) - _tpr(1)


def counterfactual_fairness_rate(
    score_factual: np.ndarray,
    score_counterfactual: np.ndarray,
    threshold: float = 0.5,
) -> float:
    """Fraction of records whose binary decision flips when the protected
    attribute is counterfactually altered. Lower is better.
    """
    yf = (np.asarray(score_factual) >= threshold).astype(int)
    yc = (np.asarray(score_counterfactual) >= threshold).astype(int)
    if len(yf) == 0:
        return float("nan")
    return float((yf != yc).mean())


def bootstrap_ci(
    values: Sequence[float],
    n_boot: int = 2000,
    ci: float = 0.95,
    seed: int = 0,
) -> Tuple[float, float, float]:
    """Mean and percentile bootstrap CI of a sample.

    Returns ``(mean, ci_low, ci_high)``. Filters out NaNs before resampling.
    """
    arr = np.asarray([v for v in values if v is not None and not np.isnan(v)], dtype=float)
    if arr.size == 0:
        return float("nan"), float("nan"), float("nan")

    rng = np.random.default_rng(seed)
    boots = np.empty(n_boot, dtype=float)
    n = arr.size
    for i in range(n_boot):
        boots[i] = arr[rng.integers(0, n, n)].mean()
    alpha = (1.0 - ci) / 2.0
    lo, hi = np.percentile(boots, [100 * alpha, 100 * (1 - alpha)])
    return float(arr.mean()), float(lo), float(hi)


def fmt_ci(values: Sequence[float], digits: int = 4, **kw) -> str:
    """Format a sample as ``mean [lo, hi]`` with the given precision."""
    m, lo, hi = bootstrap_ci(values, **kw)
    if np.isnan(m):
        return "N/A"
    return f"{m:.{digits}f} [{lo:.{digits}f}, {hi:.{digits}f}]"
