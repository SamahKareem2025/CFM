"""Path-Specific Effect (PSE) estimation for the CFAM SCM.

This is a *minimal* nested-counterfactual estimator suitable for the
expert-specified DAG used in the manuscript (Sex/AgeGroup -> mediator ->
CreditScore). It is **not** a full do-calculus engine — for general SCMs use
DoWhy or Causal-learn. Here we expose only what the CFAM pipeline needs:

    pse_via_mediator(score_fn, X, A, mediator_cols)

which estimates the path-specific effect transmitted via ``mediator_cols``
when the protected attribute ``A`` is counterfactually flipped.
"""
from __future__ import annotations

from typing import Callable, Sequence

import numpy as np
import pandas as pd


def _flip_binary(arr: np.ndarray) -> np.ndarray:
    return 1 - np.asarray(arr).astype(int)


def pse_via_mediator(
    score_fn: Callable[[pd.DataFrame], np.ndarray],
    X: pd.DataFrame,
    A_col: str,
    mediator_cols: Sequence[str],
) -> float:
    """Estimate the natural direct effect transmitted via ``mediator_cols``.

    Implementation sketch (nested counterfactual):

        E[ Y( A=1, M( A=0 ) ) - Y( A=0, M( A=0 ) ) ]

    approximated empirically by replacing the protected attribute while
    holding the mediators at their factual values.
    """
    if A_col not in X.columns:
        raise KeyError(f"protected column {A_col!r} not in X")

    X_factual = X.copy()
    X_cf = X.copy()
    X_cf[A_col] = _flip_binary(X[A_col].to_numpy())

    for m in mediator_cols:
        if m in X_cf.columns:
            X_cf[m] = X_factual[m]

    y_factual = score_fn(X_factual)
    y_cf = score_fn(X_cf)
    return float(np.mean(y_cf - y_factual))
