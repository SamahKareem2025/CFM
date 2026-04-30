"""Kamiran-Calders Reweighing pre-processor.

Reference: F. Kamiran and T. Calders, "Data preprocessing techniques for
classification without discrimination," Knowledge and Information Systems
(KAIS), 2012. Equation (8) — the joint reweighing scheme:

    w(s, y) = (P(S=s) * P(Y=y)) / P(S=s, Y=y)

so that the joint distribution of the protected attribute and the label
matches the product of its marginals after weighting.
"""
from __future__ import annotations

import numpy as np


def kamiran_calders_weights(
    sensitive: np.ndarray,
    labels: np.ndarray,
) -> np.ndarray:
    """Return Kamiran-Calders sample weights aligned with ``sensitive``/``labels``.

    Both inputs are coerced to ``int`` arrays. The output is a float array of
    the same shape as ``sensitive`` with mean approximately 1.0.
    """
    s = np.asarray(sensitive).astype(int)
    y = np.asarray(labels).astype(int)
    n = s.size
    if n == 0:
        return np.empty(0, dtype=float)
    if y.size != n:
        raise ValueError(f"sensitive and labels must have the same length; got {n} vs {y.size}")

    weights = np.ones(n, dtype=float)
    p_s_marg = {sv: float((s == sv).mean()) for sv in np.unique(s)}
    p_y_marg = {yv: float((y == yv).mean()) for yv in np.unique(y)}

    for sv in np.unique(s):
        for yv in np.unique(y):
            mask = (s == sv) & (y == yv)
            if not mask.any():
                continue
            p_joint = float(mask.mean())
            if p_joint == 0:
                continue
            weights[mask] = (p_s_marg[sv] * p_y_marg[yv]) / p_joint
    return weights
