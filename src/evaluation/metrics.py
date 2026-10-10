"""Scalar metrics on NumPy arrays and the `METRICS` registry.

Two families, distinguished by what the second argument holds:

* Regression metrics compare real-valued targets (for example the
  decoherence time) with predictions: `r2`, `mae`, `rmse`, `mape`.
* Classification metrics compare binary labels in {0, 1} with scores
  (probabilities or any monotone ranking): `auroc`, `brier`,
  `balanced_accuracy`.

`r2_unguarded` and `auroc_pairwise` are the forms R13 and R17 recorded their
numbers with; they sit in the registry under their own aliases so that those
experiments can name them.

Every registry entry has the signature
`(y_true, y_pred_or_score) -> float`. Edge cases return a number, not an
exception, so that a degenerate test split gives a row of `nan` or 0.0 in a
results table instead of aborting a sweep.
"""
from __future__ import annotations

import types
from collections.abc import Callable, Mapping

import numpy as np
from sklearn import metrics as sk_metrics

# R^2 is reported as 0.0 when the targets have (numerically) no variance.
SS_TOT_EPS = 1e-12
# MAPE ignores samples whose actual value is not above this.
MAPE_MIN_ACTUAL = 1e-8


def r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Coefficient of determination, 0.0 for (near-)constant targets.

    Args:
        y_true: Real-valued targets.
        y_pred: Predictions, same shape.

    Returns:
        1 - SS_res / SS_tot, or 0.0 when SS_tot <= 1e-12.
    """
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - y_true.mean()) ** 2)
    return float(1.0 - ss_res / ss_tot) if ss_tot > SS_TOT_EPS else 0.0


def r2_unguarded(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Coefficient of determination without the constant-target guard.

    Identical to `r2` except that constant targets give nan or -inf (with a
    NumPy warning) instead of 0.0. R13 and R17 recorded their numbers with
    this form; it is kept separate so that those CSVs stay reproducible.

    Args:
        y_true: Real-valued targets.
        y_pred: Predictions, same shape.

    Returns:
        1 - SS_res / SS_tot.
    """
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - y_true.mean()) ** 2)
    return float(1.0 - ss_res / ss_tot)


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean absolute error of real-valued predictions."""
    return float(np.mean(np.abs(y_pred - y_true)))


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Root mean squared error of real-valued predictions."""
    return float(np.sqrt(np.mean((y_pred - y_true) ** 2)))


def mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean absolute percentage error, in percent.

    Args:
        y_true: Real-valued targets; entries <= 1e-8 are left out.
        y_pred: Predictions, same shape.

    Returns:
        The error over the remaining entries, or nan if none remain.
    """
    mask = y_true > MAPE_MIN_ACTUAL
    if not mask.any():
        return float("nan")
    rel = np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])
    return float(np.mean(rel) * 100)


def auroc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Area under the ROC curve (scikit-learn).

    Args:
        y_true: Binary labels.
        y_score: Scores, higher meaning "more likely positive".

    Returns:
        The area, or nan if `y_true` holds one class only or the scores
        are not finite.
    """
    if len(np.unique(y_true)) < 2:
        return float("nan")
    try:
        return float(sk_metrics.roc_auc_score(y_true, y_score))
    except ValueError:
        return float("nan")


def auroc_pairwise(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """AUROC as P(pos > neg) + P(pos == neg) / 2 over all pairs.

    Equal to `auroc` in exact arithmetic but computed from the n_pos x n_neg
    difference matrix, so the last digits can differ. R13 recorded its AUROC
    this way. Quadratic in memory; for the large sets use `auroc`.

    Args:
        y_true: Labels; positives are entries above 0.5.
        y_score: Scores, same shape.

    Returns:
        The probability, or nan if a class is empty.
    """
    pos, neg = y_score[y_true > 0.5], y_score[y_true <= 0.5]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    diff = pos[:, None] - neg[None, :]
    return float((diff > 0).mean() + 0.5 * (diff == 0).mean())


def brier(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Brier score of predicted probabilities (scikit-learn).

    Args:
        y_true: Binary labels.
        y_score: Probabilities of the positive class.
    """
    return float(sk_metrics.brier_score_loss(y_true, y_score))


def balanced_accuracy(
    y_true: np.ndarray, y_score: np.ndarray, threshold: float = 0.5
) -> float:
    """Mean of sensitivity and specificity after thresholding the scores.

    Args:
        y_true: Binary labels.
        y_score: Scores; those >= `threshold` are predicted positive.
        threshold: Decision threshold.
    """
    return float(
        sk_metrics.balanced_accuracy_score(y_true, y_score >= threshold))


REGRESSION_METRICS = ("r2", "mae", "rmse", "mape", "r2_unguarded")
CLASSIFICATION_METRICS = (
    "auroc", "brier", "balanced_accuracy", "auroc_pairwise")

METRICS: Mapping[str, Callable[[np.ndarray, np.ndarray], float]] = (
    types.MappingProxyType({
        "r2": r2,
        "mae": mae,
        "rmse": rmse,
        "mape": mape,
        "r2_unguarded": r2_unguarded,
        "auroc": auroc,
        "brier": brier,
        "balanced_accuracy": balanced_accuracy,
        "auroc_pairwise": auroc_pairwise,
    }))
