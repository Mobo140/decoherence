"""Edge cases and registry of src/evaluation/metrics."""
from __future__ import annotations

import math

import numpy as np
import pytest
from sklearn import metrics as sk_metrics

from src.evaluation import metrics

Y = np.array([1.0, 2.0, 4.0, 8.0])
P = np.array([1.5, 2.0, 3.0, 9.0])


def test_registry_has_all_aliases_and_is_read_only():
    assert set(metrics.METRICS) == {
        "r2", "mae", "rmse", "mape", "r2_unguarded", "auroc", "brier",
        "balanced_accuracy", "auroc_pairwise"}
    with pytest.raises(TypeError):
        metrics.METRICS["r2"] = metrics.mae  # type: ignore[index]


def test_regression_values():
    assert metrics.mae(Y, P) == pytest.approx(0.625)
    assert metrics.rmse(Y, P) == pytest.approx(math.sqrt(0.5625))
    expected_r2 = 1 - np.sum((Y - P) ** 2) / np.sum((Y - Y.mean()) ** 2)
    assert metrics.r2(Y, P) == expected_r2
    assert metrics.r2_unguarded(Y, P) == expected_r2


def test_r2_of_constant_targets_is_zero_but_unguarded_is_not():
    y = np.full(5, 3.0)
    assert metrics.r2(y, y + 1.0) == 0.0
    with np.errstate(divide="ignore", invalid="ignore"):
        assert not np.isfinite(metrics.r2_unguarded(y, y + 1.0))


def test_r2_guard_threshold_is_on_ss_tot():
    y = np.array([0.0, 1e-7])  # ss_tot = 5e-15 <= 1e-12
    assert metrics.r2(y, np.array([1.0, 1.0])) == 0.0


def test_mape_masks_small_actuals():
    y = np.array([0.0, 1e-9, 2.0, 4.0])
    p = np.array([5.0, 5.0, 1.0, 5.0])
    # only 2.0 and 4.0 count: (|1|/2 + |1|/4) / 2 * 100
    assert metrics.mape(y, p) == pytest.approx(37.5)


def test_mape_without_valid_actuals_is_nan():
    assert math.isnan(metrics.mape(np.zeros(3), np.ones(3)))


def test_auroc_single_class_is_nan():
    assert math.isnan(metrics.auroc(np.ones(4), np.linspace(0, 1, 4)))
    assert math.isnan(metrics.auroc(np.zeros(4), np.linspace(0, 1, 4)))


def test_auroc_nan_scores_give_nan():
    y = np.array([0, 1, 0, 1])
    assert math.isnan(metrics.auroc(y, np.array([0.1, np.nan, 0.2, 0.9])))


def test_auroc_matches_sklearn_and_pairwise_form():
    rng = np.random.default_rng(1)
    y = rng.integers(0, 2, 50)
    s = rng.random(50)
    assert metrics.auroc(y, s) == sk_metrics.roc_auc_score(y, s)
    assert metrics.auroc_pairwise(y, s) == pytest.approx(
        metrics.auroc(y, s), abs=1e-12)


def test_auroc_pairwise_counts_ties_as_half():
    y = np.array([0, 1])
    assert metrics.auroc_pairwise(y, np.array([0.5, 0.5])) == 0.5
    assert math.isnan(metrics.auroc_pairwise(np.ones(3), np.arange(3.0)))


def test_brier_and_balanced_accuracy():
    y = np.array([0, 0, 1, 1])
    s = np.array([0.1, 0.6, 0.4, 0.9])
    assert metrics.brier(y, s) == pytest.approx(
        (0.01 + 0.36 + 0.36 + 0.01) / 4)
    # predictions [0, 1, 0, 1]: sensitivity 0.5, specificity 0.5
    assert metrics.balanced_accuracy(y, s) == 0.5
    assert metrics.balanced_accuracy(y, s, threshold=0.35) == 0.75


def test_registry_entries_take_two_arrays():
    y = np.array([0, 1, 0, 1])
    s = np.array([0.2, 0.8, 0.3, 0.7])
    for alias in ("auroc", "brier", "balanced_accuracy", "auroc_pairwise"):
        assert isinstance(metrics.METRICS[alias](y, s), float)
    for alias in ("r2", "mae", "rmse", "mape", "r2_unguarded"):
        assert isinstance(metrics.METRICS[alias](Y, P), float)
