"""Paired statistics and bootstrap helpers of src/evaluation/stats."""
from __future__ import annotations

import numpy as np
import pytest

from src.evaluation import metrics, stats


def test_paired_difference_counts_signs():
    out = stats.paired_difference([0.5, 0.4, 0.3], [0.4, 0.5, 0.1])
    assert out.n == 3
    assert (out.n_positive, out.n_negative) == (2, 1)
    assert out.mean == pytest.approx(0.2 / 3)
    assert out.differences[0] == pytest.approx(0.1)


def test_paired_difference_single_pair_has_zero_std():
    out = stats.paired_difference([1.0], [0.25])
    assert out.std == 0.0 and out.mean == 0.75


def test_paired_difference_rejects_mismatched_input():
    with pytest.raises(ValueError):
        stats.paired_difference([1.0, 2.0], [1.0])
    with pytest.raises(ValueError):
        stats.paired_difference([], [])


def test_pearson_of_linear_data_is_one():
    x = np.arange(5.0)
    assert stats.pearson(x, 2 * x + 1) == pytest.approx(1.0)


def test_resample_indices_consume_the_generator_in_order():
    a = stats.resample_indices(7, 3, np.random.default_rng(0))
    rng = np.random.default_rng(0)
    b = [rng.integers(0, 7, 7) for _ in range(3)]
    assert all(np.array_equal(x, y) for x, y in zip(a, b))


def test_group_resamples_keep_groups_whole():
    groups = np.array([5, 5, 5, 9, 9, 2])
    for idx in stats.group_resample_indices(
            groups, 20, np.random.default_rng(1)):
        # each draw is a union of whole groups: sizes 3, 2 and 1
        counts = np.bincount(groups[idx], minlength=10)
        assert counts[5] % 3 == 0 and counts[9] % 2 == 0


def test_bootstrap_interval_brackets_the_point_estimate():
    rng = np.random.default_rng(0)
    y = rng.normal(size=200)
    p = y + rng.normal(scale=0.3, size=200)
    values = stats.bootstrap_values(
        metrics.r2, stats.resample_indices(200, 200, rng), y, p)
    lo, hi = stats.percentile_interval(values)
    assert lo < metrics.r2(y, p) < hi


def test_percentile_interval_can_skip_nan():
    values = np.array([np.nan, 1.0, 2.0, 3.0])
    assert np.isnan(stats.percentile_interval(values)[0])
    lo, hi = stats.percentile_interval(values, 100.0, ignore_nan=True)
    assert (lo, hi) == (1.0, 3.0)
