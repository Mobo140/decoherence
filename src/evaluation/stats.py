"""Statistics for experiment summaries: paired differences, bootstrap.

All functions are deterministic given their arguments; bootstrap functions
take the random generator explicitly and draw from it in a fixed order
(one `rng.integers(0, n, n)` call per resample), so a seeded generator
reproduces the published intervals.
"""
from __future__ import annotations

import dataclasses
import statistics
from collections.abc import Callable, Sequence

import numpy as np


@dataclasses.dataclass(frozen=True)
class PairedDifference:
    """Summary of per-seed differences a_i - b_i.

    Attributes:
        mean: Mean difference.
        std: Sample standard deviation (n - 1), 0.0 if n < 2.
        n: Number of pairs.
        n_positive: Pairs with a_i - b_i > 0.
        n_negative: Pairs with a_i - b_i < 0.
        differences: The differences themselves, in input order.
    """

    mean: float
    std: float
    n: int
    n_positive: int
    n_negative: int
    differences: tuple[float, ...]


def mean_std(values: Sequence[float]) -> tuple[float, float]:
    """Mean and sample standard deviation, with std 0.0 for one value."""
    std = statistics.stdev(values) if len(values) > 1 else 0.0
    return statistics.mean(values), std


def paired_difference(
    a: Sequence[float], b: Sequence[float]
) -> PairedDifference:
    """Summarises a_i - b_i over matched runs (for example seeds).

    Args:
        a: Values of the first arm, one per run.
        b: Values of the second arm, same runs in the same order.

    Returns:
        The mean, standard deviation and sign counts of the differences.

    Raises:
        ValueError: If the lengths differ or are zero.
    """
    if len(a) != len(b) or not len(a):
        raise ValueError("a and b must be non-empty and of equal length")
    diffs = [x - y for x, y in zip(a, b)]
    mean, std = mean_std(diffs)
    return PairedDifference(
        mean=mean,
        std=std,
        n=len(diffs),
        n_positive=sum(d > 0 for d in diffs),
        n_negative=sum(d < 0 for d in diffs),
        differences=tuple(diffs),
    )


def pearson(x: np.ndarray, y: np.ndarray) -> float:
    """Pearson correlation coefficient of two samples."""
    return float(np.corrcoef(x, y)[0, 1])


def resample_indices(
    n: int, n_boot: int, rng: np.random.Generator
) -> list[np.ndarray]:
    """Draws `n_boot` bootstrap resamples of range(n), with replacement."""
    return [rng.integers(0, n, n) for _ in range(n_boot)]


def group_resample_indices(
    groups: np.ndarray, n_boot: int, rng: np.random.Generator
) -> list[np.ndarray]:
    """Draws bootstrap resamples of whole groups of samples.

    Use when samples are not independent, for example several windows of
    one trajectory: a group is drawn with replacement and all its samples
    enter the resample together.

    Args:
        groups: Group label of every sample (any sortable values).
        n_boot: Number of resamples.
        rng: Source of randomness.

    Returns:
        For each resample, the concatenated sample indices of the drawn
        groups.
    """
    _, label = np.unique(groups, return_inverse=True)
    n_groups = label.max() + 1
    members = [np.where(label == k)[0] for k in range(n_groups)]
    resamples = []
    for _ in range(n_boot):
        drawn = rng.integers(0, n_groups, n_groups)
        resamples.append(np.concatenate([members[k] for k in drawn]))
    return resamples


def bootstrap_values(
    statistic: Callable[..., float],
    resamples: Sequence[np.ndarray],
    *arrays: np.ndarray,
) -> np.ndarray:
    """Evaluates `statistic` on every resample.

    Args:
        statistic: Function of the indexed arrays, for example
            `metrics.r2`.
        resamples: Index arrays from `resample_indices` or
            `group_resample_indices`.
        *arrays: Equally long arrays that are indexed by each resample and
            passed to `statistic` in this order.

    Returns:
        One value per resample.
    """
    return np.array(
        [statistic(*(a[idx] for a in arrays)) for idx in resamples])


def percentile_interval(
    values: np.ndarray, level: float = 95.0, *, ignore_nan: bool = False
) -> tuple[float, float]:
    """Central percentile interval of bootstrap values.

    Args:
        values: Bootstrap statistics.
        level: Coverage in percent.
        ignore_nan: Drop nan values first (a resample where the statistic
            is undefined, such as AUROC with one class).

    Returns:
        (lower, upper) bounds.
    """
    if ignore_nan:
        values = values[~np.isnan(values)]
    tail = (100.0 - level) / 2
    lower, upper = np.percentile(values, [tail, 100.0 - tail])
    return float(lower), float(upper)
