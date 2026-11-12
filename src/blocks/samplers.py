"""Samplers: the per-trajectory parameter values of a recipe.

A sampler is a function ``(rng, n, **params) -> list[dict]`` that returns
one dict of parameter values for each of ``n`` trajectories. The value of
trajectory ``i`` must not depend on ``n``, so that a trajectory can be
recomputed alone: every sampler here draws trajectory ``i`` from the
``i``-th child of ``rng.spawn(n)``, and children do not depend on how many
siblings are spawned. Values are plain Python numbers, strings or tuples,
so that they can be hashed into a cache key.
"""
from __future__ import annotations

import itertools
import math
from collections.abc import Sequence

import numpy as np


def _check_n(n: int) -> None:
    """Raises ValueError unless `n` is a positive integer."""
    if int(n) != n or n < 1:
        raise ValueError(f"n must be a positive integer, got {n!r}.")


def _check_names(values: dict, what: str) -> None:
    """Raises ValueError if no name is given."""
    if not values:
        raise ValueError(f"{what} needs at least one parameter.")


def uniform(
    rng: np.random.Generator, n: int, **ranges: tuple[float, float]
) -> list[dict]:
    """Draws every parameter uniformly from its range.

    Args:
        rng: Parent generator; trajectory i uses its i-th spawned child.
        n: Number of trajectories.
        **ranges: Parameter name -> (low, high); parameters are drawn in
            sorted name order.

    Returns:
        One dict of floats per trajectory.

    Raises:
        ValueError: If `n` is not a positive integer, no range is given, or
            a range is not a finite pair with low <= high.
    """
    _check_n(n)
    _check_names(ranges, "uniform")
    for name, bounds in ranges.items():
        if (len(bounds) != 2 or not all(math.isfinite(b) for b in bounds)
                or bounds[0] > bounds[1]):
            raise ValueError(
                f"The range of {name!r} must be a finite (low, high) with "
                f"low <= high, got {bounds!r}.")
    out = []
    for child in rng.spawn(n):
        out.append({
            name: float(child.uniform(*ranges[name]))
            for name in sorted(ranges)})
    return out


def choice(
    rng: np.random.Generator, n: int, **options: Sequence
) -> list[dict]:
    """Draws every parameter uniformly from its list of options.

    Args:
        rng: Parent generator; trajectory i uses its i-th spawned child.
        n: Number of trajectories.
        **options: Parameter name -> non-empty sequence of values;
            parameters are drawn in sorted name order.

    Returns:
        One dict per trajectory; the values are the options themselves.

    Raises:
        ValueError: If `n` is not a positive integer, no option is given,
            or an option list is empty or a string.
    """
    _check_n(n)
    _check_names(options, "choice")
    for name, values in options.items():
        if isinstance(values, str) or len(values) == 0:
            raise ValueError(
                f"The options of {name!r} must be a non-empty sequence, "
                f"got {values!r}.")
    out = []
    for child in rng.spawn(n):
        out.append({
            name: options[name][int(child.integers(len(options[name])))]
            for name in sorted(options)})
    return out


def grid(
    rng: np.random.Generator, n: int, **axes: Sequence
) -> list[dict]:
    """Walks through the Cartesian product of the axes, then repeats.

    Trajectory i gets point ``i % size`` of the product, the last axis
    (in sorted name order) varying fastest. Nothing is drawn.

    Args:
        rng: Unused; the signature is the one of every sampler.
        n: Number of trajectories.
        **axes: Parameter name -> non-empty sequence of values.

    Returns:
        One dict per trajectory.

    Raises:
        ValueError: If `n` is not a positive integer, no axis is given, or
            an axis is empty or a string.
    """
    _check_n(n)
    _check_names(axes, "grid")
    for name, values in axes.items():
        if isinstance(values, str) or len(values) == 0:
            raise ValueError(
                f"The axis {name!r} must be a non-empty sequence, "
                f"got {values!r}.")
    names = sorted(axes)
    points = list(itertools.product(*(axes[name] for name in names)))
    return [dict(zip(names, points[i % len(points)])) for i in range(n)]


def fixed(rng: np.random.Generator, n: int, **values) -> list[dict]:
    """Gives every trajectory the same parameter values.

    Args:
        rng: Unused; the signature is the one of every sampler.
        n: Number of trajectories.
        **values: Parameter name -> value.

    Returns:
        ``n`` copies of the dict of values.

    Raises:
        ValueError: If `n` is not a positive integer or no value is given.
    """
    _check_n(n)
    _check_names(values, "fixed")
    return [dict(values) for _ in range(n)]


def from_rows(
    rng: np.random.Generator, n: int, rows: Sequence[Sequence[tuple]]
) -> list[dict]:
    """Takes the parameters of trajectory i from the i-th row.

    Used to replay a list of known configurations.

    Args:
        rng: Unused; the signature is the one of every sampler.
        n: Number of trajectories.
        rows: One row per trajectory, each a sequence of
            (name, value) pairs.

    Returns:
        The first ``n`` rows as dicts.

    Raises:
        ValueError: If `n` is not a positive integer or exceeds the number
            of rows.
    """
    _check_n(n)
    if n > len(rows):
        raise ValueError(
            f"{n} trajectories were asked for but there are {len(rows)} "
            "rows.")
    return [dict(row) for row in rows[:n]]
