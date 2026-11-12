"""blocks.samplers: per-trajectory parameter values."""
import numpy as np
import pytest

from src.blocks import samplers

NAN = float("nan")


def _rng(seed=0):
    return np.random.default_rng(np.random.SeedSequence(seed))


# --- Negative cases ---


@pytest.mark.parametrize("n", [0, -1, 1.5])
@pytest.mark.parametrize("sampler, kwargs", [
    (samplers.uniform, {"a": (0.0, 1.0)}),
    (samplers.choice, {"a": [1, 2]}),
    (samplers.grid, {"a": [1, 2]}),
    (samplers.fixed, {"a": 1}),
    (samplers.from_rows, {"rows": [(("a", 1),)]}),
])
def test_invalid_trajectory_count_raises(sampler, kwargs, n):
    with pytest.raises(ValueError, match="n must be a positive integer"):
        sampler(_rng(), n, **kwargs)


@pytest.mark.parametrize("sampler", [
    samplers.uniform, samplers.choice, samplers.grid, samplers.fixed])
def test_sampler_without_parameters_raises(sampler):
    with pytest.raises(ValueError, match="at least one parameter"):
        sampler(_rng(), 3)


@pytest.mark.parametrize("bounds", [(1.0, 0.0), (0.0, NAN), (0.0,), (0, 1, 2)])
def test_uniform_invalid_range_raises(bounds):
    with pytest.raises(ValueError, match="range of 'a'"):
        samplers.uniform(_rng(), 2, a=bounds)


@pytest.mark.parametrize("sampler", [samplers.choice, samplers.grid])
@pytest.mark.parametrize("values", [[], "abc"])
def test_choice_and_grid_empty_or_string_options_raise(sampler, values):
    with pytest.raises(ValueError, match="non-empty sequence"):
        sampler(_rng(), 2, a=values)


def test_from_rows_with_too_few_rows_raises():
    with pytest.raises(ValueError, match="3 trajectories.*2 rows"):
        samplers.from_rows(_rng(), 3, [(("a", 1),), (("a", 2),)])


# --- Positive cases ---


def test_uniform_stays_in_range_and_gives_floats():
    rows = samplers.uniform(_rng(), 50, a=(1.0, 2.0), b=(-1.0, 0.0))
    assert len(rows) == 50
    assert all(1.0 <= r["a"] <= 2.0 and -1.0 <= r["b"] <= 0.0 for r in rows)
    assert all(type(r["a"]) is float for r in rows)


@pytest.mark.parametrize("sampler, kwargs", [
    (samplers.uniform, {"a": (0.0, 1.0), "b": (2.0, 3.0)}),
    (samplers.choice, {"a": [1, 2, 3], "b": ["x", "y"]}),
    (samplers.grid, {"a": [1, 2, 3], "b": ["x", "y"]}),
    (samplers.fixed, {"a": 1, "b": 2}),
])
def test_trajectory_i_does_not_depend_on_the_number_of_trajectories(
        sampler, kwargs):
    short = sampler(_rng(4), 3, **kwargs)
    long = sampler(_rng(4), 9, **kwargs)
    assert long[:3] == short


def test_samplers_are_reproducible_and_depend_on_the_seed():
    a = samplers.uniform(_rng(1), 5, x=(0.0, 1.0))
    assert a == samplers.uniform(_rng(1), 5, x=(0.0, 1.0))
    assert a != samplers.uniform(_rng(2), 5, x=(0.0, 1.0))


def test_choice_draws_only_listed_options():
    rows = samplers.choice(_rng(), 40, a=[0.5, 2.0])
    assert {r["a"] for r in rows} == {0.5, 2.0}


def test_grid_walks_the_product_and_repeats():
    rows = samplers.grid(_rng(), 5, a=[1, 2], b=["x", "y"])
    assert rows == [
        {"a": 1, "b": "x"}, {"a": 1, "b": "y"}, {"a": 2, "b": "x"},
        {"a": 2, "b": "y"}, {"a": 1, "b": "x"}]


def test_fixed_gives_independent_copies():
    rows = samplers.fixed(_rng(), 2, a=1)
    rows[0]["a"] = 5
    assert rows[1]["a"] == 1


def test_from_rows_returns_the_leading_rows_as_dicts():
    rows = [(("a", 1), ("b", 2)), (("a", 3), ("b", 4))]
    assert samplers.from_rows(_rng(), 2, rows) == [
        {"a": 1, "b": 2}, {"a": 3, "b": 4}]
    assert samplers.from_rows(_rng(), 1, rows) == [{"a": 1, "b": 2}]
