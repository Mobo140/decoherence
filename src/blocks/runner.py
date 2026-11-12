"""Running a recipe: seeds, per-trajectory parameters, cache.

``run`` executes a ``Recipe`` for ``n_trajectories`` trajectories.

Seeds. With ``seeds="spawn"`` the root is ``SeedSequence(seed)``. The
sampler draws from the stream with spawn key ``(0,)`` and trajectory ``i``
from the one with spawn key ``(1, i)``; the generators of its channels,
initial state and observation stages are the three children of that
stream. A trajectory therefore depends only on ``(seed, i)`` and its
sampled parameters, whatever ``n_trajectories`` is and in whatever order
trajectories are computed.

With ``seeds="legacy"`` the run reproduces the published runs: the global
NumPy generator is seeded once and the trajectories are computed in order,
the initial state is drawn from the global generator, and every other
random block shares ``default_rng(seed + recipe.legacy_rng_offset)``.
"""
from __future__ import annotations

import inspect
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from src.blocks import cache, evolve, observe
from src.blocks.channels import Channel
from src.blocks.recipe import (
    Block,
    Param,
    Recipe,
    channel_role,
    freeze,
    observe_role,
)
from src.blocks.target import Target

_SAMPLER_STREAM = (0,)
_TRAJECTORY_STREAM = 1
_STAGES = ("channels", "initial", "observe")


@dataclass(frozen=True, eq=False)
class Run:
    """The result of one trajectory of a recipe.

    Attributes:
        index: Position of the trajectory in the run.
        params: The sampled parameter values.
        system_params: Resolved parameters of the system block.
        system_fn: Name of the system block's function.
        jump_op: Operator type of the first ``channels.jump`` block, or
            None if the recipe has none.
        observations: The observed series.
        target: The decoherence time.
        states: The states, only if ``keep_states`` was set.
    """

    index: int
    params: dict[str, Any]
    system_params: dict[str, Any]
    system_fn: str
    jump_op: str | None
    observations: observe.Observations
    target: Target
    states: evolve.States | None = field(default=None)


def _resolve_value(value: Any, params: dict[str, Any]) -> Any:
    """Replaces every `Param` inside a value by its sampled value."""
    if isinstance(value, Param):
        if value.name not in params:
            raise ValueError(
                f"Param {value.name!r} is not sampled; the sampler gave "
                f"{sorted(params)}.")
        return params[value.name]
    if isinstance(value, tuple):
        return tuple(_resolve_value(v, params) for v in value)
    return value


def _resolved(b: Block, params: dict[str, Any]) -> dict[str, Any]:
    """Returns the parameters of a block with the Params replaced."""
    return {k: _resolve_value(v, params) for k, v in b.params}


def _call(b: Block, params: dict, supplied: dict[str, Any]) -> Any:
    """Calls a block's function with its parameters and supplied data."""
    function = b.function()
    accepted = inspect.signature(function).parameters
    kwargs = _resolved(b, params)
    for name, value in supplied.items():
        if name in accepted and name not in kwargs:
            kwargs[name] = value
    return function(**kwargs)


def _streams(seed: int, index: int) -> dict[str, Any]:
    """Returns the generator of each stage of a trajectory (spawn mode)."""
    root = np.random.SeedSequence(
        seed, spawn_key=(_TRAJECTORY_STREAM, index))
    return {
        stage: np.random.default_rng(child)
        for stage, child in zip(_STAGES, root.spawn(len(_STAGES)))}


def _seed_identity(recipe: Recipe, seed: int, index: int) -> list:
    """Returns the JSON-serialisable identity of a trajectory's seed."""
    if recipe.seeds == "legacy":
        return ["legacy", seed, index, recipe.legacy_rng_offset]
    return ["spawn", seed, [_TRAJECTORY_STREAM, index]]


def _time_grid(t_max: float, dt: float) -> np.ndarray:
    """Builds the time grid as ``QuTipSimulator`` does."""
    n_steps = int(round(t_max / dt)) + 1
    return np.linspace(0.0, t_max, n_steps)


def _sample(recipe: Recipe, n: int, seed: int) -> list[dict[str, Any]]:
    """Draws the per-trajectory parameters; checks they are hashable data."""
    if recipe.sampler is None:
        return [{} for _ in range(n)]
    rng = np.random.default_rng(
        np.random.SeedSequence(seed, spawn_key=_SAMPLER_STREAM))
    sampled = recipe.sampler.function()(
        rng, n, **recipe.sampler.param_dict())
    if len(sampled) != n:
        raise ValueError(
            f"The sampler returned {len(sampled)} dicts for {n} "
            "trajectories.")
    return [{k: freeze(v) for k, v in row.items()} for row in sampled]


def _execute(
    recipe: Recipe,
    index: int,
    params: dict[str, Any],
    seed: int,
    streams: dict[str, Any],
    cache_dir: Path | None,
    keep_states: bool,
) -> Run:
    """Computes one trajectory."""
    t_max = _resolve_value(recipe.t_max, params)
    dt = _resolve_value(recipe.dt, params)
    times = _time_grid(t_max, dt)
    system = _call(recipe.system, params, {})

    # The initial state is drawn before anything else and before the cache
    # lookup, so the global generator advances the same on a hit.
    rho0 = _call(
        recipe.initial, params, {"system": system, "rng": streams["initial"]})

    channels: list[Channel] = []
    supplied = {"rng": streams["channels"], "t_max": t_max}
    for b in recipe.channels:
        if channel_role(b) == "producer":
            channels += _call(b, params, {**supplied, "system": system})
        else:
            channels = _call(b, params, {**supplied, "channels": channels})

    description = {
        "system": [recipe.system.fn, _resolved(recipe.system, params)],
        "channels": [[b.fn, _resolved(b, params)] for b in recipe.channels],
        "initial": [recipe.initial.fn, _resolved(recipe.initial, params)],
        "t_max": t_max, "dt": dt,
        "seed": _seed_identity(recipe, seed, index),
    }
    states = None
    key = None
    if cache_dir is not None:
        key = cache.states_key(
            description, system, channels, rho0.full(), times)
        states = cache.load(cache_dir, key)
    if states is None:
        states = evolve.master_equation(system, channels, rho0, times)
        if key is not None:
            cache.store(cache_dir, key, states)

    obs = None
    for b in recipe.observe:
        if observe_role(b) == "producer":
            part = _call(b, params, {"states": states})
            obs = part if obs is None else observe.combine(obs, part)
        else:
            obs = _call(
                b, params, {"obs": obs, "rng": streams["observe"]})
    if all(name in obs.series for name in observe.MODEL_TAIL):
        obs = observe.model_layout(obs)

    result_target = _call(
        recipe.target, params, {"states": states, "system": system})
    jump = next((b for b in recipe.channels if b.fn == "channels.jump"), None)
    return Run(
        index=index,
        params=dict(params),
        system_params=_resolved(recipe.system, params),
        system_fn=recipe.system.fn,
        jump_op=(None if jump is None
                 else _resolved(jump, params)["op_type"]),
        observations=obs,
        target=result_target,
        states=states if keep_states else None,
    )


def run(
    recipe: Recipe,
    n_trajectories: int,
    seed: int,
    cache_dir: str | Path | None = None,
    keep_states: bool = False,
    indices: Sequence[int] | None = None,
) -> list[Run]:
    """Runs a recipe.

    Args:
        recipe: The recipe.
        n_trajectories: Number of trajectories; the sampler is asked for
            this many parameter sets.
        seed: Run seed.
        cache_dir: If set, states are cached there on disk, keyed by
            content. A hit gives bitwise the same result as a miss. In
            legacy mode the initial state and the channels are built
            before the lookup, so the random generators advance the same
            on a hit.
        keep_states: Keep the states in each `Run`.
        indices: Compute only these trajectories, in this order (spawn
            seeds only; in legacy mode trajectories depend on their
            predecessors). The result for an index does not depend on the
            others.

    Returns:
        One `Run` per trajectory, in the order of `indices` or of the
        trajectories.

    Raises:
        ValueError: If `n_trajectories` is not a positive integer, `seed`
            is negative, an index is out of range, `indices` is used with
            legacy seeds, a `Param` is not sampled, or a block rejects its
            arguments.
    """
    if int(n_trajectories) != n_trajectories or n_trajectories < 1:
        raise ValueError(
            f"n_trajectories must be a positive integer, got "
            f"{n_trajectories!r}.")
    if int(seed) != seed or seed < 0:
        raise ValueError(f"seed must be a non-negative integer, got {seed!r}.")
    if indices is not None:
        if recipe.seeds == "legacy":
            raise ValueError(
                "indices cannot be used with legacy seeds: a trajectory "
                "depends on the ones before it.")
        bad = [i for i in indices if not 0 <= i < n_trajectories]
        if bad:
            raise ValueError(
                f"indices {bad} are outside 0..{n_trajectories - 1}.")
    sampled = _sample(recipe, n_trajectories, seed)
    chosen = range(n_trajectories) if indices is None else list(indices)
    cache_path = None if cache_dir is None else Path(cache_dir)

    legacy_streams = None
    if recipe.seeds == "legacy":
        np.random.seed(seed)
        shared = np.random.default_rng(seed + recipe.legacy_rng_offset)
        legacy_streams = {
            "channels": shared, "initial": None, "observe": shared}
    runs = []
    for i in chosen:
        streams = legacy_streams or _streams(seed, i)
        runs.append(_execute(
            recipe, i, sampled[i], seed, streams, cache_path, keep_states))
    return runs
