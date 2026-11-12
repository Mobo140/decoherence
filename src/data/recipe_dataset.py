"""Datasets from recipe runs.

``to_dataset`` turns the `Run`s of ``blocks.runner.run`` into the
``TrajectoryResult`` shape that ``windows.build_dataset`` cuts into
windows. It lives in ``src.data`` because ``src.blocks`` does not depend on
the dataset code.
"""
from __future__ import annotations

import dataclasses
import types
from collections.abc import Sequence

import numpy as np

from src.blocks.runner import Run
from src.data import dataset as dataset_lib
from src.data import windows
from src.domain import value_objects

_SINGLE = "system.single_qubit"
_DOUBLE = "system.two_qubit"


def _context_config(run: Run) -> types.SimpleNamespace:
    """Builds the few config attributes `windows.context_codes` reads.

    The operator type is a placeholder when the run has no jump channel;
    `to_dataset` then drops the codes that depend on it.
    """
    operator = value_objects.DissipatorType(run.jump_op or "sigma_minus")
    if run.system_fn == _SINGLE:
        return types.SimpleNamespace(
            n_qubits=value_objects.QubitCount.ONE,
            dissipator=types.SimpleNamespace(operator_type=operator),
            interaction_type=None, J=0.0)
    if run.system_fn == _DOUBLE:
        return types.SimpleNamespace(
            n_qubits=value_objects.QubitCount.TWO,
            dissipator=types.SimpleNamespace(operator_type=operator),
            interaction_type=value_objects.InteractionType(
                run.system_params["interaction"]),
            J=float(run.system_params["J"]))
    raise ValueError(
        f"to_dataset knows the systems {_SINGLE!r} and {_DOUBLE!r}, "
        f"not {run.system_fn!r}.")


def _trajectory(run: Run) -> value_objects.TrajectoryResult:
    """Converts a run to a trajectory result."""
    return value_objects.TrajectoryResult(
        times=run.observations.times,
        observables=dict(run.observations.series),
        t_decoh=run.target.time,
        system_config=_context_config(run),
        censored=run.target.censored,
    )


def to_dataset(
    runs: Sequence[Run],
    *,
    window_length: int,
    horizon: float,
    samples_per_trajectory: int,
    train_ratio: float,
    val_ratio: float,
    min_window_gap: int,
    include_censored: bool,
    rng: np.random.Generator,
) -> dataset_lib.Dataset:
    """Cuts the runs of a recipe into labelled windows.

    Context codes follow `windows.context_codes`. A code that depends on
    the dissipator is left out (unknown) when the recipe has no
    ``channels.jump`` block, for instance with a Kondo bath.

    Args:
        runs: Runs of one recipe.
        window_length: Number of time steps in a window.
        horizon: A window is at risk if decoherence follows within this
            time.
        samples_per_trajectory: Windows drawn from each trajectory.
        train_ratio: Share of trajectories in the train split.
        val_ratio: Share of trajectories in the val split.
        min_window_gap: Minimum gap in steps between windows.
        include_censored: Keep trajectories with a censored target.
        rng: Source of randomness of the split and the window ends.

    Returns:
        The dataset.

    Raises:
        ValueError: If `runs` is empty, a run has no target, the runs use
            a system the context codes do not know, or as
            `windows.build_dataset`.
        RuntimeError: If no trajectory yields a window.
    """
    if len(runs) == 0:
        raise ValueError("There are no runs to build a dataset from.")
    if any(run.target is None for run in runs):
        raise ValueError(
            "A run has no target; the recipe needs a target block.")
    dataset = windows.build_dataset(
        [_trajectory(run) for run in runs],
        window_length=window_length, horizon=horizon,
        samples_per_trajectory=samples_per_trajectory,
        train_ratio=train_ratio, val_ratio=val_ratio,
        min_window_gap=min_window_gap, include_censored=include_censored,
        seed=None, rng=rng)
    unknown = set()
    if any(run.jump_op is None for run in runs):
        unknown.add("dissipator_type")
        if any(run.system_fn == _SINGLE and run.jump_op is None
               for run in runs):
            unknown.add("interaction_type")
    context = {k: v for k, v in dataset.context.items() if k not in unknown}
    return dataclasses.replace(dataset, context=context)
