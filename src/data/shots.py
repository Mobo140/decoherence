"""Datasets of finite-shot observations of one set of simulated states.

A real experiment gives 0/1 outcomes, not ensemble means, and on two qubits
the global purity and coherence need tomography. This module simulates the
trajectories once (`simulate`) and then observes the same states under
different input sets (`InputSet`) with fresh shot noise per stream
(`shot_rng`), so that the input sets differ only in what is measured. Windows
and splits depend on the trajectories and the seed alone, hence every input
set of one seed has the same windows and the same test trajectories.
"""
from __future__ import annotations

import dataclasses
import types
from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np

from src import reproducibility
from src.blocks import legacy, observe, runner
from src.blocks.evolve import States
from src.data import dataset as dataset_lib
from src.data import recipe_dataset
from src.domain import value_objects

# kind -> what is observed.
#   exact        6 (or 3) Pauli means, purity, coherence_l1 from rho(t)
#   exact_local  the Pauli means only, from rho(t)
#   local        finite shots, one common basis x, y, z (3 settings)
#   tomography   finite shots in all 3^n product bases, rho by inversion
KINDS = ("exact", "exact_local", "local", "tomography")
# Streams of shot noise; every stream is independent of the others.
STREAMS: Mapping[str, int] = types.MappingProxyType({"train": 1, "test": 2})


@dataclasses.dataclass(frozen=True)
class InputSet:
    """What a model sees of each simulated state.

    Attributes:
        kind: One of `KINDS`.
        n_shots: Shots per setting and time for the finite-shot kinds;
            None for the exact ones.

    Raises:
        ValueError: If `kind` is unknown, an exact kind has `n_shots`, or a
            finite-shot kind lacks a positive integer `n_shots`.
    """

    kind: str
    n_shots: int | None = None

    def __post_init__(self) -> None:
        if self.kind not in KINDS:
            raise ValueError(
                f"kind must be one of {KINDS}, got {self.kind!r}.")
        if self.kind.startswith("exact"):
            if self.n_shots is not None:
                raise ValueError(
                    f"The exact kind {self.kind!r} takes no n_shots.")
        elif (self.n_shots is None or int(self.n_shots) != self.n_shots
              or self.n_shots < 1):
            raise ValueError(
                f"The kind {self.kind!r} needs a positive integer n_shots, "
                f"got {self.n_shots!r}.")

    @property
    def name(self) -> str:
        """Short label, e.g. ``tomography_1000`` or ``exact``."""
        if self.n_shots is None:
            return self.kind
        return f"{self.kind}_{self.n_shots}"

    def apply(
        self, states: States, rng: np.random.Generator
    ) -> observe.Observations:
        """Observes the states of one trajectory.

        Args:
            states: States of one or two qubits.
            rng: Source of the shot noise; unused by the exact kinds.

        Returns:
            Series with the model tail (purity, coherence_l1) last where
            the kind has it.

        Raises:
            ValueError: As the observation functions.
        """
        if self.kind == "exact":
            return observe.model_layout(observe.combine(
                observe.pauli_means(states), observe.purity(states),
                observe.coherence_l1(states)))
        if self.kind == "exact_local":
            return observe.pauli_means(states)
        obs = observe.finite_shots(
            states, int(self.n_shots), self.kind, rng)
        if self.kind == "tomography":
            return observe.model_layout(obs)
        return obs


def n_settings(kind: str, n_qubits: int) -> int | None:
    """Number of measurement bases of a finite-shot kind.

    Args:
        kind: One of `KINDS`.
        n_qubits: 1 or 2.

    Returns:
        3 for ``local`` and for one qubit, 9 for two-qubit tomography;
        None for the exact kinds, which need no measurement.

    Raises:
        ValueError: If `kind` is unknown or `n_qubits` is not 1 or 2.
    """
    if kind not in KINDS:
        raise ValueError(f"kind must be one of {KINDS}, got {kind!r}.")
    if n_qubits not in (1, 2):
        raise ValueError(f"n_qubits must be 1 or 2, got {n_qubits!r}.")
    if kind.startswith("exact"):
        return None
    return 3 if kind == "local" or n_qubits == 1 else 9


def shot_rng(
    seed: int, stream: str, n_shots: int, index: int
) -> np.random.Generator:
    """Generator of the shot noise of one trajectory.

    It depends on the arguments only, not on the order of the calls.

    Args:
        seed: Run seed.
        stream: A key of `STREAMS` ("train" or "test").
        n_shots: Shots per setting.
        index: Position of the trajectory.

    Returns:
        A new generator.

    Raises:
        KeyError: If `stream` is unknown.
    """
    if stream not in STREAMS:
        raise KeyError(
            f"Unknown stream {stream!r}. Valid streams: {sorted(STREAMS)}.")
    return np.random.default_rng(np.random.SeedSequence(
        [seed, STREAMS[stream], int(n_shots), index]))


def simulate(
    configs: Sequence[value_objects.SystemConfig],
    seed: int,
    cache_dir: str | Path | None = None,
) -> list[runner.Run]:
    """Simulates the plain physics of the configurations, keeping states.

    Same physics and same draws as ``QuTipSimulator("plain")`` after
    ``seed_numpy_global(seed)``, so the trajectories are those of
    `pipeline.generate` with the same seed, and the disk cache is shared.

    Args:
        configs: The systems, in simulation order.
        seed: Seed of the initial states.
        cache_dir: States cache directory, or None.

    Returns:
        One run per configuration, with `states` and `target`.
    """
    reproducibility.seed_numpy_global(seed)
    rng = np.random.default_rng(None)
    return [
        runner.run_with_generator(
            legacy.recipe_for(config, "plain"), rng, index=i, seed=0,
            cache_dir=cache_dir, keep_states=True)
        for i, config in enumerate(configs)]


def windowed(
    runs: Sequence[runner.Run],
    input_set: InputSet,
    *,
    seed: int,
    stream: str,
    window_length: int,
    horizon: float,
    samples_per_trajectory: int,
    train_ratio: float,
    val_ratio: float,
    min_window_gap: int = 0,
    include_censored: bool = False,
) -> dataset_lib.Dataset:
    """Observes the runs under an input set and cuts the windows.

    The split and the window ends are drawn from ``default_rng(seed)``
    only, so they are the same for every input set and every stream.

    Args:
        runs: Runs from `simulate`, with states.
        input_set: What is observed.
        seed: Seed of the split and the windows, and of the shot noise.
        stream: Noise stream, a key of `STREAMS`.
        window_length: Time steps per window.
        horizon: Risk horizon.
        samples_per_trajectory: Windows per trajectory.
        train_ratio: Share of trajectories in the train split.
        val_ratio: Share of trajectories in the validation split.
        min_window_gap: Minimum gap between window ends.
        include_censored: Keep trajectories with a censored target.

    Returns:
        The dataset.

    Raises:
        ValueError: If a run has no states, or as `to_dataset`.
        KeyError: If `stream` is unknown.
    """
    if any(run.states is None for run in runs):
        raise ValueError("windowed needs runs with states (see simulate).")
    observed = [
        dataclasses.replace(run, observations=input_set.apply(
            run.states,
            shot_rng(seed, stream, input_set.n_shots or 0, run.index)))
        for run in runs]
    return recipe_dataset.to_dataset(
        observed, window_length=window_length, horizon=horizon,
        samples_per_trajectory=samples_per_trajectory,
        train_ratio=train_ratio, val_ratio=val_ratio,
        min_window_gap=min_window_gap, include_censored=include_censored,
        rng=np.random.default_rng(seed))
