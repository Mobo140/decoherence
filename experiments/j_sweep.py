"""The TFIM coupling sweep of E7c as pipeline arms (R5, R6, R10, R20).

One Transformer per coupling J, trained and scored on the same bucket of
two-qubit systems with that J. The groups of all couplings are drawn one
after another from a generator seeded with the seed, in the order of
`J_VALUES`, as E7c did.
"""
from __future__ import annotations

import math

from src import pipeline

J_VALUES = (0.2, 0.5, 0.8, 1.0, 1.2, 1.5, 2.0)
# TwoQubitSystem hardcodes the transverse field h = 1.
H_FIELD = 1.0
MODEL = "transformer_large"
BATCH_SIZE = 64
LEARNING_RATE = 5e-4
WINDOW_LENGTH = 20


def splitting_ratio(coupling: float) -> float:
    """Returns r(J) = J / sqrt(J^2 + 4 h^2), the parity-splitting ratio."""
    return coupling / math.sqrt(coupling ** 2 + 4 * H_FIELD ** 2)


def arms(
    name: str,
    n_per_j: int,
    epochs: int,
    *,
    criterion: str | None = None,
    interaction: str | None = None,
    simulator: pipeline.SimulatorSpec = pipeline.SimulatorSpec(),
    cache: str = "disk",
    tags: pipeline.Params = (),
) -> list[pipeline.ExperimentConfig]:
    """Declares one arm per coupling.

    Args:
        name: Data name (prefix of the trajectory store's name).
        n_per_j: Trajectories per coupling.
        epochs: Training epochs.
        criterion: Value of `DecoherenceCriterion` that defines T2, or
            None for the scenario default (coherence).
        interaction: Value of `InteractionType`, or None for TFIM.
        simulator: The simulator.
        cache: "disk" or "none" (see `pipeline.DataSpec`).
        tags: Columns added to every row after the coupling `J`.

    Returns:
        The arms in the order of `J_VALUES`.
    """
    return [
        pipeline.ExperimentConfig(
            data=pipeline.DataSpec(
                name=name, coupling=coupling, coupling_sweep=J_VALUES,
                interaction=interaction, n_per_scenario=n_per_j,
                criterion=criterion, simulator=simulator, cache=cache,
                window_length=WINDOW_LENGTH),
            model=MODEL, epochs=epochs, batch_size=BATCH_SIZE,
            learning_rate=LEARNING_RATE,
            tags=(("J", coupling),) + tags)
        for coupling in J_VALUES
    ]
