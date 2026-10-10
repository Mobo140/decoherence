"""Scenario definitions and system-configuration generators.

Every experiment builds its systems here, so the physical parameters have a
single source of truth.

Scenario taxonomy (see paper/description_ru.tex):
    A: 1-qubit, sigma_-, constant gamma (the physics formula is exact).
    B: 1-qubit, sigma_-, time-dependent gamma (the core hard case).
    C: 1-qubit, sigma_z, time-dependent gamma (dephasing, other decay form).
    D: 2-qubit XXZ, time-dependent gamma (entanglement and decoherence).
    E: 2-qubit TFIM, time-dependent gamma (a different interaction).
"""
from __future__ import annotations

import dataclasses
import itertools

import numpy as np

from src.domain import ports
from src.domain import value_objects
from src.persistence import stores

SCENARIOS = {
    "A": "1-qubit, sigma_-, constant gamma",
    "B": "1-qubit, sigma_-, time-dependent gamma",
    "C": "1-qubit, sigma_z, time-dependent gamma",
    "D": "2-qubit XXZ, time-dependent gamma",
    "E": "2-qubit TFIM, time-dependent gamma",
}

# Shapes of gamma(t) cycled through by the time-dependent scenarios.
_GAMMA_SHAPES_1Q = (
    "random", "monotone_increasing", "monotone_decreasing",
    "peak", "step_up", "step_down",
)
_GAMMA_SHAPES_2Q = (
    "random", "peak", "step_up", "step_down", "monotone_increasing",
)
_GAMMA_SHAPES_FIXED_J = _GAMMA_SHAPES_2Q + ("monotone_decreasing",)


def bernstein_coeffs(
    shape: str,
    rng: np.random.Generator,
    gamma_max: float = 0.8,
) -> tuple[float, ...]:
    """Draws degree-4 Bernstein coefficients of a named gamma(t) shape.

    Args:
        shape: One of "constant", "monotone_increasing",
            "monotone_decreasing", "peak", "step_up", "step_down",
            "random".
        rng: Source of randomness; the number of draws depends on `shape`.
        gamma_max: Upper bound of the dissipation rate. Use a lower value
            for fast-dephasing (sigma_z) channels, where T2 = 1 / (2 gamma)
            is short.

    Returns:
        Five coefficients of the Bernstein polynomial.

    Raises:
        ValueError: If `shape` is unknown.
    """
    lo_edge = 0.05
    hi_edge = min(0.25, gamma_max * 0.4)
    if shape == "constant":
        g = float(rng.uniform(lo_edge, gamma_max))
        return (g, g, g, g, g)
    if shape == "monotone_increasing":
        vals = sorted(rng.uniform(lo_edge, gamma_max, 5).tolist())
        return tuple(vals)
    if shape == "monotone_decreasing":
        vals = sorted(rng.uniform(lo_edge, gamma_max, 5).tolist(),
                      reverse=True)
        return tuple(vals)
    if shape == "peak":
        mid = float(rng.uniform(gamma_max * 0.4, gamma_max))
        edge = float(rng.uniform(lo_edge, hi_edge))
        return (edge, mid * 0.6, mid, mid * 0.6, edge)
    if shape == "step_up":
        lo = float(rng.uniform(lo_edge, hi_edge))
        hi = float(rng.uniform(gamma_max * 0.5, gamma_max))
        return (lo, lo, (lo + hi) / 2, hi, hi)
    if shape == "step_down":
        lo = float(rng.uniform(lo_edge, hi_edge))
        hi = float(rng.uniform(gamma_max * 0.5, gamma_max))
        return (hi, hi, (lo + hi) / 2, lo, lo)
    if shape == "random":
        return tuple(rng.uniform(lo_edge, gamma_max, 5).tolist())
    raise ValueError(f"Unknown gamma(t) shape: {shape}")


def _make_1q_configs(
    n: int,
    dissipator_type: value_objects.DissipatorType,
    gamma_shapes: tuple[str, ...],
    rng: np.random.Generator,
) -> list[value_objects.SystemConfig]:
    """Scenarios B and C: one qubit with time-dependent gamma."""
    # sigma_z dephasing: T2 = 1/(2 gamma), so gamma <= 0.2 keeps T2 >= 2.5 s.
    # sigma_- damping: T2 ~ 2/gamma, so gamma <= 0.8 keeps T2 >= 2.5 s.
    if dissipator_type == value_objects.DissipatorType.SIGMA_Z:
        gamma_max = 0.2
    else:
        gamma_max = 0.8
    configs = []
    shape_cycle = itertools.cycle(gamma_shapes)
    for _ in range(n):
        shape = next(shape_cycle)
        coeffs = bernstein_coeffs(shape, rng, gamma_max=gamma_max)
        configs.append(value_objects.SystemConfig(
            n_qubits=value_objects.QubitCount.ONE,
            omega=float(rng.uniform(0.5, 2.0)),
            dissipator=value_objects.DissipatorConfig.time_dependent(
                coeffs, dissipator_type),
            t_max=20.0,
            dt=0.1,
            decoherence_criterion=value_objects.DecoherenceCriterion.COHERENCE,
        ))
    return configs


def _make_1q_const_configs(
    n: int,
    rng: np.random.Generator,
) -> list[value_objects.SystemConfig]:
    """Scenario A: one qubit with constant gamma."""
    configs = []
    for _ in range(n):
        # gamma >= 0.1 gives T2 = 2/gamma <= 20 s; t_max = 30 s is always
        # long enough to reach T2.
        gamma = float(rng.uniform(0.1, 0.8))
        configs.append(value_objects.SystemConfig(
            n_qubits=value_objects.QubitCount.ONE,
            omega=float(rng.uniform(0.5, 2.0)),
            dissipator=value_objects.DissipatorConfig.constant(
                gamma, value_objects.DissipatorType.SIGMA_MINUS),
            t_max=30.0,
            dt=0.1,
            decoherence_criterion=value_objects.DecoherenceCriterion.COHERENCE,
        ))
    return configs


def _make_2q_configs(
    n: int,
    interaction: value_objects.InteractionType,
    rng: np.random.Generator,
) -> list[value_objects.SystemConfig]:
    """Scenarios D and E: two qubits with random coupling J."""
    configs = []
    shape_cycle = itertools.cycle(_GAMMA_SHAPES_2Q)
    for _ in range(n):
        shape = next(shape_cycle)
        coeffs = bernstein_coeffs(shape, rng, gamma_max=0.8)
        configs.append(value_objects.SystemConfig(
            n_qubits=value_objects.QubitCount.TWO,
            omega=float(rng.uniform(0.5, 2.0)),
            J=float(rng.uniform(0.2, 1.0)),
            dissipator=value_objects.DissipatorConfig.time_dependent(
                coeffs, value_objects.DissipatorType.SIGMA_MINUS),
            interaction_type=interaction,
            t_max=20.0,
            dt=0.1,
            decoherence_criterion=value_objects.DecoherenceCriterion.COHERENCE,
        ))
    return configs


def build_configs(
    n_per_scenario: int = 100,
    seed: int = 42,
) -> dict[str, list[value_objects.SystemConfig]]:
    """Builds the configuration groups of all scenarios.

    Args:
        n_per_scenario: Number of configurations in each scenario.
        seed: Seed of the generator that draws every parameter.

    Returns:
        A dict from scenario label to its configurations. The keys are the
        labels of `SCENARIOS` and "all", the concatenation of A to E.
    """
    rng = np.random.default_rng(seed)
    groups: dict[str, list[value_objects.SystemConfig]] = {
        "A": _make_1q_const_configs(n_per_scenario, rng),
        "B": _make_1q_configs(
            n_per_scenario, value_objects.DissipatorType.SIGMA_MINUS,
            _GAMMA_SHAPES_1Q, rng),
        "C": _make_1q_configs(
            n_per_scenario, value_objects.DissipatorType.SIGMA_Z,
            _GAMMA_SHAPES_1Q, rng),
        "D": _make_2q_configs(
            n_per_scenario, value_objects.InteractionType.XXZ, rng),
        "E": _make_2q_configs(
            n_per_scenario, value_objects.InteractionType.TFIM, rng),
    }
    groups["all"] = [cfg for cfgs in groups.values() for cfg in cfgs]
    return groups


def fixed_coupling_configs(
    n: int,
    coupling: float,
    rng: np.random.Generator,
    criterion: value_objects.DecoherenceCriterion = (
        value_objects.DecoherenceCriterion.COHERENCE),
    interaction: value_objects.InteractionType = (
        value_objects.InteractionType.TFIM),
) -> list[value_objects.SystemConfig]:
    """Builds two-qubit configurations that all share the coupling J.

    Args:
        n: Number of configurations.
        coupling: The coupling J of every configuration.
        rng: Source of randomness for omega and gamma(t).
        criterion: Decoherence criterion of the configurations.
        interaction: Interaction type (TFIM by default).

    Returns:
        `n` configurations, cycling through six gamma(t) shapes.
    """
    configs = []
    shape_cycle = itertools.cycle(_GAMMA_SHAPES_FIXED_J)
    for _ in range(n):
        shape = next(shape_cycle)
        coeffs = bernstein_coeffs(shape, rng, gamma_max=0.8)
        configs.append(value_objects.SystemConfig(
            n_qubits=value_objects.QubitCount.TWO,
            omega=float(rng.uniform(0.5, 2.0)),
            J=coupling,
            dissipator=value_objects.DissipatorConfig.time_dependent(
                coeffs, value_objects.DissipatorType.SIGMA_MINUS),
            interaction_type=interaction,
            t_max=20.0,
            dt=0.1,
            decoherence_criterion=criterion,
        ))
    return configs


def with_criterion(
    configs: list[value_objects.SystemConfig],
    criterion: value_objects.DecoherenceCriterion,
) -> list[value_objects.SystemConfig]:
    """Gives every configuration another decoherence criterion.

    Args:
        configs: Configurations, e.g. one group of `build_configs`.
        criterion: The criterion that defines T2 (the prediction target).

    Returns:
        Copies of the configurations with that criterion.
    """
    return [dataclasses.replace(c, decoherence_criterion=criterion)
            for c in configs]


def trajectory_store() -> ports.ITrajectoryStore:
    """Returns the shared content-addressed cache under data/trajectories/."""
    return stores.NumpyTrajectoryStore()
