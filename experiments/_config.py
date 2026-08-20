"""Shared system-config factory for all experiments.

All experiments import `build_configs()` from here to ensure reproducibility
and a single source of truth for the physical parameters.

Scenario taxonomy (matches plan.md):
    A — 1-qubit, σ₋, constant γ          (physics formula exact)
    B — 1-qubit, σ₋, time-dependent γ    (core hard case)
    C — 1-qubit, σ_z, time-dependent γ   (dephasing, different decay form)
    D — 2-qubit XXZ, time-dependent γ    (entanglement + decoherence)
    E — 2-qubit TFIM, time-dependent γ   (different interaction)
"""
from __future__ import annotations

import itertools
from typing import Dict, List

import numpy as np

from src.domain.value_objects import (
    DecoherenceCriterion,
    DissipatorConfig,
    DissipatorType,
    InteractionType,
    QubitCount,
    SystemConfig,
)

# ---------------------------------------------------------------------------
# Bernstein coefficient shapes for diverse γ(t) profiles
# ---------------------------------------------------------------------------

def _bernstein_coeffs(shape: str, rng: np.random.Generator, gamma_max: float = 0.8) -> tuple:
    """Return Bernstein coefficients (degree-4) for named shapes.

    gamma_max controls the upper bound of dissipation rate.  Pass a lower
    value for fast-dephasing (σ_z) channels where T₂ = 1/(2γ) is short.
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
        vals = sorted(rng.uniform(lo_edge, gamma_max, 5).tolist(), reverse=True)
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
    raise ValueError(f"Unknown γ(t) shape: {shape}")


# ---------------------------------------------------------------------------
# Config builders
# ---------------------------------------------------------------------------

def _make_1q_configs(
    n: int,
    dissipator_type: DissipatorType,
    gamma_shapes: List[str],
    rng: np.random.Generator,
) -> List[SystemConfig]:
    # σ_z dephasing: T₂ = 1/(2γ) → cap γ at 0.2 so T₂_min ≥ 2.5 s
    # σ₋ amplitude damping: T₂ ≈ 2/γ → cap γ at 0.8 so T₂_min ≥ 2.5 s
    gamma_max = 0.2 if dissipator_type == DissipatorType.SIGMA_Z else 0.8
    configs = []
    shape_cycle = itertools.cycle(gamma_shapes)
    for _ in range(n):
        shape = next(shape_cycle)
        coeffs = _bernstein_coeffs(shape, rng, gamma_max=gamma_max)
        configs.append(SystemConfig(
            n_qubits=QubitCount.ONE,
            omega=float(rng.uniform(0.5, 2.0)),
            dissipator=DissipatorConfig.time_dependent(coeffs, dissipator_type),
            t_max=20.0,
            dt=0.1,
            decoherence_criterion=DecoherenceCriterion.COHERENCE,
        ))
    return configs


def _make_1q_const_configs(n: int, rng: np.random.Generator) -> List[SystemConfig]:
    configs = []
    for _ in range(n):
        # gamma_min=0.1 → T₂=2/0.1=20 s; use t_max=30 so T₂ is always reached
        gamma = float(rng.uniform(0.1, 0.8))
        configs.append(SystemConfig(
            n_qubits=QubitCount.ONE,
            omega=float(rng.uniform(0.5, 2.0)),
            dissipator=DissipatorConfig.constant(gamma, DissipatorType.SIGMA_MINUS),
            t_max=30.0,
            dt=0.1,
            decoherence_criterion=DecoherenceCriterion.COHERENCE,
        ))
    return configs


def _make_2q_configs(
    n: int,
    interaction: InteractionType,
    rng: np.random.Generator,
) -> List[SystemConfig]:
    configs = []
    shapes = ["random", "peak", "step_up", "step_down", "monotone_increasing"]
    shape_cycle = itertools.cycle(shapes)
    for _ in range(n):
        shape = next(shape_cycle)
        coeffs = _bernstein_coeffs(shape, rng, gamma_max=0.8)
        configs.append(SystemConfig(
            n_qubits=QubitCount.TWO,
            omega=float(rng.uniform(0.5, 2.0)),
            J=float(rng.uniform(0.2, 1.0)),
            dissipator=DissipatorConfig.time_dependent(coeffs, DissipatorType.SIGMA_MINUS),
            interaction_type=interaction,
            t_max=20.0,
            dt=0.1,
            decoherence_criterion=DecoherenceCriterion.COHERENCE,
        ))
    return configs


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def build_configs(
    n_per_scenario: int = 100,
    seed: int = 42,
) -> Dict[str, List[SystemConfig]]:
    """Build all scenario config groups.

    Returns a dict mapping scenario label → list of SystemConfig.
    Keys: 'A', 'B', 'C', 'D', 'E', 'all' (union of A–E).
    """
    rng = np.random.default_rng(seed)
    gamma_shapes = [
        "random", "monotone_increasing", "monotone_decreasing",
        "peak", "step_up", "step_down",
    ]

    groups: Dict[str, List[SystemConfig]] = {
        "A": _make_1q_const_configs(n_per_scenario, rng),
        "B": _make_1q_configs(n_per_scenario, DissipatorType.SIGMA_MINUS, gamma_shapes, rng),
        "C": _make_1q_configs(n_per_scenario, DissipatorType.SIGMA_Z,     gamma_shapes, rng),
        "D": _make_2q_configs(n_per_scenario, InteractionType.XXZ,  rng),
        "E": _make_2q_configs(n_per_scenario, InteractionType.TFIM, rng),
    }
    groups["all"] = [cfg for cfgs in groups.values() for cfg in cfgs]
    return groups


def trajectory_store():
    """Shared content-addressed QuTiP cache under data/trajectories/."""
    from src.infrastructure.persistence.stores import NumpyTrajectoryStore
    return NumpyTrajectoryStore()
