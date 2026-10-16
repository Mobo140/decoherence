"""QuTipSimulator and its decoherence-time logic.

Every test runs at most two tiny simulations; datasets and training are
covered under tests/integration.
"""
from __future__ import annotations

import dataclasses

import numpy as np
import pytest
import qutip as qt

from src.data.scenarios import build_configs
from src.domain.value_objects import (
    DecoherenceCriterion,
    DissipatorConfig,
    DissipatorType,
    QubitCount,
    SystemConfig,
)
from src.physics import lindblad_systems
from src.simulation import simulator
from src.simulation.simulator import QuTipSimulator, compute_t_decoh


def _plus_state(n_qubits: int = 1, seed=None) -> qt.Qobj:
    """Equal superposition so initial coherence is maximal."""
    dim = 2 ** n_qubits
    ket = qt.Qobj(np.ones((dim, 1), dtype=complex) / np.sqrt(dim))
    return qt.ket2dm(ket)


def _const_sigma_minus(
    gamma: float, t_max: float, dt: float = 0.1
) -> SystemConfig:
    return SystemConfig(
        n_qubits=QubitCount.ONE,
        omega=1.0,
        dissipator=DissipatorConfig.constant(gamma, DissipatorType.SIGMA_MINUS),
        t_max=t_max,
        dt=dt,
    )


def test_compute_t_decoh_interpolates_synthetic_exponential():
    """Crossing of C0/e on C(t)=C0·exp(-γt/2) must recover T₂=2/γ."""
    gamma = 0.4
    t2 = 2.0 / gamma  # 5.0
    times = np.linspace(0.0, 30.0, 301)  # dt = 0.1
    c0 = 1.0
    coh = c0 * np.exp(-gamma * times / 2.0)
    cfg = _const_sigma_minus(gamma, t_max=30.0)
    t_decoh, censored = compute_t_decoh(
        cfg, times, {"coherence_l1": coh},
    )
    assert censored is False
    assert abs(t_decoh - t2) < 0.01


def test_simulate_recovers_the_analytical_t2(monkeypatch):
    """γ=0.4, σ₋ → T₂ = 2/γ = 5.0 within 0.05 after interpolation."""
    monkeypatch.setattr(
        lindblad_systems, "sample_random_pure_state", _plus_state
    )
    traj = QuTipSimulator().simulate(_const_sigma_minus(0.4, t_max=30.0))
    assert traj.censored is False
    assert abs(traj.t_decoh - 5.0) < 0.05


def test_simulate_slow_decay_is_censored_at_t_max(monkeypatch):
    """γ=0.01, t_max=5 → T₂=200, threshold never crossed."""
    monkeypatch.setattr(
        lindblad_systems, "sample_random_pure_state", _plus_state
    )
    traj = QuTipSimulator().simulate(_const_sigma_minus(0.01, t_max=5.0))
    assert traj.censored is True
    assert traj.t_decoh == pytest.approx(5.0, abs=1e-9)


def test_sudden_death_time_interpolates_the_zero_of_lambda():
    t = np.linspace(0.0, 1.0, 11)
    lam = 0.5 - t                                    # zero at t = 0.5
    assert abs(simulator.sudden_death_time(t, lam)[0] - 0.5) < 1e-12
    # Separable.
    assert simulator.sudden_death_time(t, -t - 0.1) == (0.0, False)
    # Censored.
    assert simulator.sudden_death_time(t, 1.0 + t) == (1.0, True)


def test_simulate_esd_criterion_changes_only_the_target():
    cfg = dataclasses.replace(
        build_configs(n_per_scenario=2, seed=3)["E"][0],
        decoherence_criterion=DecoherenceCriterion.ENTANGLEMENT_DEATH)
    np.random.seed(11)
    traj = simulator.QuTipSimulator().simulate(cfg)
    np.random.seed(11)
    plain = simulator.QuTipSimulator().simulate(dataclasses.replace(
        cfg, decoherence_criterion=DecoherenceCriterion.COHERENCE))
    # Same dynamics and observables; only the target differs.
    np.testing.assert_array_equal(
        traj.observables["coherence_l1"], plain.observables["coherence_l1"])
    assert traj.t_decoh != plain.t_decoh
