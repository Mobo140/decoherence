"""Tests for T₂ interpolation, right-censoring, and dataset exclusion."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import qutip as qt

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.application.generate_dataset import GenerateDatasetCommand, GenerateDatasetUseCase
from src.domain.value_objects import (
    DissipatorConfig,
    DissipatorType,
    QubitCount,
    SystemConfig,
    TrajectoryResult,
)
from src.infrastructure.quantum.qutip_simulator import QuTipSimulator


def _plus_state(n_qubits: int = 1, seed=None) -> qt.Qobj:
    """Equal superposition so initial coherence is maximal."""
    dim = 2 ** n_qubits
    ket = qt.Qobj(np.ones((dim, 1), dtype=complex) / np.sqrt(dim))
    return qt.ket2dm(ket)


def _const_sigma_minus(gamma: float, t_max: float, dt: float = 0.1) -> SystemConfig:
    return SystemConfig(
        n_qubits=QubitCount.ONE,
        omega=1.0,
        dissipator=DissipatorConfig.constant(gamma, DissipatorType.SIGMA_MINUS),
        t_max=t_max,
        dt=dt,
    )


class TestInterpolation:
    def test_compute_t_decoh_interpolates_synthetic_exponential(self):
        """Crossing of C0/e on C(t)=C0·exp(-γt/2) must recover T₂=2/γ."""
        gamma = 0.4
        t2 = 2.0 / gamma  # 5.0
        times = np.linspace(0.0, 30.0, 301)  # dt = 0.1
        c0 = 1.0
        coh = c0 * np.exp(-gamma * times / 2.0)
        cfg = _const_sigma_minus(gamma, t_max=30.0)
        t_decoh, censored = QuTipSimulator._compute_t_decoh(
            cfg, times, {"coherence_l1": coh},
        )
        assert censored is False
        assert abs(t_decoh - t2) < 0.01

    def test_simulator_recovers_analytical_t2(self, monkeypatch):
        """γ=0.4, σ₋ → T₂ = 2/γ = 5.0 within 0.05 after interpolation."""
        import src.infrastructure.quantum.qutip_simulator as sim_mod

        monkeypatch.setattr(sim_mod, "sample_random_pure_state", _plus_state)
        traj = QuTipSimulator().simulate(_const_sigma_minus(0.4, t_max=30.0))
        assert traj.censored is False
        assert abs(traj.t_decoh - 5.0) < 0.05


class TestCensoring:
    def test_slow_decay_is_censored(self, monkeypatch):
        """γ=0.01, t_max=5 → T₂=200, threshold never crossed."""
        import src.infrastructure.quantum.qutip_simulator as sim_mod

        monkeypatch.setattr(sim_mod, "sample_random_pure_state", _plus_state)
        traj = QuTipSimulator().simulate(_const_sigma_minus(0.01, t_max=5.0))
        assert traj.censored is True
        assert traj.t_decoh == pytest.approx(5.0, abs=1e-9)


class _FakeSimulator:
    """Returns pre-built trajectories in order, ignoring the config."""

    def __init__(self, trajectories):
        self._trajs = list(trajectories)
        self._i = 0

    def simulate(self, config):
        traj = self._trajs[self._i]
        self._i += 1
        return traj


def _fake_traj(t_decoh: float, censored: bool, t_max: float = 10.0, dt: float = 0.1) -> TrajectoryResult:
    n = int(round(t_max / dt)) + 1
    times = np.linspace(0.0, t_max, n)
    cfg = _const_sigma_minus(0.4, t_max=t_max, dt=dt)
    obs = {
        "sigma_x": np.ones(n, dtype=np.float32),
        "sigma_y": np.zeros(n, dtype=np.float32),
        "sigma_z": np.zeros(n, dtype=np.float32),
        "coherence_l1": np.exp(-times / max(t_decoh, 1e-3)).astype(np.float32),
        "purity": np.ones(n, dtype=np.float32),
    }
    return TrajectoryResult(
        times=times,
        observables=obs,
        t_decoh=t_decoh,
        system_config=cfg,
        censored=censored,
    )


class TestDatasetExcludesCensored:
    def test_one_censored_among_ten(self):
        trajs = [_fake_traj(t_decoh=4.0, censored=False) for _ in range(9)]
        trajs.append(_fake_traj(t_decoh=10.0, censored=True))
        configs = [t.system_config for t in trajs]

        uc = GenerateDatasetUseCase(_FakeSimulator(trajs))
        ds = uc.execute(GenerateDatasetCommand(
            configs=configs,
            window_length=10,
            samples_per_trajectory=2,
            seed=0,
        ))

        assert ds.metadata["n_censored"] == 1
        assert ds.metadata["n_trajectories"] == 10
        # Every remaining window must come from an uncensored trajectory.
        assert np.all(ds.t_decoh_abs < 10.0 - 1e-6)
        assert len(ds.sequences) > 0
        assert ds.metadata["include_censored"] is False
        assert int(ds.censored.sum()) == 0

    def test_include_censored_keeps_windows(self):
        trajs = [_fake_traj(t_decoh=4.0, censored=False) for _ in range(9)]
        trajs.append(_fake_traj(t_decoh=10.0, censored=True))
        configs = [t.system_config for t in trajs]
        uc = GenerateDatasetUseCase(_FakeSimulator(trajs))
        ds = uc.execute(GenerateDatasetCommand(
            configs=configs,
            window_length=10,
            samples_per_trajectory=2,
            seed=0,
            include_censored=True,
        ))
        assert ds.metadata["n_censored"] == 1
        assert ds.metadata["n_censored_windows"] >= 1
        assert bool(ds.censored.any())
