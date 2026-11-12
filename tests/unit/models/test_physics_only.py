"""PhysicsOnlyPredictor: analytical correctness and serialisation."""
from __future__ import annotations

import numpy as np
import pytest

from src.models.physics_only import PhysicsOnlyPredictor


def _exponential_window(gamma: float = 0.5, T: int = 50, dt: float = 0.1):
    """Exact decay C(t) = exp(-gamma t / 2) in the last feature column."""
    t = np.arange(T) * dt
    window = np.zeros((T, 3), dtype=np.float32)
    window[:, -1] = np.exp(-gamma * t / 2)
    return window


def test_physics_only_is_trained_without_fitting():
    assert PhysicsOnlyPredictor().is_trained


def test_predict_exact_decay_gives_remaining_two_over_gamma_minus_t_obs():
    gamma = 0.4
    dt = 0.1
    T = 50
    t_obs = T * dt  # 5.0
    window = _exponential_window(gamma=gamma, T=T, dt=dt)
    p = PhysicsOnlyPredictor(t_max=20.0, dt=dt)
    result = p.predict(window, t_obs=t_obs, horizon=1.0)
    t2_exact = 2.0 / gamma  # = 5.0
    remaining_exact = t2_exact - t_obs  # = 0.0 (at decoherence time)
    # Should predict ~0 remaining (or t_obs ~ T2).
    assert result.t_decoh_predicted == pytest.approx(
        t_obs + max(0, remaining_exact), abs=0.5)


def test_predict_risk_score_is_a_probability():
    window = _exponential_window(gamma=0.5)
    result = PhysicsOnlyPredictor().predict(window, t_obs=3.0, horizon=1.0)
    assert 0.0 <= result.risk_score <= 1.0


def test_predict_risk_grows_with_horizon():
    window = _exponential_window(gamma=0.4, T=30, dt=0.1)
    p = PhysicsOnlyPredictor(t_max=20.0, dt=0.1)
    lo = p.predict(window, t_obs=2.0, horizon=0.2)
    hi = p.predict(window, t_obs=2.0, horizon=8.0)
    assert hi.risk_score > lo.risk_score


def test_train_is_a_noop_that_returns_the_predictor():
    p = PhysicsOnlyPredictor()
    assert p.train(object()) is p


def test_state_bytes_roundtrip_keeps_the_parameters():
    p = PhysicsOnlyPredictor(t_max=15.0, dt=0.05, risk_k=3.0)
    p2 = PhysicsOnlyPredictor.from_bytes(p.state_bytes())
    assert p2.t_max == p.t_max
    assert p2.dt == p.dt
    assert p2.risk_k == p.risk_k
