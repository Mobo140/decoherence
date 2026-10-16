"""BacktestUseCase: censored rows, Hamiltonian codes and J per window."""
from __future__ import annotations

import numpy as np

from src.application.backtest import BacktestCommand, BacktestUseCase
from src.data.dataset import Dataset
from src.domain.value_objects import PredictionResult
from src.models.physics_only import PhysicsOnlyPredictor


class _ConstPredictor(PhysicsOnlyPredictor):
    def predict(self, window, t_obs, horizon):
        return PredictionResult(
            t_decoh_predicted=4.0, risk_score=0.1, horizon=horizon)


def test_backtest_skips_censored_rows():
    n = 6
    T, F = 8, 3
    ds = Dataset(
        sequences=np.ones((n, T, F), dtype=np.float32),
        t_obs=np.full(n, 1.0, dtype=np.float32),
        t_decoh_abs=np.array(
            [4.0, 4.0, 4.0, 20.0, 20.0, 20.0], dtype=np.float32),
        remaining_time=np.array(
            [3.0, 3.0, 3.0, 19.0, 19.0, 19.0], dtype=np.float32),
        risk_labels=np.zeros(n, dtype=np.float32),
        train_idx=np.array([0], dtype=int),
        val_idx=np.array([1], dtype=int),
        test_idx=np.arange(n),
        feature_names=["a", "b", "coherence_l1"],
        metadata={"dt": 0.1, "horizon": 1.0},
        censored=np.array([False, False, False, True, True, True]),
    )
    metrics = BacktestUseCase(_ConstPredictor()).execute(
        BacktestCommand(dataset=ds, horizon=1.0))
    assert metrics.n_samples == 3
    assert metrics.mae == 0.0
    np.testing.assert_array_equal(metrics.actuals, np.array([4.0, 4.0, 4.0]))


class _CodeSpy:
    is_trained = True

    def __init__(self):
        self.inference_interaction_code = 0.0
        self.inference_dissipator_code = 0.0
        self.seen: list[tuple[float, float]] = []

    def predict(self, window, t_obs, horizon):
        self.seen.append(
            (self.inference_interaction_code, self.inference_dissipator_code))
        return PredictionResult(
            t_decoh_predicted=4.0, risk_score=0.1, horizon=horizon)


def test_backtest_sets_hamiltonian_codes_from_dataset():
    n = 3
    ds = Dataset(
        sequences=np.ones((n, 4, 2), dtype=np.float32),
        t_obs=np.full(n, 1.0, dtype=np.float32),
        t_decoh_abs=np.full(n, 4.0, dtype=np.float32),
        remaining_time=np.full(n, 3.0, dtype=np.float32),
        risk_labels=np.zeros(n, dtype=np.float32),
        train_idx=np.array([0], dtype=int),
        val_idx=np.array([1], dtype=int),
        test_idx=np.array([2], dtype=int),
        feature_names=["a", "coherence_l1"],
        metadata={"dt": 0.1, "horizon": 1.0},
        interaction_type_codes=np.array([2.0, 2.0, 2.0], dtype=np.float32),
        dissipator_type_codes=np.array([0.0, 0.0, 0.0], dtype=np.float32),
    )
    spy = _CodeSpy()
    BacktestUseCase(spy).execute(BacktestCommand(dataset=ds, horizon=1.0))
    assert spy.seen == [(2.0, 0.0)]


class _RecordJ(PhysicsOnlyPredictor):
    def __init__(self):
        super().__init__()
        self.seen_J = []
        self.inference_J = 0.0

    def predict(self, window, t_obs, horizon):
        self.seen_J.append(self.inference_J)
        return PredictionResult(
            t_decoh_predicted=4.0, risk_score=0.1, horizon=horizon)


def test_backtest_sets_inference_j_per_window():
    n, T, F = 4, 6, 3
    ds = Dataset(
        sequences=np.ones((n, T, F), dtype=np.float32),
        t_obs=np.full(n, 1.0, dtype=np.float32),
        t_decoh_abs=np.full(n, 4.0, dtype=np.float32),
        remaining_time=np.full(n, 3.0, dtype=np.float32),
        risk_labels=np.zeros(n, dtype=np.float32),
        train_idx=np.array([0], dtype=int),
        val_idx=np.array([1], dtype=int),
        test_idx=np.arange(n),
        feature_names=["a", "b", "coherence_l1"],
        metadata={"dt": 0.1, "horizon": 1.0},
        J_values=np.array([0.2, 0.4, 0.6, 0.8], dtype=np.float32),
    )
    pred = _RecordJ()
    BacktestUseCase(pred).execute(BacktestCommand(dataset=ds, horizon=1.0))
    np.testing.assert_allclose(pred.seen_J, [0.2, 0.4, 0.6, 0.8])
