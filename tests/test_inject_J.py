"""J as a physics scalar: dataset encoding, backtest wiring, LSTM flag."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.application.backtest import BacktestCommand, BacktestUseCase
from src.application.generate_dataset import (
    Dataset,
    GenerateDatasetCommand,
    GenerateDatasetUseCase,
)
from src.domain.value_objects import (
    DissipatorConfig,
    DissipatorType,
    InteractionType,
    PredictionResult,
    QubitCount,
    SystemConfig,
)
from src.infrastructure.ml.lstm_predictor import LSTMPredictor
from src.infrastructure.ml.physics_only_predictor import PhysicsOnlyPredictor
from src.infrastructure.quantum.qutip_simulator import QuTipSimulator


def _1q(gamma: float = 0.4) -> SystemConfig:
    return SystemConfig(
        n_qubits=QubitCount.ONE,
        omega=1.0,
        dissipator=DissipatorConfig.constant(gamma, DissipatorType.SIGMA_MINUS),
        t_max=20.0,
        dt=0.1,
    )


def _2q(J: float, interaction: InteractionType = InteractionType.TFIM) -> SystemConfig:
    return SystemConfig(
        n_qubits=QubitCount.TWO,
        omega=1.0,
        J=J,
        interaction_type=interaction,
        dissipator=DissipatorConfig.constant(0.3, DissipatorType.SIGMA_MINUS),
        t_max=20.0,
        dt=0.1,
    )


class _RecordJ(PhysicsOnlyPredictor):
    def __init__(self):
        super().__init__()
        self.seen_J = []
        self.inference_J = 0.0

    def predict(self, window, t_obs, horizon):
        self.seen_J.append(self.inference_J)
        return PredictionResult(t_decoh_predicted=4.0, risk_score=0.1, horizon=horizon)


def test_dataset_J_is_zero_for_1q_and_config_J_for_2q():
    sim = QuTipSimulator()
    ds = GenerateDatasetUseCase(sim).execute(GenerateDatasetCommand(
        configs=[_1q(), _2q(0.25), _2q(0.9, InteractionType.XXZ)],
        window_length=10,
        samples_per_trajectory=2,
        seed=0,
        min_window_gap=0,
    ))
    assert len(ds.J_values) == len(ds.sequences)
    ones = ds.interaction_type_codes < 1.5
    twos = ds.interaction_type_codes >= 1.5
    assert ones.any() and twos.any()
    assert np.allclose(ds.J_values[ones], 0.0)
    assert np.all((ds.J_values[twos] == 0.25) | (ds.J_values[twos] == 0.9))


def test_backtest_sets_inference_J_per_window():
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


def test_lstm_default_has_seven_physics_scalars():
    p = LSTMPredictor()
    assert p.use_J_scalar is False


def test_lstm_use_J_roundtrip_from_bytes():
    sim = QuTipSimulator()
    ds = GenerateDatasetUseCase(sim).execute(GenerateDatasetCommand(
        configs=[_2q(0.3), _2q(0.8)],
        window_length=8,
        samples_per_trajectory=3,
        seed=1,
        min_window_gap=0,
    ))
    p = LSTMPredictor(
        hidden_size=16, num_layers=1, dropout=0.0,
        use_J_scalar=True, early_stopping_patience=3,
    )
    p.train(ds, n_epochs=2, batch_size=8, lr=1e-3, verbose=False)
    assert p._net.n_physics == 8
    p2 = LSTMPredictor.from_bytes(p.state_bytes())
    assert p2.use_J_scalar is True
    assert p2._net.n_physics == 8
    out = p2.predict(ds.sequences[0], float(ds.t_obs[0]), 1.0)
    assert out.t_decoh_predicted >= 0.0


def test_lstm_without_J_from_bytes_stays_seven_scalars():
    sim = QuTipSimulator()
    ds = GenerateDatasetUseCase(sim).execute(GenerateDatasetCommand(
        configs=[_1q(0.30), _1q(0.40), _1q(0.50)],
        window_length=8,
        samples_per_trajectory=3,
        seed=2,
        min_window_gap=0,
    ))
    p = LSTMPredictor(hidden_size=16, num_layers=1, dropout=0.0, early_stopping_patience=3)
    p.train(ds, n_epochs=2, batch_size=8, lr=1e-3, verbose=False)
    assert p._net.n_physics == 7
    p2 = LSTMPredictor.from_bytes(p.state_bytes())
    assert p2.use_J_scalar is False
    assert p2._net.n_physics == 7
