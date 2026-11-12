"""J as a physics scalar: dataset encoding from real simulations, LSTM flag.

The per-window wiring in the backtest is covered in
tests/unit/application/test_backtest.py.
"""
from __future__ import annotations

import numpy as np

from src.application.generate_dataset import (
    GenerateDatasetCommand,
    GenerateDatasetUseCase,
)
from src.domain.value_objects import (
    DissipatorConfig,
    DissipatorType,
    InteractionType,
    QubitCount,
    SystemConfig,
)
from src.models.lstm import LSTMPredictor
from src.simulation.simulator import QuTipSimulator


def _1q(gamma: float = 0.4) -> SystemConfig:
    return SystemConfig(
        n_qubits=QubitCount.ONE,
        omega=1.0,
        dissipator=DissipatorConfig.constant(gamma, DissipatorType.SIGMA_MINUS),
        t_max=20.0,
        dt=0.1,
    )


def _2q(
    J: float, interaction: InteractionType = InteractionType.TFIM
) -> SystemConfig:
    return SystemConfig(
        n_qubits=QubitCount.TWO,
        omega=1.0,
        J=J,
        interaction_type=interaction,
        dissipator=DissipatorConfig.constant(0.3, DissipatorType.SIGMA_MINUS),
        t_max=20.0,
        dt=0.1,
    )


def test_execute_dataset_j_is_zero_for_1q_and_config_j_for_2q():
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


def test_lstm_with_j_scalar_trains_and_roundtrips_through_bytes():
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
    assert p.net.n_physics == 8
    p2 = LSTMPredictor.from_bytes(p.state_bytes())
    assert p2.use_J_scalar is True
    assert p2.net.n_physics == 8
    out = p2.predict(ds.sequences[0], float(ds.t_obs[0]), 1.0)
    assert out.t_decoh_predicted >= 0.0


def test_lstm_without_j_scalar_roundtrips_with_seven_scalars():
    sim = QuTipSimulator()
    ds = GenerateDatasetUseCase(sim).execute(GenerateDatasetCommand(
        configs=[_1q(0.30), _1q(0.40), _1q(0.50)],
        window_length=8,
        samples_per_trajectory=3,
        seed=2,
        min_window_gap=0,
    ))
    p = LSTMPredictor(
        hidden_size=16, num_layers=1, dropout=0.0, early_stopping_patience=3)
    p.train(ds, n_epochs=2, batch_size=8, lr=1e-3, verbose=False)
    assert p.net.n_physics == 7
    p2 = LSTMPredictor.from_bytes(p.state_bytes())
    assert p2.use_J_scalar is False
    assert p2.net.n_physics == 7
