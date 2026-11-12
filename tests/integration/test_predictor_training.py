"""Predictors trained and backtested through the use cases on QuTiP data.

Simulation, dataset, training and evaluation run together on a handful of
tiny trajectories; the predictors themselves are covered under tests/unit.
"""
from __future__ import annotations

import pytest

from src.application.backtest import BacktestCommand, BacktestUseCase
from src.application.generate_dataset import (
    GenerateDatasetCommand,
    GenerateDatasetUseCase,
)
from src.application.train_model import TrainModelCommand, TrainModelUseCase
from src.data.scenarios import build_configs
from src.models.lstm import LSTMPredictor
from src.models.transformer import TransformerPredictor
from src.simulation.simulator import QuTipSimulator


def _tiny_dataset(scenario: str, seed: int):
    groups = build_configs(n_per_scenario=8, seed=seed)
    gen = GenerateDatasetUseCase(QuTipSimulator())
    return gen.execute(GenerateDatasetCommand(
        configs=groups[scenario][:6],
        window_length=20, horizon=1.0,
        samples_per_trajectory=2, seed=seed,
    ))


@pytest.fixture
def lstm_dataset():
    return _tiny_dataset("B", seed=7)


@pytest.fixture
def transformer_dataset():
    return _tiny_dataset("A", seed=3)


def _fit(predictor, dataset, epochs: int) -> None:
    TrainModelUseCase(predictor).execute(TrainModelCommand(
        dataset=dataset, n_epochs=epochs, verbose=False))


# --- LSTM: use_physics_prior flag --------------------------------------------

def test_lstm_with_physics_prior_trains_and_backtests(lstm_dataset):
    p = LSTMPredictor(use_physics_prior=True)
    _fit(p, lstm_dataset, epochs=2)
    m = BacktestUseCase(p).execute(
        BacktestCommand(dataset=lstm_dataset, horizon=1.0))
    assert m.n_samples > 0


def test_lstm_without_physics_prior_trains_and_backtests(lstm_dataset):
    p = LSTMPredictor(use_physics_prior=False)
    _fit(p, lstm_dataset, epochs=2)
    m = BacktestUseCase(p).execute(
        BacktestCommand(dataset=lstm_dataset, horizon=1.0))
    assert m.n_samples > 0


def test_lstm_state_bytes_roundtrip_preserves_the_physics_prior_flag(
        lstm_dataset):
    p = LSTMPredictor(use_physics_prior=False)
    _fit(p, lstm_dataset, epochs=1)
    p2 = LSTMPredictor.from_bytes(p.state_bytes())
    assert p2.use_physics_prior is False


def test_lstm_predict_risk_grows_with_horizon_and_time_does_not_move(
        lstm_dataset):
    ds = lstm_dataset
    p = LSTMPredictor(use_physics_prior=True, early_stopping_patience=3)
    _fit(p, ds, epochs=2)
    window = ds.sequences[ds.test_idx[0]]
    t_obs = float(ds.t_obs[ds.test_idx[0]])
    lo = p.predict(window, t_obs, horizon=0.05)
    hi = p.predict(window, t_obs, horizon=40.0)
    assert hi.risk_score > lo.risk_score
    assert lo.t_decoh_predicted == pytest.approx(hi.t_decoh_predicted)


# --- Transformer -------------------------------------------------------------

def test_transformer_trains_and_backtests(transformer_dataset):
    ds = transformer_dataset
    p = TransformerPredictor(d_model=16, n_heads=2, n_layers=1, dim_ff=32)
    _fit(p, ds, epochs=2)
    assert p.is_trained
    m = BacktestUseCase(p).execute(BacktestCommand(dataset=ds, horizon=1.0))
    assert m.n_samples > 0


def test_transformer_predict_output_is_in_range(transformer_dataset):
    ds = transformer_dataset
    p = TransformerPredictor(
        d_model=16, n_heads=2, n_layers=1, dim_ff=32, t_max=10.0)
    _fit(p, ds, epochs=1)
    window = ds.sequences[ds.test_idx[0]]
    t_obs = float(ds.t_obs[ds.test_idx[0]])
    result = p.predict(window, t_obs=t_obs, horizon=1.0)
    assert 0.0 <= result.t_decoh_predicted <= 10.0
    assert 0.0 <= result.risk_score <= 1.0


def test_transformer_predict_risk_grows_with_horizon_and_time_does_not_move(
        transformer_dataset):
    ds = transformer_dataset
    p = TransformerPredictor(
        d_model=16, n_heads=2, n_layers=1, dim_ff=32, t_max=10.0)
    _fit(p, ds, epochs=1)
    window = ds.sequences[ds.test_idx[0]]
    t_obs = float(ds.t_obs[ds.test_idx[0]])
    lo = p.predict(window, t_obs, horizon=0.05)
    hi = p.predict(window, t_obs, horizon=40.0)
    assert hi.risk_score > lo.risk_score
    assert lo.t_decoh_predicted == pytest.approx(hi.t_decoh_predicted)
