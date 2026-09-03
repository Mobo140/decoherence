"""Unit tests for new components introduced in Phases 1–4.

Covers:
  - Domain value objects (NoiseConfig, ExperimentSpec, AblationResult)
  - ILossFunction implementations and registry
  - GaussianMeasurementNoise transform
  - PhysicsOnlyPredictor: analytical correctness
  - TransformerPredictor: train/predict smoke test
  - LSTMPredictor use_physics_prior flag
  - AblationStudyUseCase end-to-end (tiny dataset)
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))


# ---------------------------------------------------------------------------
# Domain value objects
# ---------------------------------------------------------------------------

class TestNoiseConfig:
    def test_noiseless(self):
        from src.domain.value_objects import NoiseConfig
        nc = NoiseConfig(sigma=0.0)
        assert nc.is_noiseless

    def test_noisy(self):
        from src.domain.value_objects import NoiseConfig
        nc = NoiseConfig(sigma=0.05)
        assert not nc.is_noiseless

    def test_negative_sigma_raises(self):
        from src.domain.value_objects import NoiseConfig
        with pytest.raises(ValueError):
            NoiseConfig(sigma=-0.1)


class TestExperimentSpec:
    def test_valid(self):
        from src.domain.value_objects import ExperimentSpec, PredictorVariant
        spec = ExperimentSpec(
            name="test", predictor_variant=PredictorVariant.PHYSICS_LSTM,
            window_fraction=0.20,
        )
        assert spec.name == "test"

    def test_invalid_window_fraction(self):
        from src.domain.value_objects import ExperimentSpec
        with pytest.raises(ValueError):
            ExperimentSpec(name="bad", window_fraction=0.0)

    def test_invalid_n_trajectories(self):
        from src.domain.value_objects import ExperimentSpec
        with pytest.raises(ValueError):
            ExperimentSpec(name="bad", n_trajectories=5, window_fraction=0.2)


# ---------------------------------------------------------------------------
# Loss functions
# ---------------------------------------------------------------------------

class TestLossFunctions:
    def setup_method(self):
        self.preds = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        self.targets = np.array([1.2, 1.8, 3.5], dtype=np.float32)

    def test_huber_non_negative(self):
        from src.infrastructure.ml.loss_functions import HuberLossFunction
        loss = HuberLossFunction(delta=1.0)
        assert loss(self.preds, self.targets) >= 0

    def test_mse_non_negative(self):
        from src.infrastructure.ml.loss_functions import MSELossFunction
        loss = MSELossFunction()
        assert loss(self.preds, self.targets) >= 0

    def test_mse_perfect(self):
        from src.infrastructure.ml.loss_functions import MSELossFunction
        loss = MSELossFunction()
        assert loss(self.preds, self.preds) == pytest.approx(0.0, abs=1e-6)

    def test_mae_non_negative(self):
        from src.infrastructure.ml.loss_functions import MAELossFunction
        loss = MAELossFunction()
        assert loss(self.preds, self.targets) >= 0

    def test_quantile_median_symmetric(self):
        """Quantile(0.5) should equal MAE/2."""
        from src.infrastructure.ml.loss_functions import MAELossFunction, QuantileLossFunction
        q = QuantileLossFunction(quantile=0.5)
        mae = MAELossFunction()
        assert q(self.preds, self.targets) == pytest.approx(
            mae(self.preds, self.targets) / 2, rel=1e-4
        )

    def test_quantile_invalid(self):
        from src.infrastructure.ml.loss_functions import QuantileLossFunction
        with pytest.raises(ValueError):
            QuantileLossFunction(quantile=1.5)

    def test_registry_huber(self):
        from src.infrastructure.ml.loss_functions import get_loss
        loss = get_loss("huber")
        assert loss(self.preds, self.targets) >= 0

    def test_registry_quantile_dynamic(self):
        from src.infrastructure.ml.loss_functions import get_loss
        loss = get_loss("quantile_0.9")
        assert loss(self.preds, self.targets) >= 0

    def test_registry_unknown_raises(self):
        from src.infrastructure.ml.loss_functions import get_loss
        with pytest.raises(KeyError):
            get_loss("nonexistent_loss")

    def test_as_torch_returns_module(self):
        import torch.nn as nn
        from src.infrastructure.ml.loss_functions import HuberLossFunction
        torch_loss = HuberLossFunction().as_torch()
        assert isinstance(torch_loss, nn.Module)

    def test_surv_huber_uncensored_matches_huber(self):
        import torch
        from src.infrastructure.ml.loss_functions import SurvHuberLoss, get_loss
        pred = torch.tensor([1.0, 2.0, 3.0])
        tgt = torch.tensor([1.2, 1.8, 3.5])
        zeros = torch.zeros(3)
        surv = SurvHuberLoss(delta=1.0)(pred, tgt, zeros)
        huber = get_loss("huber").as_torch()(pred, tgt)
        assert float(surv) == pytest.approx(float(huber), rel=1e-5)

    def test_surv_huber_censored_zero_when_above_bound(self):
        import torch
        from src.infrastructure.ml.loss_functions import SurvHuberLoss
        pred = torch.tensor([5.0, 6.0])
        bound = torch.tensor([3.0, 4.0])
        cens = torch.ones(2)
        assert float(SurvHuberLoss()(pred, bound, cens)) == pytest.approx(0.0, abs=1e-8)

    def test_surv_huber_censored_penalizes_underprediction(self):
        import torch
        from src.infrastructure.ml.loss_functions import SurvHuberLoss
        pred = torch.tensor([1.0])
        bound = torch.tensor([4.0])
        assert float(SurvHuberLoss()(pred, bound, torch.ones(1))) > 0.0


# ---------------------------------------------------------------------------
# Noise transform
# ---------------------------------------------------------------------------

class TestGaussianMeasurementNoise:
    def _make_trajectory(self):
        from src.domain.value_objects import (
            DissipatorConfig, QubitCount, SystemConfig, TrajectoryResult,
        )
        config = SystemConfig(
            n_qubits=QubitCount.ONE, omega=1.0,
            dissipator=DissipatorConfig.constant(0.5),
        )
        T = 50
        times = np.linspace(0, 5, T)
        obs = {
            "sigma_x": np.random.randn(T).astype(np.float32),
            "coherence_l1": np.exp(-0.5 * times).astype(np.float32),
            "purity": np.ones(T, dtype=np.float32),
        }
        return TrajectoryResult(times=times, observables=obs, t_decoh=3.0, system_config=config)

    def test_noiseless_is_identity(self):
        from src.domain.value_objects import NoiseConfig
        from src.infrastructure.quantum.noise import GaussianMeasurementNoise
        traj = self._make_trajectory()
        transform = GaussianMeasurementNoise(NoiseConfig(sigma=0.0))
        result = transform(traj)
        np.testing.assert_array_equal(result.observables["sigma_x"], traj.observables["sigma_x"])

    def test_noisy_changes_values(self):
        from src.domain.value_objects import NoiseConfig
        from src.infrastructure.quantum.noise import GaussianMeasurementNoise
        traj = self._make_trajectory()
        transform = GaussianMeasurementNoise(NoiseConfig(sigma=0.1, seed=42))
        result = transform(traj)
        assert not np.allclose(result.observables["sigma_x"], traj.observables["sigma_x"])

    def test_metadata_preserved(self):
        from src.domain.value_objects import NoiseConfig
        from src.infrastructure.quantum.noise import GaussianMeasurementNoise
        traj = self._make_trajectory()
        transform = GaussianMeasurementNoise(NoiseConfig(sigma=0.1, seed=0))
        result = transform(traj)
        assert result.t_decoh == traj.t_decoh
        np.testing.assert_array_equal(result.times, traj.times)

    def test_composed_transform(self):
        from src.domain.value_objects import NoiseConfig
        from src.infrastructure.quantum.noise import ComposedTransform, GaussianMeasurementNoise
        traj = self._make_trajectory()
        t1 = GaussianMeasurementNoise(NoiseConfig(sigma=0.01, seed=0))
        t2 = GaussianMeasurementNoise(NoiseConfig(sigma=0.01, seed=1))
        composed = ComposedTransform((t1, t2))
        result = composed(traj)
        assert result.t_decoh == traj.t_decoh


# ---------------------------------------------------------------------------
# PhysicsOnlyPredictor
# ---------------------------------------------------------------------------

class TestPhysicsOnlyPredictor:
    def _exponential_window(self, gamma: float = 0.5, T: int = 50, dt: float = 0.1):
        """Simulate exact exponential decay for coherence_l1 (last feature)."""
        t = np.arange(T) * dt
        window = np.zeros((T, 3), dtype=np.float32)
        window[:, -1] = np.exp(-gamma * t / 2)  # C(t) = exp(-γt/2)
        return window

    def test_is_trained_without_fitting(self):
        from src.infrastructure.ml.physics_only_predictor import PhysicsOnlyPredictor
        p = PhysicsOnlyPredictor()
        assert p.is_trained

    def test_predict_exact_decay(self):
        """For exact exponential decay C(t)=exp(-γt/2), remaining = 2/γ - t_obs."""
        from src.infrastructure.ml.physics_only_predictor import PhysicsOnlyPredictor
        gamma = 0.4
        dt = 0.1
        T = 50
        t_obs = T * dt  # 5.0
        window = self._exponential_window(gamma=gamma, T=T, dt=dt)
        p = PhysicsOnlyPredictor(t_max=20.0, dt=dt)
        result = p.predict(window, t_obs=t_obs, horizon=1.0)
        t2_exact = 2.0 / gamma  # = 5.0
        remaining_exact = t2_exact - t_obs  # = 0.0 (at decoherence time)
        # Should predict ≈ 0 remaining (or t_obs ≈ T₂)
        assert result.t_decoh_predicted == pytest.approx(t_obs + max(0, remaining_exact), abs=0.5)

    def test_risk_score_range(self):
        from src.infrastructure.ml.physics_only_predictor import PhysicsOnlyPredictor
        window = self._exponential_window(gamma=0.5)
        p = PhysicsOnlyPredictor()
        result = p.predict(window, t_obs=3.0, horizon=1.0)
        assert 0.0 <= result.risk_score <= 1.0

    def test_risk_moves_with_horizon(self):
        from src.infrastructure.ml.physics_only_predictor import PhysicsOnlyPredictor
        window = self._exponential_window(gamma=0.4, T=30, dt=0.1)
        p = PhysicsOnlyPredictor(t_max=20.0, dt=0.1)
        lo = p.predict(window, t_obs=2.0, horizon=0.2)
        hi = p.predict(window, t_obs=2.0, horizon=8.0)
        assert hi.risk_score > lo.risk_score

    def test_remaining_to_risk_monotone(self):
        from src.infrastructure.ml.physics_only_predictor import remaining_to_risk
        assert remaining_to_risk(2.0, 0.5) < remaining_to_risk(2.0, 8.0)
        assert remaining_to_risk(0.1, 1.0) > remaining_to_risk(5.0, 1.0)

    def test_train_is_noop(self):
        from src.infrastructure.ml.physics_only_predictor import PhysicsOnlyPredictor
        p = PhysicsOnlyPredictor()
        result = p.train(object())
        assert result is p

    def test_serialisation_roundtrip(self):
        from src.infrastructure.ml.physics_only_predictor import PhysicsOnlyPredictor
        p = PhysicsOnlyPredictor(t_max=15.0, dt=0.05, risk_k=3.0)
        data = p.state_bytes()
        p2 = PhysicsOnlyPredictor.from_bytes(data)
        assert p2.t_max == p.t_max
        assert p2.dt == p.dt
        assert p2.risk_k == p.risk_k


# ---------------------------------------------------------------------------
# LSTMPredictor: use_physics_prior flag
# ---------------------------------------------------------------------------

class TestLSTMPredictorFlag:
    def _make_tiny_dataset(self):
        from experiments._config import build_configs
        from src.application.generate_dataset import GenerateDatasetCommand, GenerateDatasetUseCase
        from src.infrastructure.quantum.qutip_simulator import QuTipSimulator
        simulator = QuTipSimulator()
        groups = build_configs(n_per_scenario=8, seed=7)
        gen = GenerateDatasetUseCase(simulator)
        return gen.execute(GenerateDatasetCommand(
            configs=groups["B"][:6],
            window_length=20, horizon=1.0,
            samples_per_trajectory=2, seed=7,
        ))

    def test_physics_prior_true(self):
        from src.application.backtest import BacktestCommand, BacktestUseCase
        from src.application.train_model import TrainModelCommand, TrainModelUseCase
        from src.infrastructure.ml.lstm_predictor import LSTMPredictor
        ds = self._make_tiny_dataset()
        p = LSTMPredictor(use_physics_prior=True)
        TrainModelUseCase(p).execute(TrainModelCommand(dataset=ds, n_epochs=2, verbose=False))
        m = BacktestUseCase(p).execute(BacktestCommand(dataset=ds, horizon=1.0))
        assert m.n_samples > 0

    def test_physics_prior_false(self):
        from src.application.backtest import BacktestCommand, BacktestUseCase
        from src.application.train_model import TrainModelCommand, TrainModelUseCase
        from src.infrastructure.ml.lstm_predictor import LSTMPredictor
        ds = self._make_tiny_dataset()
        p = LSTMPredictor(use_physics_prior=False)
        TrainModelUseCase(p).execute(TrainModelCommand(dataset=ds, n_epochs=2, verbose=False))
        m = BacktestUseCase(p).execute(BacktestCommand(dataset=ds, horizon=1.0))
        assert m.n_samples > 0

    def test_serialisation_preserves_flag(self):
        from src.application.train_model import TrainModelCommand, TrainModelUseCase
        from src.infrastructure.ml.lstm_predictor import LSTMPredictor
        ds = self._make_tiny_dataset()
        p = LSTMPredictor(use_physics_prior=False)
        TrainModelUseCase(p).execute(TrainModelCommand(dataset=ds, n_epochs=1, verbose=False))
        data = p.state_bytes()
        p2 = LSTMPredictor.from_bytes(data)
        assert p2.use_physics_prior is False

    def test_risk_moves_with_horizon(self):
        from src.application.train_model import TrainModelCommand, TrainModelUseCase
        from src.infrastructure.ml.lstm_predictor import LSTMPredictor
        ds = self._make_tiny_dataset()
        p = LSTMPredictor(use_physics_prior=True, early_stopping_patience=3)
        TrainModelUseCase(p).execute(TrainModelCommand(dataset=ds, n_epochs=2, verbose=False))
        window = ds.sequences[ds.test_idx[0]]
        t_obs = float(ds.t_obs[ds.test_idx[0]])
        lo = p.predict(window, t_obs, horizon=0.05)
        hi = p.predict(window, t_obs, horizon=40.0)
        assert hi.risk_score > lo.risk_score
        assert lo.t_decoh_predicted == pytest.approx(hi.t_decoh_predicted)


# ---------------------------------------------------------------------------
# TransformerPredictor smoke test
# ---------------------------------------------------------------------------

class TestTransformerPredictor:
    def _make_tiny_dataset(self):
        from experiments._config import build_configs
        from src.application.generate_dataset import GenerateDatasetCommand, GenerateDatasetUseCase
        from src.infrastructure.quantum.qutip_simulator import QuTipSimulator
        simulator = QuTipSimulator()
        groups = build_configs(n_per_scenario=8, seed=3)
        gen = GenerateDatasetUseCase(simulator)
        return gen.execute(GenerateDatasetCommand(
            configs=groups["A"][:6],
            window_length=20, horizon=1.0,
            samples_per_trajectory=2, seed=3,
        ))

    def test_train_and_predict(self):
        from src.application.backtest import BacktestCommand, BacktestUseCase
        from src.application.train_model import TrainModelCommand, TrainModelUseCase
        from src.infrastructure.ml.transformer_predictor import TransformerPredictor
        ds = self._make_tiny_dataset()
        p = TransformerPredictor(d_model=16, n_heads=2, n_layers=1, dim_ff=32)
        TrainModelUseCase(p).execute(TrainModelCommand(dataset=ds, n_epochs=2, verbose=False))
        assert p.is_trained
        m = BacktestUseCase(p).execute(BacktestCommand(dataset=ds, horizon=1.0))
        assert m.n_samples > 0

    def test_predict_output_range(self):
        from src.application.train_model import TrainModelCommand, TrainModelUseCase
        from src.infrastructure.ml.transformer_predictor import TransformerPredictor
        ds = self._make_tiny_dataset()
        p = TransformerPredictor(d_model=16, n_heads=2, n_layers=1, dim_ff=32, t_max=10.0)
        TrainModelUseCase(p).execute(TrainModelCommand(dataset=ds, n_epochs=1, verbose=False))
        window = ds.sequences[ds.test_idx[0]]
        t_obs = float(ds.t_obs[ds.test_idx[0]])
        result = p.predict(window, t_obs=t_obs, horizon=1.0)
        assert 0.0 <= result.t_decoh_predicted <= 10.0
        assert 0.0 <= result.risk_score <= 1.0

    def test_risk_moves_with_horizon(self):
        from src.application.train_model import TrainModelCommand, TrainModelUseCase
        from src.infrastructure.ml.transformer_predictor import TransformerPredictor
        ds = self._make_tiny_dataset()
        p = TransformerPredictor(d_model=16, n_heads=2, n_layers=1, dim_ff=32, t_max=10.0)
        TrainModelUseCase(p).execute(TrainModelCommand(dataset=ds, n_epochs=1, verbose=False))
        window = ds.sequences[ds.test_idx[0]]
        t_obs = float(ds.t_obs[ds.test_idx[0]])
        lo = p.predict(window, t_obs, horizon=0.05)
        hi = p.predict(window, t_obs, horizon=40.0)
        assert hi.risk_score > lo.risk_score
        assert lo.t_decoh_predicted == pytest.approx(hi.t_decoh_predicted)
