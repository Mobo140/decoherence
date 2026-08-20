"""Use-case: measurement noise robustness sweep (Experiment E3 from plan.md).

Trains physics+LSTM on clean data, then evaluates on test sets with
increasing Gaussian noise σ ∈ {0, 0.01, 0.02, 0.05, 0.10}.

This tests claim C4: "AUROC remains ≥0.90 at realistic noise levels."

Usage::
    cmd = NoiseSweepCommand(configs=my_configs, sigmas=[0.0, 0.02, 0.05])
    results = NoiseSweepUseCase(simulator).execute(cmd)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

import numpy as np

from ..domain.ports import ISimulator
from ..domain.value_objects import (
    AblationResult,
    ExperimentSpec,
    NoiseConfig,
    PredictorVariant,
    SystemConfig,
)
from .backtest import BacktestCommand, BacktestUseCase
from .generate_dataset import GenerateDatasetCommand, GenerateDatasetUseCase
from .train_model import TrainModelCommand, TrainModelUseCase


@dataclass(frozen=True)
class NoiseSweepCommand:
    """Input for NoiseSweepUseCase."""

    configs: List[SystemConfig]
    sigmas: List[float] = field(
        default_factory=lambda: [0.0, 0.01, 0.02, 0.05, 0.10]
    )
    window_length: int = 50
    horizon: float = 1.0
    samples_per_trajectory: int = 5
    n_epochs: int = 80
    batch_size: int = 32
    lr: float = 1e-3
    regression_loss: str = "huber"
    seed: int = 42
    verbose: bool = True
    min_window_gap: int = 0


class NoiseSweepUseCase:
    """Run E3: train on clean data, evaluate at each noise level."""

    def __init__(self, simulator: ISimulator, store=None) -> None:
        self._simulator = simulator
        self._store = store

    def execute(self, command: NoiseSweepCommand) -> List[AblationResult]:
        # Generate clean dataset once
        gen_uc = GenerateDatasetUseCase(self._simulator, store=self._store)
        clean_dataset = gen_uc.execute(GenerateDatasetCommand(
            configs=command.configs,
            window_length=command.window_length,
            horizon=command.horizon,
            samples_per_trajectory=command.samples_per_trajectory,
            seed=command.seed,
            min_window_gap=command.min_window_gap,
        ))

        # Train on clean data
        predictor = self._build_predictor()
        train_uc = TrainModelUseCase(predictor)
        train_uc.execute(TrainModelCommand(
            dataset=clean_dataset,
            n_epochs=command.n_epochs,
            batch_size=command.batch_size,
            learning_rate=command.lr,
            regression_loss=command.regression_loss,
            verbose=command.verbose,
        ))

        results: List[AblationResult] = []

        for sigma in command.sigmas:
            if command.verbose:
                print(f"\n=== Noise σ={sigma:.3f} ===")

            noisy_dataset = self._add_noise(clean_dataset, sigma, seed=command.seed)

            metrics = BacktestUseCase(predictor).execute(
                BacktestCommand(dataset=noisy_dataset, horizon=command.horizon)
            )

            spec = ExperimentSpec(
                name=f"noise_sigma{sigma:.3f}",
                predictor_variant=PredictorVariant.PHYSICS_LSTM,
                noise=NoiseConfig(sigma=sigma, seed=command.seed),
                seed=command.seed,
            )
            results.append(AblationResult(
                spec=spec,
                mae=metrics.mae,
                rmse=metrics.rmse,
                r2=metrics.r2,
                mape=metrics.mape,
                risk_auroc=metrics.risk_auroc,
                n_samples=metrics.n_samples,
            ))
            if command.verbose:
                print(f"  → {metrics.summary()}")

        return results

    # ------------------------------------------------------------------
    # Private
    # ------------------------------------------------------------------

    def _build_predictor(self):
        """Factory for the predictor used in training. Override to customise."""
        from ..infrastructure.ml.lstm_predictor import LSTMPredictor
        return LSTMPredictor(use_physics_prior=True)

    @staticmethod
    def _add_noise(dataset, sigma: float, seed: int):
        """Return a copy of dataset with Gaussian noise added to sequences."""
        import copy
        if sigma == 0.0:
            return dataset
        rng = np.random.default_rng(seed)
        noisy_seqs = dataset.sequences + rng.normal(
            0.0, sigma, size=dataset.sequences.shape
        ).astype(np.float32)
        ds = copy.copy(dataset)
        ds.sequences = noisy_seqs
        return ds
