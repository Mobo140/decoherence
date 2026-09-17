"""Use-case: window fraction sweep (Experiment E2 from plan.md).

Trains the physics+LSTM predictor at different observation-window fractions
f ∈ {0.05, 0.10, 0.15, 0.20, 0.30, 0.50} and records MAE / R² / AUROC vs f.

The window fraction is defined as:
    f = window_length * dt / median_T₂

Since T₂ varies per trajectory, we control f indirectly by fixing window_length
and varying it.  A cleaner approach (used here) is to determine the window_length
for each f *after* a pilot simulation that estimates the distribution of T₂.

Usage::
    cmd = WindowSweepCommand(configs=my_configs, fractions=[0.10, 0.20, 0.30])
    results = WindowSweepUseCase(simulator).execute(cmd)   # List[AblationResult]
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
class WindowSweepCommand:
    """Input for WindowSweepUseCase."""

    configs: List[SystemConfig]
    fractions: List[float] = field(
        default_factory=lambda: [0.05, 0.10, 0.15, 0.20, 0.30, 0.50]
    )
    noise: NoiseConfig = field(default_factory=NoiseConfig)
    horizon: float = 1.0
    samples_per_trajectory: int = 5
    n_epochs: int = 80
    batch_size: int = 32
    lr: float = 1e-3
    regression_loss: str = "huber"
    seed: int = 42
    verbose: bool = True
    # Pilot simulation: estimate median T₂ to translate fractions → window_length
    pilot_n: int = 20


class WindowSweepUseCase:
    """Run E2: train physics+LSTM at each window fraction, evaluate, return results."""

    def __init__(self, simulator: ISimulator, store=None) -> None:
        self._simulator = simulator
        self._store = store

    def execute(self, command: WindowSweepCommand) -> List[AblationResult]:
        median_t2, dt = self._estimate_median_t2(command)
        results: List[AblationResult] = []

        for f in command.fractions:
            window_length = max(5, int(round(f * median_t2 / dt)))
            if command.verbose:
                print(f"\n=== Window fraction f={f:.2f}  window_length={window_length} ===")

            gen_uc = GenerateDatasetUseCase(self._simulator, store=self._store)
            dataset = gen_uc.execute(GenerateDatasetCommand(
                configs=command.configs,
                window_length=window_length,
                horizon=command.horizon,
                samples_per_trajectory=command.samples_per_trajectory,
                seed=command.seed,
            ))

            if len(dataset.test_idx) == 0:
                if command.verbose:
                    print(f"  Skipping f={f:.2f} — no test samples.")
                continue

            from ..infrastructure.ml.lstm_predictor import LSTMPredictor
            predictor = LSTMPredictor(use_physics_prior=True)

            train_uc = TrainModelUseCase(predictor)
            train_uc.execute(TrainModelCommand(
                dataset=dataset,
                n_epochs=command.n_epochs,
                batch_size=command.batch_size,
                learning_rate=command.lr,
                regression_loss=command.regression_loss,
                verbose=command.verbose,
                seed=command.seed,
            ))

            metrics = BacktestUseCase(predictor).execute(
                BacktestCommand(dataset=dataset, horizon=command.horizon)
            )

            spec = ExperimentSpec(
                name=f"window_f{f:.2f}",
                predictor_variant=PredictorVariant.PHYSICS_LSTM,
                window_fraction=f,
                noise=command.noise,
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

    def _estimate_median_t2(self, command: WindowSweepCommand):
        """Run a small pilot simulation to estimate median T₂ and dt."""
        import random
        rng = random.Random(command.seed)
        pilot_configs = rng.choices(command.configs, k=min(command.pilot_n, len(command.configs)))
        t2_vals = []
        dt = None
        for cfg in pilot_configs:
            traj = self._simulator.simulate(cfg)
            t2_vals.append(traj.t_decoh)
            if dt is None:
                dt = float(cfg.dt)
        median_t2 = float(np.median(t2_vals)) if t2_vals else 5.0
        return median_t2, dt or 0.1
