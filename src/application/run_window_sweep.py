"""Use case: window-fraction sweep (experiment E2).

Trains the physics + LSTM predictor at different observation-window
fractions f, e.g. {0.05, 0.10, 0.15, 0.20, 0.30, 0.50}, and records MAE,
R2 and AUROC against f.

The window fraction is f = window_length * dt / median_T2. Since T2
varies per trajectory, the window length for each f is fixed after a
pilot simulation that estimates the median T2.

Example:
    command = WindowSweepCommand(configs=my_configs,
                                 fractions=[0.10, 0.20, 0.30])
    results = WindowSweepUseCase(simulator).execute(command)
"""
from __future__ import annotations

import dataclasses
import random

import numpy as np

from src.application import backtest
from src.application import generate_dataset
from src.application import train_model
from src.domain import ports
from src.domain import value_objects
from src.models import registry


@dataclasses.dataclass(frozen=True)
class WindowSweepCommand:
    """Input of `WindowSweepUseCase`.

    Attributes:
        configs: Systems to simulate.
        fractions: Window fractions to sweep.
        noise: Measurement noise recorded in each result's spec.
        horizon: Risk horizon.
        samples_per_trajectory: Windows drawn from each trajectory.
        n_epochs: Training epochs.
        batch_size: Training batch size.
        lr: Learning rate.
        regression_loss: Loss alias for the remaining-time target.
        seed: Seed of the pilot, dataset and training.
        verbose: Whether to print progress.
        pilot_n: Trajectories simulated to estimate the median T2.
    """

    configs: list[value_objects.SystemConfig]
    fractions: list[float] = dataclasses.field(
        default_factory=lambda: [0.05, 0.10, 0.15, 0.20, 0.30, 0.50])
    noise: value_objects.NoiseConfig = dataclasses.field(
        default_factory=value_objects.NoiseConfig)
    horizon: float = 1.0
    samples_per_trajectory: int = 5
    n_epochs: int = 80
    batch_size: int = 32
    lr: float = 1e-3
    regression_loss: str = "huber"
    seed: int = 42
    verbose: bool = True
    pilot_n: int = 20


class WindowSweepUseCase:
    """Trains physics + LSTM at each window fraction and scores it."""

    def __init__(
        self,
        simulator: ports.ISimulator,
        store: ports.ITrajectoryStore | None = None,
    ) -> None:
        """Wires the use case.

        Args:
            simulator: Simulates the configurations.
            store: Trajectory cache; `None` disables caching.
        """
        self._simulator = simulator
        self._store = store

    def execute(
        self, command: WindowSweepCommand
    ) -> list[value_objects.AblationResult]:
        """Trains and scores one model per window fraction.

        Args:
            command: What to run.

        Returns:
            One result per fraction, named `window_f<f>`; fractions with
            no test samples are skipped.
        """
        median_t2, dt = self._estimate_median_t2(command)
        results = []
        for fraction in command.fractions:
            window_length = max(5, int(round(fraction * median_t2 / dt)))
            if command.verbose:
                print(f"\n=== Window fraction f={fraction:.2f}  "
                      f"window_length={window_length} ===")
            dataset = generate_dataset.GenerateDatasetUseCase(
                self._simulator, store=self._store
            ).execute(generate_dataset.GenerateDatasetCommand(
                configs=command.configs,
                window_length=window_length,
                horizon=command.horizon,
                samples_per_trajectory=command.samples_per_trajectory,
                seed=command.seed,
            ))
            if len(dataset.test_idx) == 0:
                if command.verbose:
                    print(f"  Skipping f={fraction:.2f} — no test samples.")
                continue

            predictor = registry.build_model("physics_lstm")
            train_model.TrainModelUseCase(predictor).execute(
                train_model.TrainModelCommand(
                    dataset=dataset,
                    n_epochs=command.n_epochs,
                    batch_size=command.batch_size,
                    learning_rate=command.lr,
                    regression_loss=command.regression_loss,
                    verbose=command.verbose,
                    seed=command.seed,
                ))
            metrics = backtest.BacktestUseCase(predictor).execute(
                backtest.BacktestCommand(
                    dataset=dataset, horizon=command.horizon))
            spec = value_objects.ExperimentSpec(
                name=f"window_f{fraction:.2f}",
                predictor_variant=value_objects.PredictorVariant.PHYSICS_LSTM,
                window_fraction=fraction,
                noise=command.noise,
                seed=command.seed,
            )
            results.append(value_objects.AblationResult(
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

    def _estimate_median_t2(
        self, command: WindowSweepCommand
    ) -> tuple[float, float]:
        """Simulates a pilot sample to estimate the median T2 and dt."""
        rng = random.Random(command.seed)
        pilot_configs = rng.choices(
            command.configs, k=min(command.pilot_n, len(command.configs)))
        t2_values = []
        dt = None
        for config in pilot_configs:
            trajectory = self._simulator.simulate(config)
            t2_values.append(trajectory.t_decoh)
            if dt is None:
                dt = float(config.dt)
        median_t2 = float(np.median(t2_values)) if t2_values else 5.0
        return median_t2, dt or 0.1
