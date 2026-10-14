"""Use case: measurement-noise robustness sweep (experiment E3).

Trains physics + LSTM on clean data, then evaluates it on test sets with
increasing Gaussian noise, e.g. sigma in {0, 0.01, 0.02, 0.05, 0.10}.
This tests the claim that AUROC stays at 0.90 or above at realistic
noise levels.

Example:
    command = NoiseSweepCommand(configs=my_configs, sigmas=[0.0, 0.05])
    results = NoiseSweepUseCase(simulator).execute(command)
"""
from __future__ import annotations

import dataclasses
from collections.abc import Callable

import numpy as np

from src.application import backtest
from src.application import generate_dataset
from src.application import train_model
from src.data import observation
from src.domain import ports
from src.domain import value_objects
from src.models import registry

# Builds the untrained predictor to fit on the clean data.
PredictorFactory = Callable[[], ports.IPredictor]


@dataclasses.dataclass(frozen=True)
class NoiseSweepCommand:
    """Input of `NoiseSweepUseCase`.

    Attributes:
        configs: Systems to simulate; one clean dataset is built.
        sigmas: Noise standard deviations to test at.
        window_length: Observation window in time steps.
        horizon: Risk horizon.
        samples_per_trajectory: Windows drawn from each trajectory.
        n_epochs: Training epochs.
        batch_size: Training batch size.
        lr: Learning rate.
        regression_loss: Loss alias for the remaining-time target.
        seed: Seed of the dataset, training and the noise.
        verbose: Whether to print progress.
        min_window_gap: Minimum gap between window ends, in time steps.
    """

    configs: list[value_objects.SystemConfig]
    sigmas: list[float] = dataclasses.field(
        default_factory=lambda: [0.0, 0.01, 0.02, 0.05, 0.10])
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


def default_predictor() -> ports.IPredictor:
    """Builds the default physics + LSTM predictor."""
    return registry.build_model("physics_lstm")


class NoiseSweepUseCase:
    """Trains on clean data and evaluates at each noise level."""

    def __init__(
        self,
        simulator: ports.ISimulator,
        store: ports.ITrajectoryStore | None = None,
        predictor_factory: PredictorFactory = default_predictor,
    ) -> None:
        """Wires the use case.

        Args:
            simulator: Simulates the configurations.
            store: Trajectory cache; `None` disables caching.
            predictor_factory: Builds the predictor to train; the default
                is the registry's physics + LSTM model.
        """
        self._simulator = simulator
        self._store = store
        self._predictor_factory = predictor_factory

    def execute(
        self, command: NoiseSweepCommand
    ) -> list[value_objects.AblationResult]:
        """Trains on clean data and scores each noise level.

        Args:
            command: What to run.

        Returns:
            One result per sigma, named `noise_sigma<sigma>`.
        """
        clean_dataset = generate_dataset.GenerateDatasetUseCase(
            self._simulator, store=self._store
        ).execute(generate_dataset.GenerateDatasetCommand(
            configs=command.configs,
            window_length=command.window_length,
            horizon=command.horizon,
            samples_per_trajectory=command.samples_per_trajectory,
            seed=command.seed,
            min_window_gap=command.min_window_gap,
        ))

        predictor = self._predictor_factory()
        train_model.TrainModelUseCase(predictor).execute(
            train_model.TrainModelCommand(
                dataset=clean_dataset,
                n_epochs=command.n_epochs,
                batch_size=command.batch_size,
                learning_rate=command.lr,
                regression_loss=command.regression_loss,
                verbose=command.verbose,
                seed=command.seed,
            ))

        results = []
        for sigma in command.sigmas:
            if command.verbose:
                print(f"\n=== Noise σ={sigma:.3f} ===")
            noisy_dataset = observation.gaussian(
                clean_dataset, sigma=sigma,
                rng=np.random.default_rng(command.seed))
            metrics = backtest.BacktestUseCase(predictor).execute(
                backtest.BacktestCommand(
                    dataset=noisy_dataset, horizon=command.horizon))
            spec = value_objects.ExperimentSpec(
                name=f"noise_sigma{sigma:.3f}",
                predictor_variant=value_objects.PredictorVariant.PHYSICS_LSTM,
                noise=value_objects.NoiseConfig(
                    sigma=sigma, seed=command.seed),
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
