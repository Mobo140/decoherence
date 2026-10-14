"""Use case: ablation study, comparing predictor variants on one dataset.

Experiment E1: compare physics_only / lstm_only / physics_lstm /
transformer on held-out test splits of constant-gamma and time-dependent
gamma trajectories.

Example:
    command = AblationCommand(
        configs=my_system_configs,
        variants=[PredictorVariant.PHYSICS_ONLY,
                  PredictorVariant.PHYSICS_LSTM],
    )
    results = AblationStudyUseCase(simulator).execute(command)
    for result in results:
        print(result.as_dict())
"""
from __future__ import annotations

import dataclasses
from collections.abc import Callable

from src.application import backtest
from src.application import generate_dataset
from src.application import train_model
from src.domain import ports
from src.domain import value_objects
from src.models import registry

# Builds the untrained predictor of one variant for a command.
PredictorFactory = Callable[
    [value_objects.PredictorVariant, "AblationCommand"], ports.IPredictor]

# Variants that have no training step.
_UNTRAINED = (
    value_objects.PredictorVariant.PHYSICS_ONLY,
    value_objects.PredictorVariant.STRETCHED_EXP,
)


@dataclasses.dataclass(frozen=True)
class AblationCommand:
    """Input of `AblationStudyUseCase`.

    Attributes:
        configs: Systems to simulate; one dataset is built from them.
        variants: Predictor variants to train and score.
        noise: Measurement noise recorded in each result's spec.
        window_length: Observation window in time steps.
        horizon: Risk horizon.
        samples_per_trajectory: Windows drawn from each trajectory.
        n_epochs: Training epochs of the trainable variants.
        batch_size: Training batch size.
        lr: Learning rate.
        regression_loss: Loss alias for the remaining-time target.
        seed: Seed of the dataset and of training.
        verbose: Whether to print progress.
        min_window_gap: Minimum gap between window ends, in time steps.
    """

    configs: list[value_objects.SystemConfig]
    variants: list[value_objects.PredictorVariant] = dataclasses.field(
        default_factory=lambda: list(value_objects.PredictorVariant))
    noise: value_objects.NoiseConfig = dataclasses.field(
        default_factory=value_objects.NoiseConfig)
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


def default_predictor(
    variant: value_objects.PredictorVariant,
    command: AblationCommand,
) -> ports.IPredictor:
    """Builds the default predictor of a variant.

    Args:
        variant: The variant to build.
        command: The command, for `t_max` and `dt` of the configurations.

    Returns:
        A new, untrained predictor from the model registry.

    Raises:
        ValueError: If the variant is unknown.
    """
    variants = value_objects.PredictorVariant
    if variant == variants.PHYSICS_ONLY:
        t_max = max((c.t_max for c in command.configs), default=100.0)
        return registry.build_model("physics_only", t_max=t_max)
    if variant == variants.STRETCHED_EXP:
        t_max = max((c.t_max for c in command.configs), default=100.0)
        dt = command.configs[0].dt if command.configs else 0.1
        return registry.build_model("stretched_exp", t_max=t_max, dt=dt)
    if variant == variants.LSTM_ONLY:
        return registry.build_model("lstm_only")
    if variant == variants.PHYSICS_LSTM:
        return registry.build_model("physics_lstm")
    if variant == variants.TRANSFORMER:
        return registry.build_model("transformer")
    raise ValueError(f"Unknown variant: {variant}")


class AblationStudyUseCase:
    """Trains and scores every variant on one shared dataset.

    The dataset is generated once, so all variants see the same
    train/val/test split.
    """

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
            predictor_factory: Builds the predictor of a variant; the
                default builds the registry's default models.
        """
        self._simulator = simulator
        self._store = store
        self._predictor_factory = predictor_factory

    def execute(
        self, command: AblationCommand
    ) -> list[value_objects.AblationResult]:
        """Trains and scores each variant.

        Args:
            command: What to run.

        Returns:
            One result per variant, in the order of `command.variants`.
        """
        dataset = generate_dataset.GenerateDatasetUseCase(
            self._simulator, store=self._store
        ).execute(generate_dataset.GenerateDatasetCommand(
            configs=command.configs,
            window_length=command.window_length,
            horizon=command.horizon,
            samples_per_trajectory=command.samples_per_trajectory,
            seed=command.seed,
            min_window_gap=command.min_window_gap,
        ))

        results = []
        for variant in command.variants:
            if command.verbose:
                print(f"\n=== Variant: {variant.value} ===")
            predictor = self._predictor_factory(variant, command)
            if variant not in _UNTRAINED:
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
                name=variant.value,
                predictor_variant=variant,
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
