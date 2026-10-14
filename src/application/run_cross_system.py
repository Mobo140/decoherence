"""Use case: cross-system generalisation (experiments E4 and E5).

Trains ONE model on a mixture of all system types, then evaluates it on
each subsystem separately to test the claim that a single model
generalises across sigma-minus / sigma-z, one and two qubits, and
diverse gamma(t).

Example:
    command = CrossSystemCommand(
        train_configs=all_configs,
        eval_groups={"1q_sigma_minus": configs_1q, "2q_XXZ": configs_2q},
    )
    results = CrossSystemUseCase(simulator).execute(command)
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

# Builds the untrained predictor to fit on the mixture.
PredictorFactory = Callable[[], ports.IPredictor]


@dataclasses.dataclass(frozen=True)
class CrossSystemCommand:
    """Input of `CrossSystemUseCase`.

    Attributes:
        train_configs: Systems of the training mixture.
        eval_groups: Group name -> systems evaluated separately.
        window_length: Observation window in time steps.
        horizon: Risk horizon.
        samples_per_trajectory: Windows drawn from each trajectory.
        n_epochs: Training epochs.
        batch_size: Training batch size.
        lr: Learning rate.
        regression_loss: Loss alias for the remaining-time target.
        seed: Seed of training; evaluation data uses `seed + 1`.
        verbose: Whether to print progress.
        min_window_gap: Minimum gap between window ends, in time steps.
        eval_interaction_codes: Optional group name -> inference context
            code (4-class: D = 2.0, E = 3.0) set on predictors that have
            one.
    """

    train_configs: list[value_objects.SystemConfig]
    eval_groups: dict[str, list[value_objects.SystemConfig]]
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
    eval_interaction_codes: dict[str, float] = dataclasses.field(
        default_factory=dict)


def default_predictor() -> ports.IPredictor:
    """Builds the default physics + LSTM predictor."""
    return registry.build_model("physics_lstm")


def variant_of(predictor: ports.IPredictor) -> value_objects.PredictorVariant:
    """Maps a predictor instance to its CSV and paper variant label.

    Args:
        predictor: A built predictor.

    Returns:
        The variant it corresponds to.
    """
    variants = value_objects.PredictorVariant
    name = type(predictor).__name__
    if name == "TransformerPredictor":
        return variants.TRANSFORMER
    if name == "PhysicsOnlyPredictor":
        return variants.PHYSICS_ONLY
    if name == "StretchedExpPredictor":
        return variants.STRETCHED_EXP
    if getattr(predictor, "use_physics_prior", True) is False:
        return variants.LSTM_ONLY
    return variants.PHYSICS_LSTM


class CrossSystemUseCase:
    """Trains on a mixture and evaluates per subsystem."""

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
        self, command: CrossSystemCommand
    ) -> list[value_objects.AblationResult]:
        """Trains on the mixture and scores each evaluation group.

        Args:
            command: What to run.

        Returns:
            One result per evaluation group, named
            `cross_system_<group>`.
        """
        if command.verbose:
            print("=== Cross-system: training on mixture ===")
        generator = generate_dataset.GenerateDatasetUseCase(
            self._simulator, store=self._store)
        train_dataset = generator.execute(
            generate_dataset.GenerateDatasetCommand(
                configs=command.train_configs,
                window_length=command.window_length,
                horizon=command.horizon,
                samples_per_trajectory=command.samples_per_trajectory,
                seed=command.seed,
                # Keep more for training: evaluation data is separate.
                train_ratio=0.9,
                val_ratio=0.1,
                min_window_gap=command.min_window_gap,
            ))

        predictor = self._predictor_factory()
        train_model.TrainModelUseCase(predictor).execute(
            train_model.TrainModelCommand(
                dataset=train_dataset,
                n_epochs=command.n_epochs,
                batch_size=command.batch_size,
                learning_rate=command.lr,
                regression_loss=command.regression_loss,
                verbose=command.verbose,
                seed=command.seed,
            ))

        results = []
        for group_name, eval_configs in command.eval_groups.items():
            if command.verbose:
                print(f"\n--- Evaluating on: {group_name} ---")
            # A different seed gives different splits; everything goes to
            # test because nothing is retrained on it.
            eval_dataset = generator.execute(
                generate_dataset.GenerateDatasetCommand(
                    configs=eval_configs,
                    window_length=command.window_length,
                    horizon=command.horizon,
                    samples_per_trajectory=command.samples_per_trajectory,
                    seed=command.seed + 1,
                    train_ratio=0.0,
                    val_ratio=0.0,
                    min_window_gap=command.min_window_gap,
                )).as_test_only()

            codes = command.eval_interaction_codes
            if group_name in codes and hasattr(
                    predictor, "inference_interaction_code"):
                predictor.inference_interaction_code = codes[group_name]

            metrics = backtest.BacktestUseCase(predictor).execute(
                backtest.BacktestCommand(
                    dataset=eval_dataset, horizon=command.horizon))
            spec = value_objects.ExperimentSpec(
                name=f"cross_system_{group_name}",
                predictor_variant=variant_of(predictor),
                noise=value_objects.NoiseConfig(),
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
