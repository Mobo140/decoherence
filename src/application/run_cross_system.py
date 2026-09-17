"""Use-case: cross-system generalisation (Experiments E4/E5 from plan.md).

Trains ONE model on a mixture of all system types, then evaluates on each
subsystem separately to test claim C5:
  "A single model generalises across σ₋/σ_z, 1-qubit/2-qubit, diverse γ(t)."

Also supports the inverse-problem baseline comparison (E5): train a
PhysicsOnlyPredictor or TransformerPredictor on each subsystem and compare
with the jointly-trained physics+LSTM.

Usage::
    cmd = CrossSystemCommand(
        train_configs=all_configs,
        eval_groups={"1q_sigma_minus": configs_1q_sm, "2q_XXZ": configs_2q},
    )
    results = CrossSystemUseCase(simulator).execute(cmd)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

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
class CrossSystemCommand:
    """Input for CrossSystemUseCase."""

    train_configs: List[SystemConfig]
    eval_groups: Dict[str, List[SystemConfig]]
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
    # Optional per-eval-group inference context codes (4-class: D=2.0, E=3.0).
    eval_interaction_codes: Dict[str, float] = field(default_factory=dict)


class CrossSystemUseCase:
    """Train on mixture, evaluate per subsystem."""

    def __init__(self, simulator: ISimulator, store=None) -> None:
        self._simulator = simulator
        self._store = store

    def execute(self, command: CrossSystemCommand) -> List[AblationResult]:
        if command.verbose:
            print("=== Cross-system: training on mixture ===")

        # Train on the full mixture
        gen_uc = GenerateDatasetUseCase(self._simulator, store=self._store)
        train_dataset = gen_uc.execute(GenerateDatasetCommand(
            configs=command.train_configs,
            window_length=command.window_length,
            horizon=command.horizon,
            samples_per_trajectory=command.samples_per_trajectory,
            seed=command.seed,
            train_ratio=0.9,   # keep more for training when eval is separate
            val_ratio=0.1,
            min_window_gap=command.min_window_gap,
        ))

        predictor = self._build_predictor()
        TrainModelUseCase(predictor).execute(TrainModelCommand(
            dataset=train_dataset,
            n_epochs=command.n_epochs,
            batch_size=command.batch_size,
            learning_rate=command.lr,
            regression_loss=command.regression_loss,
            verbose=command.verbose,
            seed=command.seed,
        ))

        results: List[AblationResult] = []

        for group_name, eval_configs in command.eval_groups.items():
            if command.verbose:
                print(f"\n--- Evaluating on: {group_name} ---")

            eval_dataset = gen_uc.execute(GenerateDatasetCommand(
                configs=eval_configs,
                window_length=command.window_length,
                horizon=command.horizon,
                samples_per_trajectory=command.samples_per_trajectory,
                seed=command.seed + 1,   # different seed → different splits
                train_ratio=0.0,
                val_ratio=0.0,
                min_window_gap=command.min_window_gap,
                # All samples go to test — we evaluate only, no re-training
            ))

            # Ensure all samples land in test_idx
            eval_dataset = self._force_all_test(eval_dataset)

            if command.eval_interaction_codes and group_name in command.eval_interaction_codes:
                if hasattr(predictor, "inference_interaction_code"):
                    predictor.inference_interaction_code = command.eval_interaction_codes[group_name]

            metrics = BacktestUseCase(predictor).execute(
                BacktestCommand(dataset=eval_dataset, horizon=command.horizon)
            )

            spec = ExperimentSpec(
                name=f"cross_system_{group_name}",
                predictor_variant=_variant_of(predictor),
                noise=NoiseConfig(),
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
    def _force_all_test(dataset):
        """Return dataset where every sample is in test_idx."""
        import copy
        ds = copy.copy(dataset)
        ds.test_idx = np.arange(len(dataset.sequences))
        ds.train_idx = np.array([], dtype=np.int64)
        ds.val_idx = np.array([], dtype=np.int64)
        return ds


def _variant_of(predictor) -> PredictorVariant:
    """Map a live predictor instance to the CSV/paper variant label."""
    name = type(predictor).__name__
    if name == "TransformerPredictor":
        return PredictorVariant.TRANSFORMER
    if name == "PhysicsOnlyPredictor":
        return PredictorVariant.PHYSICS_ONLY
    if name == "StretchedExpPredictor":
        return PredictorVariant.STRETCHED_EXP
    if getattr(predictor, "use_physics_prior", True) is False:
        return PredictorVariant.LSTM_ONLY
    return PredictorVariant.PHYSICS_LSTM
