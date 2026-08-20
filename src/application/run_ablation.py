"""Use-case: ablation study — compare predictor variants on the same dataset.

Experiment E1 from plan.md:
    Compare physics_only / lstm_only / physics_lstm / transformer on held-out
    test splits generated from constant-γ and time-dependent-γ trajectories.

Usage::
    cmd = AblationCommand(
        configs=my_system_configs,
        variants=[PredictorVariant.PHYSICS_ONLY, PredictorVariant.PHYSICS_LSTM],
    )
    results = AblationStudyUseCase(simulator).execute(cmd)
    for r in results:
        print(r.as_dict())
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np

from ..domain.ports import IPredictor, ISimulator
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
class AblationCommand:
    """Input for AblationStudyUseCase."""

    configs: List[SystemConfig]
    variants: List[PredictorVariant] = field(
        default_factory=lambda: list(PredictorVariant)
    )
    noise: NoiseConfig = field(default_factory=NoiseConfig)
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


class AblationStudyUseCase:
    """Run the full ablation: train each variant, evaluate, return results.

    The dataset is generated once and shared across all variants to ensure
    a fair comparison (same train/val/test splits).
    """

    def __init__(
        self,
        simulator: ISimulator,
        noise_transform=None,
        store=None,
    ) -> None:
        self._simulator = simulator
        self._noise_transform = noise_transform
        self._store = store

    def execute(self, command: AblationCommand) -> List[AblationResult]:
        # --- Generate shared dataset ---
        gen_uc = GenerateDatasetUseCase(self._simulator, store=self._store)
        gen_cmd = GenerateDatasetCommand(
            configs=command.configs,
            window_length=command.window_length,
            horizon=command.horizon,
            samples_per_trajectory=command.samples_per_trajectory,
            seed=command.seed,
            min_window_gap=command.min_window_gap,
        )
        dataset = gen_uc.execute(gen_cmd)

        # Apply noise if configured
        if self._noise_transform is not None:
            dataset = self._apply_noise(dataset)

        results: List[AblationResult] = []

        for variant in command.variants:
            if command.verbose:
                print(f"\n=== Variant: {variant.value} ===")

            predictor = self._build_predictor(variant, command)

            # Physics-only and stretched-exp have no training step
            if variant not in (
                PredictorVariant.PHYSICS_ONLY,
                PredictorVariant.STRETCHED_EXP,
            ):
                train_uc = TrainModelUseCase(predictor)
                train_cmd = TrainModelCommand(
                    dataset=dataset,
                    n_epochs=command.n_epochs,
                    batch_size=command.batch_size,
                    learning_rate=command.lr,
                    regression_loss=command.regression_loss,
                    verbose=command.verbose,
                )
                train_uc.execute(train_cmd)

            # Backtest
            backtest_uc = BacktestUseCase(predictor)
            metrics = backtest_uc.execute(BacktestCommand(dataset=dataset, horizon=command.horizon))

            spec = ExperimentSpec(
                name=variant.value,
                predictor_variant=variant,
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

    def _build_predictor(self, variant: PredictorVariant, command: AblationCommand) -> IPredictor:
        from ..infrastructure.ml.lstm_predictor import LSTMPredictor
        from ..infrastructure.ml.physics_only_predictor import PhysicsOnlyPredictor
        from ..infrastructure.ml.stretched_exp_predictor import StretchedExpPredictor
        from ..infrastructure.ml.transformer_predictor import TransformerPredictor

        if variant == PredictorVariant.PHYSICS_ONLY:
            t_max = max((c.t_max for c in command.configs), default=100.0)
            return PhysicsOnlyPredictor(t_max=t_max)
        if variant == PredictorVariant.STRETCHED_EXP:
            t_max = max((c.t_max for c in command.configs), default=100.0)
            dt = command.configs[0].dt if command.configs else 0.1
            return StretchedExpPredictor(t_max=t_max, dt=dt)
        if variant == PredictorVariant.LSTM_ONLY:
            # Disable physics prior by zeroing the physics features at inference
            # We achieve this by setting alpha=0 conceptually; in practice we
            # use LSTMPredictor and override _phys_std→∞ after training so
            # phys_norm→0, making the physics estimate contribute nothing.
            # Simpler: subclass or use a flag. For now, use a dedicated kwarg.
            return LSTMPredictor(use_physics_prior=False)
        if variant == PredictorVariant.PHYSICS_LSTM:
            return LSTMPredictor(use_physics_prior=True)
        if variant == PredictorVariant.TRANSFORMER:
            return TransformerPredictor()
        raise ValueError(f"Unknown variant: {variant}")

    def _apply_noise(self, dataset):
        """Return dataset with noise applied to all sequences."""
        from ..infrastructure.quantum.noise import GaussianMeasurementNoise
        # Noise is applied at the feature-matrix level (post-simulation)
        # by adding Gaussian perturbations to the sequence arrays directly.
        noise_cfg = self._noise_transform.config
        rng = np.random.default_rng(noise_cfg.seed)
        noisy_seqs = dataset.sequences + rng.normal(
            0.0, noise_cfg.sigma, size=dataset.sequences.shape
        ).astype(np.float32)
        import copy
        noisy_ds = copy.copy(dataset)
        noisy_ds.sequences = noisy_seqs
        return noisy_ds
