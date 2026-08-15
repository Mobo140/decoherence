"""Measurement noise injection — implements ITrajectoryTransform.

Models additive Gaussian noise on each observable independently, mimicking
finite-statistics measurement uncertainty in real quantum hardware.

The transform is:
    obs_noisy[k, t] = obs_clean[k, t] + ε,   ε ~ N(0, σ²)

where σ = NoiseConfig.sigma.  The transform is pure (returns a new
TrajectoryResult without mutating the original).

Usage::
    from src.domain.value_objects import NoiseConfig
    from src.infrastructure.quantum.noise import GaussianMeasurementNoise

    noise = GaussianMeasurementNoise(NoiseConfig(sigma=0.02, seed=0))
    noisy_traj = noise(trajectory)
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ...domain.ports import ITrajectoryTransform
from ...domain.value_objects import NoiseConfig, TrajectoryResult


@dataclass(frozen=True)
class GaussianMeasurementNoise(ITrajectoryTransform):
    """Add i.i.d. Gaussian noise to all scalar observables in a trajectory.

    Density-matrix entries (key ``density_matrices``) are left untouched —
    they are never used as ML features.

    Attributes:
        config: NoiseConfig specifying σ and optional RNG seed.
    """

    config: NoiseConfig

    def __call__(self, trajectory: TrajectoryResult) -> TrajectoryResult:
        if self.config.is_noiseless:
            return trajectory

        rng = np.random.default_rng(self.config.seed)

        noisy_obs: dict[str, np.ndarray] = {}
        for key, arr in trajectory.observables.items():
            if key == "density_matrices":
                noisy_obs[key] = arr  # pass through unchanged
            else:
                noise = rng.normal(0.0, self.config.sigma, size=arr.shape).astype(
                    arr.dtype
                )
                noisy_obs[key] = arr + noise

        return TrajectoryResult(
            times=trajectory.times,
            observables=noisy_obs,
            t_decoh=trajectory.t_decoh,
            system_config=trajectory.system_config,
            censored=trajectory.censored,
        )


@dataclass(frozen=True)
class ComposedTransform(ITrajectoryTransform):
    """Apply a sequence of transforms in order (left to right).

    Allows building a pipeline::
        pipeline = ComposedTransform([noise, normaliser, ...])
        result = pipeline(trajectory)
    """

    transforms: tuple[ITrajectoryTransform, ...]

    def __call__(self, trajectory: TrajectoryResult) -> TrajectoryResult:
        for transform in self.transforms:
            trajectory = transform(trajectory)
        return trajectory
