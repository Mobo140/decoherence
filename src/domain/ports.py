"""Domain ports (interfaces).

Following the Ports & Adapters (Hexagonal) pattern:
- Ports live in the domain layer.
- Concrete adapters live in infrastructure.
- The application layer depends only on ports, never on adapters directly.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, List

import numpy as np

if TYPE_CHECKING:
    from .value_objects import (
        SystemConfig,
        TrajectoryResult,
        PredictionResult,
    )


class ISimulator(ABC):
    """Port: run a Lindblad master-equation simulation."""

    @abstractmethod
    def simulate(self, config: "SystemConfig") -> "TrajectoryResult":
        """Simulate a single quantum trajectory.

        Args:
            config: Full system specification.

        Returns:
            TrajectoryResult with observables and decoherence time.
        """


class IPredictor(ABC):
    """Port: predict decoherence from an observation window."""

    @property
    @abstractmethod
    def is_trained(self) -> bool:
        """True once the model has been fitted to data."""

    @abstractmethod
    def predict(
        self,
        window: np.ndarray,
        t_obs: float,
        horizon: float,
    ) -> "PredictionResult":
        """Predict decoherence time and imminent risk.

        Args:
            window: Float array of shape (window_length, n_features).
            t_obs:  Absolute time at the end of the window.
            horizon: Requested risk horizon. All current predictors map
                remaining → P(remaining ≤ horizon) at inference. LSTM and
                Transformer still train a binary risk head at dataset Δt;
                that logit is unused in predict().

        Returns:
            PredictionResult with t_decoh_predicted and risk_score.
        """

    @abstractmethod
    def train(self, dataset: object, **kwargs) -> object:
        """Fit the predictor to a Dataset.

        Args:
            dataset: application.generate_dataset.Dataset instance.
            **kwargs: Hyper-parameters (n_epochs, batch_size, lr, …).

        Returns:
            TrainModelResult (defined in application layer).
        """


class ITrajectoryStore(ABC):
    """Port: persist and retrieve trajectory collections."""

    @abstractmethod
    def save(self, trajectories: List["TrajectoryResult"], name: str) -> None:
        """Persist a list of trajectories under *name*."""

    @abstractmethod
    def load(self, name: str) -> List["TrajectoryResult"]:
        """Load a previously saved trajectory collection by *name*."""

    @abstractmethod
    def list_available(self) -> List[str]:
        """Return all stored trajectory collection names."""


class IModelStore(ABC):
    """Port: persist and retrieve trained predictors."""

    @abstractmethod
    def save(self, predictor: IPredictor, name: str) -> None:
        """Serialize *predictor* under *name*."""

    @abstractmethod
    def load(self, name: str) -> IPredictor:
        """Deserialize and return the predictor stored under *name*."""


class ILossFunction(ABC):
    """Port: pluggable scalar loss for training predictors.

    Concrete implementations live in infrastructure/ml/loss_functions.py.
    The application layer depends only on this interface, making it trivial
    to swap Huber → Quantile → custom without touching training logic.
    """

    @abstractmethod
    def __call__(self, predictions: "np.ndarray", targets: "np.ndarray") -> float:
        """Compute scalar loss.

        Args:
            predictions: 1-D float array of model outputs.
            targets:     1-D float array of ground-truth values.

        Returns:
            Scalar loss value (lower is better).
        """

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable identifier used in logs and result dicts."""


class ITrajectoryTransform(ABC):
    """Port: stateless transform applied to a TrajectoryResult.

    Used to inject measurement noise, normalise observables, or apply
    domain-specific pre-processing without modifying simulation logic.
    Transforms are composable (see infrastructure/quantum/noise.py).
    """

    @abstractmethod
    def __call__(self, trajectory: "TrajectoryResult") -> "TrajectoryResult":
        """Return a transformed copy of *trajectory*."""
