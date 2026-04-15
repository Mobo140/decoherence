"""Use-case: infer decoherence time and imminent risk from an observation window."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..domain.events import RiskThresholdExceeded
from ..domain.ports import IPredictor
from ..domain.value_objects import PredictionResult


@dataclass(frozen=True)
class PredictDecoherenceCommand:
    """Input for PredictDecoherenceUseCase."""

    window: np.ndarray      # (window_length, n_features) – recent observables
    t_obs: float            # absolute simulation time at the end of the window
    horizon: float = 1.0    # time horizon for risk score
    risk_alarm_threshold: float = 0.7  # emit RiskThresholdExceeded when exceeded


class PredictDecoherenceUseCase:
    """Wrap model inference in a use-case with guard-rail and event emission.

    Events are currently returned for the caller to handle (no side-effects
    inside the use-case itself).
    """

    def __init__(self, predictor: IPredictor) -> None:
        self._predictor = predictor

    def execute(
        self, command: PredictDecoherenceCommand
    ) -> tuple[PredictionResult, list[RiskThresholdExceeded]]:
        if not self._predictor.is_trained:
            raise RuntimeError(
                "Predictor is not trained. Run TrainModelUseCase first."
            )

        result = self._predictor.predict(
            command.window, command.t_obs, command.horizon
        )

        events: list[RiskThresholdExceeded] = []
        if result.risk_score >= command.risk_alarm_threshold:
            events.append(
                RiskThresholdExceeded(
                    t_obs=command.t_obs,
                    risk_score=result.risk_score,
                    alarm_threshold=command.risk_alarm_threshold,
                )
            )

        return result, events
