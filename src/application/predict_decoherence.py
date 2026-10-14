"""Use case: infer decoherence time and imminent risk from a window."""
from __future__ import annotations

import dataclasses

import numpy as np

from src.domain import events
from src.domain import ports
from src.domain import value_objects


@dataclasses.dataclass(frozen=True)
class PredictDecoherenceCommand:
    """Input of `PredictDecoherenceUseCase`.

    Attributes:
        window: Recent observables, shape (window_length, n_features).
        t_obs: Absolute simulation time at the end of the window.
        horizon: Time horizon of the risk score.
        risk_alarm_threshold: A `RiskThresholdExceeded` event is emitted
            when the risk score reaches it.
    """

    window: np.ndarray
    t_obs: float
    horizon: float = 1.0
    risk_alarm_threshold: float = 0.7


class PredictDecoherenceUseCase:
    """Wraps inference with a guard and the alarm event.

    Events are returned to the caller; the use case has no side effects.
    """

    def __init__(self, predictor: ports.IPredictor) -> None:
        """Wires the use case.

        Args:
            predictor: The trained predictor.
        """
        self._predictor = predictor

    def execute(
        self, command: PredictDecoherenceCommand
    ) -> tuple[value_objects.PredictionResult,
               list[events.RiskThresholdExceeded]]:
        """Predicts from the window and raises an alarm event if needed.

        Args:
            command: The window and alarm settings.

        Returns:
            The prediction and the alarm events (empty or one).

        Raises:
            RuntimeError: If the predictor is not trained.
        """
        if not self._predictor.is_trained:
            raise RuntimeError(
                "Predictor is not trained. Run TrainModelUseCase first.")

        result = self._predictor.predict(
            command.window, command.t_obs, command.horizon)
        alarms = []
        if result.risk_score >= command.risk_alarm_threshold:
            alarms.append(events.RiskThresholdExceeded(
                t_obs=command.t_obs,
                risk_score=result.risk_score,
                alarm_threshold=command.risk_alarm_threshold,
            ))
        return result, alarms
