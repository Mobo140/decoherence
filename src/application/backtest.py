"""Use case: evaluate a trained predictor on the held-out test split."""
from __future__ import annotations

import dataclasses

from src.data import dataset as dataset_lib
from src.domain import ports
from src.domain import value_objects
from src.evaluation import backtest


@dataclasses.dataclass(frozen=True)
class BacktestCommand:
    """Input of `BacktestUseCase`.

    Attributes:
        dataset: The dataset whose test split is scored.
        horizon: Risk horizon.
    """

    dataset: dataset_lib.Dataset
    horizon: float = 1.0


class BacktestUseCase:
    """Runs `evaluation.backtest.evaluate` for one predictor.

    Metrics: MAE, RMSE, R2 and MAPE of the predicted decoherence time,
    and the AUROC of the risk score within the horizon.
    """

    def __init__(self, predictor: ports.IPredictor) -> None:
        """Wires the use case.

        Args:
            predictor: The trained predictor to score.
        """
        self._predictor = predictor

    def execute(
        self, command: BacktestCommand
    ) -> value_objects.BacktestMetrics:
        """Scores the predictor on the command's dataset.

        Args:
            command: What to score.

        Returns:
            The metrics.
        """
        return backtest.evaluate(
            self._predictor, command.dataset, horizon=command.horizon)
