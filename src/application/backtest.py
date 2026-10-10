"""Use-case: evaluate a trained predictor on the held-out test split."""
from __future__ import annotations

from dataclasses import dataclass

from src.data import dataset as dataset_lib
from src.domain.ports import IPredictor
from src.domain.value_objects import BacktestMetrics
from src.evaluation import backtest


@dataclass(frozen=True)
class BacktestCommand:
    """Input for BacktestUseCase."""

    dataset: dataset_lib.Dataset
    horizon: float = 1.0


class BacktestUseCase:
    """Runs `evaluation.backtest.evaluate` for one predictor.

    Metrics: MAE, RMSE, R^2 and MAPE of the predicted decoherence time,
    and AUROC of the risk score within the command's horizon.
    """

    def __init__(self, predictor: IPredictor) -> None:
        self._predictor = predictor

    def execute(self, command: BacktestCommand) -> BacktestMetrics:
        """Scores the predictor on the command's dataset."""
        return backtest.evaluate(
            self._predictor, command.dataset, horizon=command.horizon)
