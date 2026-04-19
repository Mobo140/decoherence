"""Use-case: fit a predictor to a Dataset."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

from ..domain.ports import IModelStore, IPredictor
from .generate_dataset import Dataset


@dataclass
class TrainModelResult:
    """Summary produced by the training use-case."""

    predictor: IPredictor
    train_loss_history: List[float] = field(default_factory=list)
    val_loss_history: List[float] = field(default_factory=list)

    @property
    def best_val_loss(self) -> float:
        return min(self.val_loss_history) if self.val_loss_history else float("inf")


@dataclass(frozen=True)
class TrainModelCommand:
    """Input for TrainModelUseCase."""

    dataset: Dataset
    n_epochs: int = 50
    batch_size: int = 32
    learning_rate: float = 1e-3
    model_name: str = "decoherence_lstm"
    verbose: bool = True
    regression_loss: str = "huber"   # "huber" | "mse" | "mae"


class TrainModelUseCase:
    """Delegate training to the injected IPredictor, then optionally persist it.

    The use-case is intentionally thin: all ML specifics live in the
    infrastructure adapter that implements IPredictor.train().
    """

    def __init__(
        self,
        predictor: IPredictor,
        store: Optional[IModelStore] = None,
    ) -> None:
        self._predictor = predictor
        self._store = store

    def execute(self, command: TrainModelCommand) -> TrainModelResult:
        result: TrainModelResult = self._predictor.train(
            dataset=command.dataset,
            n_epochs=command.n_epochs,
            batch_size=command.batch_size,
            lr=command.learning_rate,
            verbose=command.verbose,
            regression_loss=command.regression_loss,
        )

        if self._store is not None:
            self._store.save(self._predictor, command.model_name)

        return result
