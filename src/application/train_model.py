"""Use case: fit a predictor to a Dataset."""
from __future__ import annotations

import dataclasses

from src.data import dataset as dataset_lib
from src.domain import ports
from src.training import loop

# Kept under its old name for callers of the use case.
TrainModelResult = loop.TrainResult


@dataclasses.dataclass(frozen=True)
class TrainModelCommand:
    """Input of `TrainModelUseCase`.

    Attributes:
        dataset: The dataset to fit.
        n_epochs: Training epochs.
        batch_size: Training batch size.
        learning_rate: Optimiser learning rate.
        model_name: Name under which a model store keeps the predictor.
        verbose: Whether to print progress.
        regression_loss: Loss alias: "huber", "mse" or "mae".
        seed: Seeds the ML framework (weight init, dropout, batch
            shuffling). `None` leaves the global generator alone, which
            is not reproducible.
    """

    dataset: dataset_lib.Dataset
    n_epochs: int = 50
    batch_size: int = 32
    learning_rate: float = 1e-3
    model_name: str = "decoherence_lstm"
    verbose: bool = True
    regression_loss: str = "huber"
    seed: int | None = None


class TrainModelUseCase:
    """Delegates training to the predictor, then optionally persists it.

    The use case is thin on purpose: the machine-learning details live in
    `IPredictor.train`.
    """

    def __init__(
        self,
        predictor: ports.IPredictor,
        store: ports.IModelStore | None = None,
    ) -> None:
        """Wires the use case.

        Args:
            predictor: The predictor to fit.
            store: Where to save the fitted predictor; `None` skips it.
        """
        self._predictor = predictor
        self._store = store

    def execute(self, command: TrainModelCommand) -> TrainModelResult:
        """Fits the predictor and saves it if a store was given.

        Args:
            command: What to train on and how.

        Returns:
            The result of the predictor's training.
        """
        result = self._predictor.train(
            dataset=command.dataset,
            n_epochs=command.n_epochs,
            batch_size=command.batch_size,
            lr=command.learning_rate,
            verbose=command.verbose,
            regression_loss=command.regression_loss,
            seed=command.seed,
        )
        if self._store is not None:
            self._store.save(self._predictor, command.model_name)
        return result
