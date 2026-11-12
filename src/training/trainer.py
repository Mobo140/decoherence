"""One-call training of a predictor, for experiments and notebooks."""
from __future__ import annotations

from src.application import train_model
from src.data import dataset as dataset_lib
from src.domain import ports


def train(
    model: ports.IPredictor,
    dataset: dataset_lib.Dataset,
    *,
    epochs: int,
    batch_size: int,
    lr: float,
    seed: int,
    loss: str = "huber",
    verbose: bool = False,
) -> ports.IPredictor:
    """Trains `model` on the train split of `dataset`, in place.

    A thin wrapper around `TrainModelUseCase` without a model store. The
    seed is required: it seeds torch's global generator before the weights
    are drawn, so the same call gives the same model.

    Args:
        model: An untrained predictor, e.g. from `registry.build_model`.
            Predictors without parameters (physics_only) are returned
            unchanged.
        dataset: Dataset with train and val splits.
        epochs: Maximum number of epochs.
        batch_size: Mini-batch size.
        lr: Initial learning rate.
        seed: Seed for weight initialisation, dropout and batch shuffling.
        loss: Alias in `losses.LOSSES` for the regression loss.
        verbose: Whether to print progress.

    Returns:
        The same `model`, trained.
    """
    train_model.TrainModelUseCase(model).execute(
        train_model.TrainModelCommand(
            dataset=dataset,
            n_epochs=epochs,
            batch_size=batch_size,
            learning_rate=lr,
            regression_loss=loss,
            verbose=verbose,
            seed=seed,
        ))
    return model
