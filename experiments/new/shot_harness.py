"""Shared pieces of R29 and R30: train on one input set, score on another.

Both experiments simulate the trajectories of a seed once
(`src.data.shots.simulate`) and observe the same states under different
input sets. A model is trained on the "train" noise stream of an input set
and scored on the "test" stream of the same or another set; the windows and
the split are the same for every set, so the scored trajectories are the
same.
"""
from __future__ import annotations

import dataclasses
import time

import numpy as np

from src.blocks import observe
from src.data import dataset as dataset_lib
from src.evaluation import backtest
from src.models import registry as model_registry
from src.training import trainer

METRICS = ("r2", "mae", "auroc")


def with_tail(dataset: dataset_lib.Dataset) -> bool:
    """Whether the dataset has the columns the physics models read.

    Args:
        dataset: Dataset of one input set.

    Returns:
        True if its last columns are purity and coherence_l1.
    """
    return tuple(dataset.feature_names[-2:]) == observe.MODEL_TAIL


def restrict_test(
    dataset: dataset_lib.Dataset, mask: np.ndarray
) -> dataset_lib.Dataset:
    """Keeps the test windows where `mask` is true.

    Args:
        dataset: The dataset.
        mask: Boolean per window.

    Returns:
        A dataset whose test split is the selected windows.

    Raises:
        ValueError: If `mask` does not have one entry per window.
    """
    if len(mask) != len(dataset.sequences):
        raise ValueError(
            f"mask has {len(mask)} entries for {len(dataset.sequences)} "
            "windows.")
    keep = np.asarray(dataset.test_idx)
    return dataclasses.replace(dataset, test_idx=keep[mask[keep]])


def train_model(
    alias: str,
    dataset: dataset_lib.Dataset,
    *,
    seed: int,
    epochs: int,
    batch_size: int,
    learning_rate: float,
):
    """Builds and trains a model alias on the train split.

    Args:
        alias: Key of `model_registry.MODELS`.
        dataset: Dataset whose train and val splits are used.
        seed: Seed of the weights and the batch order.
        epochs: Maximum number of epochs.
        batch_size: Mini-batch size.
        learning_rate: Initial learning rate.

    Returns:
        The trained model.
    """
    return trainer.train(
        model_registry.build_model(alias), dataset, epochs=epochs,
        batch_size=batch_size, lr=learning_rate, seed=seed)


def score(
    model, dataset: dataset_lib.Dataset, horizon: float
) -> dict[str, object]:
    """Scores a trained model on the test split of a dataset.

    Args:
        model: Trained predictor.
        dataset: Dataset whose test split is scored.
        horizon: Risk horizon.

    Returns:
        r2, mae, auroc and n_samples.

    Raises:
        ValueError: If the test split has no window.
    """
    if len(dataset.test_idx) == 0:
        raise ValueError("The test split has no window to score.")
    scores = backtest.evaluate(
        model, dataset, horizon=horizon, metrics=METRICS)
    return {**scores.extra, "n_samples": scores.n_samples}


def elapsed(start: float) -> float:
    """Seconds since `start` (from `time.perf_counter`)."""
    return round(time.perf_counter() - start, 1)
