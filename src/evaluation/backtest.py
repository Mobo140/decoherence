"""Backtest: score a trained predictor on the test windows of a dataset."""
from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from src.data import dataset as dataset_lib
from src.domain import ports
from src.domain import value_objects
from src.evaluation import metrics as metrics_lib

# Predictor attribute -> Dataset field that supplies its per-window value.
# Predictors take system context (coupling, Hamiltonian and dissipator
# type) as attributes read inside `predict`, not as arguments.
CONTEXT_ATTRIBUTES = (
    ("inference_J", "J_values"),
    ("inference_interaction_code", "interaction_type_codes"),
    ("inference_dissipator_code", "dissipator_type_codes"),
)


def context_fields(
    model: ports.IPredictor, dataset: dataset_lib.Dataset
) -> list[tuple[str, np.ndarray]]:
    """Lists the context attributes the backtest sets on `model`.

    An attribute is listed when the model has it and the dataset holds one
    value per window for it (fields that are empty, i.e. unknown, are
    skipped).

    Args:
        model: Predictor to be evaluated.
        dataset: Dataset whose windows will be predicted.

    Returns:
        Pairs of (attribute name, per-window values).
    """
    n_windows = len(dataset.sequences)
    fields = []
    for attribute, field in CONTEXT_ATTRIBUTES:
        values = getattr(dataset, field)
        if len(values) == n_windows and hasattr(model, attribute):
            fields.append((attribute, values))
    return fields


def test_indices(dataset: dataset_lib.Dataset) -> np.ndarray:
    """Returns the test windows that are scored.

    Right-censored windows are dropped (their t_decoh is only a lower
    bound, which makes R^2 meaningless) unless every test window is
    censored.
    """
    indices = dataset.test_idx
    if len(dataset.censored) == len(dataset.sequences) and len(indices) > 0:
        uncensored = ~dataset.censored[indices]
        if uncensored.any():
            indices = indices[uncensored]
    return indices


def predict_windows(
    model: ports.IPredictor,
    dataset: dataset_lib.Dataset,
    indices: Sequence[int],
    *,
    horizon: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Predicts the given windows one by one.

    Before every window the context attributes (see `CONTEXT_ATTRIBUTES`)
    of `model` are overwritten with that window's values; they keep the
    values of the last window afterwards.

    Args:
        model: Trained predictor.
        dataset: Dataset holding the windows.
        indices: Rows of `dataset.sequences` to predict.
        horizon: Risk horizon passed to `model.predict`.

    Returns:
        (predicted decoherence times as float64, risk scores).
    """
    fields = context_fields(model, dataset)
    times, risks = [], []
    for idx in indices:
        for attribute, values in fields:
            setattr(model, attribute, float(values[idx]))
        result = model.predict(
            dataset.sequences[idx], float(dataset.t_obs[idx]), horizon)
        times.append(result.t_decoh_predicted)
        risks.append(result.risk_score)
    return np.array(times, dtype=np.float64), np.array(risks)


def evaluate(
    model: ports.IPredictor,
    dataset: dataset_lib.Dataset,
    *,
    horizon: float = 1.0,
    metrics: Sequence[str] | None = None,
) -> value_objects.BacktestMetrics:
    """Scores a trained predictor on the uncensored test windows.

    MAE, RMSE, R^2 and MAPE compare the predicted with the true
    decoherence time; AUROC compares the risk score with the risk label.

    Args:
        model: Trained predictor.
        dataset: Dataset whose test split is scored.
        horizon: Risk horizon passed to `model.predict`.
        metrics: Further `metrics.METRICS` aliases to compute; they land in
            `BacktestMetrics.extra`. Regression metrics use (actual,
            predicted time), classification metrics (risk label, risk
            score).

    Returns:
        The metrics together with predictions and actuals.

    Raises:
        RuntimeError: If the model is not trained.
        KeyError: If an alias in `metrics` is unknown.
    """
    if not model.is_trained:
        raise RuntimeError(
            "Predictor is not trained. Run TrainModelUseCase first.")
    indices = test_indices(dataset)
    predicted, risk = predict_windows(
        model, dataset, indices, horizon=horizon)
    actual = dataset.t_decoh_abs[indices].astype(np.float64)
    labels = dataset.risk_labels[indices]
    extra = {}
    for alias in metrics or ():
        if alias in metrics_lib.CLASSIFICATION_METRICS:
            extra[alias] = metrics_lib.METRICS[alias](labels, risk)
        else:
            extra[alias] = metrics_lib.METRICS[alias](actual, predicted)
    return value_objects.BacktestMetrics(
        mae=metrics_lib.mae(actual, predicted),
        rmse=metrics_lib.rmse(actual, predicted),
        r2=metrics_lib.r2(actual, predicted),
        mape=metrics_lib.mape(actual, predicted),
        risk_auroc=metrics_lib.auroc(labels, risk),
        n_samples=len(indices),
        predictions=predicted,
        actuals=actual,
        extra=extra,
    )
