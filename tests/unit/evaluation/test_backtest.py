"""evaluate() and its helpers in src/evaluation/backtest."""
from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from src.domain import value_objects
from src.evaluation import backtest, metrics


class _Spy:
    """Predictor that echoes the context it was given."""

    is_trained = True

    def __init__(self) -> None:
        self.inference_J = -1.0
        self.seen: list[float] = []

    def predict(self, window, t_obs, horizon):
        self.seen.append(self.inference_J)
        return value_objects.PredictionResult(
            t_decoh_predicted=t_obs + 1.0, risk_score=0.5, horizon=horizon)


def test_evaluate_matches_the_metric_functions(toy_dataset):
    ds = toy_dataset
    out = backtest.evaluate(_Spy(), ds, metrics=["mae", "brier"])
    actual = ds.t_decoh_abs[ds.test_idx].astype(np.float64)
    predicted = ds.t_obs[ds.test_idx].astype(np.float64) + 1.0
    assert out.n_samples == len(ds.test_idx)
    assert out.mae == metrics.mae(actual, predicted)
    assert out.r2 == metrics.r2(actual, predicted)
    assert out.extra["mae"] == out.mae
    assert out.extra["brier"] == metrics.brier(
        ds.risk_labels[ds.test_idx], np.full(len(ds.test_idx), 0.5))


def test_predict_windows_sets_j_per_window_only_when_dataset_has_it(
        toy_dataset):
    ds = toy_dataset
    model = _Spy()
    backtest.predict_windows(model, ds, ds.test_idx)
    assert model.seen == [-1.0] * len(ds.test_idx)  # no J in the context

    j = np.arange(len(ds.sequences), dtype=np.float32) * 0.5
    ds = dataclasses.replace(ds, context={"J": j})
    model = _Spy()
    backtest.predict_windows(model, ds, ds.test_idx)
    assert model.seen == [float(j[i]) for i in ds.test_idx]
    assert model.inference_J == float(j[ds.test_idx[-1]])


# --- Negative cases ---


def test_evaluate_untrained_model_raises_runtime_error(toy_dataset):
    model = _Spy()
    model.is_trained = False
    with pytest.raises(RuntimeError):
        backtest.evaluate(model, toy_dataset)


def test_evaluate_unknown_metric_raises_key_error(toy_dataset):
    with pytest.raises(KeyError):
        backtest.evaluate(_Spy(), toy_dataset, metrics=["nope"])
