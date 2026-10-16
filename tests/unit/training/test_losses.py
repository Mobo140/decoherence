"""Loss functions: values, the alias registry and the survival loss."""
from __future__ import annotations

import numpy as np
import pytest
import torch
import torch.nn as nn

from src.training.losses import (
    LOSSES,
    HuberLossFunction,
    MAELossFunction,
    MSELossFunction,
    QuantileLossFunction,
    SurvHuberLoss,
    get_loss,
)

PREDS = np.array([1.0, 2.0, 3.0], dtype=np.float32)
TARGETS = np.array([1.2, 1.8, 3.5], dtype=np.float32)


def test_huber_is_non_negative():
    assert HuberLossFunction(delta=1.0)(PREDS, TARGETS) >= 0


def test_mse_is_non_negative():
    assert MSELossFunction()(PREDS, TARGETS) >= 0


def test_mse_of_perfect_predictions_is_zero():
    assert MSELossFunction()(PREDS, PREDS) == pytest.approx(0.0, abs=1e-6)


def test_mae_is_non_negative():
    assert MAELossFunction()(PREDS, TARGETS) >= 0


def test_quantile_median_is_half_the_mae():
    q = QuantileLossFunction(quantile=0.5)
    mae = MAELossFunction()
    assert q(PREDS, TARGETS) == pytest.approx(
        mae(PREDS, TARGETS) / 2, rel=1e-4
    )


def test_get_loss_huber_alias_is_callable():
    assert get_loss("huber")(PREDS, TARGETS) >= 0


def test_get_loss_builds_a_quantile_from_its_alias():
    assert get_loss("quantile_0.9")(PREDS, TARGETS) >= 0


def test_losses_mapping_matches_get_loss():
    assert {"huber", "mse", "mae", "surv_huber"} <= set(LOSSES)
    for alias, loss in LOSSES.items():
        assert get_loss(alias) is loss


def test_as_torch_returns_a_module():
    assert isinstance(HuberLossFunction().as_torch(), nn.Module)


def test_surv_huber_uncensored_matches_huber():
    pred = torch.tensor([1.0, 2.0, 3.0])
    tgt = torch.tensor([1.2, 1.8, 3.5])
    zeros = torch.zeros(3)
    surv = SurvHuberLoss(delta=1.0)(pred, tgt, zeros)
    huber = get_loss("huber").as_torch()(pred, tgt)
    assert float(surv) == pytest.approx(float(huber), rel=1e-5)


def test_surv_huber_censored_is_zero_when_prediction_exceeds_the_bound():
    pred = torch.tensor([5.0, 6.0])
    bound = torch.tensor([3.0, 4.0])
    cens = torch.ones(2)
    loss = float(SurvHuberLoss()(pred, bound, cens))
    assert loss == pytest.approx(0.0, abs=1e-8)


def test_surv_huber_censored_penalizes_underprediction():
    pred = torch.tensor([1.0])
    bound = torch.tensor([4.0])
    assert float(SurvHuberLoss()(pred, bound, torch.ones(1))) > 0.0


# --- Negative cases ---


def test_quantile_outside_unit_interval_raises_value_error():
    with pytest.raises(ValueError):
        QuantileLossFunction(quantile=1.5)


def test_get_loss_unknown_alias_raises_key_error_listing_valid_ones():
    with pytest.raises(KeyError, match="huber"):
        get_loss("nonexistent_loss")


def test_get_loss_malformed_quantile_alias_raises_key_error():
    with pytest.raises(KeyError):
        get_loss("quantile_high")
