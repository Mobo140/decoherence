"""run_cross_system.variant_of: predictor class to PredictorVariant."""
from __future__ import annotations

from src.application.run_cross_system import variant_of
from src.domain.value_objects import PredictorVariant
from src.models.lstm import LSTMPredictor
from src.models.physics_only import PhysicsOnlyPredictor
from src.models.stretched_exp import StretchedExpPredictor
from src.models.transformer import TransformerPredictor


def test_variant_of_maps_each_predictor_class_to_its_variant():
    assert variant_of(TransformerPredictor()) == PredictorVariant.TRANSFORMER
    assert variant_of(PhysicsOnlyPredictor()) == PredictorVariant.PHYSICS_ONLY
    assert variant_of(StretchedExpPredictor()) == PredictorVariant.STRETCHED_EXP
    assert (variant_of(LSTMPredictor(use_physics_prior=False))
            == PredictorVariant.LSTM_ONLY)
    assert variant_of(LSTMPredictor()) == PredictorVariant.PHYSICS_LSTM
