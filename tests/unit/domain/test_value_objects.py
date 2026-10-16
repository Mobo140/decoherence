"""Domain value objects validate themselves on construction."""
from __future__ import annotations

import pytest

from src.domain.value_objects import (
    ExperimentSpec,
    NoiseConfig,
    PredictorVariant,
)


def test_experiment_spec_accepts_valid_arguments():
    spec = ExperimentSpec(
        name="test", predictor_variant=PredictorVariant.PHYSICS_LSTM,
        window_fraction=0.20,
    )
    assert spec.name == "test"


# --- Negative cases ---


def test_noise_config_negative_sigma_raises_value_error():
    with pytest.raises(ValueError):
        NoiseConfig(sigma=-0.1)


def test_experiment_spec_zero_window_fraction_raises_value_error():
    with pytest.raises(ValueError):
        ExperimentSpec(name="bad", window_fraction=0.0)


def test_experiment_spec_too_few_trajectories_raises_value_error():
    with pytest.raises(ValueError):
        ExperimentSpec(name="bad", n_trajectories=5, window_fraction=0.2)
