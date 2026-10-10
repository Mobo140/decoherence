"""Model registry: aliases build what the experiments construct."""
from __future__ import annotations

import pytest

from src.models import lstm
from src.models import physics_only
from src.models import registry
from src.models import stretched_exp
from src.models import transformer


def test_every_alias_builds_a_predictor():
    for alias in registry.MODELS:
        assert registry.build_model(alias) is not None


def test_physics_baselines_use_the_experiment_horizon():
    p = registry.build_model("physics_only")
    assert isinstance(p, physics_only.PhysicsOnlyPredictor)
    assert p.t_max == 20.0
    s = registry.build_model("stretched_exp")
    assert isinstance(s, stretched_exp.StretchedExpPredictor)
    assert s.t_max == 20.0


def test_lstm_aliases_differ_in_prior_and_output():
    only = registry.build_model("lstm_only")
    full = registry.build_model("physics_lstm")
    feature = registry.build_model("physics_feature")
    assert all(isinstance(m, lstm.LSTMPredictor) for m in (only, full, feature))
    assert (only.use_physics_prior, only.residual_output) == (False, True)
    assert (full.use_physics_prior, full.residual_output) == (True, True)
    assert (feature.use_physics_prior, feature.residual_output) == (True, False)


def test_transformer_sizes():
    small = registry.build_model("transformer")
    large = registry.build_model("transformer_large")
    assert isinstance(small, transformer.TransformerPredictor)
    assert (small.d_model, small.n_layers, small.dim_ff) == (64, 2, 256)
    assert (large.d_model, large.n_heads, large.n_layers) == (128, 4, 3)
    assert (large.dim_ff, large.dropout) == (512, 0.1)


def test_overrides_replace_alias_defaults():
    assert registry.build_model("physics_only", t_max=7.5).t_max == 7.5
    assert registry.build_model("physics_lstm", hidden_size=8).hidden_size == 8
    assert registry.build_model("transformer_large", d_model=32).d_model == 32


def test_models_are_independent_instances():
    a = registry.build_model("physics_lstm")
    b = registry.build_model("physics_lstm")
    assert a is not b


def test_unknown_alias_lists_the_valid_ones():
    with pytest.raises(KeyError, match="physics_lstm"):
        registry.build_model("nonexistent")
