"""Model registry: build a predictor from a string alias.

`MODELS` maps an alias to a factory that constructs exactly the predictor
the experiments use under that name:

    physics_only       OLS-slope formula, no training (t_max=20)
    stretched_exp      stretched-exponential fit, no training (t_max=20)
    lstm_only          BiLSTM without the physics prior
    physics_lstm       BiLSTM with the physics prior, residual output
    physics_feature    BiLSTM with the prior as an input only, not as the
                       base of the output
    transformer        Transformer, default size (no physics prior)
    transformer_large  Transformer d_model=128, 3 layers, dim_ff=512
"""
from __future__ import annotations

import functools
import types
from collections.abc import Callable, Mapping

from src.domain import ports
from src.models import lstm
from src.models import physics_only
from src.models import stretched_exp
from src.models import transformer

MODELS: Mapping[str, Callable[..., ports.IPredictor]] = types.MappingProxyType({
    "physics_only": functools.partial(
        physics_only.PhysicsOnlyPredictor, t_max=20.0),
    "stretched_exp": functools.partial(
        stretched_exp.StretchedExpPredictor, t_max=20.0),
    "lstm_only": functools.partial(
        lstm.LSTMPredictor, use_physics_prior=False),
    "physics_lstm": functools.partial(
        lstm.LSTMPredictor, use_physics_prior=True),
    "physics_feature": functools.partial(
        lstm.LSTMPredictor, use_physics_prior=True, residual_output=False),
    "transformer": transformer.TransformerPredictor,
    "transformer_large": functools.partial(
        transformer.TransformerPredictor,
        d_model=128, n_heads=4, n_layers=3, dim_ff=512, dropout=0.1),
})


def build_model(alias: str, **overrides: object) -> ports.IPredictor:
    """Builds the predictor registered under an alias.

    Args:
        alias: A key of `MODELS`.
        **overrides: Constructor arguments that replace the alias's
            defaults, e.g. `t_max=max(c.t_max for c in configs)`.

    Returns:
        A new, untrained predictor.

    Raises:
        KeyError: If `alias` is not in `MODELS`; the message lists the
            valid aliases.
    """
    if alias not in MODELS:
        raise KeyError(
            f"Unknown model {alias!r}. Valid aliases: {sorted(MODELS)}.")
    return MODELS[alias](**overrides)
