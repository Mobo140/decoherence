"""Simulator registry: build a simulator from a string alias.

`SIMULATORS` maps an alias to the class that implements it:

    plain             QuTipSimulator, the published physics
    correlators       also records the nine two-body Pauli correlators (R23)
    entanglement      correlators plus the Wootters concurrence (R24)
    stochastic_gamma  gamma(t) times a log-normal Ornstein-Uhlenbeck factor
                      (R21); takes `s`, `tau` and `seed`
"""
from __future__ import annotations

import types
from collections.abc import Mapping

from src.domain import ports
from src.simulation import simulator
from src.simulation import variants

SIMULATORS: Mapping[str, type[ports.ISimulator]] = types.MappingProxyType({
    "plain": simulator.QuTipSimulator,
    "correlators": variants.CorrelatorSimulator,
    "entanglement": variants.EntanglementSimulator,
    "stochastic_gamma": variants.StochasticGammaSimulator,
})


def build_simulator(alias: str, **params: object) -> ports.ISimulator:
    """Builds the simulator registered under an alias.

    Args:
        alias: A key of `SIMULATORS`.
        **params: Constructor arguments, e.g. `s`, `tau` and `seed` for
            "stochastic_gamma".

    Returns:
        A new simulator.

    Raises:
        KeyError: If `alias` is not in `SIMULATORS`; the message lists the
            valid aliases.
    """
    if alias not in SIMULATORS:
        raise KeyError(
            f"Unknown simulator {alias!r}. "
            f"Valid aliases: {sorted(SIMULATORS)}.")
    return SIMULATORS[alias](**params)
