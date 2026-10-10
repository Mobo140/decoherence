"""Simulator registry: aliases build the variants the experiments use."""
import pytest

from src.simulation import registry
from src.simulation import simulator
from src.simulation import variants


def test_aliases_build_the_variants():
    assert type(registry.build_simulator("plain")) is simulator.QuTipSimulator
    assert isinstance(registry.build_simulator("correlators"),
                      variants.CorrelatorSimulator)
    assert isinstance(registry.build_simulator("entanglement"),
                      variants.EntanglementSimulator)


def test_parameters_reach_the_constructor():
    sim = registry.build_simulator(
        "stochastic_gamma", s=0.3, tau=2.0, seed=4)
    assert (sim.s, sim.tau) == (0.3, 2.0)


def test_unknown_alias_lists_the_valid_ones():
    with pytest.raises(KeyError, match="plain"):
        registry.build_simulator("nope")
