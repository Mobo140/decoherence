"""Simulator registry: aliases build the variants the experiments use."""
import pytest
import qutip as qt

from src.data import scenarios
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


def test_parity_sector_projects_the_initial_operator_from_the_left():
    config = scenarios.build_configs(1, 3)["E"][0]
    for sector in (+1, -1):
        sim = registry.build_simulator("parity_sector", sector=sector)
        for _ in range(5):
            # The hook is the whole difference to the plain simulator.
            prepared = sim._initial_state(config)
            projector = (qt.qeye([2, 2]) + sector * variants.PARITY) / 2
            assert (projector * prepared - prepared).norm() < 1e-9


def test_parity_sector_leaves_one_qubit_alone_and_checks_the_sector():
    with pytest.raises(ValueError):
        registry.build_simulator("parity_sector", sector=0)
    config = scenarios.build_configs(1, 3)["B"][0]
    sim = registry.build_simulator("parity_sector", sector=1)
    assert sim._initial_state(config).shape == (2, 2)
