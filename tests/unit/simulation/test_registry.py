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


def test_every_registered_alias_builds_its_class():
    arguments = {
        "stochastic_gamma": dict(s=0.1, tau=1.0, seed=0),
        "parity_sector": dict(sector=1),
        "kondo": dict(rho_J=0.1, kT=0.05, kF_d=1.0, dim=3),
    }
    assert set(registry.SIMULATORS) == {
        "plain", "correlators", "entanglement", "stochastic_gamma",
        "parity_sector", "kondo"}
    for alias, cls in registry.SIMULATORS.items():
        built = registry.build_simulator(alias, **arguments.get(alias, {}))
        assert type(built) is cls


def test_each_call_builds_a_new_simulator():
    assert registry.build_simulator("plain") is not (
        registry.build_simulator("plain"))


def test_parameters_reach_the_constructor():
    sim = registry.build_simulator(
        "stochastic_gamma", s=0.3, tau=2.0, seed=4)
    assert (sim.s, sim.tau) == (0.3, 2.0)


def test_kondo_parameters_reach_the_constructor():
    sim = registry.build_simulator(
        "kondo", rho_J=0.2, kT=0.1, kF_d=2.0, dim=2)
    assert (sim.rho_J, sim.kT, sim.kF_d, sim.dim) == (0.2, 0.1, 2.0, 2)


def test_parity_sector_projects_the_initial_operator_from_the_left():
    config = scenarios.build_configs(1, 3)["E"][0]
    for sector in (+1, -1):
        sim = registry.build_simulator("parity_sector", sector=sector)
        for _ in range(5):
            # The hook is the whole difference to the plain simulator.
            prepared = sim._initial_state(config)
            projector = (qt.qeye([2, 2]) + sector * variants.PARITY) / 2
            assert (projector * prepared - prepared).norm() < 1e-9


# --- Negative cases ---


def test_unknown_alias_lists_the_valid_ones():
    with pytest.raises(KeyError, match="plain"):
        registry.build_simulator("nope")


def test_unknown_alias_message_names_the_alias_and_every_valid_one():
    with pytest.raises(KeyError) as excinfo:
        registry.build_simulator("nope")
    message = str(excinfo.value)
    assert "'nope'" in message
    for alias in registry.SIMULATORS:
        assert alias in message


@pytest.mark.parametrize("alias", ["Plain", "PLAIN", " plain", "", None, 3])
def test_alias_is_matched_exactly(alias):
    with pytest.raises(KeyError, match="Unknown simulator"):
        registry.build_simulator(alias)


def test_registry_cannot_be_modified():
    with pytest.raises(TypeError):
        registry.SIMULATORS["extra"] = simulator.QuTipSimulator


def test_parameter_the_simulator_does_not_take_raises_type_error():
    with pytest.raises(TypeError, match="foo|takes no arguments"):
        registry.build_simulator("plain", foo=1)


def test_missing_required_parameters_raise_type_error():
    with pytest.raises(TypeError, match="missing"):
        registry.build_simulator("stochastic_gamma")
    with pytest.raises(TypeError, match="sector"):
        registry.build_simulator("parity_sector")


def test_invalid_parameters_reach_the_constructor_validation():
    with pytest.raises(ValueError, match="sector must be"):
        registry.build_simulator("parity_sector", sector=0)
    with pytest.raises(ValueError, match="tau must be"):
        registry.build_simulator("stochastic_gamma", s=0.3, tau=0, seed=1)


def test_parity_sector_leaves_one_qubit_alone_and_checks_the_sector():
    with pytest.raises(ValueError):
        registry.build_simulator("parity_sector", sector=0)
    config = scenarios.build_configs(1, 3)["B"][0]
    sim = registry.build_simulator("parity_sector", sector=1)
    assert sim._initial_state(config).shape == (2, 2)
