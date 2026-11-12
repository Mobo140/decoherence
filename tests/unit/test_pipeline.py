"""pipeline: DataSpec validation, configuration generation and hashing."""
from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from src import pipeline
from src.data import scenarios

SEED = 3


def _data(**changes) -> pipeline.DataSpec:
    spec = pipeline.DataSpec(
        name="t", scenarios=("B",), n_per_scenario=10, window_length=20,
        samples_per_trajectory=3)
    return dataclasses.replace(spec, **changes)


def test_system_configs_coupling_group_replays_the_generator_of_the_sweep():
    sweep = (0.5, 1.0, 2.0)
    rng = np.random.default_rng(SEED)
    drawn = [scenarios.fixed_coupling_configs(4, j, rng) for j in sweep]
    for coupling, expected in zip(sweep, drawn):
        spec = pipeline.DataSpec(
            name="t", coupling=coupling, coupling_sweep=sweep,
            n_per_scenario=4)
        assert pipeline.system_configs(spec, SEED) == expected


def test_system_configs_nested_sets_are_prefixes_and_replace_the_criterion():
    big = pipeline.DataSpec(
        name="t", scenarios=("D",), n_per_scenario=8,
        criterion="entanglement_death")
    small = dataclasses.replace(big, n_trajectories=3)
    all_configs = pipeline.system_configs(big, SEED)
    assert pipeline.system_configs(small, SEED) == all_configs[:3]
    assert {c.decoherence_criterion.value for c in all_configs} == {
        "entanglement_death"}


def test_system_configs_concatenate_scenario_groups_in_order():
    spec = pipeline.DataSpec(name="t", scenarios=("C", "B"),
                             n_per_scenario=3)
    groups = scenarios.build_configs(3, SEED)
    assert pipeline.system_configs(spec, SEED) == groups["C"] + groups["B"]


def test_experiment_config_is_hashable_and_comparable():
    a = pipeline.ExperimentConfig(data=_data(), model="lstm_only")
    b = pipeline.ExperimentConfig(data=_data(), model="lstm_only")
    assert a == b and len({a, b}) == 1


def test_simulator_stochastic_gamma_gets_the_offset_seed():
    spec = pipeline.SimulatorSpec(
        "stochastic_gamma", (("s", 0.3), ("tau", 2.0)), seed_offset=13)
    sim = pipeline._simulator(spec, 5)
    twin = pipeline._simulator(spec, 5)
    assert (sim.s, sim.tau) == (0.3, 2.0)
    assert sim.rng.standard_normal() == twin.rng.standard_normal()
    other = pipeline._simulator(spec, 6)
    assert sim.rng.standard_normal() != other.rng.standard_normal()


def test_system_configs_fixed_coupling_group_takes_an_interaction():
    sweep = (0.5, 1.0)
    spec = pipeline.DataSpec(
        name="t", coupling=1.0, coupling_sweep=sweep, interaction="XXZ",
        n_per_scenario=3, criterion="coherence_energy")
    configs = pipeline.system_configs(spec, SEED)
    assert {c.interaction_type.value for c in configs} == {"XXZ"}
    assert {c.decoherence_criterion.value for c in configs} == {
        "coherence_energy"}
    default = pipeline.system_configs(
        dataclasses.replace(spec, interaction=None, criterion=None), SEED)
    assert {c.interaction_type.value for c in default} == {"TFIM"}


# --- Negative cases ---


@pytest.mark.parametrize("changes", [
    {"scenarios": ()},
    {"scenarios": ("Z",)},
    {"scenarios": ("B",), "cache": "tape"},
    {"scenarios": ("B",), "criterion": "nonsense"},
    {"scenarios": ("B",), "cache": "disk",
     "simulator": pipeline.SimulatorSpec("correlators")},
    {"scenarios": ("B",), "coupling": 1.0, "coupling_sweep": (1.0,)},
    {"scenarios": (), "coupling": 1.0, "coupling_sweep": (0.5,)},
])
def test_data_spec_invalid_arguments_raise_value_error(changes):
    with pytest.raises(ValueError):
        pipeline.DataSpec(name="t", **changes)


@pytest.mark.parametrize("changes", [
    {"scenarios": ("B",), "interaction": "XXZ"},
    {"scenarios": (), "coupling": 1.0, "coupling_sweep": (1.0,),
     "interaction": "ising"},
])
def test_data_spec_invalid_interaction_raises_value_error(changes):
    with pytest.raises(ValueError):
        pipeline.DataSpec(name="t", **changes)
