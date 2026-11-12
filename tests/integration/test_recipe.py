"""Recipes end to end: legacy equivalence, composition, seeds, cache."""
import numpy as np
import pytest

from src.blocks import runner
from src.blocks.recipe import Param, Recipe, block
from src.data import recipe_dataset
from src.data.scenarios import build_configs
from src.simulation.simulator import QuTipSimulator


def _row(config):
    """The recipe parameters of a legacy config, as sampler rows."""
    dissipator = config.dissipator
    return tuple(sorted({
        "omega": config.omega, "J": config.J,
        "interaction": config.interaction_type.value,
        "op_type": dissipator.operator_type.value,
        "coeffs": dissipator.gamma_coeffs,
        "t_max": config.t_max, "dt": config.dt,
        "threshold": config.decoherence_threshold,
        "criterion": config.decoherence_criterion.value,
    }.items()))


def _legacy_recipe(system_block, rows):
    return Recipe(
        system=system_block,
        channels=(
            block("channels.jump", op_type=Param("op_type"), gamma=1.0),
            block("channels.bernstein_profile", coeffs=Param("coeffs"))),
        initial=block("initial.random_pure"),
        observe=(block("observe.pauli_means"), block("observe.purity"),
                 block("observe.coherence_l1")),
        target=block("target.coherence_crossing",
                     threshold=Param("threshold"),
                     criterion=Param("criterion")),
        t_max=Param("t_max"), dt=Param("dt"),
        sampler=block("samplers.from_rows", rows=rows), seeds="legacy")


# --- (i) legacy equivalence ---


@pytest.mark.parametrize("scenario, system_block", [
    ("B", block("system.single_qubit", omega=Param("omega"))),
    ("D", block("system.two_qubit", J=Param("J"),
                interaction=Param("interaction"))),
])
def test_legacy_seeds_reproduce_the_quantum_simulator_bit_for_bit(
        scenario, system_block):
    configs = build_configs(3, 0)[scenario]
    recipe = _legacy_recipe(system_block, tuple(_row(c) for c in configs))
    runs = runner.run(recipe, len(configs), seed=11)

    np.random.seed(11)
    for config, run_ in zip(configs, runs):
        legacy = QuTipSimulator().simulate(config)
        np.testing.assert_array_equal(run_.observations.times, legacy.times)
        assert list(run_.observations.series) == legacy.feature_names
        for name, values in legacy.observables.items():
            np.testing.assert_array_equal(
                run_.observations.series[name], values)
        assert run_.target.time == legacy.t_decoh
        assert run_.target.censored == legacy.censored


# --- (ii) composition ---


def test_kondo_and_stochastic_rate_compose_on_two_qubits():
    def make(with_ou):
        modifiers = (block("channels.ou_modulation", s=0.5, tau=1.0),
                     ) if with_ou else ()
        return Recipe(
            system=block("system.two_qubit", J=0.8, interaction="XXZ"),
            channels=(block("channels.kondo_bath", rho_J=0.2, kT=0.3,
                            k_F=1.0, dim=2),) + modifiers,
            initial=block("initial.random_pure"),
            observe=(block("observe.pauli_means"),
                     block("observe.purity"),
                     block("observe.coherence_l1")),
            target=block("target.coherence_crossing", threshold=0.5),
            t_max=20.0, dt=0.5)

    plain = runner.run(make(False), 1, seed=4, keep_states=True)[0]
    modulated = runner.run(make(True), 1, seed=4, keep_states=True)[0]
    # Same initial state (the streams of the stages are independent), but
    # the modulated rates change the evolution.
    np.testing.assert_array_equal(
        plain.states.rho[0], modulated.states.rho[0])
    assert not np.allclose(
        plain.observations.series["purity"],
        modulated.observations.series["purity"])
    assert modulated.observations.series["purity"][-1] < 1.0


# --- (iii) spawn determinism ---


def _spawn_recipe():
    return Recipe(
        system=block("system.two_qubit", J=Param("J"), interaction="XXZ"),
        channels=(block("channels.jump", op_type="sigma_minus", gamma=0.2),
                  block("channels.ou_modulation", s=Param("s"), tau=1.0)),
        initial=block("initial.random_pure"),
        observe=(block("observe.pauli_means"), block("observe.purity"),
                 block("observe.coherence_l1"),
                 block("observe.ar1_noise", sigma=0.01, phi=0.5)),
        target=block("target.coherence_crossing", threshold=0.5),
        t_max=5.0, dt=0.25,
        sampler=block("samplers.uniform", J=(0.2, 1.0), s=(0.1, 0.5)))


def _equal(a, b):
    assert a.params == b.params and a.target == b.target
    for name, values in a.observations.series.items():
        np.testing.assert_array_equal(values, b.observations.series[name])


def test_spawned_trajectory_depends_only_on_the_seed_and_its_index():
    few = runner.run(_spawn_recipe(), 2, seed=9)
    many = runner.run(_spawn_recipe(), 6, seed=9)
    for a, b in zip(few, many):
        _equal(a, b)
    reordered = runner.run(_spawn_recipe(), 6, seed=9, indices=[5, 1, 0])
    for run_, i in zip(reordered, [5, 1, 0]):
        _equal(run_, many[i])


# --- (iv) cache ---


def test_cache_hit_equals_miss(tmp_path):
    miss = runner.run(_spawn_recipe(), 3, seed=9, cache_dir=tmp_path)
    hit = runner.run(_spawn_recipe(), 3, seed=9, cache_dir=tmp_path)
    assert len(list(tmp_path.glob("*.npz"))) == 3
    for a, b in zip(miss, hit):
        _equal(a, b)


# --- (v) dataset ---


def test_runs_make_a_valid_dataset():
    runs = runner.run(_spawn_recipe(), 8, seed=9)
    ds = recipe_dataset.to_dataset(
        runs, window_length=8, horizon=1.0, samples_per_trajectory=2,
        train_ratio=0.6, val_ratio=0.2, min_window_gap=0,
        include_censored=True, rng=np.random.default_rng(0))
    assert len(ds.sequences) > 0
    assert ds.sequences.shape[2] == len(ds.feature_names)
    assert set(ds.context) == {"interaction_type", "dissipator_type", "J"}
    assert np.all(ds.remaining_time > 0)
