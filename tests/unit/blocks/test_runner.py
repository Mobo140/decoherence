"""blocks.runner and blocks.cache: seeds, parameters and the state cache."""
import numpy as np
import pytest
import qutip as qt

from src.blocks import cache, channels, evolve, runner, system
from src.blocks.recipe import Param, Recipe, block


def _recipe(**overrides):
    fields = dict(
        system=block("system.single_qubit", omega=Param("omega")),
        channels=(
            block("channels.jump", op_type="sigma_minus", gamma=0.3),
            block("channels.ou_modulation", s=0.4, tau=1.0)),
        initial=block("initial.random_pure"),
        observe=(block("observe.pauli_means"), block("observe.purity"),
                 block("observe.coherence_l1"),
                 block("observe.gaussian_noise", sigma=0.01)),
        target=block("target.coherence_crossing", threshold=0.5),
        t_max=2.0, dt=0.1,
        sampler=block("samplers.uniform", omega=(0.5, 1.5)))
    fields.update(overrides)
    return Recipe(**fields)


def _same(a, b):
    assert a.index == b.index and a.params == b.params
    assert list(a.observations.series) == list(b.observations.series)
    for name, values in a.observations.series.items():
        np.testing.assert_array_equal(values, b.observations.series[name])
    assert a.target == b.target


# --- Negative cases ---


@pytest.mark.parametrize("n", [0, -2, 1.5])
def test_invalid_trajectory_count_raises(n):
    with pytest.raises(ValueError, match="n_trajectories must be"):
        runner.run(_recipe(), n, seed=0)


@pytest.mark.parametrize("seed", [-1, 1.5])
def test_invalid_seed_raises(seed):
    with pytest.raises(ValueError, match="seed must be a non-negative"):
        runner.run(_recipe(), 1, seed=seed)


def test_index_out_of_range_raises():
    with pytest.raises(ValueError, match=r"indices \[3\] are outside 0..2"):
        runner.run(_recipe(), 3, seed=0, indices=[0, 3])


def test_indices_with_legacy_seeds_raise():
    with pytest.raises(ValueError, match="legacy seeds"):
        runner.run(_recipe(seeds="legacy"), 2, seed=0, indices=[1])


def test_param_that_the_sampler_does_not_give_raises():
    bad = _recipe(sampler=block("samplers.fixed", other=1.0))
    with pytest.raises(ValueError, match="Param 'omega' is not sampled"):
        runner.run(bad, 1, seed=0)


def test_sampler_with_too_few_rows_raises():
    rows = (tuple({"omega": 1.0}.items()),)
    bad = _recipe(sampler=block("samplers.from_rows", rows=rows))
    with pytest.raises(ValueError, match="2 trajectories were asked for"):
        runner.run(bad, 2, seed=0)


def test_cache_miss_returns_none(tmp_path):
    assert cache.load(tmp_path, "0" * 40) is None


# --- Positive cases: parameters and seeds ---


def test_run_returns_one_run_per_trajectory_with_resolved_parameters():
    runs = runner.run(_recipe(), 3, seed=1)
    assert [r.index for r in runs] == [0, 1, 2]
    for r in runs:
        assert r.system_params == {"omega": r.params["omega"]}
        assert r.system_fn == "system.single_qubit"
        assert r.jump_op == "sigma_minus" and r.states is None
        assert 0.5 <= r.params["omega"] <= 1.5
        assert r.observations.times.shape == (21,)
        assert list(r.observations.series)[-2:] == ["purity", "coherence_l1"]


def test_trajectory_i_is_the_same_for_any_number_of_trajectories():
    short = runner.run(_recipe(), 3, seed=5)
    long = runner.run(_recipe(), 8, seed=5)
    for a, b in zip(short, long):
        _same(a, b)


def test_trajectory_i_is_the_same_in_any_evaluation_order():
    full = runner.run(_recipe(), 4, seed=5)
    shuffled = runner.run(_recipe(), 4, seed=5, indices=[3, 0, 2])
    for run_, i in zip(shuffled, [3, 0, 2]):
        _same(run_, full[i])


def test_different_seeds_and_different_trajectories_differ():
    a = runner.run(_recipe(), 2, seed=1)
    b = runner.run(_recipe(), 2, seed=2)
    assert a[0].params != b[0].params
    assert a[0].params != a[1].params


def test_stages_have_independent_streams():
    # Changing the noise block must not change the sampled parameters or
    # the states, only the observations.
    base = runner.run(_recipe(), 1, seed=3, keep_states=True)[0]
    noisier = _recipe(observe=_recipe().observe[:3] + (
        block("observe.gaussian_noise", sigma=0.05),))
    other = runner.run(noisier, 1, seed=3, keep_states=True)[0]
    assert base.params == other.params
    for x, y in zip(base.states.rho, other.states.rho):
        np.testing.assert_array_equal(x, y)
    assert not np.array_equal(
        base.observations.series["sigma_x"],
        other.observations.series["sigma_x"])


def test_keep_states_returns_the_states():
    run_ = runner.run(_recipe(), 1, seed=0, keep_states=True)[0]
    assert isinstance(run_.states, evolve.States)
    assert len(run_.states.rho) == 21


def test_time_params_are_resolved_per_trajectory():
    recipe = _recipe(
        t_max=Param("t_max"), dt=Param("dt"),
        sampler=block("samplers.grid", omega=[1.0], t_max=[1.0, 2.0],
                      dt=[0.1]))
    runs = runner.run(recipe, 2, seed=0)
    assert [len(r.observations.times) for r in runs] == [11, 21]


def test_composition_of_kondo_and_ou_modulates_the_bath_rates():
    recipe = _recipe(
        system=block("system.two_qubit", J=0.8, interaction="XXZ"),
        channels=(
            block("channels.kondo_bath", rho_J=0.1, kT=0.5, k_F=1.0, dim=2),
            block("channels.ou_modulation", s=0.4, tau=1.0)),
        sampler=None)
    run_ = runner.run(recipe, 1, seed=0, keep_states=True)[0]
    assert run_.jump_op is None
    assert run_.observations.series["purity"][-1] < 1.0
    assert run_.observations.series["sigma_x1"].shape == (21,)


# --- Positive cases: cache ---


def test_cache_hit_equals_miss_bitwise(tmp_path):
    miss = runner.run(_recipe(), 3, seed=2, cache_dir=tmp_path)
    assert len(list(tmp_path.glob("*.npz"))) == 3
    hit = runner.run(_recipe(), 3, seed=2, cache_dir=tmp_path)
    assert len(list(tmp_path.glob("*.npz"))) == 3
    plain = runner.run(_recipe(), 3, seed=2)
    for a, b, c in zip(miss, hit, plain):
        _same(a, b)
        _same(a, c)


def test_cache_hit_does_not_call_the_integrator(tmp_path, monkeypatch):
    runner.run(_recipe(), 1, seed=2, cache_dir=tmp_path)

    def fail(*args, **kwargs):
        raise AssertionError("mesolve was called on a cache hit")

    monkeypatch.setattr(runner.evolve, "master_equation", fail)
    runner.run(_recipe(), 1, seed=2, cache_dir=tmp_path)


def test_cache_key_changes_with_the_physics_but_not_with_the_observation(
        tmp_path):
    runner.run(_recipe(), 1, seed=2, cache_dir=tmp_path)
    runner.run(_recipe(), 1, seed=3, cache_dir=tmp_path)
    assert len(list(tmp_path.glob("*.npz"))) == 2
    other_target = _recipe(
        target=block("target.coherence_crossing", threshold=0.3))
    runner.run(other_target, 1, seed=2, cache_dir=tmp_path)
    assert len(list(tmp_path.glob("*.npz"))) == 2


def test_stored_states_are_complex128_and_round_trip(tmp_path):
    sys_ = system.single_qubit(1.0)
    chans = channels.jump(sys_, "sigma_z", 0.1)
    rho0 = qt.fock_dm(2, 0)
    times = np.linspace(0.0, 1.0, 5)
    states = evolve.master_equation(sys_, chans, rho0, times)
    key = cache.states_key({"a": 1}, sys_, chans, rho0.full(), times)
    cache.store(tmp_path, key, states)
    loaded = cache.load(tmp_path, key)
    np.testing.assert_array_equal(loaded.times, states.times)
    for a, b in zip(loaded.rho, states.rho):
        assert a.dtype == np.complex128
        np.testing.assert_array_equal(a, b)
    assert cache.states_key(
        {"a": 2}, sys_, chans, rho0.full(), times) != key


# --- Positive cases: legacy seeds ---


def test_legacy_seeds_follow_the_global_generator_in_order():
    np.random.seed(7)
    expected = np.random.randn(2) + 1j * np.random.randn(2)
    expected /= np.linalg.norm(expected)
    runs = runner.run(_recipe(seeds="legacy"), 1, seed=7, keep_states=True)
    np.testing.assert_allclose(
        runs[0].states.rho[0], np.outer(expected, expected.conj()))


def test_legacy_global_generator_advances_the_same_on_a_cache_hit(tmp_path):
    recipe = _recipe(seeds="legacy")
    runner.run(recipe, 3, seed=7, cache_dir=tmp_path)
    after_miss = np.random.get_state()[1].copy()
    pos_miss = np.random.get_state()[2]
    hit = runner.run(recipe, 3, seed=7, cache_dir=tmp_path)
    assert np.random.get_state()[2] == pos_miss
    np.testing.assert_array_equal(np.random.get_state()[1], after_miss)
    plain = runner.run(recipe, 3, seed=7)
    for a, b in zip(hit, plain):
        _same(a, b)


def test_legacy_shared_generator_offset_changes_the_ou_path():
    a = runner.run(_recipe(seeds="legacy"), 1, seed=7)[0]
    b = runner.run(
        _recipe(seeds="legacy", legacy_rng_offset=1), 1, seed=7)[0]
    assert not np.array_equal(
        a.observations.series["purity"], b.observations.series["purity"])


# --- evolution stage and finite shots ---


def _sequential_recipe(**overrides):
    fields = dict(
        system=block("system.single_qubit", omega=1.0),
        channels=(block("channels.jump", op_type="sigma_minus", gamma=0.3),),
        initial=block("initial.random_pure"),
        evolve=block("evolve.sequential_record", basis="z"),
        observe=(block("observe.record_series"),), target=None,
        t_max=2.0, dt=0.1)
    fields.update(overrides)
    return Recipe(**fields)


def _shots_recipe(settings="tomography"):
    return _recipe(
        observe=(block("observe.finite_shots", n_shots=50,
                       settings=settings),),
        sampler=block("samplers.fixed", omega=1.0))


def test_states_cache_with_sequential_record_raises(tmp_path):
    with pytest.raises(ValueError, match="master_equation only"):
        runner.run(_sequential_recipe(), 1, seed=0, cache_dir=tmp_path)


def test_sequential_record_recipe_runs_without_a_target():
    run_ = runner.run(_sequential_recipe(), 1, seed=3)[0]
    assert run_.target is None and run_.states is None
    assert run_.record.outcomes.shape == (21, 1)
    assert list(run_.observations.series) == ["outcome_q1"]
    assert set(run_.observations.series["outcome_q1"]) <= {0.0, 1.0}


def test_sequential_record_is_deterministic_per_seed_and_index():
    a = runner.run(_sequential_recipe(), 4, seed=3)
    b = runner.run(_sequential_recipe(), 2, seed=3)
    for x, y in zip(a, b):
        np.testing.assert_array_equal(x.record.outcomes, y.record.outcomes)
    assert not all(np.array_equal(a[0].record.outcomes, r.record.outcomes)
                   for r in a[1:])


def test_finite_shots_is_a_recipe_observation_with_its_own_stream():
    runs = runner.run(_shots_recipe(), 2, seed=5)
    again = runner.run(_shots_recipe(), 2, seed=5)
    exact = runner.run(
        _recipe(observe=(block("observe.pauli_means"),),
                sampler=block("samplers.fixed", omega=1.0)), 2, seed=5)
    for a, b, c in zip(runs, again, exact):
        for name, values in a.observations.series.items():
            np.testing.assert_array_equal(values, b.observations.series[name])
        assert a.params == c.params
        assert not np.array_equal(
            a.observations.series["sigma_z"], c.observations.series["sigma_z"])
        assert list(a.observations.series)[-2:] == ["purity", "coherence_l1"]


def test_two_qubits_with_local_shots_have_no_model_tail():
    recipe = Recipe(
        system=block("system.two_qubit", J=0.8, interaction="XXZ"),
        channels=(block("channels.jump", op_type="sigma_z", gamma=0.2),),
        initial=block("initial.random_pure"),
        observe=(block("observe.finite_shots", n_shots=20, settings="local"),
                 block("observe.model_layout")),
        target=None, t_max=1.0, dt=0.5)
    with pytest.raises(ValueError, match="need tomography"):
        runner.run(recipe, 1, seed=0)
