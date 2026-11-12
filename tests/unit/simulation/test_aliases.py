"""QuTipSimulator aliases: the published variants as recipes."""
import numpy as np
import pytest

from src.blocks.observe import CORR
from src.domain import value_objects
from src.simulation.simulator import QuTipSimulator

NAN = float("nan")


def _config(n_qubits, gamma=0.5, **overrides):
    params = dict(
        n_qubits=n_qubits, omega=1.0, J=0.8, t_max=4.0, dt=0.1,
        dissipator=value_objects.DissipatorConfig.constant(gamma))
    params.update(overrides)
    return value_objects.SystemConfig(**params)


def _varying(n_qubits, **overrides):
    return _config(
        n_qubits, dissipator=value_objects.DissipatorConfig(
            gamma_coeffs=(0.4, 0.5, 0.3)), **overrides)


ONE, TWO = value_objects.QubitCount.ONE, value_objects.QubitCount.TWO


def _run(sim, config, seed=3):
    np.random.seed(seed)
    return sim.simulate(config)


def _same(a, b):
    assert a.feature_names == b.feature_names
    for name in a.feature_names:
        np.testing.assert_array_equal(a.observables[name], b.observables[name])
    assert (a.t_decoh, a.censored) == (b.t_decoh, b.censored)


# --- Negative cases ---


def test_unknown_alias_raises_key_error_naming_the_valid_ones():
    with pytest.raises(KeyError, match="plain"):
        QuTipSimulator("nope")


def test_every_alias_with_missing_arguments_raises_type_error():
    for alias in ("stochastic_gamma", "parity_sector", "kondo"):
        with pytest.raises(TypeError, match="missing"):
            QuTipSimulator(alias)


def test_unexpected_argument_raises_type_error():
    with pytest.raises(TypeError, match="unexpected"):
        QuTipSimulator("plain", foo=1)


@pytest.mark.parametrize("s", [-0.1, NAN])
def test_stochastic_gamma_invalid_strength_raises_value_error(s):
    with pytest.raises(ValueError, match="s must be"):
        QuTipSimulator("stochastic_gamma", s=s, tau=1.0, seed=0)


def test_stochastic_gamma_negative_seed_raises_value_error():
    with pytest.raises(ValueError, match="non-negative"):
        QuTipSimulator("stochastic_gamma", s=0.3, tau=1.0, seed=-1)


def test_parity_sector_zero_raises_value_error():
    with pytest.raises(ValueError, match=r"sector must be \+1 or -1"):
        QuTipSimulator("parity_sector", sector=0)


def test_kondo_invalid_distance_raises_value_error():
    with pytest.raises(ValueError, match="kF_d"):
        QuTipSimulator("kondo", rho_J=0.1, kT=0.1, kF_d=-1.0, dim=3)


def test_simulate_of_a_non_config_raises_type_error():
    with pytest.raises(TypeError, match="SystemConfig"):
        QuTipSimulator().simulate("B")


# --- Positive cases: stochastic gamma ---


def test_stochastic_gamma_zero_strength_reproduces_the_plain_simulator():
    config = _varying(ONE)
    sto = QuTipSimulator("stochastic_gamma", s=0.0, tau=0.5, seed=1)
    _same(_run(QuTipSimulator(), config), _run(sto, config))


def test_stochastic_gamma_changes_the_trajectory_of_a_varying_rate():
    config = _varying(ONE)
    sto = QuTipSimulator("stochastic_gamma", s=0.6, tau=0.5, seed=1)
    a, b = _run(QuTipSimulator(), config), _run(sto, config)
    assert not np.allclose(
        a.observables["coherence_l1"], b.observables["coherence_l1"])


def test_stochastic_gamma_leaves_a_constant_rate_alone_but_still_draws():
    config = _config(ONE)
    sto = QuTipSimulator("stochastic_gamma", s=0.6, tau=0.5, seed=1)
    _same(_run(QuTipSimulator(), config), _run(sto, config))
    # The path was drawn although it was not used: the stream moved on.
    twin = QuTipSimulator("stochastic_gamma", s=0.6, tau=0.5, seed=1)
    assert sto._rng.standard_normal() != twin._rng.standard_normal()


def test_stochastic_gamma_same_seed_same_path_and_the_stream_advances():
    config = _varying(ONE)
    a = QuTipSimulator("stochastic_gamma", s=0.4, tau=1.0, seed=8)
    b = QuTipSimulator("stochastic_gamma", s=0.4, tau=1.0, seed=8)
    c = QuTipSimulator("stochastic_gamma", s=0.4, tau=1.0, seed=9)
    first_a, first_b, first_c = (_run(x, config) for x in (a, b, c))
    _same(first_a, first_b)
    assert not np.allclose(first_a.observables["coherence_l1"],
                           first_c.observables["coherence_l1"])
    # The second trajectory of one simulator uses the next part of the path.
    second_a = _run(a, config)
    assert not np.allclose(first_a.observables["coherence_l1"],
                           second_a.observables["coherence_l1"])


def test_stochastic_gamma_without_a_fixed_strength_draws_it_per_trajectory():
    config = _varying(ONE)
    sim = QuTipSimulator("stochastic_gamma", s=None, tau=None, seed=2)
    a, b = _run(sim, config), _run(sim, config)
    assert not np.allclose(a.observables["coherence_l1"],
                           b.observables["coherence_l1"])


# --- Positive cases: extra observables ---


def test_correlators_add_the_nine_columns_before_purity_and_coherence():
    names = _run(QuTipSimulator("correlators"), _config(TWO)).feature_names
    assert names[-2:] == ["purity", "coherence_l1"]
    assert names[6:-2] == list(CORR)


def test_entanglement_adds_the_concurrence_after_the_correlators():
    names = _run(QuTipSimulator("entanglement"), _config(TWO)).feature_names
    assert names[-3:] == ["concurrence", "purity", "coherence_l1"]


def test_extra_observables_leave_the_base_columns_unchanged():
    plain = _run(QuTipSimulator(), _config(TWO))
    full = _run(QuTipSimulator("entanglement"), _config(TWO))
    for name in plain.feature_names:
        np.testing.assert_array_equal(
            plain.observables[name], full.observables[name])
    assert plain.t_decoh == full.t_decoh


@pytest.mark.parametrize("alias", ["correlators", "entanglement"])
def test_extra_observables_on_a_one_qubit_system_add_nothing(alias):
    # Kept: the extras belong to the two-qubit branch, a one-qubit run
    # records the plain set.
    traj = _run(QuTipSimulator(alias), _config(ONE))
    assert traj.feature_names == [
        "sigma_x", "sigma_y", "sigma_z", "purity", "coherence_l1"]


# --- Positive cases: parity sector ---


def test_parity_sector_changes_two_qubit_runs_only():
    config = _config(TWO)
    sector = QuTipSimulator("parity_sector", sector=1)
    plain, projected = _run(QuTipSimulator(), config), _run(sector, config)
    assert not np.allclose(
        plain.observables["sigma_z1"], projected.observables["sigma_z1"])
    _same(_run(QuTipSimulator(), _config(ONE)), _run(sector, _config(ONE)))


# --- Positive cases: Kondo bath ---


def _kondo(**overrides):
    params = dict(rho_J=0.3, kT=0.5, kF_d=1.0, dim=3)
    params.update(overrides)
    return QuTipSimulator("kondo", **params)


@pytest.mark.parametrize("n_qubits", [ONE, TWO])
def test_kondo_returns_a_finite_decaying_trajectory(n_qubits):
    result = _run(_kondo(), _config(n_qubits, t_max=40.0, dt=1.0))
    assert result.times.shape == result.observables["purity"].shape
    assert np.all(np.isfinite(result.observables["coherence_l1"]))
    assert result.observables["purity"][-1] < result.observables["purity"][0]


def test_kondo_ignores_the_phenomenological_rate():
    a = _run(_kondo(), _config(TWO, gamma=0.5, t_max=10.0, dt=0.5))
    b = _run(_kondo(), _config(TWO, gamma=2.0, t_max=10.0, dt=0.5))
    _same(a, b)


def test_kondo_without_coupling_conserves_purity():
    result = _run(_kondo(rho_J=0.0), _config(TWO))
    np.testing.assert_allclose(result.observables["purity"], 1.0, atol=1e-5)


def test_kondo_two_qubit_result_depends_on_the_pair_distance():
    config = _config(TWO, t_max=30.0, dt=1.0)
    a = _run(_kondo(kF_d=0.0, dim=1), config, seed=4)
    b = _run(_kondo(kF_d=1.5, dim=1), config, seed=4)
    assert not np.allclose(a.observables["purity"], b.observables["purity"])


# --- Positive cases: disk cache ---


@pytest.mark.parametrize("alias, params", [
    ("plain", {}), ("entanglement", {}),
    ("stochastic_gamma", dict(s=0.4, tau=1.0, seed=2)),
    ("kondo", dict(rho_J=0.3, kT=0.5, kF_d=1.0, dim=2))])
def test_disk_cache_hit_equals_a_miss_for_every_alias(
        alias, params, tmp_path):
    config = _varying(TWO)
    uncached = _run(QuTipSimulator(alias, **params), config)
    miss = _run(QuTipSimulator(alias, cache_dir=tmp_path, **params), config)
    n_files = len(list(tmp_path.glob("*.npz")))
    after_miss = np.random.get_state()[1].copy()
    hit = _run(QuTipSimulator(alias, cache_dir=tmp_path, **params), config)
    assert n_files == 1 and len(list(tmp_path.glob("*.npz"))) == 1
    _same(uncached, miss)
    _same(miss, hit)
    # The global generator moved the same way on the miss and on the hit.
    assert np.array_equal(np.random.get_state()[1], after_miss)
