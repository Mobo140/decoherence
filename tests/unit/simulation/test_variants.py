"""Simulator variants: stochastic gamma(t), correlators, parity sectors."""
import dataclasses

import numpy as np
import pytest
import qutip as qt

from src.blocks import channels, evolve, initial, observe
from src.data.scenarios import build_configs
from src.domain import value_objects
from src.simulation import simulator
from src.simulation import variants
from src.simulation.simulator import QuTipSimulator
from src.simulation.variants import (
    CORR,
    CorrelatorSimulator,
    EntanglementSimulator,
    ParitySectorSimulator,
    StochasticGammaSimulator,
)

NAN = float("nan")
INF = float("inf")


def _two_qubit_configs():
    g = build_configs(n_per_scenario=3, seed=0)
    return g["D"] + g["E"]


def _unit_rate_config(config):
    """The config with the time-dependent rate gamma(t) = 1."""
    return dataclasses.replace(
        config, dissipator=value_objects.DissipatorConfig(
            gamma_coeffs=(1.0,)))


def _factor_path(sim, config, n=50):
    """gamma(t) / gamma at n points, after the hook modulated the rate."""
    config = _unit_rate_config(config)
    rate = sim._channels(simulator.build_system(config), config)[0].rate
    return np.array([rate(t) for t in np.linspace(0.0, config.t_max, n)])


# --- stochastic gamma ------------------------------------------------------

def test_stochastic_gamma_zero_strength_reproduces_the_plain_simulator(
        scenario_configs):
    cfg = scenario_configs["B"]
    np.random.seed(5)
    plain = QuTipSimulator().simulate(cfg)
    np.random.seed(5)
    sto = StochasticGammaSimulator(0.0, 0.5, seed=1).simulate(cfg)
    assert plain.t_decoh == sto.t_decoh
    np.testing.assert_array_equal(plain.observables["coherence_l1"],
                                  sto.observables["coherence_l1"])


@pytest.mark.slow
def test_stochastic_gamma_keeps_its_mean_and_changes_the_trajectory(
        scenario_configs):
    cfg = scenario_configs["B"]
    sim = StochasticGammaSimulator(0.6, 0.5, seed=1)
    np.random.seed(5)
    traj = sim.simulate(cfg)
    np.random.seed(5)
    plain = QuTipSimulator().simulate(cfg)
    assert not np.allclose(traj.observables["coherence_l1"],
                           plain.observables["coherence_l1"])

    # The log-normal factor exp(s x - s^2/2) applied by the hook has mean one.
    sim = StochasticGammaSimulator(0.6, 0.5, seed=3)
    unit = _unit_rate_config(cfg)
    system = simulator.build_system(unit)
    ts = np.linspace(0.0, cfg.t_max, 201)
    vals = []
    for _ in range(100):
        rate = sim._channels(system, unit)[0].rate
        vals.append([rate(t) for t in ts])
    assert abs(np.mean(vals) - 1.0) < 0.05


def test_stochastic_gamma_same_seed_draws_the_same_path(scenario_configs):
    cfg = scenario_configs["B"]
    a = _factor_path(StochasticGammaSimulator(0.4, 1.0, seed=8), cfg)
    b = _factor_path(StochasticGammaSimulator(0.4, 1.0, seed=8), cfg)
    c = _factor_path(StochasticGammaSimulator(0.4, 1.0, seed=9), cfg)
    np.testing.assert_array_equal(a, b)
    assert not np.allclose(a, c)


def test_stochastic_gamma_paths_are_positive_and_vary_with_the_strength(
        scenario_configs):
    cfg = scenario_configs["B"]
    weak = _factor_path(StochasticGammaSimulator(0.05, 1.0, seed=1), cfg)
    strong = _factor_path(StochasticGammaSimulator(0.6, 1.0, seed=1), cfg)
    assert weak.min() > 0 and strong.min() > 0
    assert strong.std() > weak.std()


def test_stochastic_gamma_without_a_fixed_strength_draws_it_per_trajectory(
        scenario_configs):
    cfg = scenario_configs["B"]
    sim = StochasticGammaSimulator(None, None, seed=2)
    first = _factor_path(sim, cfg)
    second = _factor_path(sim, cfg)
    assert sim.s is None and sim.tau is None
    assert not np.allclose(first, second)
    assert first.min() > 0


def test_stochastic_gamma_leaves_a_constant_rate_alone(scenario_configs):
    config = scenario_configs["A"]
    result = StochasticGammaSimulator(0.6, 0.5, seed=1)._channels(
        simulator.build_system(config), config)
    assert all(c.rate == config.dissipator.gamma_const for c in result)


def test_stochastic_gamma_zero_strength_ignores_tau(scenario_configs):
    sim = StochasticGammaSimulator(0.0, None, seed=1)
    np.testing.assert_array_equal(
        _factor_path(sim, scenario_configs["B"]), 1.0)


def test_stochastic_gamma_accepts_a_missing_seed():
    assert StochasticGammaSimulator(0.1, 1.0, seed=None).rng is not None


# --- correlators and entanglement ------------------------------------------

def test_correlator_pauli_expectations_reconstruct_the_purity():
    for cfg in _two_qubit_configs():
        np.random.seed(1)
        obs = CorrelatorSimulator().simulate(cfg).observables
        single = sum(obs[f"sigma_{a}{q}"] ** 2 for a in "xyz" for q in "12")
        two = sum(obs[name] ** 2 for name in CORR)
        np.testing.assert_allclose(
            (1 + single + two) / 4, obs["purity"], atol=1e-8)


def test_correlators_sit_before_purity_and_coherence():
    np.random.seed(1)
    names = CorrelatorSimulator().simulate(
        _two_qubit_configs()[0]).feature_names
    assert names[-2:] == ["purity", "coherence_l1"]
    assert set(CORR) <= set(names[:-2])


def test_correlator_table_has_the_nine_two_body_products():
    assert sorted(CORR) == sorted(
        f"corr_{a}{b}" for a in "xyz" for b in "xyz")
    assert all(op.shape == (4, 4) for op in CORR.values())


def _states(*rhos):
    return evolve.States(np.arange(len(rhos), dtype=float), list(rhos))


def _extra(sim, *rhos):
    """The series of the hook's observations by name."""
    return observe.combine(*sim._extra_observables(_states(*rhos))).series


def test_correlator_extra_observables_of_known_states():
    up_down = qt.ket2dm(qt.basis([2, 2], [0, 1])).full()
    bell = qt.ket2dm(
        (qt.basis([2, 2], [0, 0]) + qt.basis([2, 2], [1, 1])).unit()).full()
    extra = _extra(CorrelatorSimulator(), up_down, bell)
    assert extra["corr_zz"] == pytest.approx([-1.0, 1.0])
    assert extra["corr_xx"] == pytest.approx([0.0, 1.0])
    assert extra["corr_yy"] == pytest.approx([0.0, -1.0])
    assert set(extra) == set(CORR)


def test_entanglement_simulator_adds_the_concurrence_to_the_correlators():
    bell = qt.ket2dm(
        (qt.basis([2, 2], [0, 0]) + qt.basis([2, 2], [1, 1])).unit()).full()
    product = qt.ket2dm(qt.basis([2, 2], [0, 0])).full()
    extra = _extra(EntanglementSimulator(), bell, product)
    assert set(extra) == set(CORR) | {"concurrence"}
    assert extra["concurrence"] == pytest.approx([1.0, 0.0], abs=1e-9)


def test_entanglement_simulator_records_the_concurrence_column():
    np.random.seed(1)
    names = EntanglementSimulator().simulate(
        _two_qubit_configs()[0]).feature_names
    assert names[-3:] == ["concurrence", "purity", "coherence_l1"]


# --- parity sector ---------------------------------------------------------

def test_parity_projector_selects_the_requested_eigenspace():
    for sector in (+1, -1):
        sim = ParitySectorSimulator(sector)
        projector = (qt.qeye([2, 2]) + sector * variants.PARITY) / 2
        assert (sim._projector - projector).norm() < 1e-12
        assert sim.sector == sector
        assert (sim._projector * sim._projector - sim._projector).norm() < 1e-12


def test_parity_sector_state_is_unit_norm_in_the_trace_norm(scenario_configs):
    sim = ParitySectorSimulator(+1)
    np.random.seed(0)
    prepared = sim._initial_state(scenario_configs["E"])
    assert prepared.norm() == pytest.approx(1.0)


# --- Kondo bath ------------------------------------------------------------

def _kondo(**overrides):
    params = dict(rho_J=0.1, kT=0.05, kF_d=1.0, dim=3)
    params.update(overrides)
    return variants.KondoSimulator(**params)


def _kondo_config(n_qubits, **overrides):
    params = dict(
        n_qubits=n_qubits, omega=1.0, J=0.8, t_max=4.0, dt=0.1,
        dissipator=value_objects.DissipatorConfig.constant(0.5))
    params.update(overrides)
    return value_objects.SystemConfig(**params)


def test_kondo_simulator_replaces_the_phenomenological_jump_operators():
    config = _kondo_config(value_objects.QubitCount.TWO)
    system = simulator.build_system(config)
    kondo = _kondo()._channels(system, config)
    plain = simulator.build_channels(system, config)
    assert all(c.rate == 1.0 for c in kondo)
    assert len(kondo) > 0
    assert all(np.linalg.norm(c.operator.full()) > 0 for c in kondo)
    assert len(kondo) != len(plain)


def test_kondo_simulator_single_qubit_relaxes_to_the_thermal_state():
    config = _kondo_config(value_objects.QubitCount.ONE, t_max=2000.0, dt=50.0)
    sim = _kondo(rho_J=0.3, kT=0.5)
    np.random.seed(1)
    system = simulator.build_system(config)
    result = qt.mesolve(
        system.H, qt.fock_dm(2, 0), np.array([0.0, 2000.0]),
        channels.to_mesolve(sim._channels(system, config))).states[-1]
    gibbs = (-system.H / 0.5).expm()
    assert (result - gibbs / gibbs.tr()).norm() < 1e-3


@pytest.mark.parametrize("n_qubits", [value_objects.QubitCount.ONE,
                                      value_objects.QubitCount.TWO])
def test_kondo_simulator_returns_a_decaying_trajectory(n_qubits):
    config = _kondo_config(n_qubits, t_max=40.0, dt=1.0)
    np.random.seed(3)
    result = _kondo(rho_J=0.3, kT=0.5).simulate(config)
    assert result.times.shape == result.observables["purity"].shape
    assert np.all(np.isfinite(result.observables["coherence_l1"]))


def test_kondo_simulator_without_coupling_conserves_purity():
    config = _kondo_config(value_objects.QubitCount.TWO)
    np.random.seed(3)
    result = _kondo(rho_J=0.0).simulate(config)
    np.testing.assert_allclose(result.observables["purity"], 1.0, atol=1e-5)


def test_kondo_two_qubit_result_depends_on_the_pair_distance():
    config = _kondo_config(value_objects.QubitCount.TWO, t_max=30.0, dt=1.0)
    out = []
    for k_f_d in (0.0, 1.5):
        np.random.seed(4)
        out.append(_kondo(rho_J=0.3, kF_d=k_f_d, dim=1).simulate(config))
    assert not np.allclose(out[0].observables["purity"],
                           out[1].observables["purity"])


# --- Negative cases ---


@pytest.mark.parametrize("s", [-0.1, NAN, INF, -INF])
def test_stochastic_gamma_invalid_strength_raises_value_error(s):
    with pytest.raises(ValueError, match="s must be"):
        StochasticGammaSimulator(s, 1.0, seed=0)


@pytest.mark.parametrize("tau", [0.0, -1.0, NAN, INF, None])
def test_stochastic_gamma_invalid_correlation_time_raises_value_error(tau):
    with pytest.raises(ValueError, match="tau must be"):
        StochasticGammaSimulator(0.3, tau, seed=0)


def test_stochastic_gamma_negative_seed_raises_value_error():
    with pytest.raises(ValueError, match="non-negative"):
        StochasticGammaSimulator(0.3, 1.0, seed=-1)


def test_stochastic_gamma_requires_all_three_arguments():
    with pytest.raises(TypeError, match="missing"):
        StochasticGammaSimulator()


@pytest.mark.parametrize("sector", [0, 2, -2, None, "1", "+"])
def test_parity_sector_other_than_plus_or_minus_one_raises_value_error(
        sector):
    with pytest.raises(ValueError, match=r"sector must be \+1 or -1"):
        ParitySectorSimulator(sector)


def test_parity_sector_accepts_equal_numeric_types():
    # Debatable, kept: 1.0 and True compare equal to +1.
    assert ParitySectorSimulator(1.0).sector == 1.0
    assert ParitySectorSimulator(True).sector is True


def test_parity_sector_that_cannot_be_reached_raises_runtime_error(
        monkeypatch, scenario_configs):
    # No normalised state has a projection norm above one, so every draw is
    # rejected.
    monkeypatch.setattr(variants, "MIN_SECTOR_NORM", 2.0)
    monkeypatch.setattr(variants, "MAX_SECTOR_DRAWS", 3)
    sim = ParitySectorSimulator(+1)
    with pytest.raises(RuntimeError, match="inside the requested sector"):
        sim._initial_state(scenario_configs["D"])


def test_parity_sector_redraws_a_state_with_a_vanishing_weight(
        monkeypatch, scenario_configs):
    # First draw lies in the other sector (norm 0 after projection), the
    # second one is the projected state itself.
    states = iter([
        qt.ket2dm(qt.basis([2, 2], [0, 1])),    # parity -1
        qt.ket2dm(qt.basis([2, 2], [0, 0])),    # parity +1
    ])
    monkeypatch.setattr(
        initial, "random_pure", lambda system, rng=None: next(states))
    prepared = ParitySectorSimulator(+1)._initial_state(
        scenario_configs["D"])
    assert prepared.norm() == pytest.approx(1.0)
    assert prepared.full()[0, 0] == pytest.approx(1.0)
    assert next(states, None) is None


def test_correlators_on_a_one_qubit_system_add_no_observables(
        scenario_configs):
    # Debatable, kept: the extra observables belong to the two-qubit branch,
    # so a one-qubit run silently records the plain set (pad_features fills
    # the columns when such data is mixed with two-qubit data).
    for sim in (CorrelatorSimulator(), EntanglementSimulator()):
        np.random.seed(1)
        traj = sim.simulate(scenario_configs["A"])
        assert traj.feature_names == [
            "sigma_x", "sigma_y", "sigma_z", "purity", "coherence_l1"]


def test_correlator_extra_observables_of_one_qubit_states_raise():
    with pytest.raises(ValueError):
        CorrelatorSimulator()._extra_observables(_states(np.eye(2) / 2))


def test_kondo_simulator_invalid_parameters_raise_value_error():
    for bad in (dict(rho_J=-0.1), dict(rho_J=1.0), dict(kT=0.0),
                dict(kF_d=-1.0), dict(dim=4)):
        with pytest.raises(ValueError):
            _kondo(**bad)


def test_kondo_simulator_requires_all_bath_arguments():
    with pytest.raises(TypeError):
        variants.KondoSimulator(rho_J=0.1, kT=0.1)
