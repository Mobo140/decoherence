"""Simulator variants: stochastic gamma(t), correlators, parity sectors."""
import numpy as np
import pytest
import qutip as qt

from src.data.scenarios import build_configs
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


class _TimeDependentSystem:
    """Stand-in for a Lindblad system with gamma(t) = 1."""

    gamma_const = None

    def get_gamma(self, t):
        return 1.0


class _ConstantSystem:
    """Stand-in for a Lindblad system with a constant rate."""

    gamma_const = 0.3

    def get_gamma(self, t):
        return 0.3


def _factor_path(sim, config, n=50):
    """gamma(t) / gamma at n points, after the hook modulated the system."""
    system = sim._prepare_system(_TimeDependentSystem(), config)
    return np.array([system.get_gamma(t)
                     for t in np.linspace(0.0, config.t_max, n)])


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
    class Unit:
        gamma_const = None

        def get_gamma(self, t):
            return 1.0

    sim = StochasticGammaSimulator(0.6, 0.5, seed=3)
    ts = np.linspace(0.0, cfg.t_max, 201)
    vals = [[sim._prepare_system(Unit(), cfg).get_gamma(t) for t in ts]
            for _ in range(100)]
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
    system = _ConstantSystem()
    result = StochasticGammaSimulator(0.6, 0.5, seed=1)._prepare_system(
        system, scenario_configs["A"])
    assert result is system and result.get_gamma(1.0) == 0.3


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


def test_correlator_extra_observables_of_known_states():
    up_down = qt.ket2dm(qt.basis([2, 2], [0, 1])).full()
    bell = qt.ket2dm(
        (qt.basis([2, 2], [0, 0]) + qt.basis([2, 2], [1, 1])).unit()).full()
    extra = CorrelatorSimulator()._extra_observables([up_down, bell])
    assert extra["corr_zz"] == pytest.approx([-1.0, 1.0])
    assert extra["corr_xx"] == pytest.approx([0.0, 1.0])
    assert extra["corr_yy"] == pytest.approx([0.0, -1.0])
    assert set(extra) == set(CORR)


def test_entanglement_simulator_adds_the_concurrence_to_the_correlators():
    bell = qt.ket2dm(
        (qt.basis([2, 2], [0, 0]) + qt.basis([2, 2], [1, 1])).unit()).full()
    product = qt.ket2dm(qt.basis([2, 2], [0, 0])).full()
    extra = EntanglementSimulator()._extra_observables([bell, product])
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
        variants.simulator.QuTipSimulator, "_initial_state",
        lambda self, config: next(states))
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
        CorrelatorSimulator()._extra_observables([np.eye(2) / 2])
