"""Simulator variants: stochastic gamma(t) and two-body correlators."""
import numpy as np
import pytest

from src.data.scenarios import build_configs
from src.simulation.simulator import QuTipSimulator
from src.simulation.variants import (
    CORR,
    CorrelatorSimulator,
    StochasticGammaSimulator,
)


def _two_qubit_configs():
    g = build_configs(n_per_scenario=3, seed=0)
    return g["D"] + g["E"]


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
