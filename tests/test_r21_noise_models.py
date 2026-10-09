"""R21 noise models: AR(1) measurement noise and the stochastic gamma(t) hook."""
import numpy as np

from experiments._config import build_configs
from experiments.r21_stochastic_gamma_ar1 import StochasticGammaSimulator
from src.infrastructure.quantum.noise import ar1_filter as ar1
from src.infrastructure.quantum.qutip_simulator import QuTipSimulator


def test_ar1_is_stationary_with_unit_variance_and_lag_one_correlation_phi():
    eta = np.random.default_rng(0).standard_normal((4000, 200))
    for phi in (0.0, 0.5, 0.9):
        e = ar1(eta, phi)
        assert abs(e.var() - 1.0) < 0.03
        assert abs(e[:, 0].var() - 1.0) < 0.08          # stationary from t = 0
        r1 = np.mean(e[:, 1:] * e[:, :-1])
        assert abs(r1 - phi) < 0.02


def test_ar1_accepts_one_correlation_per_window():
    eta = np.random.default_rng(1).standard_normal((3000, 100, 2))
    phi = np.repeat([0.0, 0.9], 1500)
    e = ar1(eta, phi)
    np.testing.assert_array_equal(e[:1500], eta[:1500])
    r1 = np.mean(e[1500:, 1:] * e[1500:, :-1])
    assert abs(r1 - 0.9) < 0.02 and abs(e[1500:].var() - 1.0) < 0.05


def test_zero_strength_reproduces_the_plain_simulator():
    cfg = build_configs(n_per_scenario=1, seed=0)["B"][0]
    np.random.seed(5)
    plain = QuTipSimulator().simulate(cfg)
    np.random.seed(5)
    sto = StochasticGammaSimulator(0.0, 0.5, seed=1).simulate(cfg)
    assert plain.t_decoh == sto.t_decoh
    np.testing.assert_array_equal(plain.observables["coherence_l1"],
                                  sto.observables["coherence_l1"])


def test_fluctuating_rate_keeps_its_mean_and_changes_the_trajectory():
    cfg = build_configs(n_per_scenario=1, seed=0)["B"][0]
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
