"""R24: Wootters concurrence and the entanglement-sudden-death target."""
import dataclasses

import numpy as np
import qutip as qt

from experiments._config import build_configs
from src.domain.value_objects import DecoherenceCriterion
from src.infrastructure.quantum.qutip_simulator import QuTipSimulator, concurrence, wootters_lambda


def test_concurrence_of_known_states():
    bell = (qt.basis(4, 1) + qt.basis(4, 2)).unit()
    assert abs(concurrence(bell * bell.dag()) - 1.0) < 1e-12
    prod = qt.tensor(qt.basis(2, 0), (qt.basis(2, 0) + qt.basis(2, 1)).unit())
    assert concurrence(prod * prod.dag()) < 1e-12
    a, b = np.cos(0.3), np.sin(0.3)
    psi = a * qt.basis(4, 0) + b * qt.basis(4, 3)
    assert abs(concurrence(psi * psi.dag()) - 2 * a * b) < 1e-12
    for p in (0.2, 0.5, 0.9):                       # Werner states
        w = p * (bell * bell.dag()).full() + (1 - p) * np.eye(4) / 4
        assert abs(concurrence(w) - max(0.0, (3 * p - 1) / 2)) < 1e-12
        assert abs(wootters_lambda(w) - (3 * p - 1) / 2) < 1e-12   # unclipped


def test_sudden_death_time_interpolates_the_zero_of_lambda():
    t = np.linspace(0.0, 1.0, 11)
    lam = 0.5 - t                                    # zero at t = 0.5
    assert abs(QuTipSimulator._sudden_death_time(t, lam)[0] - 0.5) < 1e-12
    assert QuTipSimulator._sudden_death_time(t, -t - 0.1) == (0.0, False)   # separable
    assert QuTipSimulator._sudden_death_time(t, 1.0 + t) == (1.0, True)     # censored


def test_esd_target_matches_the_concurrence_of_the_trajectory():
    cfg = dataclasses.replace(build_configs(n_per_scenario=2, seed=3)["E"][0],
                              decoherence_criterion=DecoherenceCriterion.ENTANGLEMENT_DEATH)
    np.random.seed(11)
    traj = QuTipSimulator().simulate(cfg)
    np.random.seed(11)
    plain = QuTipSimulator().simulate(dataclasses.replace(
        cfg, decoherence_criterion=DecoherenceCriterion.COHERENCE))
    # Same dynamics and observables; only the target differs.
    np.testing.assert_array_equal(traj.observables["coherence_l1"], plain.observables["coherence_l1"])
    assert traj.t_decoh != plain.t_decoh
