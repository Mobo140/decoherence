"""Observables computed from a density matrix."""
import numpy as np

from src.physics.measures import coherence_l1, purity
from src.physics.observables import compute_observables_single_qubit


def test_single_qubit_observables_agree_with_the_measures():
    rho = np.array([[0.6, 0.4], [0.4, 0.4]], dtype=complex)
    obs = compute_observables_single_qubit(rho)
    assert 'sigma_x' in obs and 'sigma_y' in obs and 'sigma_z' in obs
    assert np.isclose(obs['coherence_l1'], coherence_l1(rho))
    assert np.isclose(obs['purity'], purity(rho))
