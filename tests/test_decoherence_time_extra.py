"""Additional tests for decoherence utilities."""
import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.physics.decoherence_time import (
    coherence_l1,
    purity,
    compute_observables_single_qubit,
)


def test_coherence_and_purity_basic():
    # Pure state |0><0|
    rho_pure = np.array([[1.0, 0.0], [0.0, 0.0]], dtype=complex)
    assert np.isclose(purity(rho_pure), 1.0)
    assert np.isclose(coherence_l1(rho_pure), 0.0)

    # Maximally mixed state
    rho_mix = 0.5 * np.eye(2, dtype=complex)
    assert np.isclose(purity(rho_mix), 0.5)
    assert np.isclose(coherence_l1(rho_mix), 0.0)


def test_compute_observables_single_qubit_consistency():
    rho = np.array([[0.6, 0.4], [0.4, 0.4]], dtype=complex)
    obs = compute_observables_single_qubit(rho)
    assert 'sigma_x' in obs and 'sigma_y' in obs and 'sigma_z' in obs
    assert np.isclose(obs['coherence_l1'], coherence_l1(rho))
    assert np.isclose(obs['purity'], purity(rho))
