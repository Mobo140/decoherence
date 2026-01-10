"""Additional tests for decoherence utilities."""
import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.physics.decoherence_time import (
    coherence_l1,
    purity,
    calculate_decoherence_time,
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


def test_calculate_decoherence_time_purity():
    # Create trajectory where purity decays linearly from 1 to 0.5
    n = 10
    times = np.linspace(0, 9, n)
    density_matrices = []
    for i in range(n):
        p = 1.0 - 0.5 * (i / (n - 1))
        # build diagonal density matrix with purity p by mixing |0> and |1>
        # For diagonal rho = diag(a, 1-a), purity = a^2 + (1-a)^2
        # Solve for a given purity p: a = (1 +/- sqrt(2p-1)) / 2
        if p < 0.5:
            a = 0.5
        else:
            a = (1.0 + np.sqrt(max(0.0, 2 * p - 1.0))) / 2.0
        rho = np.array([[a, 0.0], [0.0, 1.0 - a]], dtype=complex)
        density_matrices.append(rho)

    # Use absolute threshold: find first time purity < 0.9
    t_decoh = calculate_decoherence_time(times, density_matrices, criterion='purity', threshold=0.9)
    assert t_decoh is not None
    assert t_decoh >= times[0]


def test_compute_observables_single_qubit_consistency():
    rho = np.array([[0.6, 0.4], [0.4, 0.4]], dtype=complex)
    obs = compute_observables_single_qubit(rho)
    assert 'sigma_x' in obs and 'sigma_y' in obs and 'sigma_z' in obs
    assert np.isclose(obs['coherence_l1'], coherence_l1(rho))
    assert np.isclose(obs['purity'], purity(rho))
