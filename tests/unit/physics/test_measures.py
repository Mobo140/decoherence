"""Decoherence measures: purity, l1-coherence, concurrence."""
import numpy as np
import qutip as qt

from src.physics.measures import (
    coherence_l1,
    concurrence,
    purity,
    wootters_lambda,
)


def test_purity_and_coherence_of_pure_and_mixed_states():
    # Pure state |0><0|
    rho_pure = np.array([[1.0, 0.0], [0.0, 0.0]], dtype=complex)
    assert np.isclose(purity(rho_pure), 1.0)
    assert np.isclose(coherence_l1(rho_pure), 0.0)

    # Maximally mixed state
    rho_mix = 0.5 * np.eye(2, dtype=complex)
    assert np.isclose(purity(rho_mix), 0.5)
    assert np.isclose(coherence_l1(rho_mix), 0.0)


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
