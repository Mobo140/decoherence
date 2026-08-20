"""Tests for lindblad_systems utilities."""
import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.physics.lindblad_systems import (
    bernstein_polynomial,
    bernstein_gamma,
    SingleQubitSystem,
    sample_random_pure_state,
)


def test_bernstein_polynomial_edge_cases():
    # k out of range
    assert bernstein_polynomial(3, -1, 0.5) == 0.0
    assert bernstein_polynomial(3, 4, 0.5) == 0.0

    # sum of Bernstein basis at t should be 1
    vals = [bernstein_polynomial(4, k, 0.3) for k in range(5)]
    assert np.isclose(sum(vals), 1.0)


def test_bernstein_gamma_nonnegative():
    coeffs = np.array([0.1, 0.2, 0.3])
    for t in [0.0, 0.5, 1.0]:
        g = bernstein_gamma(t, coeffs, t_max=2.0)
        assert g >= 0.0


def test_get_gamma_and_lindblad_ops():
    sys = SingleQubitSystem(gamma_const=0.2)
    assert np.isclose(sys.get_gamma(0.0), 0.2)
    ops = sys.get_lindblad_operators(0.0)
    assert len(ops) > 0 and isinstance(ops[0][0], object)


def test_sample_random_pure_state_reproducible():
    s1 = sample_random_pure_state(n_qubits=1, seed=123)
    s2 = sample_random_pure_state(n_qubits=1, seed=123)
    assert np.allclose(s1.full(), s2.full())