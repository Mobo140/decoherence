"""Pauli matrices as read-only NumPy constants.

Every module that needs sigma_x, sigma_y, sigma_z or the 2x2 identity as a
plain array takes them from here, so the project has one definition.
"""
from __future__ import annotations

import numpy as np

SIGMA_X = np.array([[0, 1], [1, 0]], dtype=complex)
SIGMA_Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
SIGMA_Z = np.array([[1, 0], [0, -1]], dtype=complex)
IDENTITY = np.eye(2, dtype=complex)

# Keyed by axis name, for code that loops over "xyz".
PAULI = {"x": SIGMA_X, "y": SIGMA_Y, "z": SIGMA_Z}

for _matrix in (SIGMA_X, SIGMA_Y, SIGMA_Z, IDENTITY):
    # Shared module constants: make accidental in-place edits fail loudly.
    _matrix.setflags(write=False)
del _matrix
