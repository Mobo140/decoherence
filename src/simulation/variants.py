"""Simulator variants used by individual experiments.

Each overrides one hook of ``simulator.QuTipSimulator`` and leaves the
physics and the decoherence-time logic untouched.
"""
from __future__ import annotations

import numpy as np
import qutip as qt

from src.physics import measures, paulis
from src.simulation import simulator

# Two-body Pauli correlators <sigma_a x sigma_b>, keyed by feature name.
CORR = {
    f"corr_{a}{b}": np.kron(paulis.PAULI[a], paulis.PAULI[b])
    for a in "xyz" for b in "xyz"
}

# Step of the Ornstein-Uhlenbeck grid that modulates gamma(t).
OU_GRID_DT = 0.01

# Parity of the two-qubit Hamiltonian the code builds,
# H = J sx1 sx2 + h (sz1 + sz2).
PARITY = qt.tensor(qt.sigmaz(), qt.sigmaz())
# Smallest norm of a projected initial state that is kept.
MIN_SECTOR_NORM = 1e-3
# Redraws of an initial state with a vanishing weight in the sector.
MAX_SECTOR_DRAWS = 100


class CorrelatorSimulator(simulator.QuTipSimulator):
    """Also records <sigma_a x sigma_b> for all a, b (R23)."""

    def _extra_observables(self, density_matrices) -> dict:
        rhos = np.asarray(density_matrices)
        return {
            name: np.real(np.einsum("tij,ji->t", rhos, op))
            for name, op in CORR.items()
        }


class EntanglementSimulator(CorrelatorSimulator):
    """Also records the Wootters concurrence next to the correlators (R24)."""

    def _extra_observables(self, density_matrices) -> dict:
        out = super()._extra_observables(density_matrices)
        out["concurrence"] = np.array(
            [measures.concurrence(r) for r in density_matrices]
        )
        return out


class StochasticGammaSimulator(simulator.QuTipSimulator):
    """Multiplies gamma(t) by a log-normal Ornstein-Uhlenbeck factor (R21).

    One OU path is drawn per simulated trajectory from the simulator's own
    generator, in the order trajectories are simulated. Two instances with
    the same seed therefore drive the same configurations with the same
    standard-normal increments, whatever ``s`` and ``tau`` are.

    Attributes:
        s: Strength of the fluctuation, or None to draw it per trajectory
            from U[0, 0.6].
        tau: Correlation time, or None to draw it per trajectory from
            {0.5, 2.0}.
        rng: Generator that drives the paths.
    """

    def __init__(self, s: float | None, tau: float | None, seed: int):
        """Initialises the simulator.

        Args:
            s: Fixed strength, or None to draw it for each trajectory.
            tau: Fixed correlation time; ignored when ``s`` is None.
            seed: Seed of the generator of the OU paths.
        """
        self.s, self.tau = s, tau
        self.rng = np.random.default_rng(seed)

    def _prepare_system(self, system, config):
        n = int(round(config.t_max / OU_GRID_DT)) + 1
        eta = self.rng.standard_normal(n)
        s, tau = self.s, self.tau
        if s is None:
            s = float(self.rng.uniform(0.0, 0.6))
            tau = float(self.rng.choice([0.5, 2.0]))
        # A constant rate has no time dependence to modulate.
        if s == 0.0 or system.gamma_const is not None:
            return system
        a = np.exp(-OU_GRID_DT / tau)
        x = np.empty(n)
        x[0] = eta[0]
        for k in range(1, n):
            x[k] = a * x[k - 1] + np.sqrt(1.0 - a * a) * eta[k]
        grid = np.linspace(0.0, config.t_max, n)
        # Mean one, so the average rate is the unmodulated one.
        factor = np.exp(s * x - 0.5 * s * s)
        base = system.get_gamma
        system.get_gamma = lambda t: base(t) * float(np.interp(t, grid, factor))
        return system


class ParitySectorSimulator(simulator.QuTipSimulator):
    """Projects each initial two-qubit state onto one parity sector (R6).

    Only the state preparation changes; the Hamiltonian, the dissipator and
    the integration are the ordinary ones, so any difference in the results
    is attributable to how many sector frequencies the signal contains.

    Known defect, kept because R6's recorded numbers come from it: the
    drawn initial state is a density matrix rho, and the projector P acts
    on it from the left only, so the prepared operator is P rho / |P rho|
    (trace norm) instead of P rho P / tr(P rho P). It is not Hermitian and
    its trace is below one, so the sector arms of R6 start from an
    operator that is not a state.

    Attributes:
        sector: +1 or -1, the eigenvalue of the parity operator.
    """

    def __init__(self, sector: int) -> None:
        """Initialises the simulator.

        Args:
            sector: Parity eigenvalue of the prepared states, +1 or -1.

        Raises:
            ValueError: If `sector` is not +1 or -1.
        """
        if sector not in (+1, -1):
            raise ValueError("sector must be +1 or -1")
        self.sector = sector
        self._projector = (qt.qeye([2, 2]) + sector * PARITY) / 2

    def _initial_state(self, config):
        psi = super()._initial_state(config)
        if config.n_qubits.value != 2:
            return psi
        # Redraw rather than renormalise a vanishing component: amplifying
        # a near-zero projection would make the state an artefact of the
        # noise in the discarded sector.
        for _ in range(MAX_SECTOR_DRAWS):
            projected = self._projector * psi
            if projected.norm() > MIN_SECTOR_NORM:
                return projected.unit()
            psi = super()._initial_state(config)
        raise RuntimeError(
            "could not draw a state inside the requested sector")
