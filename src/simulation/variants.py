"""Simulator variants used by individual experiments.

Each overrides one hook of ``simulator.QuTipSimulator`` and leaves the
physics and the decoherence-time logic untouched.
"""
from __future__ import annotations

import numpy as np

from src.physics import measures, paulis
from src.simulation import simulator

# Two-body Pauli correlators <sigma_a x sigma_b>, keyed by feature name.
CORR = {
    f"corr_{a}{b}": np.kron(paulis.PAULI[a], paulis.PAULI[b])
    for a in "xyz" for b in "xyz"
}

# Step of the Ornstein-Uhlenbeck grid that modulates gamma(t).
OU_GRID_DT = 0.01


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
