"""Simulator variants used by individual experiments.

Each overrides one hook of ``simulator.QuTipSimulator`` and leaves the
physics and the decoherence-time logic untouched.
"""
from __future__ import annotations

import dataclasses
import math

import numpy as np
import qutip as qt

from src.blocks import channels, initial, observe
from src.physics import kondo_bath
from src.simulation import simulator

# Two-body Pauli correlators <sigma_a x sigma_b>, keyed by feature name.
CORR = observe.CORR

# Step of the Ornstein-Uhlenbeck grid that modulates gamma(t).
OU_GRID_DT = channels.OU_GRID_DT

# Parity of the two-qubit Hamiltonian the code builds,
# H = J sx1 sx2 + h (sz1 + sz2).
PARITY = initial.PARITY
# Smallest norm of a projected initial state that is kept.
MIN_SECTOR_NORM = initial.MIN_SECTOR_NORM
# Redraws of an initial state with a vanishing weight in the sector.
MAX_SECTOR_DRAWS = initial.MAX_SECTOR_DRAWS


class CorrelatorSimulator(simulator.QuTipSimulator):
    """Also records <sigma_a x sigma_b> for all a, b (R23)."""

    def _extra_observables(self, states):
        return [observe.correlators(states)]


class EntanglementSimulator(CorrelatorSimulator):
    """Also records the Wootters concurrence next to the correlators (R24)."""

    def _extra_observables(self, states):
        return super()._extra_observables(states) + [
            observe.concurrence(states)]


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
            tau: Fixed correlation time; ignored when ``s`` is None or 0.
            seed: Seed of the generator of the OU paths.

        Raises:
            ValueError: If `s` is negative or not finite, or `s` is positive
                and `tau` is not a finite positive number.
        """
        if s is not None and not (math.isfinite(s) and s >= 0):
            raise ValueError(
                f"s must be finite and non-negative, got {s!r}.")
        if s and not (tau is not None and math.isfinite(tau) and tau > 0):
            raise ValueError(
                f"tau must be finite and positive, got {tau!r}.")
        self.s, self.tau = s, tau
        self.rng = np.random.default_rng(seed)

    def _channels(self, system, config):
        base = super()._channels(system, config)
        # The path is drawn for every trajectory, so the stream does not
        # depend on the configuration; a constant rate has no time
        # dependence to modulate and keeps its channels.
        modulated = channels.ou_modulation(
            base, self.s, self.tau, config.t_max, self.rng)
        return base if config.dissipator.gamma_const is not None \
            else modulated


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
        system = simulator.build_system(config)
        if system.n_qubits != 2:
            return super()._initial_state(config)
        return initial.parity_sector(
            system, self.sector, min_norm=MIN_SECTOR_NORM,
            max_draws=MAX_SECTOR_DRAWS)


class KondoSimulator(simulator.QuTipSimulator):
    """Replaces the phenomenological jump operators by a Kondo bath (R28).

    The Hamiltonian of the system is the published one. The sigma_minus or
    sigma_z jump operators and the rate gamma(t) are replaced by the
    collapse operators of `kondo_bath.kondo_collapse_operators` for spin-1/2
    qubits (S = sigma / 2) coupled to a d-dimensional electron gas. Two
    qubits sit at distance d, so that k_F d = `kF_d`. The dissipator thus
    depends on H, on rho_J, on kT and on the geometry, and a time-dependent
    gamma(t) of the configuration is not used.

    Attributes:
        rho_J: Dimensionless Kondo coupling.
        kT: Temperature in the units of the Hamiltonian.
        kF_d: Product of the Fermi wave number and the qubit distance.
        dim: Dimension of the electron gas, 1, 2 or 3.
    """

    def __init__(self, rho_J: float, kT: float, kF_d: float, dim: int):
        """Initialises the simulator.

        Args:
            rho_J: Dimensionless coupling, in [0, 1).
            kT: Temperature in the units of the Hamiltonian, positive.
            kF_d: k_F times the distance of the two qubits, non-negative;
                a single qubit does not depend on it.
            dim: Dimension of the electron gas, 1, 2 or 3.

        Raises:
            ValueError: If a parameter is out of range.
        """
        if not (math.isfinite(kF_d) and kF_d >= 0):
            raise ValueError(
                f"kF_d must be finite and non-negative, got {kF_d!r}.")
        kondo_bath.validate_bath_parameters(
            rho_J, kT, 1.0, dim, tol=channels.KONDO_TOLERANCE)
        self.rho_J, self.kT, self.kF_d, self.dim = rho_J, kT, kF_d, dim

    def _channels(self, system, config):
        # The sites sit at 0 and kF_d, in the unit of 1 / k_F.
        spaced = dataclasses.replace(
            system, positions=system.positions * self.kF_d)
        return channels.kondo_bath(
            spaced, self.rho_J, self.kT, 1.0, self.dim)
