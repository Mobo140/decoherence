"""Dissipation channels: a jump operator with a rate, and rate modifiers.

A channel is a pair (L, rate) with a constant or time-dependent rate.
Channels combine by list concatenation, and a modifier maps a list of
channels to a list of channels, so a Kondo bath with a stochastic rate is
``ou_modulation(kondo_bath(...), ...)``. ``to_mesolve`` is the one place
that knows QuTiP's collapse-operator format.
"""
from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Literal

import numpy as np
import qutip as qt

from src.blocks.system import System
from src.physics import kondo_bath as bath
from src.physics import lindblad_systems

# Step of the Ornstein-Uhlenbeck grid that modulates a rate.
OU_GRID_DT = 0.01
# Bohr frequencies closer than this are one frequency in the Kondo bath.
KONDO_TOLERANCE = 1e-9
# Draw range of the strength and the choices of the correlation time when
# ou_modulation draws them itself.
_OU_S_RANGE = (0.0, 0.6)
_OU_TAU_CHOICES = [0.5, 2.0]

Rate = float | Callable[[float], float]


@dataclass(frozen=True)
class Channel:
    """A Lindblad dissipation channel.

    Attributes:
        operator: The jump operator L.
        rate: A non-negative constant, or a function of time.
    """

    operator: qt.Qobj
    rate: Rate

    def __post_init__(self) -> None:
        """Validates the rate.

        Raises:
            ValueError: If a constant rate is negative or not finite.
        """
        if not callable(self.rate) and not (
                math.isfinite(self.rate) and self.rate >= 0):
            raise ValueError(
                "A constant rate must be finite and non-negative, "
                f"got {self.rate!r}.")


def _rate_at(rate: Rate, t: float) -> float:
    """Evaluates a constant or time-dependent rate at ``t``."""
    return rate(t) if callable(rate) else rate


def _scaled(
    channels: Sequence[Channel],
    factor: Callable[[float], float],
    modulate_constant: bool = True,
) -> list[Channel]:
    """Multiplies every rate by ``factor(t)``, making it time dependent.

    With ``modulate_constant`` False, channels with a constant rate are
    returned unchanged.
    """
    # Channels that shared a rate keep sharing one function, see to_mesolve.
    scaled = {}
    for c in channels:
        if not modulate_constant and not callable(c.rate):
            continue
        if id(c.rate) not in scaled:
            scaled[id(c.rate)] = (
                lambda t, base=c.rate: _rate_at(base, t) * factor(t))
    return [
        Channel(c.operator, scaled[id(c.rate)]) if id(c.rate) in scaled
        else c
        for c in channels]


def jump(
    system: System,
    op_type: Literal["sigma_minus", "sigma_z"],
    gamma: float,
) -> list[Channel]:
    """Builds the jump channel of each qubit with a constant rate.

    Args:
        system: A one- or two-qubit system.
        op_type: 'sigma_minus' or 'sigma_z'.
        gamma: Rate of every channel.

    Returns:
        One channel per qubit, in the order of the qubits.

    Raises:
        ValueError: If `op_type` is unknown, `gamma` is negative or not
            finite, or the system has more than two qubits.
    """
    if system.n_qubits == 1:
        operators = [lindblad_systems.single_qubit_jump_operator(op_type)]
    elif system.n_qubits == 2:
        operators = [
            lindblad_systems.two_qubit_jump_operator(op_type, qubit)
            for qubit in (0, 1)
        ]
    else:
        raise ValueError(
            f"jump supports one or two qubits, got {system.n_qubits}.")
    return [Channel(op, gamma) for op in operators]


def bernstein_profile(
    channels: Sequence[Channel], coeffs: Sequence[float], t_max: float
) -> list[Channel]:
    """Multiplies the rates by the Bernstein polynomial gamma(t).

    With channels of unit rate the new rate is gamma(t) itself.

    Args:
        channels: Channels to modify.
        coeffs: Bernstein coefficients of gamma(t).
        t_max: Time that is mapped to 1 in the polynomial argument.

    Returns:
        Channels with the time-dependent rate rate * gamma(t).

    Raises:
        ValueError: If `coeffs` is empty or `t_max` is not finite and
            positive.
    """
    if len(coeffs) == 0:
        raise ValueError("Bernstein coefficients must not be empty.")
    if not (math.isfinite(t_max) and t_max > 0):
        raise ValueError(f"t_max must be finite and positive, got {t_max!r}.")
    coeffs = np.asarray(coeffs)
    return _scaled(
        channels,
        lambda t: lindblad_systems.bernstein_gamma(t, coeffs, t_max))


def ou_modulation(
    channels: Sequence[Channel],
    s: float | None,
    tau: float | None,
    t_max: float,
    rng: np.random.Generator,
    modulate_constant: bool = True,
) -> list[Channel]:
    """Multiplies the rates by a log-normal Ornstein-Uhlenbeck factor (R21).

    The factor exp(s x - s^2 / 2) has mean one, x being a unit-variance OU
    process with correlation time ``tau`` on a grid of step ``OU_GRID_DT``.
    The generator is consumed in a fixed order: the standard-normal
    increments first, then, if ``s`` is None, the strength and the
    correlation time. The increments are drawn even when ``s`` is zero, so
    the stream does not depend on the parameters.

    Args:
        channels: Channels to modify.
        s: Strength of the fluctuation, or None to draw it from U[0, 0.6].
        tau: Correlation time, or None to draw it from {0.5, 2.0} together
            with ``s``; ignored when ``s`` is zero.
        t_max: End of the time grid of the path.
        rng: Generator that drives the path.
        modulate_constant: If False, channels with a constant rate are left
            alone (the path is still drawn); the published runs of R21 do
            this.

    Returns:
        The channels with the modulated rates; the channels themselves if
        the strength is zero.

    Raises:
        ValueError: If `s` is negative or not finite, `s` is positive and
            `tau` is not finite and positive, or `t_max` is not finite and
            positive.
    """
    if s is not None and not (math.isfinite(s) and s >= 0):
        raise ValueError(f"s must be finite and non-negative, got {s!r}.")
    if s and not (tau is not None and math.isfinite(tau) and tau > 0):
        raise ValueError(f"tau must be finite and positive, got {tau!r}.")
    if not (math.isfinite(t_max) and t_max > 0):
        raise ValueError(f"t_max must be finite and positive, got {t_max!r}.")
    n = int(round(t_max / OU_GRID_DT)) + 1
    eta = rng.standard_normal(n)
    if s is None:
        s = float(rng.uniform(*_OU_S_RANGE))
        tau = float(rng.choice(_OU_TAU_CHOICES))
    if s == 0.0:
        return list(channels)
    a = np.exp(-OU_GRID_DT / tau)
    x = np.empty(n)
    x[0] = eta[0]
    for k in range(1, n):
        x[k] = a * x[k - 1] + np.sqrt(1.0 - a * a) * eta[k]
    grid = np.linspace(0.0, t_max, n)
    # Mean one, so the average rate is the unmodulated one.
    factor = np.exp(s * x - 0.5 * s * s)
    return _scaled(
        channels, lambda t: float(np.interp(t, grid, factor)),
        modulate_constant)


def kondo_bath(
    system: System,
    rho_J: float,
    kT: float,
    k_F: float,
    dim: Literal[1, 2, 3],
    spacing: float = 1.0,
) -> list[Channel]:
    """Builds the Davies channels of a Kondo bath (R28).

    The rates are part of the operators (see
    `bath.kondo_collapse_operators`), so each channel has rate one.
    The kernel depends on k_F times the distance of the sites, which is
    ``spacing`` times the distance in ``system.positions``.

    Args:
        system: The spins and their positions.
        rho_J: Dimensionless coupling, in [0, 1).
        kT: Temperature in the units of the Hamiltonian, positive.
        k_F: Fermi wave number, positive.
        dim: Dimension of the electron gas, 1, 2 or 3.
        spacing: Factor on the site positions, non-negative; 0 puts all
            sites at one point.

    Returns:
        One channel per collapse operator.

    Raises:
        ValueError: If a parameter is out of range.
    """
    if not (math.isfinite(spacing) and spacing >= 0):
        raise ValueError(
            f"spacing must be finite and non-negative, got {spacing!r}.")
    bath.validate_bath_parameters(
        rho_J, kT, k_F, dim, tol=KONDO_TOLERANCE)
    operators = bath.kondo_collapse_operators(
        system.H, system.spin_ops, system.positions * spacing,
        rho_J=rho_J, kT=kT, k_F=k_F, dim=dim, tol=KONDO_TOLERANCE)
    return [Channel(op, 1.0) for op in operators]


def _collapse_operator(channel: Channel, coefficients: dict):
    """Returns the QuTiP collapse operator of one channel.

    ``coefficients`` maps a time-dependent rate to its QuTiP coefficient, so
    channels that share a rate function share one coefficient function.
    """
    if not callable(channel.rate):
        return np.sqrt(channel.rate) * channel.operator
    rate = channel.rate
    if id(rate) in coefficients:
        return [channel.operator, coefficients[id(rate)]]

    def sqrt_rate(t, args):
        """Returns sqrt(rate(t)) in QuTiP's coefficient format.

        Args:
            t: Time.
            args: Unused; required by the QuTiP coefficient signature.
        """
        value = rate(t)
        return np.sqrt(value) if value > 0 else 0.0

    coefficients[id(rate)] = sqrt_rate
    return [channel.operator, sqrt_rate]


def to_mesolve(channels: Sequence[Channel]) -> list:
    """Converts channels to the collapse operators of ``qutip.mesolve``.

    Args:
        channels: The channels.

    Returns:
        For a constant rate r the operator sqrt(r) * L, for a time-dependent
        rate the pair [L, sqrt_coeff] with sqrt_coeff(t, args) equal to
        sqrt(rate(t)) where that is positive and 0.0 elsewhere. Channels
        with the same rate function get the same coefficient function, as
        QuTiP then treats them alike.
    """
    coefficients = {}
    return [_collapse_operator(c, coefficients) for c in channels]
