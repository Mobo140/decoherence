"""Kondo bath: weak exchange coupling of localised spins to an electron gas.

The spins at positions r_l couple to a free electron gas of dimension d by
V = J sum_l S_l . s(r_l). To second order in rho_J, in the secular
(Davies) limit of the Bloch-Redfield equation, the spins relax between the
eigenstates |M> (energies E_M) of their Hamiltonian H at the rates

    Gamma(M -> M') = pi (rho J)^2 / 4 * G(E_M - E_M')
        * sum_{l,l'} K_d(k_F |r_l - r_l'|)
          sum_a <M'|S^a_l|M> <M|S^a_l'|M'>,

with the Korringa function G(x) = x / (1 - exp(-x / kT)), G(0) = kT, and the
pair kernel K_d from the Fermi-surface average of the Bloch phases:
K_1 = cos^2, K_2 = J0^2, K_3 = sinc^2. Units: hbar = 1; energies and kT in
the units of H. Delgado and Fernandez-Rossier, Phys. Rev. B 95, 075413
(2017), Eqs. 12-13, 20, 26, 31-34, and Surf. Sci. 630, 337 (2014), Eqs.
14-16.

`kondo_collapse_operators` turns the same rates into Lindblad collapse
operators for `qutip.mesolve`. It is a pure function of the Hamiltonian,
the spin operators, the positions and the bath parameters, so any
simulator can reuse it.
"""
from __future__ import annotations

import dataclasses
import math
import warnings
from collections.abc import Sequence

import numpy as np
import qutip as qt
from scipy import special

DIMENSIONS = (1, 2, 3)
# Above this the second-order treatment is not trustworthy.
WEAK_COUPLING_LIMIT = 0.3
# Collapse operators with a smaller norm are dropped.
_MIN_OPERATOR_NORM = 1e-14
# Eigenvalues of the kernel matrix below this are numerical zeros.
_MIN_KERNEL_EIGENVALUE = 1e-12
# Below this |x| / kT the Korringa function is expanded around zero.
_SMALL_ARGUMENT = 1e-8
# Relative accuracy with which supplied states must diagonalise H.
_EIGENVECTOR_TOLERANCE = 1e-8
_HERMITIAN_TOLERANCE = 1e-10

SpinOperators = Sequence[Sequence[qt.Qobj]]


@dataclasses.dataclass(frozen=True)
class KondoRates:
    """Rates of the Pauli master equation and the decoherence rates.

    Index m labels the eigenstates in the order of `energies`.

    Attributes:
        energies: Eigenvalues of H, ascending.
        states: Eigenvectors as columns, shape (n, n).
        rates: ``rates[m, k]`` is the rate Gamma(m -> k); the diagonal is 0.
        dephasing_adiabatic: ``[m, k]`` is the pure-dephasing rate of the
            coherence between m and k (PRB Eq. 15).
        dephasing: ``[m, k]`` is the full decoherence rate 1/T2, the sum of
            the adiabatic part and half the sum of the rates that leave m
            and k (PRB Eqs. 14-15). It is not valid for coherences between
            degenerate states.
    """

    energies: np.ndarray
    states: np.ndarray
    rates: np.ndarray
    dephasing_adiabatic: np.ndarray
    dephasing: np.ndarray


def validate_bath_parameters(
    rho_J: float, kT: float, k_F: float, dim: int, tol: float
) -> None:
    """Checks the parameters of the bath.

    Args:
        rho_J: Dimensionless coupling, density of states times J.
        kT: Temperature in energy units.
        k_F: Fermi wave number, in the inverse unit of the positions.
        dim: Dimension of the electron gas, 1, 2 or 3.
        tol: Tolerance for grouping Bohr frequencies as equal.

    Raises:
        ValueError: If `rho_J` is not in [0, 1), `kT`, `k_F` or `tol` is
            not finite and positive, or `dim` is not 1, 2 or 3.

    Warns:
        UserWarning: If `rho_J` exceeds `WEAK_COUPLING_LIMIT`.
    """
    if not (math.isfinite(rho_J) and 0.0 <= rho_J < 1.0):
        raise ValueError(f"rho_J must be in [0, 1), got {rho_J!r}.")
    if rho_J > WEAK_COUPLING_LIMIT:
        warnings.warn(
            f"rho_J = {rho_J} is not small; the weak coupling expansion "
            "to second order may fail.", UserWarning, stacklevel=2)
    for name, value in (("kT", kT), ("k_F", k_F), ("tol", tol)):
        if not (math.isfinite(value) and value > 0):
            raise ValueError(
                f"{name} must be finite and positive, got {value!r}.")
    if dim not in DIMENSIONS or int(dim) != dim:
        raise ValueError(f"dim must be one of {DIMENSIONS}, got {dim!r}.")


def pair_kernel(dim: int, x: np.ndarray | float) -> np.ndarray:
    """Evaluates the Fermi-surface averaged pair kernel K_d(x).

    Args:
        dim: Dimension of the electron gas: cos^2(x) for 1, J0(x)^2 for 2,
            (sin x / x)^2 for 3.
        x: Distances in units of 1 / k_F, non-negative.

    Returns:
        K_d at `x`, equal to 1 at 0.

    Raises:
        ValueError: If `dim` is not 1, 2 or 3 or `x` has a negative entry.
    """
    if dim not in DIMENSIONS:
        raise ValueError(f"dim must be one of {DIMENSIONS}, got {dim!r}.")
    x = np.asarray(x, dtype=float)
    if np.any(x < 0):
        raise ValueError("x must be non-negative.")
    if dim == 1:
        return np.cos(x) ** 2
    if dim == 2:
        return special.j0(x) ** 2
    return np.sinc(x / math.pi) ** 2


def bath_rate(delta: np.ndarray | float, kT: float) -> np.ndarray | float:
    """Evaluates the Korringa function G(delta) = delta / (1 - e^(-delta/kT)).

    G is the rate of emission of an electron-hole pair with energy `delta`:
    proportional to `delta` for delta >> kT, `kT` at 0, and exponentially
    small for negative `delta` (detailed balance).

    Args:
        delta: Energy released by the spins; either sign.
        kT: Temperature in energy units, positive.

    Returns:
        G(delta), a float for a scalar input.

    Raises:
        ValueError: If `kT` is not finite and positive.
    """
    if not (math.isfinite(kT) and kT > 0):
        raise ValueError(f"kT must be finite and positive, got {kT!r}.")
    x = np.asarray(delta, dtype=float) / kT
    magnitude = np.abs(x)
    with np.errstate(over="ignore", invalid="ignore"):
        ratio = np.where(
            magnitude < _SMALL_ARGUMENT,
            1.0 + magnitude / 2.0,
            magnitude / -np.expm1(-magnitude))
    # G(-x) = G(x) exp(-x): the Boltzmann factor for absorption.
    result = kT * ratio * np.where(x >= 0, 1.0, np.exp(-magnitude))
    return float(result) if result.ndim == 0 else result


def chain_spin_operators(
    n_sites: int, spin: float = 0.5
) -> list[tuple[qt.Qobj, qt.Qobj, qt.Qobj]]:
    """Builds the spin operators of a chain of identical spins.

    Args:
        n_sites: Number of spins, at least 1.
        spin: Spin S, a positive integer or half-integer.

    Returns:
        For each site the tuple (S^x, S^y, S^z) acting on the whole
        tensor-product space of dimension (2S + 1)^n_sites.

    Raises:
        ValueError: If `n_sites` is not a positive integer or `spin` is not
            a positive integer or half-integer.
    """
    if int(n_sites) != n_sites or n_sites < 1:
        raise ValueError(
            f"n_sites must be a positive integer, got {n_sites!r}.")
    if not (spin > 0 and abs(2 * spin - round(2 * spin)) < 1e-12):
        raise ValueError(
            f"spin must be a positive integer or half-integer, got {spin!r}.")
    dim = int(round(2 * spin)) + 1
    local = [qt.jmat(spin, axis) for axis in "xyz"]
    chain = []
    for site in range(int(n_sites)):
        components = []
        for op in local:
            factors = [qt.qeye(dim)] * int(n_sites)
            factors[site] = op
            components.append(qt.tensor(*factors))
        chain.append(tuple(components))
    return chain


def kernel_matrix(
    positions: np.ndarray | Sequence[float], *, k_F: float, dim: int
) -> np.ndarray:
    """Builds the matrix K_{ll'} = K_d(k_F |r_l - r_l'|).

    Args:
        positions: Site positions, shape (n,) for a line or (n, m) points.
        k_F: Fermi wave number, positive.
        dim: Dimension of the electron gas, 1, 2 or 3.

    Returns:
        The symmetric, positive semi-definite matrix, shape (n, n).

    Raises:
        ValueError: If `positions` is empty or not finite, or `k_F` or
            `dim` is invalid.
    """
    points = np.asarray(positions, dtype=float)
    if points.ndim == 1:
        points = points[:, None]
    if points.ndim != 2 or points.size == 0 or not np.all(
            np.isfinite(points)):
        raise ValueError(
            "positions must be a non-empty finite array of shape (n,) or "
            "(n, m).")
    if not (math.isfinite(k_F) and k_F > 0):
        raise ValueError(f"k_F must be finite and positive, got {k_F!r}.")
    distance = np.linalg.norm(points[:, None, :] - points[None, :, :], axis=2)
    return pair_kernel(dim, k_F * distance)


def _check_system(
    hamiltonian: qt.Qobj, spin_ops: SpinOperators, positions: np.ndarray
) -> np.ndarray:
    """Validates H, the spin operators and the positions; returns H."""
    ham = hamiltonian.full()
    if ham.ndim != 2 or ham.shape[0] != ham.shape[1] or not np.allclose(
            ham, ham.conj().T, atol=_HERMITIAN_TOLERANCE):
        raise ValueError("The Hamiltonian must be a Hermitian matrix.")
    if len(spin_ops) == 0:
        raise ValueError("spin_ops must contain at least one site.")
    for site in spin_ops:
        if len(site) != 3:
            raise ValueError(
                "Each site needs three spin components (x, y, z), "
                f"got {len(site)}.")
        for op in site:
            if op.shape != hamiltonian.shape:
                raise ValueError(
                    f"A spin operator has shape {op.shape} but the "
                    f"Hamiltonian has shape {hamiltonian.shape}.")
            if not np.allclose(op.full(), op.full().conj().T,
                               atol=_HERMITIAN_TOLERANCE):
                raise ValueError("Spin operators must be Hermitian.")
    n_points = np.asarray(positions).shape[0] if np.ndim(positions) else 0
    if n_points != len(spin_ops):
        raise ValueError(
            f"positions has {n_points} entries for {len(spin_ops)} sites.")
    return ham


def _eigensystem(
    ham: np.ndarray, states: np.ndarray | None
) -> tuple[np.ndarray, np.ndarray]:
    """Returns eigenvalues and eigenvectors, validating supplied states."""
    if states is None:
        return np.linalg.eigh(ham)
    states = np.asarray(states, dtype=complex)
    if states.shape != ham.shape:
        raise ValueError(
            f"states has shape {states.shape} but H has shape {ham.shape}.")
    projected = states.conj().T @ ham @ states
    energies = np.real(np.diag(projected))
    scale = max(1.0, float(np.max(np.abs(energies))))
    if (np.abs(projected - np.diag(energies)).max()
            > _EIGENVECTOR_TOLERANCE * scale
            or not np.allclose(states.conj().T @ states, np.eye(len(ham)),
                               atol=_EIGENVECTOR_TOLERANCE)):
        raise ValueError(
            "states must be orthonormal eigenvectors of the Hamiltonian.")
    order = np.argsort(energies, kind="stable")
    return energies[order], states[:, order]


def _site_elements(
    spin_ops: SpinOperators, states: np.ndarray
) -> np.ndarray:
    """Returns <m|S^a_l|k> as an array of shape (3, sites, n, n)."""
    return np.array([
        [states.conj().T @ site[a].full() @ states for site in spin_ops]
        for a in range(3)
    ])


def kondo_rates(
    hamiltonian: qt.Qobj,
    spin_ops: SpinOperators,
    positions: np.ndarray | Sequence[float],
    *,
    rho_J: float,
    kT: float,
    k_F: float,
    dim: int,
    states: np.ndarray | None = None,
) -> KondoRates:
    """Computes the Kondo relaxation and dephasing rates between eigenstates.

    Args:
        hamiltonian: Hamiltonian of the spins, Hermitian.
        spin_ops: For each site the Hermitian operators (S^x, S^y, S^z) on
            the full space; see `chain_spin_operators`.
        positions: Site positions, shape (n_sites,) or (n_sites, m), in the
            inverse unit of `k_F`.
        rho_J: Dimensionless coupling in [0, 1).
        kT: Temperature in the units of the Hamiltonian.
        k_F: Fermi wave number.
        dim: Dimension of the electron gas, 1, 2 or 3.
        states: Orthonormal eigenvectors of H as columns, to fix the basis
            inside degenerate levels (the rates of a degenerate level
            depend on the choice); by default `numpy.linalg.eigh` decides.

    Returns:
        The `KondoRates`; its energies are ascending, and `states` are the
        columns used.

    Raises:
        ValueError: If a parameter is invalid (`validate_bath_parameters`),
            H or a spin operator is not Hermitian, shapes or the number of
            positions do not match, or `states` are not orthonormal
            eigenvectors of H.
    """
    validate_bath_parameters(rho_J, kT, k_F, dim, tol=1.0)
    ham = _check_system(hamiltonian, spin_ops, positions)
    kernel = kernel_matrix(positions, k_F=k_F, dim=dim)
    energies, vectors = _eigensystem(ham, states)
    elements = _site_elements(spin_ops, vectors)
    prefactor = math.pi * rho_J ** 2 / 4.0

    # weight[k, m] = sum_a sum_{ll'} K_ll' <k|S_l|m> <m|S_l'|k>
    weight = np.einsum(
        "lL,alkm,aLkm->km", kernel, elements, elements.conj()).real
    bohr = energies[:, None] - energies[None, :]
    rates = prefactor * np.asarray(bath_rate(bohr, kT)) * weight.T
    np.fill_diagonal(rates, 0.0)

    diagonal = np.einsum("alkk->alk", elements).real
    jump = diagonal[:, :, :, None] - diagonal[:, :, None, :]
    adiabatic = 0.5 * prefactor * kT * np.einsum(
        "lL,alkm,aLkm->km", kernel, jump, jump)
    out = rates.sum(axis=1)
    dephasing = adiabatic + 0.5 * (out[:, None] + out[None, :])
    return KondoRates(energies, vectors, rates, adiabatic, dephasing)


def _frequency_groups(
    bohr: np.ndarray, tol: float
) -> tuple[np.ndarray, np.ndarray]:
    """Groups Bohr frequencies that differ by at most `tol`.

    Args:
        bohr: Matrix of frequencies.
        tol: Largest gap inside a group.

    Returns:
        The labels of each matrix entry (same shape as `bohr`) and the mean
        frequency of each group.
    """
    flat = bohr.ravel()
    order = np.argsort(flat, kind="stable")
    sorted_values = flat[order]
    new_group = np.concatenate([[True], np.diff(sorted_values) > tol])
    group_of_sorted = np.cumsum(new_group) - 1
    labels = np.empty(flat.size, dtype=int)
    labels[order] = group_of_sorted
    means = np.array([
        sorted_values[group_of_sorted == g].mean()
        for g in range(group_of_sorted[-1] + 1)])
    return labels.reshape(bohr.shape), means


def kondo_collapse_operators(
    hamiltonian: qt.Qobj,
    spin_ops: SpinOperators,
    positions: np.ndarray | Sequence[float],
    *,
    rho_J: float,
    kT: float,
    k_F: float,
    dim: int,
    tol: float = 1e-9,
) -> list[qt.Qobj]:
    """Builds the Davies collapse operators of the Kondo bath.

    The kernel matrix K_{ll'} is diagonalised, K = sum_c lambda_c u_c u_c^T,
    which gives the collective operators A_{c,a} = sum_l u_c(l) S^a_l. Each
    is split by Bohr frequency, A_{c,a}(w) = sum over E_M - E_M' = w of
    |M'><M'|A_{c,a}|M><M|, and

        L_{c,a,w} = sqrt(pi (rho J)^2 / 4 * lambda_c * G(w)) A_{c,a}(w)

    is a collapse operator. Frequencies that differ by at most `tol` count
    as equal, so degenerate spectra keep every coherent transition inside a
    level; w = 0 gives pure dephasing with the rate proportional to kT.
    The steady state is the Gibbs state of H, and the populations obey the
    master equation with the rates of `kondo_rates`.

    Args:
        hamiltonian: Hamiltonian of the spins, Hermitian.
        spin_ops: For each site the Hermitian operators (S^x, S^y, S^z) on
            the full space; see `chain_spin_operators`.
        positions: Site positions, shape (n_sites,) or (n_sites, m), in the
            inverse unit of `k_F`.
        rho_J: Dimensionless coupling in [0, 1).
        kT: Temperature in the units of the Hamiltonian.
        k_F: Fermi wave number.
        dim: Dimension of the electron gas, 1, 2 or 3.
        tol: Bohr frequencies closer than this are treated as equal.

    Returns:
        Collapse operators with the rates included, for `qutip.mesolve`;
        empty if `rho_J` is 0.

    Raises:
        ValueError: If a parameter is invalid (`validate_bath_parameters`),
            H or a spin operator is not Hermitian, or shapes or the number
            of positions do not match.
    """
    validate_bath_parameters(rho_J, kT, k_F, dim, tol)
    ham = _check_system(hamiltonian, spin_ops, positions)
    kernel = kernel_matrix(positions, k_F=k_F, dim=dim)
    weights, modes = np.linalg.eigh(kernel)
    energies, vectors = np.linalg.eigh(ham)
    # bohr[k, m] = E_m - E_k: the energy released by the jump m -> k.
    bohr = energies[None, :] - energies[:, None]
    labels, frequencies = _frequency_groups(bohr, tol)
    prefactor = math.pi * rho_J ** 2 / 4.0
    dims = hamiltonian.dims

    collapse = []
    for weight, mode in zip(weights, modes.T):
        if weight < _MIN_KERNEL_EIGENVALUE:
            continue
        for axis in range(3):
            collective = sum(
                amplitude * site[axis].full()
                for amplitude, site in zip(mode, spin_ops))
            eigenbasis = vectors.conj().T @ collective @ vectors
            for group, frequency in enumerate(frequencies):
                part = np.where(labels == group, eigenbasis, 0.0)
                rate = prefactor * weight * float(
                    bath_rate(frequency, kT))
                operator = math.sqrt(rate) * (
                    vectors @ part @ vectors.conj().T)
                if np.linalg.norm(operator) > _MIN_OPERATOR_NORM:
                    collapse.append(qt.Qobj(operator, dims=dims))
    return collapse


def level_shifts(
    hamiltonian: qt.Qobj,
    spin_ops: SpinOperators,
    *,
    rho_J: float,
    kT: float,
    bandwidth: float,
) -> np.ndarray:
    """Computes the Kondo shifts of the energy levels of a single spin.

    Leading logarithm of the real part of the self-energy (Surf. Sci. 630,
    337, Eqs. 17-20): with Re I(D) = Re I(0) - D/4 (rho J)^2 ln(2W / (pi kT)),

        dE_n = -(rho J)^2 / 4 * ln(2W / (pi kT))
               * sum_{n',a} |<n|S_a|n'>|^2 (E_n - E_n'),

    where the level-independent constant Re I(0) is dropped because it is
    not observable. It is meaningful for W >> kT.

    Args:
        hamiltonian: Hamiltonian of the spin, Hermitian.
        spin_ops: One site, the operators (S^x, S^y, S^z).
        rho_J: Dimensionless coupling in [0, 1).
        kT: Temperature in the units of the Hamiltonian.
        bandwidth: Half-width W of the electron band, same units; must
            satisfy 2 W > pi kT.

    Returns:
        The shifts of the eigenvalues of H, in ascending order of energy.

    Raises:
        ValueError: If a parameter is invalid, `spin_ops` has more than one
            site, `bandwidth` is not larger than pi kT / 2, or H or a spin
            operator is not Hermitian.
    """
    validate_bath_parameters(rho_J, kT, 1.0, 1, tol=1.0)
    if len(spin_ops) != 1:
        raise ValueError(
            "level_shifts is defined for a single spin; "
            f"got {len(spin_ops)} sites.")
    if not (math.isfinite(bandwidth)
            and 2.0 * bandwidth > math.pi * kT):
        raise ValueError(
            "bandwidth must satisfy 2 * bandwidth > pi * kT, "
            f"got {bandwidth!r}.")
    ham = _check_system(hamiltonian, spin_ops, [0.0])
    energies, vectors = np.linalg.eigh(ham)
    elements = _site_elements(spin_ops, vectors)[:, 0]
    strength = np.einsum("akm,akm->km", elements, elements.conj()).real
    gaps = energies[:, None] - energies[None, :]
    log = math.log(2.0 * bandwidth / (math.pi * kT))
    return -(rho_J ** 2) / 4.0 * log * np.sum(strength * gaps, axis=1)


def shifted_transitions(
    hamiltonian: qt.Qobj,
    spin_ops: SpinOperators,
    *,
    rho_J: float,
    kT: float,
    bandwidth: float,
) -> np.ndarray:
    """Computes the renormalised transition energies of a single spin.

    Args:
        hamiltonian: Hamiltonian of the spin, Hermitian.
        spin_ops: One site, the operators (S^x, S^y, S^z).
        rho_J: Dimensionless coupling in [0, 1).
        kT: Temperature in the units of the Hamiltonian.
        bandwidth: Half-width of the band, see `level_shifts`.

    Returns:
        The matrix ``D[n, m] = E_n - E_m + dE_n - dE_m`` (Surf. Sci. Eq. 21)
        over the eigenstates in ascending order of energy.

    Raises:
        ValueError: As `level_shifts`.
    """
    shifts = level_shifts(
        hamiltonian, spin_ops, rho_J=rho_J, kT=kT, bandwidth=bandwidth)
    energies = np.linalg.eigvalsh(hamiltonian.full())
    shifted = energies + shifts
    return shifted[:, None] - shifted[None, :]
