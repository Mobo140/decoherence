"""R28 — a Kondo bath for the Lindblad simulator, checked against two papers.

Why
---
The phenomenological sigma_minus / sigma_z jump operators with a Bernstein
gamma(t) have no microscopic origin. `src.physics.kondo_bath` derives the
dissipator of spins exchange-coupled to a free electron gas (Delgado and
Fernandez-Rossier, PRB 95, 075413 (2017); Surf. Sci. 630, 337 (2014)). This
script reproduces their results from that generic rate code and exact
diagonalisation (nothing is typed in from the closed forms except for the
overlay lines), then uses the new `kondo` simulator once.

  dimer   PRB Fig. 2(b, c): S = 1/2 Heisenberg dimer, J_H = 5 kT.
  swave   PRB Fig. 3: lowest spin waves of an S = 3/2 ferromagnetic chain
          (the Sz = 3N/2 and 3N/2 - 1 sectors only), N = 4, 5, 6, 8.
  ising   PRB Fig. 4: pure dephasing of Ising chains, Lambda^FM / N^2 and
          Lambda^AFM versus N k_F d / pi at k_F d / pi = 0.01.
  surf    Surf. Sci. Table 1 / Fig. 2: Korringa function, renormalisation of
          the transition energy of S = 3/2 and S = 1.
  demo    new: T2 and T2^E of two qubits versus k_F d under the `kondo`
          simulator, d = 1, 2, 3, XXZ and TFIM.

Writes ONLY into experiments/results/r28/ (CSV files, their provenance, PNG
figures and deviations.csv, the maximum deviation of each check).

Usage
-----
    python -m experiments.r28_kondo_reproduction --fast --tag smoke
    python -m experiments.r28_kondo_reproduction --part dimer
"""
from __future__ import annotations

import argparse
import math
import pathlib
from collections.abc import Callable, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import qutip as qt  # noqa: E402
import scipy.integrate  # noqa: E402
import scipy.sparse  # noqa: E402
from scipy import special  # noqa: E402

from experiments import common  # noqa: E402
from src.domain import value_objects  # noqa: E402
from src.physics import kondo_bath  # noqa: E402
from src.simulation import variants  # noqa: E402

OUT_DIR = common.RESULTS_DIR / "r28"
PARTS = ("dimer", "swave", "ising", "surf", "demo")
# Colours of the dimensions in the papers' figures.
DIM_COLORS = {1: "black", 2: "tab:red", 3: "tab:blue"}
RHO_J = 0.1
SEED = 0


def _gamma0(delta: float, kT: float, rho_J: float = RHO_J) -> float:
    """Single spin-1/2 Zeeman relaxation rate, PRB Eq. 22."""
    return math.pi * rho_J ** 2 / 8.0 * float(kondo_bath.bath_rate(delta, kT))


def _save(fig: plt.Figure, name: str) -> pathlib.Path:
    """Saves a figure under the r28 results directory."""
    path = OUT_DIR / name
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return path


def _write(rows: list[dict], name: str, config: str) -> None:
    """Writes a CSV with provenance into the r28 directory."""
    common.write_results(rows, OUT_DIR / name, config=config, seeds=[SEED])


def _deviation(figure: str, quantity: str, a: np.ndarray, b: np.ndarray,
               note: str = "") -> dict:
    """Records the maximum absolute deviation between two arrays."""
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    return {"figure": figure, "quantity": quantity,
            "max_abs_deviation": float(np.max(np.abs(a - b))),
            "n_points": int(a.size), "note": note}


# --- PRB Fig. 2: Heisenberg dimer -------------------------------------------

def _dimer_hamiltonian(j_h: float, field: float) -> qt.Qobj:
    """H = J_H S1.S2 + field (S1z + S2z) of two spins 1/2."""
    ops = kondo_bath.chain_spin_operators(2)
    return j_h * sum(ops[0][a] * ops[1][a] for a in range(3)) + field * (
        ops[0][2] + ops[1][2])


def _state_index(vectors: np.ndarray, target: np.ndarray) -> int:
    """Returns the column of `vectors` with the largest overlap with target."""
    return int(np.argmax(np.abs(vectors.conj().T @ target)))


def part_dimer(fast: bool) -> list[dict]:
    """Reproduces PRB Fig. 2(b, c) from the generic rates.

    Returns:
        The deviation records.
    """
    kT, j_h, field = 1.0, 5.0, 10.0
    x = np.linspace(0.0, 2.0, 21 if fast else 201)  # k_F d / pi
    ops = kondo_bath.chain_spin_operators(2)
    zero_plus = (qt.tensor(qt.basis(2, 0), qt.basis(2, 1))
                 + qt.tensor(qt.basis(2, 1), qt.basis(2, 0))).full().ravel()
    zero_minus = (qt.tensor(qt.basis(2, 0), qt.basis(2, 1))
                  - qt.tensor(qt.basis(2, 1), qt.basis(2, 0))).full().ravel()
    down = qt.tensor(qt.basis(2, 1), qt.basis(2, 1)).full().ravel()
    zero_plus, zero_minus = zero_plus / math.sqrt(2), zero_minus / math.sqrt(2)

    rows, records = [], []
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.6))
    for dim in (1, 2, 3):
        kernel = kondo_bath.pair_kernel(dim, np.pi * x)
        num_b, num_c, num_c2 = [], [], []
        for xi in x:
            pos = [0.0, math.pi * xi]
            kw = dict(rho_J=RHO_J, kT=kT, k_F=1.0, dim=dim)
            r0 = kondo_bath.kondo_rates(_dimer_hamiltonian(j_h, 0.0), ops,
                                        pos, **kw)
            # Ascending energies: the singlet first, then the triplet.
            num_b.append(np.mean(
                [r0.rates[t, 0] for t in (1, 2, 3)]) / _gamma0(j_h, kT))
            r1 = kondo_bath.kondo_rates(
                _dimer_hamiltonian(j_h, field), ops, pos, **kw)
            i_up, i_down = (_state_index(r1.states, zero_plus),
                            _state_index(r1.states, down))
            i_sing = _state_index(r1.states, zero_minus)
            num_c.append(r1.rates[i_up, i_down] / _gamma0(field, kT))
            num_c2.append(r1.rates[i_sing, i_down]
                          / _gamma0(field - j_h, kT))
        num_b, num_c, num_c2 = map(np.array, (num_b, num_c, num_c2))
        closed_b, closed_c = 1.0 - kernel, 1.0 + kernel
        records += [
            _deviation("PRB Fig. 2(b)", f"Gamma_T->S/Gamma0, d={dim}",
                       num_b, closed_b, "closed form 1 - K_d, Eqs. 28, 31-32"),
            _deviation("PRB Fig. 2(c)", f"Gamma_10->1-1/Gamma0, d={dim}",
                       num_c, closed_c, "closed form 1 + K_d, Eqs. 30, 33-34"),
            _deviation("PRB Sec. IV C 3", f"Gamma_00->1-1/Gamma0, d={dim}",
                       num_c2, closed_b,
                       "closed form (1 - K_d), Gamma0 at B - J_H, Eq. 28"),
        ]
        axes[0].plot(x, closed_b, color=DIM_COLORS[dim], lw=1.2,
                     label=f"{dim}D closed form")
        axes[0].plot(x[::8], num_b[::8], "o", color=DIM_COLORS[dim], ms=4)
        axes[1].plot(x, closed_c, color=DIM_COLORS[dim], lw=1.2,
                     label=f"{dim}D closed form")
        axes[1].plot(x[::8], num_c[::8], "o", color=DIM_COLORS[dim], ms=4)
        axes[2].plot(x, closed_b, color=DIM_COLORS[dim], lw=1.2)
        axes[2].plot(x[::8], num_c2[::8], "o", color=DIM_COLORS[dim], ms=4)
        rows += [{"dim": dim, "kFd_over_pi": float(xi), "kernel": float(k),
                  "triplet_to_singlet": b, "t10_to_t1m1": c,
                  "singlet_to_t1m1": c2}
                 for xi, k, b, c, c2 in zip(x, kernel, num_b, num_c, num_c2)]
    titles = ["(b) B = 0: triplet -> singlet", "(c) large B: |1,0> -> |1,-1>",
              "subradiant: |0,0> -> |1,-1>"]
    for ax, title in zip(axes, titles):
        ax.set_title(title, fontsize=10)
        ax.set_xlabel(r"$k_F d / \pi$")
        ax.set_ylabel(r"$\Gamma / \Gamma_0$")
    axes[0].legend(fontsize=8, title="dots: generic code")
    fig.suptitle("PRB 95, 075413 Fig. 2: S=1/2 dimer, J_H = 5 kT, "
                 "g mu_B B = 10 kT", fontsize=10)
    _save(fig, "fig2_dimer.png")
    _write(rows, "r28_dimer.csv", "PRB Fig. 2 from kondo_rates")
    return records


# --- PRB Fig. 3: spin waves of an S = 3/2 ferromagnetic chain ---------------

def _sector_problem(
    n_sites: int, j_h: float, d_aniso: float
) -> tuple[qt.Qobj, list[tuple[qt.Qobj, qt.Qobj, qt.Qobj]]]:
    """Builds H and the spin operators on the Sz = 3N/2, 3N/2 - 1 sectors.

    The Hamiltonian D sum Sz^2 + J_H sum S_l.S_(l+1) of N spins 3/2 is built
    sparsely on the whole 4^N space and projected onto the N + 1 states that
    have at most one spin lowered; H conserves Sz, so the projection is
    exact for it, and the projected spin operators give exactly the matrix
    elements between these states.

    Args:
        n_sites: Number of spins.
        j_h: Heisenberg exchange.
        d_aniso: Single-ion anisotropy D.

    Returns:
        The sector Hamiltonian and, for each site, (Sx, Sy, Sz) projected.
    """
    local = [scipy.sparse.csr_matrix(qt.jmat(1.5, a).full()) for a in "xyz"]
    eye = scipy.sparse.identity(4, format="csr")

    def site_op(op: scipy.sparse.spmatrix, site: int) -> scipy.sparse.spmatrix:
        out = scipy.sparse.identity(1, format="csr")
        for k in range(n_sites):
            out = scipy.sparse.kron(out, op if k == site else eye,
                                    format="csr")
        return out

    full = [[site_op(local[a], s) for a in range(3)] for s in range(n_sites)]
    hamiltonian = sum(d_aniso * full[s][2] @ full[s][2]
                      for s in range(n_sites))
    for s in range(n_sites - 1):
        hamiltonian = hamiltonian + j_h * sum(
            full[s][a] @ full[s + 1][a] for a in range(3))
    # Basis index with digit 0 = m 3/2 and 1 = m 1/2, site 1 most significant.
    keep = [0] + [4 ** (n_sites - i) for i in range(1, n_sites + 1)]
    h_sector = qt.Qobj(hamiltonian[keep][:, keep].toarray())
    ops = [tuple(qt.Qobj(full[s][a][keep][:, keep].toarray())
                 for a in range(3)) for s in range(n_sites)]
    return h_sector, ops


def _swave_rates(
    n_sites: int, x: np.ndarray, kT: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
    """Relaxation of SW1, SW2 to the ground state versus k_F d / pi.

    Returns:
        The rates of SW1 and SW2 in units of the single-spin S = 3/2 rate
        at the excitation energy of SW1, the same from the semi-analytic
        formula sum_{ll'} K_ll' C_l C_l' with the ED amplitudes, the
        amplitudes C(i) of SW1 and SW2 (a 2 x N array) and Delta_SW1.
    """
    h_sector, ops = _sector_problem(n_sites, -1.0, -1.0)
    kw = dict(rho_J=RHO_J, kT=kT, dim=3, k_F=1.0)
    out = np.zeros((2, x.size))
    semi = np.zeros((2, x.size))
    for ix, xi in enumerate(x):
        positions = math.pi * xi * np.arange(n_sites)
        rates = kondo_bath.kondo_rates(h_sector, ops, positions, **kw)
        amplitudes = rates.states[1:, 1:3].T  # <i | SW>, SW1 and SW2 rows
        if ix == 0:
            # SW1 is the lowest excitation, SW2 the next one.
            order = np.argsort(-np.abs(amplitudes.sum(axis=1)))
            if order[0] != 0:
                raise RuntimeError(
                    "the lowest excitation is not the uniform spin wave")
        kernel = kondo_bath.kernel_matrix(positions, k_F=1.0, dim=3)
        # Both waves are measured against the single-spin rate at the
        # energy of SW1, as the size dependence of the SW2 energy shows up
        # in the paper's figure.
        norm = (math.pi * RHO_J ** 2 / 4.0 * 1.5 * float(
            kondo_bath.bath_rate(rates.energies[1] - rates.energies[0], kT)))
        for k in range(2):
            out[k, ix] = rates.rates[k + 1, 0] / norm
            c = amplitudes[k].real
            # G(Delta) = Delta for Delta >> kT.
            semi[k, ix] = (c @ kernel @ c) * (
                rates.energies[k + 1] - rates.energies[0]) / (
                rates.energies[1] - rates.energies[0])
        if ix == 0:
            gap1 = rates.energies[1] - rates.energies[0]
            coeff = amplitudes.real.copy()
            coeff *= np.sign(coeff.sum(axis=1, keepdims=True) + 1e-300)
            if coeff[1, -1] < coeff[1, 0]:
                coeff[1] *= -1.0
    return out[0], out[1], semi, coeff, gap1


def part_swave(fast: bool) -> list[dict]:
    """Reproduces PRB Fig. 3 for S = 3/2 FM chains with D = J_H = -1 meV.

    Returns:
        The deviation records.
    """
    kT = 0.02  # meV, far below the 2 meV excitation: G(Delta) = Delta.
    x = np.linspace(0.0, 2.0, 21 if fast else 81)
    sizes = (4, 5, 6, 8)
    rows, records = [], []
    fig, axes = plt.subplots(2, 2, figsize=(10, 7))
    coefficient_rows = []
    for n_sites in sizes:
        sw1, sw2, semi, coeff, gap = _swave_rates(n_sites, x, kT)
        records += [
            _deviation("PRB Fig. 3", f"SW1 rate, N={n_sites}", sw1, semi[0],
                       "kondo_rates vs sum K_ll' C_l C_l' (ED amplitudes), G = Delta"),
            _deviation("PRB Fig. 3", f"SW2 rate, N={n_sites}", sw2, semi[1],
                       "kondo_rates vs sum K_ll' C_l C_l' (ED amplitudes), G = Delta"),
            _deviation("PRB Eq. 37", f"SW1 rate at k_F d = 0, N={n_sites}",
                       np.array([sw1[0]]), np.array([float(n_sites)]),
                       "ratio to the single-spin rate vs N"),
            _deviation("PRB Fig. 3(d)", f"SW2 rate at k_F d = 0, N={n_sites}",
                       np.array([sw2[0]]), np.array([0.0]),
                       "SW2 is orthogonal to SW1: no decay at k_F d = 0"),
            _deviation("PRB Sec. V A", f"Delta_SW1 / (2S-1)|J|, N={n_sites}",
                       np.array([gap]), np.array([2.0]), "meV"),
        ]
        axes[1, 0].plot(x, sw1, label=f"N={n_sites}")
        axes[1, 1].plot(x, sw2, label=f"N={n_sites}")
        rows += [{"N": n_sites, "kFd_over_pi": float(xi), "sw1": a, "sw2": b}
                 for xi, a, b in zip(x, sw1, sw2)]
        if n_sites == 8:
            sites = np.arange(1, n_sites + 1)
            axes[0, 0].bar(sites, coeff[0], color="black")
            axes[0, 1].bar(sites, coeff[1], color="tab:red")
            coefficient_rows = [
                {"site": int(s), "C_sw1": float(a), "C_sw2": float(b)}
                for s, a, b in zip(sites, coeff[0], coeff[1])]
    axes[0, 0].set_title("SW1 amplitudes C(i), N=8")
    axes[0, 1].set_title("SW2 amplitudes C(i), N=8")
    axes[1, 0].set_title(r"$\Gamma_{SW1}$ / single-spin rate, 3D gas")
    axes[1, 1].set_title(r"$\Gamma_{SW2}$ / single-spin rate, 3D gas")
    for ax in axes[1]:
        ax.set_xlabel(r"$k_F d/\pi$")
        ax.legend(fontsize=8)
    for ax in axes[0]:
        ax.set_xlabel("site")
    fig.suptitle("PRB 95, 075413 Fig. 3: S=3/2 FM chain, D = J_H = -1 meV",
                 fontsize=10)
    _save(fig, "fig3_spin_waves.png")
    _write(rows, "r28_swave.csv", "PRB Fig. 3(c, d) from kondo_rates")
    _write(coefficient_rows, "r28_swave_coefficients_n8.csv",
           "PRB Fig. 3(a, b), N=8 amplitudes")
    return records


# --- PRB Fig. 4: Ising chains -------------------------------------------------

def _ising_lambda(signs: np.ndarray, xi: float, dim: int) -> float:
    """Lambda = 4 s^T K s for the classical states with S^z(l) = s_l S."""
    positions = xi * np.arange(1, signs.size + 1)
    kernel = kondo_bath.kernel_matrix(positions, k_F=1.0, dim=dim)
    return float(4.0 * signs @ kernel @ signs)


def _paper_lambda_1d(xi: float, n: int, kind: str) -> float:
    """Closed forms (C3) and (C5) of PRB for one dimension.

    (C6), for odd N, is not used: as printed it does not tend to 4 for
    k_F d -> 0, which the text states, so it is misprinted.
    """
    if kind == "fm":
        return 2.0 * (n ** 2 + math.sin(n * xi) ** 2 / math.sin(xi) ** 2
                      * math.cos(xi * (n + 1)) ** 2)
    if kind != "afm_even":
        raise ValueError(f"unknown kind {kind!r}")
    return 0.5 / math.sin(2 * xi) ** 2 * (
        math.sin(2 * xi) + math.sin(2 * n * xi)
        - math.sin(2 * (1 + n) * xi)) ** 2


def _paper_lambda_sum(xi: float, n: int, dim: int, sign: str) -> float:
    """Explicit sums (C4) and the AFM sums of PRB for d = 2, 3."""
    total = 0.0
    for a in range(1, n + 1):
        for b in range(1, n + 1):
            u = xi * abs(a - b)
            k = special.j0(u) ** 2 if dim == 2 else (
                1.0 if u == 0 else (math.sin(u) / u) ** 2)
            total += (-1) ** (a + b) * k if sign == "afm" else k
    return 4.0 * total


def _ising_cos_cos(xi: float, n: int, kind: str) -> float:
    """1D Lambda with the phases cos(q r_l) cos(q r_l') of PRB Eq. C1.

    Pure arithmetic, used to show where the paper's 1D formulas come from.
    """
    sites = np.arange(1, n + 1)
    weight = np.ones(n) if kind == "fm" else (-1.0) ** sites
    q0 = weight.sum() ** 2
    qf = (weight * np.cos(2 * xi * sites)).sum() ** 2
    return 2.0 * (q0 + qf)


def part_ising(fast: bool) -> list[dict]:
    """Reproduces PRB Fig. 4 for Ising chains; validates against ED.

    Returns:
        The deviation records.
    """
    ratio = 0.01  # k_F d / pi
    xi = math.pi * ratio
    n_max = 40 if fast else 200
    sizes = np.arange(1, n_max + 1)
    x = sizes * ratio
    rows, records = [], []

    # ED of small chains: the generic rates between the two degenerate
    # ground states, against the kernel sum used for the large chains.
    ed_dev = {"fm": 0.0, "afm": 0.0}
    for dim in (1, 2, 3):
        for n_sites in range(2, 9):
            ops = kondo_bath.chain_spin_operators(n_sites)
            for kind in ("fm", "afm"):
                sign = 1.0 if kind == "fm" else -1.0
                ham = -sign * sum(
                    4 * ops[i][2] * ops[i + 1][2] for i in range(n_sites - 1))
                rates = kondo_bath.kondo_rates(
                    ham, ops, xi * np.arange(1, n_sites + 1),
                    states=np.eye(2 ** n_sites), rho_J=RHO_J, kT=1.0,
                    k_F=1.0, dim=dim)
                # Ground doublet: all up / all down (FM), Neel pair (AFM).
                lo = rates.energies[0]
                ground = [i for i in range(2 ** n_sites)
                          if rates.energies[i] < lo + 1e-9]
                gamma = rates.dephasing_adiabatic[ground[0], ground[1]]
                # Lambda = gamma / (pi (rho J)^2 kT / 8 * S^2), S = 1/2.
                lam_ed = gamma / (math.pi * RHO_J ** 2 / 8.0 * 0.25)
                signs = (np.ones(n_sites) if kind == "fm"
                         else (-1.0) ** np.arange(1, n_sites + 1))
                lam_sum = _ising_lambda(signs, xi, dim)
                ed_dev[kind] = max(ed_dev[kind], abs(lam_ed - lam_sum))
    records += [
        _deviation("PRB Eqs. 42-44", f"ED vs kernel sum, {kind.upper()}",
                   np.array([ed_dev[kind]]), np.array([0.0]),
                   "N = 2..8, d = 1, 2, 3, kondo_rates between the two "
                   "degenerate ground states")
        for kind in ed_dev]

    series, doubled = {}, {}
    for dim in (1, 2, 3):
        fm = np.array([_ising_lambda(np.ones(n), xi, dim) for n in sizes])
        afm = np.array([_ising_lambda((-1.0) ** np.arange(1, n + 1), xi, dim)
                        for n in sizes])
        series[dim] = (fm, afm)
        # The same sums with the kernel argument 2 k_F |r - r'|.
        doubled[dim] = (
            np.array([_ising_lambda(np.ones(n), 2 * xi, dim) for n in sizes]),
            np.array([_ising_lambda((-1.0) ** np.arange(1, n + 1), 2 * xi,
                                    dim) for n in sizes]))
        rows += [{"dim": dim, "N": int(n), "N_kFd_over_pi": float(v),
                  "lambda_fm_over_n2": float(a / n ** 2),
                  "lambda_afm": float(b),
                  "lambda_fm_over_n2_arg_2xi": float(c / n ** 2),
                  "lambda_afm_arg_2xi": float(e)}
                 for n, v, a, b, c, e in zip(
                     sizes, x, fm, afm, *doubled[dim])]
    odd, even = sizes % 2 == 1, sizes % 2 == 0

    # Overlay: the paper's formulas as printed, and the cos-cos arithmetic.
    paper_fm = np.array([_paper_lambda_1d(xi, int(n), "fm") for n in sizes])
    paper_even = np.array([_paper_lambda_1d(xi, int(n), "afm_even")
                           for n in sizes[even]])
    paper_odd = np.array([_ising_cos_cos(xi, int(n), "afm")
                          for n in sizes[odd]])
    cc_fm = np.array([_ising_cos_cos(xi, int(n), "fm") for n in sizes])
    cc_even = np.array([_ising_cos_cos(xi, int(n), "afm")
                        for n in sizes[even]])
    fm1, afm1 = series[1]
    records += [
        _deviation("PRB Fig. 4(a)", "Lambda^FM_1D/N^2 vs Eq. C3",
                   fm1 / sizes ** 2, paper_fm / sizes ** 2,
                   "closed form as printed; see README"),
        _deviation("PRB Fig. 4(c)", "Lambda^AFM_1D even N vs Eq. C5",
                   afm1[even], paper_even, "closed form as printed"),
        _deviation("PRB Fig. 4(b)", "Lambda^AFM_1D odd N vs cos-cos form",
                   afm1[odd], paper_odd,
                   "Eq. C6 is misprinted; cos-cos arithmetic of Eq. C1"),
        _deviation("PRB Eq. C3", "Eq. C3 vs cos(q r_l) cos(q r_l') sum",
                   paper_fm, cc_fm,
                   "the paper's 1D formula is the cos-cos part only"),
        _deviation("PRB Eq. C5", "Eq. C5 vs cos-cos sum, even N",
                   paper_even, cc_even, "as above"),
    ]
    records += [
        _deviation("PRB Fig. 4(a)", "Lambda^FM_1D/N^2, arg 2xi, vs Eq. C3",
                   doubled[1][0] / sizes ** 2, paper_fm / sizes ** 2,
                   "same sums with kernel argument 2 k_F |r-r'|"),
        _deviation("PRB Fig. 4(c)", "Lambda^AFM_1D even, arg 2xi, vs Eq. C5",
                   doubled[1][1][even], paper_even,
                   "same sums with kernel argument 2 k_F |r-r'|"),
    ]
    for dim in (2, 3):
        sub = slice(0, min(n_max, 40))
        fm_sum = np.array([_paper_lambda_sum(xi, int(n), dim, "fm")
                           for n in sizes[sub]])
        afm_sum = np.array([_paper_lambda_sum(xi, int(n), dim, "afm")
                            for n in sizes[sub]])
        records += [
            _deviation("PRB Fig. 4(a)", f"Lambda^FM_{dim}D vs Eq. C4",
                       series[dim][0][sub], fm_sum, "N <= 40"),
            _deviation("PRB Fig. 4(b, c)", f"Lambda^AFM_{dim}D vs C-sums",
                       series[dim][1][sub], afm_sum, "N <= 40"),
        ]
    records.append(_deviation(
        "PRB Fig. 4(a)", "Lambda^FM/N^2 at N=1 vs 4",
        np.array([series[d][0][0] for d in (1, 2, 3)]), np.full(3, 4.0),
        "k_F d -> 0 limit is 4 N^2"))

    fig, axes = plt.subplots(3, 1, figsize=(6.5, 9), sharex=True)
    for dim in (1, 2, 3):
        fm, afm = series[dim]
        color = DIM_COLORS[dim]
        axes[0].plot(x, fm / sizes ** 2, color=color, label=f"{dim}D")
        axes[1].plot(x[odd], afm[odd], color=color)
        axes[2].plot(x[even], afm[even], color=color)
    for dim in (1, 2, 3):
        axes[0].plot(x, doubled[dim][0] / sizes ** 2, "--", lw=0.9,
                     color=DIM_COLORS[dim])
        axes[1].plot(x[odd], doubled[dim][1][odd], "--", lw=0.9,
                     color=DIM_COLORS[dim])
        axes[2].plot(x[even], doubled[dim][1][even], "--", lw=0.9,
                     color=DIM_COLORS[dim])
    axes[0].plot([], [], "k--", lw=0.9, label="kernel argument 2 k_F d")
    axes[0].plot(x, paper_fm / sizes ** 2, "k:", lw=1.5,
                 label="1D, Eq. C3 as printed")
    axes[1].plot(x[odd], paper_odd, "k:", lw=1.5)  # cos-cos form (C1)
    axes[2].plot(x[even], paper_even, "k:", lw=1.5)
    axes[0].set_ylabel(r"$\Lambda^{FM}/N^2$")
    axes[1].set_ylabel(r"$\Lambda^{AFM}$, odd N")
    axes[2].set_ylabel(r"$\Lambda^{AFM}$, even N")
    axes[2].set_xlabel(r"$N k_F d/\pi$   ($k_F d/\pi = 0.01$)")
    axes[0].legend(fontsize=8)
    fig.suptitle("PRB 95, 075413 Fig. 4: Ising-chain dephasing "
                 "(solid: kondo_bath kernel, dashed: argument 2 k_F d,\n"
                 "dotted: paper's 1D formula)",
                 fontsize=9)
    _save(fig, "fig4_ising_dephasing.png")
    _write(rows, "r28_ising.csv", "PRB Fig. 4 from the kernel sums")
    return records


# --- Surf. Sci.: shifts and Korringa rates ----------------------------------

def _korringa_integral(delta: float, kT: float) -> float:
    """Evaluates int f(e) (1 - f(e + delta)) de numerically (Eq. 15)."""
    def fermi(e: float) -> float:
        return 0.5 * (1.0 - math.tanh(e / (2.0 * kT)))

    span = 40.0 * kT + abs(delta)
    value, _ = scipy.integrate.quad(
        lambda e: fermi(e) * (1.0 - fermi(e + delta)),
        -span - abs(delta), span, limit=400, points=[0.0, -delta])
    return value


def _spin_hamiltonian(spin: float, d_aniso: float, e_aniso: float) -> tuple:
    """Returns H = D Sz^2 + E (Sx^2 - Sy^2) and the single-site spin ops."""
    ops = kondo_bath.chain_spin_operators(1, spin)
    sx, sy, sz = ops[0]
    return d_aniso * sz * sz + e_aniso * (sx * sx - sy * sy), ops


def part_surf(fast: bool) -> list[dict]:
    """Checks the Surf. Sci. Korringa function and level renormalisation.

    Returns:
        The deviation records.
    """
    records, rows = [], []
    kT = 0.1
    deltas = np.linspace(-6.0, 12.0, 37) * kT
    numeric = np.array([_korringa_integral(d, kT) for d in deltas])
    closed = np.array([float(kondo_bath.bath_rate(d, kT)) for d in deltas])
    records.append(_deviation(
        "SurfSci Eq. 15", "G(Delta) vs quadrature of f(1-f)", closed, numeric,
        "Delta/kT in [-6, 12]"))

    # Korringa rates of a spin-1/2 Zeeman doublet from the generic code.
    ops = kondo_bath.chain_spin_operators(1)
    grid = np.logspace(-2, 2, 41) * kT
    rates = np.array([kondo_bath.kondo_rates(
        d / 2 * qt.sigmaz(), ops, [0.0], rho_J=RHO_J, kT=kT, k_F=1.0,
        dim=3).rates[1, 0] for d in grid])
    expected = np.array([_gamma0(d, kT) for d in grid])
    records += [
        _deviation("SurfSci Eq. 14 / PRB Eq. 22", "Gamma(Delta) vs Gamma0",
                   rates, expected, "relaxation of a spin 1/2"),
        _deviation("SurfSci Eq. 14", "Delta >> kT: Gamma / (pi (rJ)^2 Delta/8)",
                   np.array([rates[-1] / (math.pi * RHO_J ** 2 / 8 * grid[-1])]),
                   np.array([1.0]), "linear in Delta"),
        _deviation("SurfSci Eq. 14", "Delta -> 0: Gamma / (pi (rJ)^2 kT/8)",
                   np.array([rates[0] / (math.pi * RHO_J ** 2 / 8 * kT)]),
                   np.array([1.0]), "Korringa limit, Gamma proportional to kT"),
    ]
    rows += [{"kind": "korringa", "x": float(d / kT), "value": float(r),
              "reference": float(e)} for d, r, e in zip(grid, rates, expected)]

    # Transition-energy renormalisation, S = 3/2 (Co on Cu2N) and S = 1.
    big_w, kt_surf = 1000.0, 0.1  # meV
    log = math.log(2 * big_w / (math.pi * kt_surf))
    rho_j = np.linspace(0.0, 0.3, 31)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for name, spin, d_an, e_an, ax in (
            ("S=3/2, D=-3.5, E=2 meV", 1.5, -3.5, 2.0, axes[0]),
            ("S=3/2, D=-3.5, E=0 meV", 1.5, -3.5, 0.0, axes[0]),
            ("S=1, D=-3.5, E=1.5 meV", 1.0, -3.5, 1.5, axes[1])):
        ham, sops = _spin_hamiltonian(spin, d_an, e_an)
        energies = np.linalg.eigvalsh(ham.full())
        numeric = []
        for value in rho_j:
            mat = kondo_bath.shifted_transitions(
                ham, sops, rho_J=float(value), kT=kt_surf, bandwidth=big_w)
            numeric.append(mat)
        numeric = np.array(numeric)
        if spin == 1.5:
            # Excitation between the two Kramers doublets: levels 2 vs 0.
            i, j = 2, 0
            delta0 = energies[i] - energies[j]
            ratio = numeric[:, i, j] / delta0
            lam = _lambda_half(sops, ham)
            lam_paper = 1.5 * (1.0 + (e_an / d_an) ** 2)
            eq23_ed = 1.0 - rho_j ** 2 * lam / 8.0 * log
            eq23 = 1.0 - rho_j ** 2 * lam_paper / 8.0 * log
            eq21 = 1.0 - rho_j ** 2 * log / 4.0 * _eq21_sum(ham, sops, i, j,
                                                            delta0)
            ax.plot(rho_j, ratio, "o", ms=3, label=f"{name}: generic Eq. 21")
            ax.plot(rho_j, eq23, "-", lw=1,
                    label="Eq. 23 as printed (Lambda = 3/2 (1 + E^2/D^2))")
            records += [
                _deviation("SurfSci Eq. 23", f"{name}: generic vs Eq. 23",
                           ratio, eq23, f"Lambda = {lam_paper:.4f}, paper"),
                _deviation("SurfSci Eq. 23",
                           f"{name}: generic vs Eq. 23 with Lambda from ED",
                           ratio, eq23_ed, f"Lambda = {lam:.4f}"),
                _deviation("SurfSci Eq. 21", f"{name}: generic vs hand sum",
                           ratio, eq21, "same formula, independent code"),
            ]
            rows += [{"kind": f"shift {name}", "x": float(r), "value": float(v),
                      "reference": float(w)}
                     for r, v, w in zip(rho_j, ratio, eq23)]
        else:
            order = np.argsort(energies)  # ascending: alpha, beta, 0
            delta0 = energies[order[1]] - energies[order[0]]
            ratio = numeric[:, order[1], order[0]] / delta0
            eq24 = 1.0 - 1.5 * rho_j ** 2 * log
            ax.plot(rho_j, ratio, "o", ms=3, label=f"{name}: generic Eq. 21")
            ax.plot(rho_j, eq24, "-", lw=1, label="Eq. 24 as printed")
            records.append(_deviation(
                "SurfSci Eq. 24", f"{name}: generic vs Eq. 24", ratio, eq24,
                "closed form as printed"))
            records.append(_deviation(
                "SurfSci Eq. 24", f"{name}: generic vs Eq. 24 with 3/4",
                ratio, 1.0 - 0.75 * rho_j ** 2 * log,
                "coefficient 3/4 instead of the printed 3/2"))
            rows += [{"kind": f"shift {name}", "x": float(r), "value": float(v),
                      "reference": float(w)}
                     for r, v, w in zip(rho_j, ratio, eq24)]
    for ax in axes:
        ax.set_xlabel(r"$\rho J$")
        ax.set_ylabel(r"$\tilde\Delta / \Delta_0$")
        ax.legend(fontsize=7)
    fig.suptitle(f"Surf. Sci. 630, 337 Eqs. 21, 23, 24: W = {big_w:.0f} meV, "
                 f"kT = {kt_surf} meV", fontsize=10)
    _save(fig, "fig_surfsci_shift.png")

    fig, ax = plt.subplots(1, 2, figsize=(10, 3.8))
    grid_x = np.logspace(-2, 2, 200)
    ax[0].loglog(grid_x, [float(kondo_bath.bath_rate(g, 1.0)) for g in grid_x],
                 "-", label="G(Delta), Eq. 15")
    ax[0].loglog(grid / kT, rates / (math.pi * RHO_J ** 2 / 8.0 * kT), "o",
                 ms=3, label="generic rate / [pi (rJ)^2 kT / 8]")
    ax[0].loglog(grid_x, grid_x, "k:", lw=1, label="Delta")
    ax[0].axhline(1.0, color="gray", lw=1, ls=":")
    ax[0].set_xlabel(r"$\Delta / k_B T$")
    ax[0].set_ylabel(r"$G / k_B T$")
    ax[0].legend(fontsize=7)
    ax[1].plot(deltas / kT, closed, "-", label="Eq. 15 closed form")
    ax[1].plot(deltas / kT, numeric_g(deltas, kT), "o", ms=3,
               label="quadrature of f(e)(1 - f(e + Delta))")
    ax[1].set_xlabel(r"$\Delta / k_B T$")
    ax[1].legend(fontsize=7)
    _save(fig, "fig_surfsci_korringa.png")
    _write(rows, "r28_surfsci.csv", "Surf. Sci. Eqs. 14, 15, 21, 23, 24")
    return records


def numeric_g(deltas: np.ndarray, kT: float) -> np.ndarray:
    """Quadrature of the Korringa function on a grid of energies."""
    return np.array([_korringa_integral(d, kT) for d in deltas])


def _lambda_half(sops: Sequence, ham: qt.Qobj) -> float:
    """Lambda = sum_a |<1/2|S_a|3/2>|^2 between the two Kramers doublets.

    Summed over the upper doublet and averaged over the lower one, so that
    it does not depend on the basis chosen inside a doublet.
    """
    _, vectors = np.linalg.eigh(ham.full())
    lam = 0.0
    for a in range(3):
        element = vectors.conj().T @ sops[0][a].full() @ vectors
        lam += np.sum(np.abs(element[:2, 2:]) ** 2) / 2.0
    return float(lam)


def _eq21_sum(ham: qt.Qobj, sops: Sequence, n: int, m: int,
              delta: float) -> float:
    """Evaluates sum_{n'a}[|S_nn'|^2 D_nn' - |S_mn'|^2 D_mn'] / delta."""
    energies, vectors = np.linalg.eigh(ham.full())
    total = 0.0
    for a in range(3):
        el = np.abs(vectors.conj().T @ sops[0][a].full() @ vectors) ** 2
        total += np.sum(el[n] * (energies[n] - energies)) - np.sum(
            el[m] * (energies[m] - energies))
    return float(total / delta)


# --- new demo: T2 versus k_F d under the kondo simulator --------------------

def part_demo(fast: bool) -> list[dict]:
    """Runs two-qubit trajectories under the `kondo` simulator.

    T2 (computational-basis l1 coherence) and T2^E (l1 coherence in the
    energy basis) versus k_F d / pi for d = 1, 2, 3 at rho J = 0.1.

    Returns:
        An empty list; the demo has no paper to compare with.
    """
    kT, j_coupling, t_max, dt = 0.2, 0.8, 1200.0, 1.0
    x = np.linspace(0.0, 2.0, 5 if fast else 17)
    seeds = (11, 12) if fast else (11, 12, 13)
    rows = []
    criteria = {"t2": value_objects.DecoherenceCriterion.COHERENCE,
                "t2_energy":
                    value_objects.DecoherenceCriterion.COHERENCE_ENERGY}
    interactions = (value_objects.InteractionType.XXZ,
                    value_objects.InteractionType.TFIM)
    for interaction in interactions:
        configs = {name: value_objects.SystemConfig(
            n_qubits=value_objects.QubitCount.TWO, omega=1.0, J=j_coupling,
            interaction_type=interaction,
            dissipator=value_objects.DissipatorConfig.constant(1.0),
            t_max=t_max, dt=dt, decoherence_criterion=crit)
            for name, crit in criteria.items()}
        for dim in (1, 2, 3):
            for xi in x:
                sim = variants.KondoSimulator(
                    rho_J=RHO_J, kT=kT, kF_d=math.pi * float(xi), dim=dim)
                for seed in seeds:
                    row = {"system": interaction.value, "dim": dim,
                           "kFd_over_pi": float(xi), "seed": seed}
                    for name, config in configs.items():
                        np.random.seed(seed)
                        result = sim.simulate(config)
                        row[name] = result.t_decoh
                        row[f"{name}_censored"] = result.censored
                    rows.append(row)
            print(f"R28 demo | {interaction.value} d={dim} done", flush=True)
    _write(rows, "r28_demo_t2.csv",
           f"kondo simulator, rho_J={RHO_J}, kT={kT}, J={j_coupling}, "
           f"t_max={t_max}, seeds={list(seeds)}")

    fig, axes = plt.subplots(2, 2, figsize=(11, 7), sharex=True)
    for r, interaction in enumerate(interactions):
        for c, name in enumerate(criteria):
            ax = axes[r, c]
            for dim in (1, 2, 3):
                values = []
                for xi in x:
                    cell = [row[name] for row in rows
                            if row["system"] == interaction.value
                            and row["dim"] == dim
                            and abs(row["kFd_over_pi"] - xi) < 1e-12
                            and not row[f"{name}_censored"]]
                    values.append(np.median(cell) if cell else np.nan)
                ax.plot(x, values, "o-", color=DIM_COLORS[dim], ms=3,
                        label=f"{dim}D")
            ax.set_title(f"{interaction.value}: "
                         + ("T2, computational basis" if c == 0
                            else "T2^E, energy basis"), fontsize=10)
            ax.set_ylabel("median decoherence time")
    for ax in axes[1]:
        ax.set_xlabel(r"$k_F d/\pi$")
    axes[0, 0].legend(fontsize=8)
    fig.suptitle(f"kondo simulator: rho J = {RHO_J}, kT = {kT}, "
                 "uncensored trajectories only", fontsize=10)
    _save(fig, "fig_demo_t2.png")
    return []


# --- main ----------------------------------------------------------------------

_RUNNERS: dict[str, Callable[[bool], list[dict]]] = {
    "dimer": part_dimer, "swave": part_swave, "ising": part_ising,
    "surf": part_surf, "demo": part_demo,
}


def add_flags(parser: argparse.ArgumentParser) -> None:
    """Adds the choice of part to the common flags."""
    parser.add_argument("--part", choices=["all", *PARTS], default="all")


def main() -> None:
    """Runs the reproduction parts and writes deviations.csv.

    Writes experiments/results/r28/*.csv, the PNG figures and
    deviations.csv (only for the parts that were run).
    """
    args = common.parse_args(__doc__, extra=add_flags, seeds=None, out=False)
    parts = PARTS if args.part == "all" else (args.part,)
    records = []
    for part in parts:
        print(f"R28 | part {part}", flush=True)
        records += _RUNNERS[part](args.fast)
    if records:
        suffix = f"_{args.part}" if args.part != "all" else ""
        name = f"deviations{suffix}{'_fast' if args.fast else ''}.csv"
        common.write_results(
            records, OUT_DIR / name,
            config="max deviation of the generic code from the closed forms")
        for record in records:
            print(f"{record['figure']:<18} {record['quantity']:<48} "
                  f"{record['max_abs_deviation']:.3e}", flush=True)


if __name__ == "__main__":
    main()
