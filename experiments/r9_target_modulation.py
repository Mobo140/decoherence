"""R9 — is the TFIM predictability loss a property of the target?

Background
----------
Paper 2 found that TFIM predictability falls as the coupling J grows and
then flattens, and that two mechanisms offered for it fail their direct
tests (R6). Both mechanisms were about the *dynamics*. This probe looks at
the *target* instead.

T2 is the first time the l1-coherence C(t) = sum_{i!=j} |rho_ij| falls
below C(0)/e, and C is measured in the computational basis. The TFIM
Hamiltonian H = J sx1 sx2 + h(sz1 + sz2) is not diagonal in that basis, so
unitary evolution alone modulates C(t). In the P = +1 sector the block is
[[2h, J], [J, -2h]], whose eigenbasis is rotated from the computational
one by tan(2 theta) = J / 2h: the rotation, and with it the modulation
depth, grows with J and saturates. When the modulation is deep, the first
1/e crossing depends on the oscillation phase at the moment the decaying
envelope reaches the threshold, not only on the decay -- and C(t) can climb
back above the threshold afterwards.

The XXZ Hamiltonian conserves magnetisation and is diagonal on |00>, |11>;
only the |01>, |10> block mixes, maximally and for every J, so its
modulation depth should not depend on J.

What this probe measures (simulation only, no training)
-------------------------------------------------------
Part 1, gamma = 0: pure unitary evolution from Haar-random states.
  - frac_unitary_cross: share of states whose C(t) dips below C(0)/e with
    no dissipation at all;
  - median_depth: median of 1 - min_t C(t)/C(0).
Part 2, Bernstein gamma(t) as in the experiments (gamma_max = 0.8, six
shapes): per J and model,
  - frac_nonmonotone: share of trajectories where C(t) returns above the
    threshold after the first crossing;
  - mean_gap: mean time between the first crossing and the last moment
    above the threshold (the ambiguity of T2 as a first-passage time);
  - corr_energy / resid_sd: correlation of T2 with T2_E, the same 1/e
    crossing computed for the l1-coherence in the eigenbasis of H, which
    unitary evolution leaves unchanged (only phases rotate there), and the
    residual s.d. of a linear fit T2 ~ T2_E.

Usage
-----
    python -m experiments.r9_target_modulation          # ~5 min on a laptop
    python -m experiments.r9_target_modulation --fast   # smoke test

Writes ONLY to experiments/results/r9_target_modulation.csv.
"""
from __future__ import annotations

import argparse
import csv
import itertools
from pathlib import Path

import numpy as np
import qutip as qt

from experiments._config import _bernstein_coeffs
from src.physics.lindblad_systems import TwoQubitSystem

RESULTS = Path(__file__).parent / "results" / "r9_target_modulation.csv"

T_MAX, DT = 20.0, 0.1
TIMES = np.linspace(0.0, T_MAX, int(round(T_MAX / DT)) + 1)
J_GRID = [0.2, 0.5, 0.8, 1.0, 1.2, 1.5, 2.0]
SHAPES = ["random", "monotone_increasing", "monotone_decreasing",
          "peak", "step_up", "step_down"]
THR = np.exp(-1.0)
MODELS = ["XXZ", "TFIM"]


def _haar(rng: np.random.Generator) -> np.ndarray:
    v = rng.standard_normal(4) + 1j * rng.standard_normal(4)
    return v / np.linalg.norm(v)


def _l1(rho: np.ndarray) -> float:
    return float(np.sum(np.abs(rho)) - np.sum(np.abs(np.diag(rho))))


def _first_crossing(rel: np.ndarray) -> float | None:
    below = np.where(rel < THR)[0]
    if below.size == 0:
        return None
    i = below[0]
    if i == 0:
        return float(TIMES[0])
    return float(TIMES[i - 1] + (rel[i - 1] - THR) / (rel[i - 1] - rel[i] + 1e-12) * DT)


def _unitary_min(H: np.ndarray, psi0: np.ndarray) -> float:
    evals, evecs = np.linalg.eigh(H)
    c0 = evecs.conj().T @ psi0
    vals = np.empty_like(TIMES)
    for k, t in enumerate(TIMES):
        a = np.abs(evecs @ (np.exp(-1j * evals * t) * c0))
        vals[k] = a.sum() ** 2 - (a ** 2).sum()
    return float((vals / vals[0]).min())


def _coef(system: TwoQubitSystem):
    def g(t, args):
        v = system.get_gamma(t)
        return np.sqrt(v) if v > 0 else 0.0
    return g


def part1(n_states: int, seed: int) -> list[dict]:
    rng = np.random.default_rng(seed)
    states = [_haar(rng) for _ in range(n_states)]
    rows = []
    for model in MODELS:
        for J in J_GRID:
            H = TwoQubitSystem(J=J, interaction_type=model).H.full()
            mins = np.array([_unitary_min(H, s) for s in states])
            rows.append({
                "part": 1, "model": model, "J": J, "n": n_states,
                "frac_unitary_cross": float(np.mean(mins < THR)),
                "median_depth": float(np.median(1.0 - mins)),
            })
    return rows


def part2(n_traj: int, seed: int) -> list[dict]:
    rng = np.random.default_rng(seed)
    rows = []
    for model in MODELS:
        for J in J_GRID:
            shapes = itertools.cycle(SHAPES)
            t2, t2e, gaps, nonmono, censored = [], [], [], 0, 0
            for _ in range(n_traj):
                coeffs = np.array(_bernstein_coeffs(next(shapes), rng, gamma_max=0.8))
                system = TwoQubitSystem(J=J, interaction_type=model,
                                        gamma_coeffs=coeffs, gamma_t_max=T_MAX)
                psi = qt.Qobj(_haar(rng).reshape(-1, 1), dims=[[2, 2], [1, 1]])
                out = qt.mesolve(system.H, psi * psi.dag(), TIMES,
                                 [[L, _coef(system)] for L in system.L_list],
                                 e_ops=[], options={"nsteps": 10000})
                rhos = [s.full() for s in out.states]
                U = np.linalg.eigh(system.H.full())[1]
                c = np.array([_l1(r) for r in rhos])
                ce = np.array([_l1(U.conj().T @ r @ U) for r in rhos])
                rel = c / c[0]
                tf = _first_crossing(rel)
                if tf is None:
                    censored += 1
                    continue
                above = np.where(rel >= THR)[0]
                tl = float(TIMES[above[-1]]) if above.size else float(TIMES[0])
                te = _first_crossing(ce / ce[0])
                t2.append(tf)
                t2e.append(te if te is not None else np.nan)
                gaps.append(max(0.0, tl - tf))
                nonmono += int(tl > tf + 1e-9)
            a, b = np.array(t2), np.array(t2e)
            ok = np.isfinite(b)
            if ok.sum() > 5:
                k, c0 = np.polyfit(b[ok], a[ok], 1)
                corr = float(np.corrcoef(b[ok], a[ok])[0, 1])
                resid = float(np.std(a[ok] - (k * b[ok] + c0)))
            else:
                corr = resid = float("nan")
            n = len(t2)
            rows.append({
                "part": 2, "model": model, "J": J, "n": n, "censored": censored,
                "frac_nonmonotone": nonmono / n if n else float("nan"),
                "mean_gap": float(np.mean(gaps)) if n else float("nan"),
                "corr_energy": corr, "resid_sd": resid,
                "mean_T2": float(np.mean(a)) if n else float("nan"),
                "std_T2": float(np.std(a)) if n else float("nan"),
            })
            print(f"R9 | {model} J={J}: nonmono={rows[-1]['frac_nonmonotone']:.2f} "
                  f"gap={rows[-1]['mean_gap']:.2f} corr={corr:.3f}", flush=True)
    return rows


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--n-states", type=int, default=400)
    p.add_argument("--n-traj", type=int, default=150)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--fast", action="store_true")
    a = p.parse_args()
    if a.fast:
        a.n_states, a.n_traj = 20, 6
    rows = part1(a.n_states, a.seed) + part2(a.n_traj, a.seed + 1)
    keys = sorted({k for r in rows for k in r}, key=lambda k: (k not in ("part", "model", "J", "n"), k))
    RESULTS.parent.mkdir(parents=True, exist_ok=True)
    with RESULTS.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    print(f"R9 | wrote {RESULTS}")


if __name__ == "__main__":
    main()
