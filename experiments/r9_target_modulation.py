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

Writes ONLY to experiments/results/r9_target_modulation[_<tag>][_fast].csv.
"""
from __future__ import annotations

import itertools
from collections.abc import Sequence

import numpy as np
import qutip as qt

from experiments import common
from src.data.scenarios import bernstein_coeffs
from src.evaluation import stats
from src.physics.lindblad_systems import TwoQubitSystem

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
    step = (rel[i - 1] - THR) / (rel[i - 1] - rel[i] + 1e-12)
    return float(TIMES[i - 1] + step * DT)


def _unitary_min(H: np.ndarray, psi0: np.ndarray) -> float:
    evals, evecs = np.linalg.eigh(H)
    c0 = evecs.conj().T @ psi0
    vals = np.empty_like(TIMES)
    for k, t in enumerate(TIMES):
        a = np.abs(evecs @ (np.exp(-1j * evals * t) * c0))
        vals[k] = a.sum() ** 2 - (a ** 2).sum()
    return float((vals / vals[0]).min())


def _coef(system: TwoQubitSystem):
    def sqrt_gamma(t, args):
        """Returns sqrt(gamma(t)), or 0 where the rate is not positive.

        Args:
            t: Time.
            args: Unused; required by the QuTiP coefficient signature.
        """
        rate = system.get_gamma(t)
        return np.sqrt(rate) if rate > 0 else 0.0
    return sqrt_gamma


def part1(n_states: int, seed: int) -> list[dict]:
    """Measures unitary-only coherence dips from Haar-random states.

    Args:
        n_states: Number of Haar-random initial states per (model, J).
        seed: Seed of the state generator.

    Returns:
        One row per model and J with the share of states whose coherence
        crosses the threshold and the median modulation depth.
    """
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
    """Measures T2 ambiguity on dissipative Bernstein-gamma trajectories.

    Args:
        n_traj: Number of trajectories per (model, J).
        seed: Seed of the generator of rates and initial states.

    Returns:
        One row per model and J with the non-monotone share, the mean
        crossing gap, the correlation with the energy-basis T2 and T2
        statistics.
    """
    rng = np.random.default_rng(seed)
    rows = []
    for model in MODELS:
        for J in J_GRID:
            shapes = itertools.cycle(SHAPES)
            t2, t2e, gaps, nonmono, censored = [], [], [], 0, 0
            for _ in range(n_traj):
                coeffs = np.array(bernstein_coeffs(
                    next(shapes), rng, gamma_max=0.8))
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
                corr = stats.pearson(b[ok], a[ok])
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
            row = rows[-1]
            print(f"R9 | {model} J={J}: "
                  f"nonmono={row['frac_nonmonotone']:.2f} "
                  f"gap={row['mean_gap']:.2f} corr={corr:.3f}", flush=True)
    return rows


def _add_flags(parser) -> None:
    parser.add_argument("--n-states", type=int, default=400)
    parser.add_argument("--n-traj", type=int, default=150)
    parser.add_argument("--seed", type=int, default=0)


def main(argv: Sequence[str] | None = None) -> None:
    """Runs both parts and writes the CSV."""
    args = common.parse_args(
        __doc__.split("\n")[0], extra=_add_flags, argv=argv, seeds=None)
    n_states, n_traj = (20, 6) if args.fast else (args.n_states, args.n_traj)
    rows = part1(n_states, args.seed) + part2(n_traj, args.seed + 1)
    columns = sorted(
        {k for r in rows for k in r},
        key=lambda k: (k not in ("part", "model", "J", "n"), k))
    path = common.results_path("r9_target_modulation", args)
    common.write_results(
        rows, path, columns=columns, seeds=[args.seed, args.seed + 1],
        config={"n_states": n_states, "n_traj": n_traj, "J_grid": J_GRID})
    print(f"R9 | wrote {path}")


if __name__ == "__main__":
    main()
