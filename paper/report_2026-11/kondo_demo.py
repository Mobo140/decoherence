"""Two-qubit demonstration of the Kondo bath through `src.blocks`.

Run from the repository root:

    .venv/bin/python paper/report_2026-11/kondo_demo.py [--n 30] [--points 9]

For XXZ and TFIM (J = 0.8), d = 1, 2, 3, and k_F d / pi on a grid in [0, 2]:
n Haar-random pure initial states, the Kondo channels with rho J = 0.1,
kT = 0.2, k_F = 1, positions 0 and k_F d, t_max = 1200, dt = 1. The target
is the 1/e crossing of the l1 coherence in the energy basis (T2^E); T2 in
the computational basis is computed from the same states. A trajectory that
does not cross is censored at t_max; the median is taken over all
trajectories with censored ones counted at t_max, which is exact while fewer
than half are censored. The 95% interval is a bootstrap of the median
(2000 resamples, fixed seed).

Checks of the parameter choice, for every (system, d, k_F d):
  Born--Markov and secular: the largest transition rate and the largest
  decay rate against the smallest nonzero distance between distinct Bohr
  frequencies; rho J; the rates against kT.
Check of `channels.bernstein_profile` on the Kondo channels.

Writes into paper/report_2026-11/:
  figures/kondo_demo.png, table_kondo_demo.tex, table_kondo_validity.tex,
  numbers_kondo_demo.json
The states are cached in data/states (untracked).
"""
from __future__ import annotations

import argparse
import json
import math
import multiprocessing
import pathlib
import sys
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))

from src.blocks import channels, evolve, initial, system, target  # noqa: E402
from src.blocks.recipe import Recipe, block  # noqa: E402
from src.blocks.runner import run  # noqa: E402
from src.physics import kondo_bath  # noqa: E402

FIG = HERE / "figures"
CACHE = ROOT / "data" / "states"
RHO_J, KT, K_F, J = 0.1, 0.2, 1.0, 0.8
T_MAX, DT = 1200.0, 1.0
THRESHOLD = 1 / math.e
SEED = 2026
BOOT, BOOT_SEED = 2000, 7
SYSTEMS = ("XXZ", "TFIM")
DIMS = (1, 2, 3)
BOHR_TOL = 1e-9


def spacing_of(x_over_pi: float) -> float:
    """Factor on the site positions 0, 1 so that the distance is k_F d."""
    return math.pi * x_over_pi / K_F


def recipe_for(interaction: str, dim: int, x_over_pi: float) -> Recipe:
    return Recipe(
        system=block("system.two_qubit", J=J, interaction=interaction),
        channels=(block("channels.kondo_bath", rho_J=RHO_J, kT=KT, k_F=K_F,
                        dim=dim, spacing=spacing_of(x_over_pi)),),
        initial=block("initial.random_pure"),
        observe=(block("observe.purity"),),
        target=block("target.energy_basis_crossing", threshold=THRESHOLD),
        t_max=T_MAX, dt=DT, seeds="spawn")


def bohr_gap(energies: np.ndarray) -> float:
    """Smallest nonzero distance between distinct Bohr frequencies."""
    bohr = np.sort((energies[:, None] - energies[None, :]).ravel())
    distinct = [bohr[0]]
    for value in bohr[1:]:
        if value - distinct[-1] > BOHR_TOL:
            distinct.append(value)
    return float(np.diff(distinct).min())


def validity(interaction: str, dim: int, x_over_pi: float) -> dict:
    """Rates and gaps of the secular (Davies) treatment for one point."""
    sys_ = system.two_qubit(J, interaction)
    rates = kondo_bath.kondo_rates(
        sys_.H, sys_.spin_ops, sys_.positions * spacing_of(x_over_pi),
        rho_J=RHO_J, kT=KT, k_F=K_F, dim=dim)
    off_diag = np.abs(rates.energies[:, None]
                      - rates.energies[None, :]) > BOHR_TOL
    out = rates.rates.sum(axis=1)
    deph = rates.dephasing[off_diag]
    gap = bohr_gap(rates.energies)
    return {"gamma_transition_max": float(rates.rates.max()),
            "gamma_out_max": float(out.max()),
            "gamma_dephasing_max": float(deph.max()),
            "gamma_decay_max": float(max(out.max(), deph.max())),
            "bohr_gap_min": gap,
            "ratio_transition": float(rates.rates.max() / gap),
            "ratio_decay": float(max(out.max(), deph.max()) / gap),
            "gamma_decay_over_kT": float(max(out.max(), deph.max()) / KT)}


def bootstrap_median(values: np.ndarray, n_boot: int = BOOT) -> tuple:
    rng = np.random.default_rng(BOOT_SEED)
    draws = rng.choice(values, size=(n_boot, values.size), replace=True)
    med = np.median(draws, axis=1)
    return float(np.percentile(med, 2.5)), float(np.percentile(med, 97.5))


def cell(task: tuple) -> dict:
    """Runs n trajectories for one (system, d, k_F d / pi)."""
    interaction, dim, x, n, cache = task
    recipe = recipe_for(interaction, dim, x)
    sys_ = system.two_qubit(J, interaction)
    runs = run(recipe, n, seed=SEED, cache_dir=cache, keep_states=True)
    t2e = np.array([r.target.time for r in runs])
    t2e_cens = np.array([r.target.censored for r in runs])
    t2 = [target.coherence_crossing(r.states, sys_, THRESHOLD) for r in runs]
    t2c = np.array([t.time for t in t2])
    t2c_cens = np.array([t.censored for t in t2])
    out = {"system": interaction, "dim": dim, "x_over_pi": x, "n": n,
           "validity": validity(interaction, dim, x)}
    for name, values, cens in (("t2e", t2e, t2e_cens),
                               ("t2", t2c, t2c_cens)):
        lo, hi = bootstrap_median(values)
        median = float(np.median(values))
        out[name] = {"median": median, "ci_low": lo, "ci_high": hi,
                     "censored": int(cens.sum()),
                     "median_is_censored": bool(median >= T_MAX),
                     "ci_reaches_t_max": bool(hi >= T_MAX),
                     "values": values.tolist()}
    return out


def bernstein_check() -> dict:
    """Constant Bernstein coefficients c scale the Kondo rates by c.

    (1) The QuTiP coefficients of the channels after `bernstein_profile`
    with coefficients (c, c, c) equal sqrt(c) at several times.
    (2) One trajectory: T2^E with the profile, with all rates multiplied by
    c in the channel list, and the unmodulated T2^E divided by c.
    """
    interaction, dim, x = "XXZ", 2, 0.5
    sys_ = system.two_qubit(J, interaction)
    base = channels.kondo_bath(sys_, RHO_J, KT, K_F, dim,
                               spacing=spacing_of(x))
    times = np.linspace(0.0, T_MAX, int(round(T_MAX / DT)) + 1)
    rho0 = initial.random_pure(sys_, np.random.default_rng(1))

    def t2e(chs) -> float:
        states = evolve.master_equation(sys_, chs, rho0, times)
        return target.energy_basis_crossing(states, sys_, THRESHOLD)

    ref = t2e(base)
    rows = []
    for c in (0.5, 2.0):
        profile = channels.bernstein_profile(base, (c, c, c), T_MAX)
        coeff_dev = 0.0
        for item in channels.to_mesolve(profile):
            for t in (0.0, 100.0, 600.0, T_MAX):
                coeff_dev = max(coeff_dev,
                                abs(item[1](t, None) - math.sqrt(c)))
        scaled = [channels.Channel(ch.operator, ch.rate * c) for ch in base]
        with_profile, with_rates = t2e(profile), t2e(scaled)
        rows.append({
            "c": c, "n_channels": len(base),
            "max_abs_dev_coefficient_from_sqrt_c": coeff_dev,
            "t2e_unmodulated": ref.time,
            "t2e_profile": with_profile.time,
            "t2e_rates_times_c": with_rates.time,
            "t2e_unmodulated_over_c": ref.time / c,
            "censored": [ref.censored, with_profile.censored,
                         with_rates.censored],
            "relative_difference_profile_vs_rates_times_c": abs(
                with_profile.time - with_rates.time) / with_rates.time,
            "relative_difference_profile_vs_unmodulated_over_c": abs(
                with_profile.time - ref.time / c) / (ref.time / c)})
    return {"system": interaction, "dim": dim, "k_F d / pi": x,
            "dt": DT, "rows": rows}


def plot(cells: list[dict], xs: list[float]) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0), sharey=False)
    colors = {1: "black", 2: "tab:red", 3: "tab:blue"}
    for ax, interaction in zip(axes, SYSTEMS):
        for dim in DIMS:
            part = sorted((c for c in cells
                           if c["system"] == interaction and c["dim"] == dim),
                          key=lambda c: c["x_over_pi"])
            x = [c["x_over_pi"] for c in part]
            med = np.array([c["t2e"]["median"] for c in part])
            lo = np.array([c["t2e"]["ci_low"] for c in part])
            hi = np.array([c["t2e"]["ci_high"] for c in part])
            ax.fill_between(x, lo, hi, color=colors[dim], alpha=0.2, lw=0)
            ax.plot(x, med, "o-", ms=3, lw=1, color=colors[dim],
                    label=f"{dim}D")
        ax.set_title(interaction)
        ax.set_xlabel(r"$k_Fd/\pi$")
        ax.set_ylabel(r"$T_2^E$, медиана")
        ax.grid(alpha=0.3)
    axes[0].legend()
    fig.tight_layout()
    fig.savefig(FIG / "kondo_demo.png", dpi=150)
    plt.close(fig)


def write_table(cells: list[dict], xs: list[float]) -> None:
    """T2^E: median [95% interval] for every point; no trajectory censored."""
    def get(interaction, dim, x):
        return next(c["t2e"] for c in cells if c["system"] == interaction
                    and c["dim"] == dim and abs(c["x_over_pi"] - x) < 1e-12)

    lines = [r"\begin{tabular}{lccc}", r"\toprule",
             r"$k_Fd/\pi$ & $d=1$ & $d=2$ & $d=3$ \\"]
    for interaction in SYSTEMS:
        lines += [r"\midrule",
                  rf"\multicolumn{{4}}{{l}}{{\emph{{{interaction}}}}} \\"]
        for x in xs:
            row = []
            for dim in DIMS:
                r = get(interaction, dim, x)
                star = "$^{*}$" if r["censored"] else ""
                row.append(f"{r['median']:.0f} [{r['ci_low']:.0f}, "
                           f"{r['ci_high']:.0f}]{star}")
            lines.append(f"{x:g} & " + " & ".join(row) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (HERE / "table_kondo_demo.tex").write_text("\n".join(lines) + "\n")


def summary(cells: list[dict]) -> dict:
    """Numbers quoted in the report, from the cells."""
    out = {}
    for interaction in SYSTEMS:
        part = [c for c in cells if c["system"] == interaction]
        item = {}
        for name in ("t2e", "t2"):
            med = np.array([c[name]["median"] for c in part])
            item[name] = {
                "median_min": float(med.min()), "median_max": float(med.max()),
                "ratio_max_to_min": float(med.max() / med.min()),
                "censored_total": int(sum(c[name]["censored"] for c in part)),
                "trajectories_total": int(sum(c["n"] for c in part)),
                "cells_with_median_censored": int(sum(
                    c[name]["median_is_censored"] for c in part)),
                "cells_with_any_censored": int(sum(
                    c[name]["censored"] > 0 for c in part)),
                "cells": len(part)}
        per_dim = {}
        for dim in DIMS:
            med = np.array([c["t2e"]["median"] for c in part
                            if c["dim"] == dim])
            per_dim[str(dim)] = {"min": float(med.min()),
                                 "max": float(med.max()),
                                 "ratio": float(med.max() / med.min())}
        item["t2e_by_dim"] = per_dim
        width = [(c["t2e"]["ci_high"] - c["t2e"]["ci_low"])
                 / c["t2e"]["median"] for c in part]
        item["t2e_ci_width_over_median"] = {"min": float(min(width)),
                                            "max": float(max(width))}
        v = {k: max(c["validity"][k] for c in part)
             for k in ("ratio_transition", "ratio_decay",
                       "gamma_decay_over_kT", "gamma_transition_max",
                       "gamma_decay_max")}
        worst = max(part, key=lambda c: c["validity"]["ratio_decay"])
        v["bohr_gap_min"] = min(c["validity"]["bohr_gap_min"] for c in part)
        v["worst_cell_ratio_decay"] = {
            "dim": worst["dim"], "x_over_pi": worst["x_over_pi"]}
        product = np.array([c["validity"]["gamma_decay_max"]
                            * c["t2e"]["median"] for c in part])
        v["decay_rate_times_t2e_min"] = float(product.min())
        v["decay_rate_times_t2e_max"] = float(product.max())
        v["kT_times_t2e_min"] = float(KT * min(
            c["t2e"]["median"] for c in part))
        item["validity"] = v
        out[interaction] = item
    return out


def write_validity_table(summ: dict) -> None:
    """Born--Markov and secular checks, the worst case over the grid."""
    lines = [r"\begin{tabular}{lcccccc}", r"\toprule",
             r"& $\Delta_{\min}$ & $\Gamma_{\rm tr}^{\max}$ & "
             r"$\Gamma_{\rm dec}^{\max}$ & $\Gamma_{\rm dec}^{\max}/\Delta_{\min}$ & "
             r"$\Gamma_{\rm dec}^{\max}/\kT$ & $\Gamma_{\rm dec}^{\max}T_2^E$ \\",
             r"\midrule"]
    for name in SYSTEMS:
        v = summ[name]["validity"]
        lines.append(
            f"{name} & {v['bohr_gap_min']:.3f} & "
            f"{v['gamma_transition_max']:.4f} & {v['gamma_decay_max']:.4f} & "
            f"{v['ratio_decay']:.4f} & {v['gamma_decay_over_kT']:.3f} & "
            f"{v['decay_rate_times_t2e_min']:.2f}--"
            f"{v['decay_rate_times_t2e_max']:.2f} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (HERE / "table_kondo_validity.tex").write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=30)
    parser.add_argument("--points", type=int, default=9,
                        help="grid points in [0, 2]")
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--limit", type=int, default=None,
                        help="only the first LIMIT cells (timing)")
    parser.add_argument("--cache", type=pathlib.Path, default=CACHE,
                        help="states cache (default data/states)")
    parser.add_argument("--no-write", action="store_true")
    args = parser.parse_args()
    FIG.mkdir(exist_ok=True)
    xs = [float(v) for v in np.linspace(0.0, 2.0, args.points)]
    cache_was_empty = not any(args.cache.glob("*.npz"))
    tasks = [(s, d, x, args.n, args.cache)
             for s in SYSTEMS for d in DIMS for x in xs]
    if args.limit:
        tasks = tasks[:args.limit]
    start = time.time()
    with multiprocessing.get_context("spawn").Pool(args.workers) as pool:
        cells = []
        for k, c in enumerate(pool.imap_unordered(cell, tasks), 1):
            cells.append(c)
            print(f"{k}/{len(tasks)} {c['system']} d={c['dim']} "
                  f"x={c['x_over_pi']:g} T2E={c['t2e']['median']:.0f} "
                  f"({time.time() - start:.0f} s)", flush=True)
    runtime = time.time() - start
    if args.no_write:
        return
    check = bernstein_check()
    cells.sort(key=lambda c: (c["system"], c["dim"], c["x_over_pi"]))
    plot(cells, xs)
    write_table(cells, xs)
    write_validity_table(summary(cells))
    numbers = {
        "settings": {"rho_J": RHO_J, "kT": KT, "k_F": K_F, "J": J,
                     "t_max": T_MAX, "dt": DT, "threshold": THRESHOLD,
                     "n_initial_states": args.n, "grid": xs,
                     "seed": SEED, "bootstrap": BOOT,
                     "bootstrap_seed": BOOT_SEED,
                     "workers": args.workers,
                     "runtime_seconds": runtime,
                     "cache_was_empty": cache_was_empty},
        "summary": summary(cells), "bernstein_check": check,
        "cells": cells}
    (HERE / "numbers_kondo_demo.json").write_text(
        json.dumps(numbers, indent=1) + "\n")
    print("done", round(runtime), "s")


if __name__ == "__main__":
    main()
