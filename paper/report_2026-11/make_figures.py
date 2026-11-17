"""Figures, tables and numbers for the report of 2026-11.

Run from the repository root:

    .venv/bin/python paper/report_2026-11/make_figures.py

Writes into paper/report_2026-11/:
  figures/fig_shots_series.png    <sigma_x>, l1 coherence: exact and finite shots
  figures/fig_records.png         Zeno check and amplitude damping on records
  table_shots_std.tex             empirical std of the estimate vs the binomial
  table_shots_bias.tex            bias of the purity and l1 coherence estimates
  table_records.tex               excited fraction on records vs exp(-gamma t)
  table_r28_dev.tex               maximum deviations from the R28 CSV files
  numbers.json                    every number the text quotes from here

The R28 figures are not made here; they are copied from
experiments/results/r28/ (see the README of that folder).
"""
from __future__ import annotations

import json
import math
import pathlib
import sys
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))

from src.blocks import observe  # noqa: E402
from src.blocks.evolve import States  # noqa: E402
from src.blocks.recipe import Recipe, block  # noqa: E402
from src.blocks.runner import run  # noqa: E402
from src.data.scenarios import build_configs  # noqa: E402

FIG = HERE / "figures"
R28 = ROOT / "experiments" / "results" / "r28"
DPI = 150
plt.rcParams.update({
    "font.size": 9, "axes.titlesize": 9, "axes.labelsize": 9,
    "legend.fontsize": 8, "figure.dpi": DPI, "savefig.dpi": DPI,
    "axes.grid": True, "grid.alpha": 0.3,
})
NUMBERS: dict = {}
T0 = time.time()


def save(fig, name: str) -> None:
    fig.tight_layout()
    fig.savefig(FIG / name)
    plt.close(fig)


def sci(x: float) -> str:
    """LaTeX scientific notation with two significant digits."""
    mantissa, exponent = f"{x:.1e}".split("e")
    return rf"${mantissa}\cdot 10^{{{int(exponent)}}}$"


# --- 1. finite shots, scenario B ------------------------------------------
SEED_B = 5            # the seed of notebooks/constructor.ipynb, fixed in advance
N_SHOTS = (100, 1000)
REPEATS = 2000
CHECK_TIMES = (0.0, 2.0, 5.0, 10.0, 15.0, 20.0)


def scenario_b_recipe(observation) -> Recipe:
    """Single qubit of scenario B (sigma_minus, Bernstein gamma(t))."""
    config = build_configs(n_per_scenario=1, seed=0)["B"][0]
    dissipator = config.dissipator
    return Recipe(
        system=block("system.single_qubit", omega=config.omega),
        channels=(
            block("channels.jump", op_type=dissipator.operator_type.value,
                  gamma=1.0),
            block("channels.bernstein_profile",
                  coeffs=dissipator.gamma_coeffs)),
        initial=block("initial.random_pure"),
        observe=observation,
        target=None,
        t_max=config.t_max, dt=config.dt)


def finite_shots_part() -> None:
    config = build_configs(n_per_scenario=1, seed=0)["B"][0]
    exact_blocks = (block("observe.pauli_means"), block("observe.purity"),
                    block("observe.coherence_l1"))
    exact = run(scenario_b_recipe(exact_blocks), 1, seed=SEED_B,
                keep_states=True)[0]
    series = exact.observations.series
    times = exact.observations.times
    NUMBERS["scenario_b"] = {
        "omega": config.omega, "gamma_coeffs": list(
            config.dissipator.gamma_coeffs),
        "t_max": config.t_max, "dt": config.dt, "seed": SEED_B,
        "sigma_initial": [float(series[k][0]) for k in
                          ("sigma_x", "sigma_y", "sigma_z")],
        "purity_initial": float(series["purity"][0]),
    }
    shots = {}
    for n in N_SHOTS:
        shots[n] = run(
            scenario_b_recipe((block("observe.finite_shots", n_shots=n,
                                     settings="tomography"),)),
            1, seed=SEED_B)[0].observations
    NUMBERS["scenario_b"]["std_over_grid"] = {
        str(n): {key: float(np.std(shots[n].series[key] - series[key]))
                 for key in ("sigma_x", "sigma_y", "sigma_z", "purity",
                             "coherence_l1")}
        for n in N_SHOTS}

    fig, axes = plt.subplots(1, 2, figsize=(6.4, 2.5))
    for ax, key, label in ((axes[0], "sigma_x", r"$\langle\sigma_x\rangle$"),
                           (axes[1], "coherence_l1", r"$C_{l_1}$")):
        ax.plot(times, series[key], "k-", lw=1.2, label="точное")
        ax.plot(times, shots[100].series[key], ".", ms=3, color="tab:red",
                alpha=0.8, label=r"$N=100$")
        ax.plot(times, shots[1000].series[key], ".", ms=3, color="tab:blue",
                alpha=0.8, label=r"$N=1000$")
        ax.set_xlabel("$t$")
        ax.set_ylabel(label)
    axes[0].legend(loc="upper right")
    save(fig, "fig_shots_series.png")

    # Repeated draws on a few times, same exact states.
    index = [int(round(t / config.dt)) for t in CHECK_TIMES]
    sub = States(times=times[index],
                 rho=[exact.states.rho[i] for i in index])
    exact_pauli = {a: np.array([series[f"sigma_{a}"][i] for i in index])
                   for a in "xyz"}
    exact_purity = np.array([series["purity"][i] for i in index])
    exact_coh = np.array([series["coherence_l1"][i] for i in index])
    stats = {}
    rng = np.random.default_rng(np.random.SeedSequence(2026))
    for n in N_SHOTS:
        draws = {k: np.empty((REPEATS, len(index))) for k in (
            "sigma_x", "sigma_y", "sigma_z", "purity", "coherence_l1")}
        for r in range(REPEATS):
            obs = observe.finite_shots(sub, n, "tomography", rng)
            for k in draws:
                draws[k][r] = obs.series[k]
        stats[n] = draws
    worst, n_near_pure = 0.0, 0
    result = {}
    for n in N_SHOTS:
        d = stats[n]
        for a in "xyz":
            emp = d[f"sigma_{a}"].std(axis=0, ddof=1)
            pred = np.sqrt((1 - exact_pauli[a] ** 2) / n)
            interior = np.abs(exact_pauli[a]) < 0.99
            worst = max(worst, float(np.max(
                np.abs(emp / pred - 1)[interior])))
            n_near_pure += int((~interior).sum())
            result[f"{n}_{a}"] = {"empirical": emp.tolist(),
                                  "binomial": pred.tolist()}
        r2 = sum(exact_pauli[a] ** 2 for a in "xyz")
        result[f"{n}_purity_bias"] = {
            "empirical": (d["purity"].mean(axis=0) - exact_purity).tolist(),
            "standard_error": (d["purity"].std(axis=0, ddof=1)
                               / math.sqrt(REPEATS)).tolist(),
            "predicted": ((3 - r2) / (2 * n)).tolist()}
        bias = result[f"{n}_purity_bias"]
        bias["z"] = ((np.array(bias["empirical"]) - np.array(bias["predicted"]))
                     / np.array(bias["standard_error"])).tolist()
        result[f"{n}_coherence_bias"] = {
            "empirical": (d["coherence_l1"].mean(axis=0) - exact_coh).tolist(),
            "standard_error": (d["coherence_l1"].std(axis=0, ddof=1)
                               / math.sqrt(REPEATS)).tolist()}
    NUMBERS["shots_repeats"] = {
        "repeats": REPEATS, "times": list(CHECK_TIMES),
        "sigma_exact": {a: exact_pauli[a].tolist() for a in "xyz"},
        "purity_exact": exact_purity.tolist(),
        "coherence_exact": exact_coh.tolist(),
        "max_relative_deviation_std_axes_with_abs_m_below_0.99": worst,
        "axis_time_pairs_with_abs_m_at_least_0.99_per_N": n_near_pure // 2,
        "relative_error_of_an_empirical_std": 1 / math.sqrt(2 * (REPEATS - 1)),
        **result}

    lines = [r"\begin{tabular}{rrrrrr}", r"\toprule",
             r"& & \multicolumn{2}{c}{$N=100$} & \multicolumn{2}{c}{$N=1000$}"
             r" \\", r"\cmidrule(lr){3-4}\cmidrule(lr){5-6}",
             r"$t$ & $\langle\sigma_x\rangle$ & эмпир. & $\sqrt{(1-m^2)/N}$"
             r" & эмпир. & $\sqrt{(1-m^2)/N}$ \\", r"\midrule"]
    for j, t in enumerate(CHECK_TIMES):
        lines.append(
            f"{t:g} & {exact_pauli['x'][j]:.3f} & "
            f"{result['100_x']['empirical'][j]:.4f} & "
            f"{result['100_x']['binomial'][j]:.4f} & "
            f"{result['1000_x']['empirical'][j]:.4f} & "
            f"{result['1000_x']['binomial'][j]:.4f} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (HERE / "table_shots_std.tex").write_text("\n".join(lines) + "\n")

    lines = [r"\begin{tabular}{rrrrrr}", r"\toprule",
             r"& \multicolumn{3}{c}{$N=100$} & \multicolumn{2}{c}{$N=1000$}"
             r" \\", r"\cmidrule(lr){2-4}\cmidrule(lr){5-6}",
             r"$t$ & $\Delta P$ & $(3-|\mathbf r|^2)/2N$ & $\Delta C_{l_1}$"
             r" & $\Delta P$ & $\Delta C_{l_1}$ \\", r"\midrule"]
    for j, t in enumerate(CHECK_TIMES):
        a, b = result["100_purity_bias"], result["100_coherence_bias"]
        c, e = result["1000_purity_bias"], result["1000_coherence_bias"]
        lines.append(
            f"{t:g} & {a['empirical'][j]:+.4f} & {a['predicted'][j]:.4f} & "
            f"{b['empirical'][j]:+.4f} & {c['empirical'][j]:+.5f} & "
            f"{e['empirical'][j]:+.5f} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (HERE / "table_shots_bias.tex").write_text("\n".join(lines) + "\n")


# --- 2. records of sequential measurements ---------------------------------
GAMMA = 0.3
RECORD_DT, RECORD_TMAX = 0.5, 6.0
ZENO_RECORDS = 1000
DAMPING_RECORDS = 4000


def record_recipe(system_block, gamma: float) -> Recipe:
    return Recipe(
        system=system_block,
        channels=(block("channels.jump", op_type="sigma_z", gamma=gamma),),
        initial=block("initial.random_pure"),
        observe=(block("observe.record_series"),),
        target=None, t_max=RECORD_TMAX, dt=RECORD_DT,
        evolve=block("evolve.sequential_record", basis="z"))


def records_part() -> None:
    # (a) Zeno: H = omega sigma_z and no dissipation; contrast with an H that
    # does not commute with sigma_z.
    diag = run(record_recipe(block("system.single_qubit", omega=1.0), 0.0),
               ZENO_RECORDS, seed=101)
    xxz = run(record_recipe(
        block("system.two_qubit", J=0.8, interaction="XXZ"), 0.0),
        ZENO_RECORDS, seed=102)
    out_diag = np.array([r.record.outcomes[:, 0] for r in diag])
    out_xxz = np.array([r.record.outcomes[:, 0] for r in xxz])
    times = diag[0].record.times
    flip_diag = (out_diag != out_diag[:, :1]).mean(axis=0)
    flip_xxz = (out_xxz != out_xxz[:, :1]).mean(axis=0)
    first_diag = float((out_diag[:, 0] == 0).mean())

    # (b) amplitude damping sigma_minus, H = omega sigma_z.
    damp_recipe = Recipe(
        system=block("system.single_qubit", omega=1.0),
        channels=(block("channels.jump", op_type="sigma_minus", gamma=GAMMA),),
        initial=block("initial.random_pure"),
        observe=(block("observe.record_series"),), target=None,
        t_max=RECORD_TMAX, dt=RECORD_DT,
        evolve=block("evolve.sequential_record", basis="z"))
    damp = run(damp_recipe, DAMPING_RECORDS, seed=103)
    out = np.array([r.record.outcomes[:, 0] for r in damp])
    excited = out == 0          # outcome 0 is sigma_z = +1, the state |0>
    first = excited[:, 0]
    n0 = int(first.sum())
    p_hat = excited[first].mean(axis=0)
    p_model = np.exp(-GAMMA * times)
    err = np.sqrt(p_model * (1 - p_model) / n0)
    z = (p_hat - p_model) / np.where(err > 0, err, np.nan)
    returned = int(excited[~first][:, 1:].any(axis=1).sum())
    NUMBERS["records"] = {
        "dt": RECORD_DT, "t_max": RECORD_TMAX, "gamma": GAMMA,
        "zeno": {
            "records": ZENO_RECORDS,
            "outcome_changes_diag_H": int((out_diag != out_diag[:, :1]).sum()),
            "outcomes_checked_diag_H": int(out_diag[:, 1:].size),
            "first_outcome_0_fraction": first_diag,
            "flip_fraction_xxz_by_time": flip_xxz.tolist(),
            "flip_fraction_xxz_mean_t_ge_1": float(flip_xxz[2:].mean()),
        },
        "damping": {
            "records": DAMPING_RECORDS, "first_outcome_0": n0,
            "first_outcome_0_fraction": n0 / DAMPING_RECORDS,
            "times": times.tolist(), "p_hat": p_hat.tolist(),
            "p_model": p_model.tolist(), "binomial_error": err.tolist(),
            "z": [None if math.isnan(v) else float(v) for v in z],
            "max_abs_z": float(np.nanmax(np.abs(z))),
            "returned_to_excited_after_first_outcome_1": returned,
            "records_with_first_outcome_1": DAMPING_RECORDS - n0,
        },
    }

    fig, axes = plt.subplots(1, 2, figsize=(6.4, 2.6))
    ax = axes[0]
    ax.plot(times, flip_diag, "o-", ms=3, color="k",
            label=r"$H=\omega\sigma_z$")
    ax.plot(times, flip_xxz, "s-", ms=3, color="tab:red",
            label=r"$H=J(XX{+}YY{+}ZZ)$")
    ax.set_xlabel("$t$")
    ax.set_ylabel("доля записей,\nгде исход $\\neq$ первого")
    ax.set_ylim(-0.02, 0.4)
    ax.legend(loc="center right")
    ax = axes[1]
    fine = np.linspace(0, RECORD_TMAX, 200)
    ax.plot(fine, np.exp(-GAMMA * fine), "k-", lw=1, label=r"$e^{-\gamma t}$")
    ax.errorbar(times, p_hat, yerr=np.sqrt(
        np.clip(p_hat * (1 - p_hat), 0, None) / n0), fmt="o", ms=3,
        color="tab:blue", capsize=2, label=f"записи, $n_0={n0}$")
    ax.set_xlabel("$t$")
    ax.set_ylabel("доля исходов\n«возбуждён»")
    ax.legend(loc="upper right")
    save(fig, "fig_records.png")

    lines = [r"\begin{tabular}{rrrrr}", r"\toprule",
             r"$t$ & $\hat p(t)$ & $e^{-\gamma t}$ & $\sigma_{\rm bin}$ & "
             r"$(\hat p-e^{-\gamma t})/\sigma_{\rm bin}$ \\", r"\midrule"]
    for k in range(0, len(times), 2):
        zz = "--" if math.isnan(z[k]) else f"{z[k]:+.2f}"
        lines.append(f"{times[k]:g} & {p_hat[k]:.3f} & {p_model[k]:.3f} & "
                     f"{err[k]:.3f} & {zz} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (HERE / "table_records.tex").write_text("\n".join(lines) + "\n")


# --- 3. maximum deviations from the R28 CSV files ----------------------------
def r28_part() -> None:
    dev = {name: pd.read_csv(R28 / f"deviations_{name}.csv")
           for name in ("dimer", "swave", "ising", "surf")}

    def worst(name: str, mask) -> float:
        frame = dev[name]
        selected = frame[mask(frame)]
        assert len(selected), name
        return float(selected.max_abs_deviation.max())

    q = lambda pattern: (lambda f: f.quantity.str.contains(pattern, regex=False))
    fig_is = lambda label: (lambda f: f.figure == label)
    spec = [
        ("PRB рис.~2(b): $1-K_d$, ур.~(28), (31)--(32)", "dimer",
         fig_is("PRB Fig. 2(b)")),
        ("PRB рис.~2(c): $1+K_d$, ур.~(30), (33)--(34)", "dimer",
         fig_is("PRB Fig. 2(c)")),
        ("PRB рис.~3: скорости SW1, SW2 против $\\sum K_{ll'}C_lC_{l'}$",
         "swave", lambda f: f.quantity.str.match(r"SW[12] rate, N=\d$")),
        ("PRB ур.~(37): $\\Gamma_{SW1}(N)/\\Gamma_{SW1}(1)=N$", "swave",
         lambda f: f.figure == "PRB Eq. 37"),
        ("PRB: $\\Lambda^{FM,AFM}$, ED против суммы по ядру, $N=2..8$",
         "ising", fig_is("PRB Eqs. 42-44")),
        ("PRB ур.~(C3), (C5), (C6 исправл.) против суммы ур.~(16)",
         "ising", lambda f: f.figure.isin(["PRB Eq. C3", "PRB Eq. C5",
                                           "PRB Eq. C6"])),
        ("PRB ур.~(C4) как напечатано против инвариантной суммы", "ising",
         fig_is("PRB Eq. C4")),
        ("SS ур.~(15): $G(\\Delta)$ против квадратуры", "surf",
         fig_is("SurfSci Eq. 15")),
        ("SS ур.~(21) против независимого расчёта той же суммы", "surf",
         lambda f: (f.figure == "SurfSci Eq. 21")),
        ("SS ур.~(24) с коэффициентом $3/4$ вместо напечатанного $3/2$",
         "surf", lambda f: f.quantity.str.endswith("with 3/4")),
    ]
    contrast = [
        ("SS ур.~(24) как напечатано", "surf",
         lambda f: (f.figure == "SurfSci Eq. 24")
         & ~f.quantity.str.endswith("with 3/4")),
        ("SS ур.~(23) как напечатано, $E=0$ и $E=2$ мэВ", "surf",
         lambda f: (f.figure == "SurfSci Eq. 23")),
        ("PRB рис.~4: инвариантная форма против ур.~(16), $\\Lambda^{FM}/N^2$",
         "ising", lambda f: f.quantity.str.startswith("Lambda^FM_")
         & f.quantity.str.endswith("invariant vs Eq. 16")),
        ("PRB рис.~4: инвариантная форма против ур.~(16), $\\Lambda^{AFM}$",
         "ising", lambda f: f.quantity.str.startswith("Lambda^AFM_")
         & f.quantity.str.endswith("invariant vs Eq. 16")),
    ]
    numbers = {}
    lines = [r"\begin{tabular}{lr}", r"\toprule",
             r"Сравнение & max.\ откл. \\", r"\midrule"]
    for label, name, mask in spec:
        value = worst(name, mask)
        numbers[label] = value
        lines.append(f"{label} & {sci(value)} \\\\")
    lines.append(r"\midrule")
    lines.append(r"\multicolumn{2}{l}{\emph{расхождения, а не совпадения}} "
                 r"\\")
    for label, name, mask in contrast:
        value = worst(name, mask)
        numbers[label] = value
        lines.append(f"{label} & {value:.2f} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (HERE / "table_r28_dev.tex").write_text("\n".join(lines) + "\n")
    NUMBERS["r28_deviations"] = numbers

    # Numbers of the demo, recomputed from the CSV of the committed run.
    demo = pd.read_csv(R28 / "r28_demo_t2.csv")
    out = {"trajectories": int(len(demo)),
           "kFd_points": int(demo.kFd_over_pi.nunique()),
           "initial_states_per_point": int(demo.seed.nunique()),
           "t2_censored_fraction_all": float(demo.t2_censored.mean()),
           "t2_censored_count_all": int(demo.t2_censored.sum()),
           "t2_energy_censored_count": int(demo.t2_energy_censored.sum())}
    for system in ("XXZ", "TFIM"):
        part = demo[demo.system == system]
        out[system] = {
            "t2_censored_fraction": float(part.t2_censored.mean()),
            "t2_censored_count": int(part.t2_censored.sum()),
            "runs": int(len(part))}
        for column, flag in (("t2", "t2_censored"),
                             ("t2_energy", "t2_energy_censored")):
            kept = part[~part[flag]]
            med = kept.groupby(["dim", "kFd_over_pi"])[column].median()
            out[system][column] = {
                "median_min": float(med.min()), "median_max": float(med.max()),
                "ratio_max_to_min": float(med.max() / med.min()),
                "uncensored_runs": int(len(kept))}
            if column == "t2_energy" and system == "XXZ":
                out[system]["t2_energy"]["median_at_kFd_0"] = float(
                    med.loc[(1, 0.0)])
    NUMBERS["demo"] = out


def main() -> None:
    FIG.mkdir(exist_ok=True)
    finite_shots_part()
    print("finite shots", round(time.time() - T0, 1), "s", flush=True)
    records_part()
    print("records", round(time.time() - T0, 1), "s", flush=True)
    r28_part()
    (HERE / "numbers.json").write_text(json.dumps(NUMBERS, indent=1) + "\n")
    print("done", round(time.time() - T0, 1), "s")


if __name__ == "__main__":
    main()
