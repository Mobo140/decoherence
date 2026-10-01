"""Figures for Paper 2 built from the replicated results (R1, R5, R9).

    python paper/paper_2/figures/replicated_figures.py

Reads experiments/results/{r1_multiseed_ablation_full,r5_multiseed_J_sweep,
r9_target_modulation}.csv and simulates one illustrative TFIM trajectory.
Writes fig_r1_ablation.pdf, fig_r5_r9_J.pdf and fig_r9_example.pdf next to
this file.
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import qutip as qt

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from experiments._config import _bernstein_coeffs  # noqa: E402
from src.physics.lindblad_systems import TwoQubitSystem  # noqa: E402

OUT = Path(__file__).parent
RES = ROOT / "experiments" / "results"
plt.rcParams.update({"font.size": 9, "figure.dpi": 150})


def fig_ablation() -> None:
    d = pd.read_csv(RES / "r1_multiseed_ablation_full.csv")
    variants = ["lstm_only", "physics_lstm", "transformer"]
    fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.4))
    for ax, metric, label in zip(axes, ["r2", "auroc"], ["$R^2$", "AUROC"]):
        for k, sc in enumerate(["D", "E"]):
            g = d[d.scenario == sc].groupby("variant")[metric]
            m, s = g.mean()[variants], g.std()[variants]
            x = np.arange(len(variants)) + (k - 0.5) * 0.36
            ax.bar(x, m, 0.34, yerr=s, capsize=2,
                   label="D (XXZ)" if sc == "D" else "E (TFIM)")
        ax.set_xticks(range(len(variants)))
        ax.set_xticklabels(variants, fontsize=7)
        ax.set_ylabel(label)
        ax.set_ylim(0, 1)
    axes[0].legend(fontsize=7, loc="upper left")
    fig.tight_layout()
    fig.savefig(OUT / "fig_r1_ablation.pdf")


def fig_J() -> None:
    r5 = pd.read_csv(RES / "r5_multiseed_J_sweep.csv").groupby("J")["r2"]
    r9 = pd.read_csv(RES / "r9_target_modulation.csv")
    r9 = r9[r9.part == 2]
    fig, ax = plt.subplots(figsize=(3.3, 2.4))
    J = r5.mean().index.values
    ax.errorbar(J, r5.mean().values, yerr=r5.std().values, fmt="o-", color="C0",
                capsize=2, label="$R^2$ (R5)")
    ax.set_xlabel("$J$  ($h=1$)")
    ax.set_ylabel("$R^2$", color="C0")
    ax.set_ylim(0.2, 0.9)
    ax2 = ax.twinx()
    for model, style in [("TFIM", "s-"), ("XXZ", "s--")]:
        sub = r9[r9.model == model]
        ax2.plot(sub.J, sub.frac_nonmonotone, style, color="C3", ms=3,
                 label=f"non-monotone, {model}")
    ax2.set_ylabel("non-monotone fraction", color="C3")
    ax2.set_ylim(0, 0.9)
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, fontsize=6, loc="lower left")
    fig.tight_layout()
    fig.savefig(OUT / "fig_r5_r9_J.pdf")


def _trajectory(J: float, seed: int):
    rng = np.random.default_rng(seed)
    coeffs = np.array(_bernstein_coeffs("monotone_increasing", rng, gamma_max=0.8))
    s = TwoQubitSystem(J=J, interaction_type="TFIM", gamma_coeffs=coeffs, gamma_t_max=20.0)
    v = rng.standard_normal(4) + 1j * rng.standard_normal(4)
    psi = qt.Qobj((v / np.linalg.norm(v)).reshape(-1, 1), dims=[[2, 2], [1, 1]])

    def g(t, args):
        x = s.get_gamma(t)
        return np.sqrt(x) if x > 0 else 0.0

    times = np.linspace(0, 20, 201)
    out = qt.mesolve(s.H, psi * psi.dag(), times, [[L, g] for L in s.L_list],
                     e_ops=[], options={"nsteps": 10000})
    U = np.linalg.eigh(s.H.full())[1]
    l1 = lambda m: float(np.abs(m).sum() - np.abs(np.diag(m)).sum())
    c = np.array([l1(r.full()) for r in out.states])
    ce = np.array([l1(U.conj().T @ r.full() @ U) for r in out.states])
    return times, c / c[0], ce / ce[0]


def fig_example() -> None:
    times, c, ce = _trajectory(1.5, 0)
    thr = np.exp(-1)
    first = times[np.argmax(c < thr)]
    last = times[np.where(c >= thr)[0][-1]]
    fig, ax = plt.subplots(figsize=(3.3, 2.3))
    ax.plot(times, c, "k-", lw=1, label="computational basis")
    ax.plot(times, ce, "C0--", lw=1, label="energy eigenbasis")
    ax.axhline(thr, color="gray", ls=":", lw=0.8)
    ax.axvspan(first, last, color="C3", alpha=0.15)
    ax.set_xlim(0, 14)
    ax.set_xlabel("$t$")
    ax.set_ylabel("$C(t)/C(0)$")
    ax.legend(fontsize=6)
    fig.tight_layout()
    fig.savefig(OUT / "fig_r9_example.pdf")


if __name__ == "__main__":
    fig_ablation()
    fig_J()
    fig_example()
