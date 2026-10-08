"""Figures for Paper 1 built from the replicated results (R13, R15).

    python paper/paper_1/figures/replicated_figures.py

Reads experiments/results/{r13_independent_test,r15_noise_variants}.csv and
writes fig_r13_independent.pdf and fig_r15_noise.pdf next to this file.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
RES = ROOT / "experiments" / "results"
OUT = Path(__file__).parent
plt.rcParams.update({"font.size": 9, "figure.dpi": 150})

LABELS = {
    "physics_only": "physics_only",
    "stretched_exp": "stretched_exp",
    "lstm_only": "lstm_only",
    "physics_feature": "physics_feature",
    "physics_lstm": "physics_lstm",
    "transformer": "transformer",
}


def fig_r13() -> None:
    d = pd.read_csv(RES / "r13_independent_test.csv")
    g = d.groupby(["scenario", "variant"])[["r2", "r2_lo", "r2_hi"]].mean()
    variants = list(LABELS)
    fig, ax = plt.subplots(figsize=(6.6, 2.6))
    w = 0.13
    for k, v in enumerate(variants):
        x = np.arange(2) + (k - (len(variants) - 1) / 2) * w
        m = np.array([g.loc[(sc, v), "r2"] for sc in "BC"])
        lo = m - np.array([g.loc[(sc, v), "r2_lo"] for sc in "BC"])
        hi = np.array([g.loc[(sc, v), "r2_hi"] for sc in "BC"]) - m
        ax.bar(x, m, w, yerr=[lo, hi], capsize=2, label=LABELS[v])
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["B: amplitude damping", "C: dephasing"])
    ax.set_ylabel("$R^2$")
    ax.set_ylim(0.45, 1.02)
    ax.legend(fontsize=6, ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.28))
    fig.tight_layout()
    fig.savefig(OUT / "fig_r13_independent.pdf")


def fig_r15() -> None:
    d = pd.read_csv(RES / "r15_noise_variants.csv")
    g = d.groupby(["variant", "sigma"])["r2"]
    sig = sorted(d.sigma.unique())
    pos = np.arange(len(sig))
    fig, ax = plt.subplots(figsize=(3.3, 2.5))
    for v, style in [("physics_only", "o-"), ("lstm_only", "s-"),
                     ("physics_lstm", "D-"), ("transformer", "^-")]:
        ax.plot(pos, g.mean()[v].values, style, ms=3, label=v)
    ax.set_xticks(pos)
    ax.set_xticklabels([f"{x:g}" for x in sig])
    ax.set_xlabel(r"noise $\sigma$")
    ax.set_ylabel("$R^2$")
    ax.set_ylim(-0.3, 1.05)
    ax.legend(fontsize=6, loc="lower left")
    fig.tight_layout()
    fig.savefig(OUT / "fig_r15_noise.pdf")


if __name__ == "__main__":
    fig_r13()
    fig_r15()
