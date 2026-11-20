"""Singlet rate of the dimer, 1 - K_d(k_F d), for d = 1, 2, 3.

Checks the claim after PRB 95, 075413 Eq. 34 that at large k_F d the singlet
decays at half the single-spin rate. Run from the repository root:

    .venv/bin/python paper/report_2026-11/dimer_one_minus_k.py

Writes into paper/report_2026-11/:
  figures/dimer_1mK.png  --- 1 - K_d with its running mean over one period pi;
  table_dimer_1mK.tex    --- tabular of window means of 1 - K_d.
"""
from __future__ import annotations

import pathlib
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from physics.kondo_bath import pair_kernel  # noqa: E402

OUT = pathlib.Path(__file__).resolve().parent
X_MAX = 40.0
WINDOWS = [(0, 1), (1, 2), (2, 4), (4, 8), (8, 16), (32, 64)]  # in units of pi
COLORS = {1: "black", 2: "tab:red", 3: "tab:blue"}


def running_mean(x: np.ndarray, y: np.ndarray, width: float) -> np.ndarray:
    """Mean of y over [x - width/2, x + width/2] on a uniform grid."""
    n = int(round(width / (x[1] - x[0])))
    kernel = np.ones(n) / n
    return np.convolve(y, kernel, mode="same")


def pi_label(n: int) -> str:
    return {0: "0", 1: "\\pi"}.get(n, f"{n}\\pi")


def window_mean(dim: int, lo: float, hi: float) -> float:
    x = np.linspace(lo, hi, 200_001)
    return float(np.mean(1.0 - pair_kernel(dim, x)))


def main() -> None:
    x = np.linspace(0.0, X_MAX + np.pi, 40_001)
    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    for dim in (1, 2, 3):
        y = 1.0 - pair_kernel(dim, x)
        ax.plot(x, y, color=COLORS[dim], lw=1.0, label=f"{dim}D: $1-K_{dim}$")
        avg = running_mean(x, y, np.pi)
        ok = (x > np.pi / 2) & (x < X_MAX)
        ax.plot(x[ok], avg[ok], color=COLORS[dim], lw=2.0, ls="--", alpha=0.8,
                label=f"{dim}D: среднее за период $\\pi$")
    ax.axhline(1.0, color="gray", lw=0.8)
    ax.axhline(0.5, color="gray", lw=0.8, ls=":")
    ax.text(X_MAX, 1.02, "одиночный спин", ha="right", va="bottom", fontsize=8)
    ax.text(X_MAX, 0.52, "«половина» (текст статьи)", ha="right", va="bottom",
            fontsize=8)
    ax.set_xlim(0.0, X_MAX)
    ax.set_ylim(0.0, 1.25)
    ax.set_xlabel("$k_F d$")
    ax.set_ylabel(r"$\Gamma_{|1,0\rangle\to|0,0\rangle}/\Gamma_0=1-K_d$")
    ax.legend(fontsize=7, ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.17), frameon=False)
    fig.tight_layout()
    fig.savefig(OUT / "figures" / "dimer_1mK.png", dpi=200)

    rows = []
    for lo, hi in WINDOWS:
        vals = " & ".join(f"{window_mean(d, lo * np.pi, hi * np.pi):.3f}" for d in (1, 2, 3))
        rows.append(f"$[{pi_label(lo)},\\,{pi_label(hi)}]$ & {vals} \\\\")
    table = (
        "\\begin{tabular}{lccc}\\hline\n"
        "окно $k_Fd$ & 1D & 2D & 3D \\\\\\hline\n"
        + "\n".join(rows)
        + "\\hline\n\\end{tabular}\n"
    )
    (OUT / "table_dimer_1mK.tex").write_text(table, encoding="utf-8")
    print(table)


if __name__ == "__main__":
    main()
