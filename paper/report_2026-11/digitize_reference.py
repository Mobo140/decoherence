"""Digitises PRB 95, 075413 Figs. 2(b, c) and 4 and overlays our curves.

Run from the repository root:

    .venv/bin/python paper/report_2026-11/digitize_reference.py

Needs docks/Delgado_RKKY_PRB2017.pdf (untracked) and `pdfimages`. The
embedded figure images are extracted to a temporary directory and are not
kept. Curves are found by colour (black 1D, red 2D, blue 3D); per pixel
column the mean row of the masked pixels is the curve value if their spread
is under 12 px. The digitised points are compared with candidate formulas.

Writes into paper/report_2026-11/:
  figures/cmp_fig2.png, figures/cmp_fig4.png
  table_digitize.tex, numbers_digitize.json
"""
from __future__ import annotations

import json
import pathlib
import subprocess
import sys
import tempfile

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))

from src.physics import kondo_bath  # noqa: E402
from experiments.new.r28_kondo_reproduction import _paper_lambda  # noqa: E402

PDF = ROOT / "docks" / "Delgado_RKKY_PRB2017.pdf"
FIG = HERE / "figures"
MAX_SPREAD = 12          # px
# Columns within 14 px (0.03) of the ends of the x axis are dropped: the
# ticks of the frame are inside the plot there.
EDGE = 0.03
RATIO = 0.01             # k_F d / pi in Fig. 4
XI = np.pi * RATIO
DIMS = {1: "black", 2: "red", 3: "blue"}
PLOT_COLOR = {1: "black", 2: "tab:red", 3: "tab:blue"}
plt.rcParams.update({
    "font.size": 9, "axes.titlesize": 9, "axes.labelsize": 9,
    "legend.fontsize": 7.5, "figure.dpi": 150, "savefig.dpi": 150,
    "axes.grid": True, "grid.alpha": 0.3})


def extract_images(tmp: pathlib.Path) -> tuple[np.ndarray, np.ndarray]:
    """Extracts the embedded images of the PDF; returns Fig. 2 and Fig. 4."""
    if not PDF.exists():
        raise FileNotFoundError(PDF)
    subprocess.run(["pdfimages", "-png", str(PDF), str(tmp / "img")],
                   check=True)
    fig2 = np.asarray(Image.open(tmp / "img-001.png").convert("RGB"))
    fig4 = np.asarray(Image.open(tmp / "img-002.png").convert("RGB"))
    assert fig2.shape[:2] == (1452, 1020), fig2.shape
    assert fig4.shape[:2] == (1100, 999), fig4.shape
    return fig2.astype(int), fig4.astype(int)


def colour_mask(img: np.ndarray, colour: str) -> np.ndarray:
    r, g, b = img[..., 0], img[..., 1], img[..., 2]
    if colour == "black":
        return (r < 80) & (g < 80) & (b < 80)
    if colour == "red":
        return (r > 170) & (g < 110) & (b < 110)
    return (b > 170) & (r < 110) & (g < 110)


def trace(mask: np.ndarray, rows: tuple[int, int], cols: tuple[int, int],
          frame: bool) -> tuple[np.ndarray, np.ndarray]:
    """Curve of a mask inside a panel: (pixel columns, mean pixel row).

    With `frame` the black frame lines (rows and columns that are mostly
    masked) are removed first.
    """
    sub = mask[rows[0]:rows[1] + 1, cols[0]:cols[1] + 1].copy()
    if frame:
        sub[sub.mean(axis=1) > 0.5, :] = False
        sub[:, sub.mean(axis=0) > 0.5] = False
    xs, ys = [], []
    for j in range(sub.shape[1]):
        idx = np.flatnonzero(sub[:, j])
        if idx.size and idx.max() - idx.min() < MAX_SPREAD:
            xs.append(cols[0] + j)
            ys.append(rows[0] + idx.mean())
    return np.array(xs), np.array(ys)


def blank(mask: np.ndarray, boxes: list[tuple[int, int, int, int]]) -> None:
    """Clears boxes (y0, y1, x0, x1) of a mask in place."""
    for y0, y1, x0, x1 in boxes:
        mask[y0:y1, x0:x1] = False


def stats(deviation: np.ndarray) -> dict:
    d = np.abs(deviation)
    return {"median": float(np.median(d)),
            "p95": float(np.percentile(d, 95)), "points": int(d.size)}


# --- Fig. 2 -------------------------------------------------------------------
F2_X = (131.0, 996.0, 2.0)          # px at 0 and at 2.0
F2_PANELS = {"b": (940.0, 584.0, 0.0, 1.0), "c": (1346.0, 990.0, 1.0, 2.0)}
F2_MASK = [(640, 700, 200, 270)]    # the label "1D"


def fig2_curves(img: np.ndarray) -> dict:
    out = {}
    for panel, (y0, y1, v0, v1) in F2_PANELS.items():
        rows = (int(y1) - 8, int(y0) + 8)
        for dim, colour in DIMS.items():
            mask = colour_mask(img, colour)
            blank(mask, F2_MASK)
            px, py = trace(mask, rows, (int(F2_X[0]) - 4, int(F2_X[1]) + 4),
                           frame=(colour == "black"))
            x = (px - F2_X[0]) / (F2_X[1] - F2_X[0]) * F2_X[2]
            v = v0 + (y0 - py) / (y0 - y1) * (v1 - v0)
            keep = (x >= EDGE) & (x <= 2.0 - EDGE)
            out[(panel, dim)] = (x[keep], v[keep])
    return out


def kernel(dim: int, x_over_pi: np.ndarray) -> np.ndarray:
    return kondo_bath.pair_kernel(dim, np.pi * x_over_pi)


def fig2_candidates(panel: str, dim: int, x: np.ndarray) -> dict:
    k = kernel(dim, x)
    if panel == "b":
        return {"1-K": 1 - k, "(1-K)/2": (1 - k) / 2}
    return {"1+K": 1 + k}


# --- Fig. 4 -------------------------------------------------------------------
F4_X = (119.5, 990.0, 2.0)
F4_PANELS = {"a": (330.0, 10.0), "b": (670.0, 354.0), "c": (1010.0, 694.0)}
F4_MASK = [(20, 150, 720, 900), (540, 665, 125, 310), (700, 840, 125, 360),
           # text labels inside the frames
           (110, 150, 360, 420), (180, 215, 255, 310), (215, 250, 425, 485),
           (360, 415, 770, 935), (700, 765, 770, 935)]
_LAMBDA_CACHE: dict = {}


def lambda_pair(panel: str, dim: int, n: int) -> tuple[float, float]:
    """(invariant 4 s^T K s, Eq. 16 as printed) for N sites; FM / N^2 in (a)."""
    key = (panel, dim, n)
    if key not in _LAMBDA_CACHE:
        signs = (np.ones(n) if panel == "a"
                 else (-1.0) ** np.arange(1, n + 1))
        positions = XI * np.arange(1, n + 1)
        kern = kondo_bath.kernel_matrix(positions, k_F=1.0, dim=dim)
        inv = float(4.0 * signs @ kern @ signs)
        eq16 = _paper_lambda(signs, XI, dim)
        if panel == "a":
            inv, eq16 = inv / n ** 2, eq16 / n ** 2
        _LAMBDA_CACHE[key] = (inv, eq16)
    return _LAMBDA_CACHE[key]


def chain_length(panel: str, x: float) -> int:
    """N = round(x / 0.01), moved to the parity of the panel; at least 1."""
    n = int(round(x / RATIO))
    if panel == "b" and n % 2 == 0:
        n += 1 if x / RATIO >= n else -1
    if panel == "c" and n % 2 == 1:
        n += 1 if x / RATIO >= n else -1
    return max(n, 1 if panel != "c" else 2)


def fig4_curves(img: np.ndarray) -> dict:
    out = {}
    for panel, (y0, y1) in F4_PANELS.items():
        rows = (int(y1) - 6, int(y0) + 6)
        for dim, colour in DIMS.items():
            mask = colour_mask(img, colour)
            blank(mask, F4_MASK)
            px, py = trace(mask, rows, (int(F4_X[0]) - 4, int(F4_X[1]) + 4),
                           frame=(colour == "black"))
            x = (px - F4_X[0]) / (F4_X[1] - F4_X[0]) * F4_X[2]
            v = 4.0 * (y0 - py) / (y0 - y1)
            keep = (x >= EDGE) & (x <= 2.0 - EDGE)
            out[(panel, dim)] = (x[keep], v[keep])
    return out


def fig4_candidates(panel: str, dim: int, x: np.ndarray) -> dict:
    pairs = np.array([lambda_pair(panel, dim, chain_length(panel, v))
                      for v in x])
    return {"invariant": pairs[:, 0], "Eq. 16": pairs[:, 1]}


# --- output -------------------------------------------------------------------
def plot_fig2(curves: dict, cands: dict) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.6))
    for ax, panel, ylabel in ((axes[0], "b", r"$\Gamma_{S\to T}/\Gamma_0$"),
                              (axes[1], "c", r"$\Gamma_{|1,0\rangle\to|1,-1\rangle}/\Gamma_0$")):
        for dim in DIMS:
            x, v = curves[(panel, dim)]
            ax.plot(x, v, ".", ms=1.6, color="0.55",
                    label="оцифровка" if dim == 1 else None, zorder=1)
            grid = np.linspace(0, 2, 801)
            for style, (name, y) in zip(("-", "--"),
                                        fig2_candidates(panel, dim, grid).items()):
                ax.plot(grid, y, style, lw=0.9, color=PLOT_COLOR[dim],
                        label=f"{dim}D, {name}", zorder=2)
        ax.set_xlabel(r"$k_Fd/\pi$")
        ax.set_ylabel(ylabel)
        ax.set_title(f"({panel})")
    handles, labels = axes[0].get_legend_handles_labels()
    h2, l2 = axes[1].get_legend_handles_labels()
    handles += h2[1:]
    labels += l2[1:]
    fig.legend(handles, labels, loc="lower center", ncol=5, fontsize=7)
    fig.tight_layout(rect=(0, 0.15, 1, 1))
    fig.savefig(FIG / "cmp_fig2.png")
    plt.close(fig)


def plot_fig4(curves: dict) -> None:
    fig, axes = plt.subplots(3, 1, figsize=(6.4, 7.6), sharex=True)
    titles = {"a": r"(a) $\Lambda^{FM}/N^2$", "b": r"(b) $\Lambda^{AFM}$, нечётное $N$",
              "c": r"(c) $\Lambda^{AFM}$, чётное $N$"}
    for ax, panel in zip(axes, "abc"):
        for dim in DIMS:
            x, v = curves[(panel, dim)]
            ax.plot(x, v, ".", ms=1.6, color="0.55",
                    label="оцифровка" if dim == 1 else None, zorder=1)
            step = 1 if panel == "a" else 2
            ns = np.arange(1 if panel != "c" else 2, 201, step)
            c = fig4_candidates(panel, dim, ns * RATIO)
            ax.plot(ns * RATIO, c["invariant"], "-", lw=0.9,
                    color=PLOT_COLOR[dim],
                    label=f"{dim}D, $4s^TKs$ (инвариантная)", zorder=2)
            ax.plot(ns * RATIO, c["Eq. 16"], "--", lw=0.9,
                    color=PLOT_COLOR[dim],
                    label=f"{dim}D, ур.~16 как напечатано".replace("~", " "),
                    zorder=2)
        ax.set_ylabel(titles[panel])
        ax.set_ylim(-0.1, 4.2)
    axes[0].legend(loc="upper right", ncol=2, fontsize=6)
    axes[2].set_xlabel(r"$N k_Fd/\pi$")
    fig.tight_layout()
    fig.savefig(FIG / "cmp_fig4.png")
    plt.close(fig)


def write_table(res2: dict, res4: dict) -> None:
    def cell(s: dict) -> str:
        return f"{s['median']:.3f} / {s['p95']:.3f}"

    lines = [r"\begin{tabular}{llcc}", r"\toprule",
             r"Кривая & & формула 1 & формула 2 \\", r"\midrule",
             r"\multicolumn{4}{l}{\emph{рис.~2: формула 1 --- $1-K_d$ (в (c): $1+K_d$), "
             r"формула 2 --- $(1-K_d)/2$}} \\"]
    for panel in "bc":
        for dim in DIMS:
            r = res2[f"{panel}_{dim}D"]
            if panel == "b":
                a, b = cell(r["1-K"]), cell(r["(1-K)/2"])
            else:
                a, b = cell(r["1+K"]), "--"
            lines.append(f"({panel}) & {dim}D & {a} & {b} \\\\")
    lines += [r"\midrule",
              r"\multicolumn{4}{l}{\emph{рис.~4: формула 1 --- $4s^TKs$, "
              r"формула 2 --- ур.~16 как напечатано}} \\"]
    for panel in "abc":
        for dim in DIMS:
            r = res4[f"{panel}_{dim}D"]
            lines.append(f"({panel}) & {dim}D & {cell(r['invariant'])} & "
                         f"{cell(r['Eq. 16'])} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (HERE / "table_digitize.tex").write_text("\n".join(lines) + "\n")


def main() -> None:
    FIG.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        fig2, fig4 = extract_images(pathlib.Path(tmp))
    c2, c4 = fig2_curves(fig2), fig4_curves(fig4)
    res2, res4 = {}, {}
    for (panel, dim), (x, v) in c2.items():
        res2[f"{panel}_{dim}D"] = {
            name: stats(v - y)
            for name, y in fig2_candidates(panel, dim, x).items()}
    for (panel, dim), (x, v) in c4.items():
        res4[f"{panel}_{dim}D"] = {
            name: stats(v - y)
            for name, y in fig4_candidates(panel, dim, x).items()}
    plot_fig2(c2, None)
    plot_fig4(c4)
    write_table(res2, res4)
    numbers = {
        "method": {"max_spread_px": MAX_SPREAD,
                   "fig4_k_F_d_over_pi": RATIO,
                   "fig4_N": "round(x/0.01), moved to the parity of the panel"},
        "fig2": res2, "fig4": res4}
    (HERE / "numbers_digitize.json").write_text(
        json.dumps(numbers, indent=1) + "\n")
    for label, res in (("fig2", res2), ("fig4", res4)):
        for key, cands in res.items():
            print(label, key, {k: (round(v["median"], 4), round(v["p95"], 4),
                                   v["points"]) for k, v in cands.items()})


if __name__ == "__main__":
    main()
