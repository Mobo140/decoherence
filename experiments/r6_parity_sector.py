"""R6 — does the two-mode picture explain the TFIM predictability loss?

Background
----------
Paper 2 explained the loss by the amplitude-damping dissipator mixing the
two parity sectors of the TFIM Hamiltonian. That step is wrong for the
system actually simulated, on two counts:

  * The paper writes H = J sz sz + h(sx (x) I + I (x) sx) and takes
    P = sx (x) sx. The code implements H = J sx1 sx2 + h(sz1 + sz2),
    whose parity is P = sz (x) sz. The spectra match, but sigma_minus is
    defined in the computational basis either way, so the two setups are
    not related by a basis rotation.
  * More fundamentally, L enters the dissipator quadratically:
    D[L]rho = L rho L^dag - 1/2 {L^dag L, rho}. When P L P^dag = ±L the
    sign cancels and the symmetry survives. Here sigma_minus -> -sigma_minus,
    so it does. Checked directly: the Liouvillian commutes with the parity
    superoperator at every J, for both sigma_minus and sigma_z, while
    deliberately asymmetric controls give a norm of 32.0 and 25.9.

So the sectors never mix, and a sigma_minus vs sigma_z contrast would
isolate nothing -- both conserve parity exactly.

What can still be tested
------------------------
The rest of the argument does not need mixing. Sectors stay separate, but
a generic initial state populates both (98% of random states carry more
than 5% weight in each), so the observed coherence is a sum of modes at
the two sector frequencies, whose ratio is
r(J) = J / sqrt(J^2 + 4h^2). When the two are comparable, a short early
window has two competing decay/oscillation channels to extrapolate rather
than one -- which is the actual claim about predictability.

That is falsifiable: prepare initial states inside a single parity
sector, leaving one frequency instead of two. The mechanism predicts
markedly better predictability and a flatter dependence on J. If the
J-dependence survives single-sector preparation, the two-mode picture is
not what drives it.

Arms (three seeds each, TFIM J sweep, everything else identical):

    generic    initial states as published
    sector+    initial states projected onto P = +1
    sector-    initial states projected onto P = -1

The cache is disabled for every arm: its keys do not include the
simulator, so the projected arms would otherwise read the ordinary
trajectories straight back.

Writes ONLY to experiments/results/r6_parity_sector.csv.

Usage
-----
    python -m experiments.r6_parity_sector --fast
    python -m experiments.r6_parity_sector --seeds 42 43 44
"""
from __future__ import annotations

import argparse
import csv
import math
import statistics as st
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import qutip as qt

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from experiments.e7_paper2_extended import run_e7c
from src.infrastructure.quantum.qutip_simulator import QuTipSimulator

RESULTS_DIR = Path(__file__).parent / "results"
SCRATCH = RESULTS_DIR / "_r6_scratch.csv"
H_FIELD = 1.0

# Parity of the Hamiltonian the code actually builds, H = J sx1 sx2 + h(sz1+sz2).
PARITY = qt.tensor(qt.sigmaz(), qt.sigmaz())


class ParitySectorSimulator(QuTipSimulator):
    """Projects each initial pure state onto one parity sector.

    Only the state preparation changes; the Hamiltonian, the dissipator and
    the integration are the ordinary ones, so any difference in the results
    is attributable to how many sector frequencies the signal contains.
    """

    def __init__(self, sector: int) -> None:
        if sector not in (+1, -1):
            raise ValueError("sector must be +1 or -1")
        self.sector = sector
        eye = qt.qeye([2, 2])
        self._proj = (eye + sector * PARITY) / 2

    def _initial_state(self, config):
        psi = super()._initial_state(config)
        if config.n_qubits.value != 2:
            return psi
        # Redraw rather than renormalise a vanishing component: amplifying a
        # near-zero projection would make the state an artefact of the noise
        # in the discarded sector.
        for _ in range(100):
            proj = self._proj * psi
            if proj.norm() > 1e-3:
                return proj.unit()
            psi = super()._initial_state(config)
        raise RuntimeError("could not draw a state inside the requested sector")


def _append(path: Path, row: dict, first: bool) -> None:
    path.parent.mkdir(exist_ok=True)
    with path.open("w" if first else "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(row.keys()))
        if first:
            w.writeheader()
        w.writerow(row)


def _stats(values):
    good = [v for v in values if math.isfinite(v)]
    if not good:
        return float("nan"), float("nan")
    return st.mean(good), (st.stdev(good) if len(good) > 1 else 0.0)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    p.add_argument("--fast", action="store_true")
    p.add_argument("--out", type=str, default="r6_parity_sector.csv")
    p.add_argument("--arms", nargs="+", default=["generic", "sector+", "sector-"])
    a = p.parse_args()

    n_per_J = 30 if a.fast else 200
    epochs = 10 if a.fast else 150
    out = RESULTS_DIR / a.out

    print(f"R6 | seeds={a.seeds} arms={a.arms} n_per_J={n_per_J} epochs={epochs}")
    print(f"writing -> {out}  (cache disabled, champion CSVs untouched)\n", flush=True)

    sims = {"generic": QuTipSimulator(),
            "sector+": ParitySectorSimulator(+1),
            "sector-": ParitySectorSimulator(-1)}
    first, t0, collected = True, time.time(), []

    for seed in a.seeds:
        for arm in a.arms:
            ts = time.time()
            print(f"--- seed {seed} | {arm} ---", flush=True)
            rows = run_e7c(sims[arm], n_per_J=n_per_J, n_epochs=epochs,
                           verbose=False, seed=seed, config_seed=seed,
                           out_path=SCRATCH, store=None)
            for r in rows:
                row = {"seed": seed, "arm": arm, "J": r["J"], "r2": r["r2"],
                       "mae": r["mae"], "auroc": r["auroc"],
                       "n_samples": r["n_samples"]}
                _append(out, row, first)
                first = False
                collected.append(row)
            print(f"    [{time.time() - ts:.0f}s]", flush=True)

    SCRATCH.unlink(missing_ok=True)

    g = defaultdict(list)
    for r in collected:
        g[(r["arm"], float(r["J"]))].append(float(r["r2"]))

    Js = sorted({float(r["J"]) for r in collected})
    print("\n" + "=" * 70)
    print(f"{'J':>5}{'r(J)':>7}" + "".join(f"{arm:>19}" for arm in a.arms))
    for J in Js:
        line = f"{J:>5.1f}{J / math.sqrt(J * J + 4 * H_FIELD ** 2):>7.3f}"
        for arm in a.arms:
            m, sd = _stats(g[(arm, J)])
            line += f"{m:>12.3f}±{sd:<6.3f}"
        print(line)

    print("\nslope across J (mean R2 at J<=0.5 minus mean at J>=0.8):")
    for arm in a.arms:
        lo = [v for J in Js if J <= 0.5 for v in g[(arm, J)]]
        hi = [v for J in Js if J >= 0.8 for v in g[(arm, J)]]
        if lo and hi:
            m_lo, _ = _stats(lo)
            m_hi, _ = _stats(hi)
            print(f"  {arm:<9} weak {m_lo:6.3f}  strong {m_hi:6.3f}  "
                  f"drop {m_lo - m_hi:+.3f}")
    print("\nThe mechanism predicts a markedly smaller drop for the single-sector")
    print("arms. A drop that survives them is not the two-mode picture.")
    print(f"\ntotal {time.time() - t0:.0f}s -> {out}")


if __name__ == "__main__":
    main()
