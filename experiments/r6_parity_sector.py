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

Writes ONLY to experiments/results/r6_parity_sector*.csv.
Each CSV is accompanied by <name>.csv.provenance.json (configuration,
seeds, software environment).

Usage
-----
    python -m experiments.r6_parity_sector --fast --tag smoke
    python -m experiments.r6_parity_sector --seeds 42 43 44
"""
from __future__ import annotations

import argparse
import collections
import math
from collections.abc import Mapping, Sequence

from experiments import common
from experiments import j_sweep
from src import pipeline
from src.evaluation import stats

COLUMNS = ("seed", "arm", "J", "r2", "mae", "auroc", "n_samples")
# arm -> parity sector of the initial states (None: as published)
SECTORS = {"generic": None, "sector+": +1, "sector-": -1}
# J below which the coupling is weak, and from which it is strong.
WEAK_MAX, STRONG_MIN = 0.5, 0.8


def simulator_spec(arm: str) -> pipeline.SimulatorSpec:
    """Returns the simulator that prepares the initial states of an arm."""
    sector = SECTORS[arm]
    if sector is None:
        return pipeline.SimulatorSpec()
    return pipeline.SimulatorSpec("parity_sector", (("sector", sector),))


def arms(
    names: Sequence[str], n_per_j: int, epochs: int
) -> list[pipeline.ExperimentConfig]:
    """Declares the J sweep of every arm.

    The cache is off for every arm: its keys do not name the simulator, so
    the projected arms would otherwise read the ordinary trajectories back.

    Args:
        names: Chosen arms among `SECTORS`.
        n_per_j: Trajectories per coupling.
        epochs: Training epochs.

    Returns:
        The arms in the order of the result rows.
    """
    configs = []
    for name in names:
        configs += j_sweep.arms(
            "r6", n_per_j, epochs, simulator=simulator_spec(name),
            cache="none", tags=(("arm", name),))
    return configs


def finite_mean_std(values: Sequence[float]) -> tuple[float, float]:
    """Mean and s.d. over the finite entries; nan if there are none."""
    good = [v for v in values if math.isfinite(v)]
    if not good:
        return float("nan"), float("nan")
    return stats.mean_std(good)


def summarize(
    rows: Sequence[Mapping[str, object]], names: Sequence[str]
) -> None:
    """Prints R^2 by J for every arm and the drop from weak to strong J.

    Args:
        rows: Result rows with `arm`, `J` and `r2`.
        names: The arms, in print order.
    """
    r2 = collections.defaultdict(list)
    for row in rows:
        r2[(row["arm"], float(row["J"]))].append(float(row["r2"]))
    couplings = sorted({float(row["J"]) for row in rows})

    print("\n" + "=" * 70)
    print(f"{'J':>5}{'r(J)':>7}" + "".join(f"{arm:>19}" for arm in names))
    for coupling in couplings:
        line = f"{coupling:>5.1f}{j_sweep.splitting_ratio(coupling):>7.3f}"
        for arm in names:
            mean, std = finite_mean_std(r2[(arm, coupling)])
            line += f"{mean:>12.3f}±{std:<6.3f}"
        print(line)

    print("\nslope across J (mean R2 at J<=0.5 minus mean at J>=0.8):")
    for arm in names:
        weak = [v for c in couplings if c <= WEAK_MAX for v in r2[(arm, c)]]
        strong = [v for c in couplings if c >= STRONG_MIN
                  for v in r2[(arm, c)]]
        if weak and strong:
            mean_weak = finite_mean_std(weak)[0]
            mean_strong = finite_mean_std(strong)[0]
            print(f"  {arm:<9} weak {mean_weak:6.3f}  "
                  f"strong {mean_strong:6.3f}  "
                  f"drop {mean_weak - mean_strong:+.3f}")
    print("\nThe mechanism predicts a markedly smaller drop for the "
          "single-sector")
    print("arms. A drop that survives them is not the two-mode picture.")


def add_flags(parser: argparse.ArgumentParser) -> None:
    """Adds the choice of arms to the common flags."""
    parser.add_argument("--arms", nargs="+", choices=list(SECTORS),
                        default=list(SECTORS))


def main() -> None:
    """Runs the parity-sector arms and prints the summary.

    Writes experiments/results/r6_parity_sector*.csv.
    """
    args = common.parse_args(__doc__, extra=add_flags)
    n_per_j, epochs = (30, 10) if args.fast else (200, 150)
    rows = common.run_arms(
        "R6", arms(args.arms, n_per_j, epochs), args,
        common.results_path("r6_parity_sector", args), COLUMNS)
    summarize(rows, args.arms)


if __name__ == "__main__":
    main()
