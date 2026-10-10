"""R20 — the energy-basis target on the Heisenberg model and on the cross-system model.

Why
---
R10 showed that on TFIM the loss of R^2 with J disappears when the target is
the 1/e crossing of the l1-coherence in the eigenbasis of H instead of the
computational basis. The explanation is that the TFIM Hamiltonian rotates
coherence between bases by an angle tan(2 theta) = J / 2h, so a
computational-basis target is modulated by unitary motion.

Two checks of that explanation:

  (a) XXZ J-sweep, both targets. The XXZ Hamiltonian mixes |01> and |10>
      maximally for every J, so the mixing angle does not depend on J.
      Prediction: no J-trend under either target. This is the control.
  (b) Cross-system Transformer (E11a, window 20) trained on D+E with the
      energy-basis target. Compared with R3 arm e11a_w20 (computational
      target, same configurations and seeds). Prediction: E (TFIM) gains,
      D (XXZ) changes little.

Arms
----
  xxz_coh      the E7c J sweep on XXZ, computational-basis target
  xxz_energy   the E7c J sweep on XXZ, energy-basis target
  cross_energy the E11a Transformer on D+E (500 trajectories each),
               energy-basis target

Writes ONLY to experiments/results/r20a_xxz_J_sweep*.csv and
experiments/results/r20b_cross_energy*.csv (suffix set by --tag; --out is
not accepted because there are two files). Each CSV is accompanied by
<name>.csv.provenance.json (configuration, seeds, software environment).

Usage
-----
    python -m experiments.r20_energy_target_xxz --fast --tag smoke
    python -m experiments.r20_energy_target_xxz --arms xxz_coh --seeds 42 --tag s42
"""
from __future__ import annotations

import argparse
import dataclasses

from experiments import common
from experiments import j_sweep
from src import pipeline
from src.domain import value_objects

ARMS = ("xxz_coh", "xxz_energy", "cross_energy")
ENERGY = value_objects.DecoherenceCriterion.COHERENCE_ENERGY.value
XXZ = value_objects.InteractionType.XXZ.value
COLUMNS_A = ("seed", "target", "J", "r2", "mae", "auroc", "n_samples")
COLUMNS_B = ("seed", "scenario", "r2", "mae", "rmse", "auroc", "n_samples")
# E11a: window, gap between the windows of a trajectory, batch and rate.
WINDOW_LENGTH = 20
MIN_WINDOW_GAP = 10
CROSS_BATCH_SIZE = 64
CROSS_LEARNING_RATE = 5e-4


def sweep_arms(
    arms: list[str], n_per_j: int, epochs: int
) -> list[pipeline.ExperimentConfig]:
    """Declares the XXZ J-sweep arms, one per target.

    Args:
        arms: Chosen arms among "xxz_coh" and "xxz_energy".
        n_per_j: Trajectories per coupling.
        epochs: Training epochs.

    Returns:
        The arms in the order of the result rows.
    """
    targets = {"xxz_coh": (None, "computational"),
               "xxz_energy": (ENERGY, "energy")}
    configs = []
    for arm in arms:
        criterion, target = targets[arm]
        configs += j_sweep.arms(
            "r20a", n_per_j, epochs, criterion=criterion, interaction=XXZ,
            tags=(("target", target),))
    return configs


def cross_arm(n_per: int, epochs: int) -> pipeline.ExperimentConfig:
    """Declares the Transformer trained on D+E with the energy-basis target.

    It is scored on fresh trajectories of the training configurations of
    each scenario (data seed + 1), as E11a is.

    Args:
        n_per: Trajectories per scenario.
        epochs: Training epochs.

    Returns:
        The arm; its two tests give the rows of D and E.
    """
    data = pipeline.DataSpec(
        name="r20b", scenarios=("D", "E"), n_per_scenario=n_per,
        criterion=ENERGY, cache="disk", window_length=WINDOW_LENGTH,
        min_window_gap=MIN_WINDOW_GAP, train_ratio=0.9, val_ratio=0.1)
    tests = tuple(
        pipeline.EvalSpec(
            data=dataclasses.replace(
                data, name=f"r20b_eval_{scenario}", scenarios=(scenario,),
                train_ratio=0.0, val_ratio=0.0, data_seed_offset=1),
            tags=(("scenario", scenario),))
        for scenario in ("D", "E"))
    return pipeline.ExperimentConfig(
        data=data, model=j_sweep.MODEL, epochs=epochs,
        batch_size=CROSS_BATCH_SIZE, learning_rate=CROSS_LEARNING_RATE,
        metrics=("r2", "mae", "rmse", "auroc"), tests=tests)


def add_flags(parser: argparse.ArgumentParser) -> None:
    """Adds the choice of arms to the common flags."""
    parser.add_argument("--arms", nargs="+", choices=ARMS,
                        default=list(ARMS))


def main() -> None:
    args = common.parse_args(__doc__, extra=add_flags)
    if args.out:
        raise SystemExit("R20 writes two files: use --tag, not --out")
    n_per_j, n_per, epochs = (30, 50, 10) if args.fast else (200, 500, 150)
    sweeps = [arm for arm in args.arms if arm != "cross_energy"]
    if sweeps:
        common.run_arms(
            "R20a", sweep_arms(sweeps, n_per_j, epochs), args,
            common.results_path("r20a_xxz_J_sweep", args), COLUMNS_A)
    if "cross_energy" in args.arms:
        common.run_arms(
            "R20b", [cross_arm(n_per, epochs)], args,
            common.results_path("r20b_cross_energy", args), COLUMNS_B)


if __name__ == "__main__":
    main()
