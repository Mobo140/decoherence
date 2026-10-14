"""R23 — two-body Pauli correlators as inputs for two qubits.

Why
---
The two-qubit models see ten channels derived from eight observables: the
six single-qubit expectations <sigma_a x I>, <I x sigma_a>, the
l1-coherence and the purity. A two-qubit state has fifteen real
parameters; the nine two-body correlators <sigma_a x sigma_b> carry the
rest. In particular the populations of |00>, |11> and the coherence
between them, which the TFIM Hamiltonian rotates, enter only through
<ZZ>, <XX> - <YY> and <XY> + <YX>, so the phase of the unitary modulation
that limits the computational-basis T2 target on TFIM (R9, R10) is not
directly visible in the inputs. With all fifteen expectations the window
holds the full state, and the only unknown left is the future rate.

  (a) Single-scenario models on D (XXZ) and E (TFIM), 500 trajectories,
      window 20, Transformer (as in E11a) and BiLSTM without the slope
      estimate, each with the 8 base observables and with the base plus
      the 9 correlators. Scored on 200 independent trajectories per
      scenario (generator seed + 1000), all windows.
  (b) TFIM coupling sweep (R5 protocol, 200 trajectories per J, Transformer,
      computational-basis target) with both feature sets.
  (c) No training: how the two targets respond to a small change of the
      rate. For 100 configurations per (system, J) every trajectory is
      simulated with gamma(t) and with 1.02 gamma(t), same initial state,
      and the relative change of T2 (computational basis) and of T2^E
      (energy basis) is recorded. A smooth function of the envelope changes
      by about 2%; a first crossing that can jump to another oscillation
      period changes by much more.

Both feature sets come from one simulation: the base set is the full set
with the correlator columns removed, so the comparison is paired on
identical windows. Trajectories are simulated without the cache because
the cache does not store correlators.

Writes ONLY to experiments/results/r23_two_body_correlators*.csv and
r23c_target_sensitivity*.csv.
Each CSV is accompanied by <name>.csv.provenance.json (configuration,
seeds, software environment).

Usage
-----
    python -m experiments.r23_two_body_correlators --fast --tag smoke
    python -m experiments.r23_two_body_correlators --part a --seeds 42 --tag a42
"""
from __future__ import annotations

import argparse
import dataclasses
import pathlib

import numpy as np

from experiments import common
from src import pipeline
from src import reproducibility
from src.data import scenarios
from src.domain import value_objects
from src.simulation import simulator

J_VALUES = (0.2, 0.5, 0.8, 1.0, 1.2, 1.5, 2.0)
COLUMNS = ("part", "seed", "scenario", "J", "model", "features", "test", "r2",
           "mae", "auroc", "n_samples")

# CSV label -> model alias
MODELS = {"transformer": "transformer_large", "lstm_only": "lstm_only"}
BATCH_SIZE = 64
LEARNING_RATE = 5e-4


def part_a(
    n_per: int, n_eval: int, epochs: int
) -> list[pipeline.ExperimentConfig]:
    """Declares the single-scenario arms of part (a).

    Scenarios D and E, each model with the base set and with the base set
    plus correlators, scored on the own test split and on independent
    trajectories (generator seed + 1000). Both feature sets are cut from
    one simulation, so the comparison is paired on identical windows.

    Args:
        n_per: Training trajectories per scenario.
        n_eval: Independent trajectories per scenario.
        epochs: Training epochs.

    Returns:
        The arms in the order of the result rows.
    """
    arms = []
    for scenario in ("D", "E"):
        data = pipeline.DataSpec(
            name="r23a", scenarios=(scenario,), n_per_scenario=n_per,
            simulator=pipeline.SimulatorSpec("correlators"),
            window_length=20)
        independent = dataclasses.replace(
            data, name="r23a_eval", n_per_scenario=n_eval, train_ratio=0.0,
            val_ratio=0.0, config_seed_offset=1000, data_seed_offset=1)
        tests = (
            pipeline.EvalSpec(tags=(("test", "own"),)),
            pipeline.EvalSpec(
                data=independent, tags=(("test", "independent"),)),
        )
        for feats in ("base", "corr"):
            for model, alias in MODELS.items():
                arms.append(pipeline.ExperimentConfig(
                    data=data, model=alias, features=feats, epochs=epochs,
                    batch_size=BATCH_SIZE, learning_rate=LEARNING_RATE,
                    tests=tests,
                    tags=(("part", "a"), ("scenario", scenario), ("J", ""),
                          ("model", model), ("features", feats))))
    return arms


def part_b(
    n_per_j: int, epochs: int
) -> list[pipeline.ExperimentConfig]:
    """Declares the TFIM coupling-sweep arms of part (b).

    One Transformer per coupling and feature set, scored on its own test
    split. The groups of all couplings are drawn from one generator
    seeded with the seed, in the order of `J_VALUES`.

    Args:
        n_per_j: Trajectories per coupling.
        epochs: Training epochs.

    Returns:
        The arms in the order of the result rows.
    """
    arms = []
    for coupling in J_VALUES:
        data = pipeline.DataSpec(
            name="r23b", coupling=coupling, coupling_sweep=J_VALUES,
            n_per_scenario=n_per_j,
            simulator=pipeline.SimulatorSpec("correlators"),
            window_length=20)
        for feats in ("base", "corr"):
            arms.append(pipeline.ExperimentConfig(
                data=data, model="transformer_large", features=feats,
                epochs=epochs, batch_size=BATCH_SIZE,
                learning_rate=LEARNING_RATE,
                tests=(pipeline.EvalSpec(tags=(("test", "own"),)),),
                tags=(("part", "b"), ("scenario", "E"), ("J", coupling),
                      ("model", "transformer"), ("features", feats))))
    return arms


def part_c(seed: int, n: int, out_path: pathlib.Path) -> None:
    """Measures how T2 and T2^E respond to a 2% change of the rate.

    Not an `ExperimentConfig` arm: nothing is trained; every trajectory is
    simulated twice with the same initial state. Writes the CSV.

    Args:
        seed: Seed of the configurations and the initial states.
        n: Configurations per (system, J).
        out_path: The CSV file.
    """
    sim = simulator.QuTipSimulator()
    rows = []
    for inter in (value_objects.InteractionType.TFIM,
                  value_objects.InteractionType.XXZ):
        rng = np.random.default_rng(seed)
        for coupling in (0.2, 0.8, 1.5):
            configs = scenarios.fixed_coupling_configs(
                n, coupling, rng, interaction=inter)
            for i, cfg in enumerate(configs):
                gamma = tuple(1.02 * c for c in cfg.dissipator.gamma_coeffs)
                scaled = dataclasses.replace(
                    cfg, dissipator=dataclasses.replace(
                        cfg.dissipator, gamma_coeffs=gamma))
                row = {"seed": seed, "system": inter.value, "J": coupling,
                       "index": i}
                for crit, tag in (
                        (value_objects.DecoherenceCriterion.COHERENCE, "t2"),
                        (value_objects.DecoherenceCriterion.COHERENCE_ENERGY,
                         "t2e")):
                    out = []
                    for c in (cfg, scaled):
                        reproducibility.seed_numpy_global(seed * 100000 + i)
                        traj = sim.simulate(dataclasses.replace(
                            c, decoherence_criterion=crit))
                        out.append((traj.t_decoh, traj.censored))
                    row[tag], row[f"{tag}_scaled"] = out[0][0], out[1][0]
                    row[f"{tag}_censored"] = out[0][1] or out[1][1]
                rows.append(row)
            print(f"R23c | seed={seed} {inter.value} J={coupling} done",
                  flush=True)
    common.write_results(
        rows, out_path, seeds=[seed],
        config="R23 (c): T2 and T2^E under a 2% larger gamma(t), "
               f"{n} configurations per (system, J)")


def add_flags(parser: argparse.ArgumentParser) -> None:
    """Adds the choice of parts to the common flags."""
    parser.add_argument("--part", choices=["a", "b", "ab", "c"],
                        default="ab")


def main() -> None:
    """Runs the two-body correlator experiment (parts a, b, c).

    Writes experiments/results/r23_two_body_correlators*.csv for
    parts a and b and r23c_target_sensitivity*.csv for part c.
    """
    args = common.parse_args(__doc__, extra=add_flags)
    n_per, n_eval, n_per_j, epochs = (
        (30, 20, 20, 5) if args.fast else (500, 200, 200, 150))
    if args.part == "c":
        part_c(args.seeds[0], 10 if args.fast else 100,
               common.results_path("r23c_target_sensitivity", args))
        return
    arms = []
    if "a" in args.part:
        arms += part_a(n_per, n_eval, epochs)
    if "b" in args.part:
        arms += part_b(n_per_j, epochs)
    common.run_arms(
        "R23", arms, args,
        common.results_path("r23_two_body_correlators", args), COLUMNS)


if __name__ == "__main__":
    main()
