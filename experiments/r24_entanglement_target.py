"""R24 — entanglement as the target: predicting entanglement sudden death.

Why
---
Coherence measures how far the pair is from a classical mixture in one
basis; entanglement is what a two-qubit device actually uses. Under local
amplitude damping the Wootters concurrence C(t) can reach zero in finite
time (entanglement sudden death, ESD). The target here is T_ESD, the first
time C = 0, interpolated on the unclipped Wootters lambda; trajectories
still entangled at t_max are censored and, as everywhere in this project,
dropped, so the models predict T_ESD given that it occurs before t_max.
The 1/e crossing of C was considered first and rejected: its median is
1.3 time units, shorter than a 20-step window for two thirds of the
trajectories, and the interaction makes C re-cross the threshold in about
half of them.

Concurrence is a function of the full two-qubit state, not of the local
Bloch vectors, so inputs should matter more than for T2.

  (a) Single-scenario models on D (XXZ) and E (TFIM), 500 trajectories,
      window 20; Transformer (as in E11a) and BiLSTM without the slope
      estimate; three input sets from one simulation:
          base        the 8 observables used everywhere else
          corr        + the 9 two-body correlators (full tomography, R23)
          corr_conc   + the concurrence itself as a channel
      Scored on 200 independent trajectories per scenario. Same
      configurations, seeds and initial states as R23 (a), so the
      predictability of T_ESD can be set against that of T2.
  (b) No training: for 100 configurations per (system, J) the fraction of
      trajectories reaching ESD before t_max, the fraction in which
      entanglement revives after dying, and the response of T_ESD to a 2%
      increase of the rate (as R23 (c) did for T2).

Writes ONLY to experiments/results/r24_entanglement_target*.csv and
r24b_esd_statistics*.csv.
Each CSV is accompanied by <name>.csv.provenance.json (configuration,
seeds, software environment).

Usage
-----
    python -m experiments.r24_entanglement_target --fast --tag smoke
    python -m experiments.r24_entanglement_target --part a --seeds 42 --tag a42
"""
from __future__ import annotations

import argparse
import dataclasses
import pathlib

import numpy as np

from experiments import common
from experiments.r23_two_body_correlators import (
    BATCH_SIZE, LEARNING_RATE, MODELS)
from src import pipeline
from src import reproducibility
from src.data import scenarios
from src.domain import value_objects
from src.physics import lindblad_systems
from src.physics import measures
from src.physics import simulate_qutip
from src.simulation import simulator

ESD = value_objects.DecoherenceCriterion.ENTANGLEMENT_DEATH
COLUMNS = ("seed", "scenario", "model", "features", "test", "r2", "mae",
           "auroc", "n_samples", "n_censored")


def esd_configs(groups):
    """Scenario groups D and E with entanglement death as the criterion."""
    return {sc: [dataclasses.replace(c, decoherence_criterion=ESD)
                 for c in groups[sc]]
            for sc in ("D", "E")}


def part_a(
    n_per: int, n_eval: int, epochs: int
) -> list[pipeline.ExperimentConfig]:
    """Declares the single-scenario arms of part (a).

    Same configurations, seeds and initial states as R23 (a); the three
    input sets are cut from one simulation.

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
            name="r24a", scenarios=(scenario,), n_per_scenario=n_per,
            criterion=ESD.value,
            simulator=pipeline.SimulatorSpec("entanglement"),
            window_length=20)
        independent = dataclasses.replace(
            data, name="r24a_eval", n_per_scenario=n_eval, train_ratio=0.0,
            val_ratio=0.0, config_seed_offset=1000, data_seed_offset=1)
        tests = (
            pipeline.EvalSpec(tags=(("test", "own"),)),
            pipeline.EvalSpec(
                data=independent, tags=(("test", "independent"),)),
        )
        for feats in ("base", "corr", "corr_conc"):
            for model, alias in MODELS.items():
                arms.append(pipeline.ExperimentConfig(
                    data=data, model=alias, features=feats, epochs=epochs,
                    batch_size=BATCH_SIZE, learning_rate=LEARNING_RATE,
                    tests=tests,
                    tags=(("scenario", scenario), ("model", model),
                          ("features", feats))))
    return arms


def part_b(seed: int, n: int, out_path: pathlib.Path) -> None:
    """Measures ESD statistics and the response of T_ESD to a 2% rate change.

    Not an `ExperimentConfig` arm: nothing is trained. For each (system, J)
    it records the share of trajectories that reach ESD, the revival of
    entanglement after it, and T_ESD at the rate and at 1.02 times the
    rate. Writes the CSV.

    Args:
        seed: Seed of the configurations and the initial states.
        n: Configurations per (system, J).
        out_path: The CSV file.
    """
    rows = []
    for inter in (value_objects.InteractionType.XXZ,
                  value_objects.InteractionType.TFIM):
        rng = np.random.default_rng(seed)
        for coupling in (0.2, 0.8, 1.5):
            configs = scenarios.fixed_coupling_configs(
                n, coupling, rng, interaction=inter)
            for i, cfg in enumerate(configs):
                times = np.linspace(
                    0.0, cfg.t_max, int(round(cfg.t_max / cfg.dt)) + 1)
                row = {"seed": seed, "system": inter.value, "J": coupling,
                       "index": i}
                for tag, scale in (("", 1.0), ("_scaled", 1.02)):
                    system = lindblad_systems.TwoQubitSystem(
                        J=cfg.J, interaction_type=cfg.interaction_type.value,
                        gamma_coeffs=scale * np.array(
                            cfg.dissipator.gamma_coeffs),
                        gamma_t_max=cfg.t_max,
                        jump_operators=[cfg.dissipator.operator_type.value])
                    reproducibility.seed_numpy_global(seed * 100000 + i)
                    psi = simulator.sample_initial_state(cfg)
                    rhos = simulate_qutip.simulate_two_qubit(
                        system, psi, times)["density_matrices"]
                    lam = np.array([measures.wootters_lambda(r)
                                    for r in rhos])
                    t_esd, censored = simulator.sudden_death_time(times, lam)
                    after = (lam[times > t_esd] if not censored
                             else np.array([]))
                    row[f"c0{tag}"] = max(0.0, lam[0])
                    row[f"t_esd{tag}"] = t_esd
                    row[f"censored{tag}"] = censored
                    row[f"revival{tag}"] = bool((after > 1e-3).any())
                rows.append(row)
            print(f"R24b | seed={seed} {inter.value} J={coupling} done",
                  flush=True)
    common.write_results(
        rows, out_path, seeds=[seed],
        config="R24 (b): ESD statistics and the response of T_ESD to a "
               f"2% larger gamma(t), {n} configurations per (system, J)")


def add_flags(parser: argparse.ArgumentParser) -> None:
    """Adds the choice of part to the common flags."""
    parser.add_argument("--part", choices=["a", "b"], default="a")


def main() -> None:
    args = common.parse_args(__doc__, extra=add_flags)
    n_per, n_eval, epochs = (30, 20, 5) if args.fast else (500, 200, 150)
    if args.part == "b":
        part_b(args.seeds[0], 10 if args.fast else 100,
               common.results_path("r24b_esd_statistics", args))
        return
    common.run_arms(
        "R24", part_a(n_per, n_eval, epochs), args,
        common.results_path("r24_entanglement_target", args), COLUMNS)


if __name__ == "__main__":
    main()
