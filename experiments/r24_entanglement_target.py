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

Usage
-----
    python -m experiments.r24_entanglement_target --fast
    python -m experiments.r24_entanglement_target --part a --seeds 42 --tag a42
"""
from __future__ import annotations

import argparse
import copy
import csv
import dataclasses
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from experiments._config import build_configs
from experiments.e7_paper2_extended import _make_tfim_configs_fixed_J
from experiments.r23_two_body_correlators import CORR, MODELS, CorrelatorSimulator, _all_test, _train
from src.application.backtest import BacktestCommand, BacktestUseCase
from src.application.generate_dataset import GenerateDatasetCommand, GenerateDatasetUseCase
from src.domain.value_objects import DecoherenceCriterion, InteractionType
from src.infrastructure.quantum.qutip_simulator import QuTipSimulator, concurrence, wootters_lambda
from src.physics.lindblad_systems import TwoQubitSystem
from src.physics.simulate_qutip import simulate_two_qubit

RESULTS_DIR = Path(__file__).parent / "results"
ESD = DecoherenceCriterion.ENTANGLEMENT_DEATH


class EntanglementSimulator(CorrelatorSimulator):
    """Records the two-body correlators and the concurrence."""

    def _extra_observables(self, density_matrices) -> dict:
        out = super()._extra_observables(density_matrices)
        out["concurrence"] = np.array([concurrence(r) for r in density_matrices])
        return out


def feature_set(ds, name):
    drop = {"base": set(CORR) | {"concurrence"}, "corr": {"concurrence"}, "corr_conc": set()}[name]
    keep = [i for i, n in enumerate(ds.feature_names) if n not in drop]
    out = copy.copy(ds)
    out.sequences = np.ascontiguousarray(ds.sequences[:, :, keep])
    out.feature_names = [ds.feature_names[i] for i in keep]
    return out


def _esd(groups):
    return {sc: [dataclasses.replace(c, decoherence_criterion=ESD) for c in groups[sc]]
            for sc in ("D", "E")}


def part_a(seed, n_per, n_eval, epochs, w, f):
    gen = GenerateDatasetUseCase(EntanglementSimulator(), store=None)
    groups = _esd(build_configs(n_per_scenario=n_per, seed=seed))
    eval_groups = _esd(build_configs(n_per_scenario=n_eval, seed=seed + 1000))
    for sc in ("D", "E"):
        ts = time.time()
        full = gen.execute(GenerateDatasetCommand(
            configs=groups[sc], window_length=20, horizon=1.0,
            samples_per_trajectory=5, seed=seed, dataset_name=f"r24a_{seed}_{sc}"))
        full_ev = _all_test(gen.execute(GenerateDatasetCommand(
            configs=eval_groups[sc], window_length=20, horizon=1.0,
            samples_per_trajectory=5, seed=seed + 1, train_ratio=0.0, val_ratio=0.0,
            dataset_name=f"r24a_eval_{seed}_{sc}")))
        print(f"R24a | seed={seed} {sc} data [{time.time() - ts:.0f}s] "
              f"censored={full.metadata.get('n_censored')} windows={len(full.sequences)}",
              flush=True)
        for feats in ("base", "corr", "corr_conc"):
            tr, ev = feature_set(full, feats), feature_set(full_ev, feats)
            for model, build in MODELS.items():
                pred = _train(build(), tr, epochs, seed)
                for split, ds in (("own", tr), ("independent", ev)):
                    m = BacktestUseCase(pred).execute(BacktestCommand(dataset=ds, horizon=1.0))
                    w.writerow({"seed": seed, "scenario": sc, "model": model,
                                "features": feats, "test": split, "r2": m.r2, "mae": m.mae,
                                "auroc": m.risk_auroc, "n_samples": m.n_samples,
                                "n_censored": full.metadata.get("n_censored")})
                    print(f"R24a | seed={seed} {sc} {model:<11} {feats:<9} {split:<11} "
                          f"R2={m.r2:.3f} AUROC={m.risk_auroc:.3f}", flush=True)
                f.flush()


def part_b(seed, n, out_path):
    times = None
    rows = []
    for inter in (InteractionType.XXZ, InteractionType.TFIM):
        rng = np.random.default_rng(seed)
        for J in (0.2, 0.8, 1.5):
            for i, cfg in enumerate(_make_tfim_configs_fixed_J(n, J, rng, interaction=inter)):
                times = np.linspace(0.0, cfg.t_max, int(round(cfg.t_max / cfg.dt)) + 1)
                row = {"seed": seed, "system": inter.value, "J": J, "index": i}
                for tag, scale in (("", 1.0), ("_scaled", 1.02)):
                    sys_ = TwoQubitSystem(
                        J=cfg.J, interaction_type=cfg.interaction_type.value,
                        gamma_coeffs=scale * np.array(cfg.dissipator.gamma_coeffs),
                        gamma_t_max=cfg.t_max,
                        jump_operators=[cfg.dissipator.operator_type.value])
                    np.random.seed(seed * 100000 + i)
                    psi = QuTipSimulator()._initial_state(cfg)
                    rhos = simulate_two_qubit(sys_, psi, times)["density_matrices"]
                    lam = np.array([wootters_lambda(r) for r in rhos])
                    t_esd, cens = QuTipSimulator._sudden_death_time(times, lam)
                    after = lam[times > t_esd] if not cens else np.array([])
                    row[f"c0{tag}"] = max(0.0, lam[0])
                    row[f"t_esd{tag}"] = t_esd
                    row[f"censored{tag}"] = cens
                    row[f"revival{tag}"] = bool((after > 1e-3).any())
                rows.append(row)
            print(f"R24b | seed={seed} {inter.value} J={J} done", flush=True)
    with out_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    p.add_argument("--part", choices=["a", "b"], default="a")
    p.add_argument("--fast", action="store_true")
    p.add_argument("--tag", type=str, default="")
    a = p.parse_args()
    n_per, n_eval, epochs = (30, 20, 5) if a.fast else (500, 200, 150)
    sfx = (f"_{a.tag}" if a.tag else "") + ("_fast" if a.fast else "")
    print(f"R24 | seeds={a.seeds} part={a.part} epochs={epochs}", flush=True)
    if a.part == "b":
        part_b(a.seeds[0], 10 if a.fast else 100, RESULTS_DIR / f"r24b_esd_statistics{sfx}.csv")
        return
    out = RESULTS_DIR / f"r24_entanglement_target{sfx}.csv"
    with out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["seed", "scenario", "model", "features", "test",
                                          "r2", "mae", "auroc", "n_samples", "n_censored"])
        w.writeheader()
        for seed in a.seeds:
            ts = time.time()
            part_a(seed, n_per, n_eval, epochs, w, f)
            print(f"R24 | seed={seed} done [{time.time() - ts:.0f}s]", flush=True)


if __name__ == "__main__":
    main()
