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

Usage
-----
    python -m experiments.r23_two_body_correlators --fast
    python -m experiments.r23_two_body_correlators --part a --seeds 42 --tag a42
"""
from __future__ import annotations

import argparse
import copy
import csv
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from experiments._config import build_configs
from experiments.e7_paper2_extended import _make_tfim_configs_fixed_J
from src.application.backtest import BacktestCommand, BacktestUseCase
from src.application.generate_dataset import GenerateDatasetCommand, GenerateDatasetUseCase
from src.application.train_model import TrainModelCommand, TrainModelUseCase
from src.infrastructure.ml.lstm_predictor import LSTMPredictor
from src.infrastructure.ml.transformer_predictor import TransformerPredictor
from src.infrastructure.quantum.qutip_simulator import QuTipSimulator

RESULTS_DIR = Path(__file__).parent / "results"
PAULI = {
    "x": np.array([[0, 1], [1, 0]], dtype=complex),
    "y": np.array([[0, -1j], [1j, 0]], dtype=complex),
    "z": np.array([[1, 0], [0, -1]], dtype=complex),
}
CORR = {f"corr_{a}{b}": np.kron(PAULI[a], PAULI[b]) for a in "xyz" for b in "xyz"}
J_VALUES = [0.2, 0.5, 0.8, 1.0, 1.2, 1.5, 2.0]


class CorrelatorSimulator(QuTipSimulator):
    """QuTiP simulator that also records <sigma_a x sigma_b> for all a, b."""

    def _extra_observables(self, density_matrices) -> dict:
        rhos = np.asarray(density_matrices)
        return {name: np.real(np.einsum("tij,ji->t", rhos, op)) for name, op in CORR.items()}


def base_features(ds):
    """The same dataset without the correlator columns."""
    keep = [i for i, n in enumerate(ds.feature_names) if n not in CORR]
    out = copy.copy(ds)
    out.sequences = np.ascontiguousarray(ds.sequences[:, :, keep])
    out.feature_names = [ds.feature_names[i] for i in keep]
    return out


def _all_test(ds):
    out = copy.copy(ds)
    out.test_idx = np.arange(len(ds.sequences))
    out.train_idx = np.array([], dtype=np.int64)
    out.val_idx = np.array([], dtype=np.int64)
    return out


def _transformer():
    return TransformerPredictor(d_model=128, n_heads=4, n_layers=3, dim_ff=512, dropout=0.1)


def _train(pred, ds, epochs, seed):
    TrainModelUseCase(pred).execute(TrainModelCommand(
        dataset=ds, n_epochs=epochs, batch_size=64, learning_rate=5e-4,
        regression_loss="huber", verbose=False, seed=seed))
    return pred


MODELS = {"transformer": _transformer,
          "lstm_only": lambda: LSTMPredictor(use_physics_prior=False)}


def part_a(seed, n_per, n_eval, epochs, w):
    gen = GenerateDatasetUseCase(CorrelatorSimulator(), store=None)
    groups = build_configs(n_per_scenario=n_per, seed=seed)
    eval_groups = build_configs(n_per_scenario=n_eval, seed=seed + 1000)
    for sc in ("D", "E"):
        ts = time.time()
        full = gen.execute(GenerateDatasetCommand(
            configs=groups[sc], window_length=20, horizon=1.0,
            samples_per_trajectory=5, seed=seed, dataset_name=f"r23a_{seed}_{sc}"))
        full_ev = _all_test(gen.execute(GenerateDatasetCommand(
            configs=eval_groups[sc], window_length=20, horizon=1.0,
            samples_per_trajectory=5, seed=seed + 1, train_ratio=0.0, val_ratio=0.0,
            dataset_name=f"r23a_eval_{seed}_{sc}")))
        print(f"R23a | seed={seed} {sc} data [{time.time() - ts:.0f}s]", flush=True)
        for feats, (tr, ev) in {"base": (base_features(full), base_features(full_ev)),
                                "corr": (full, full_ev)}.items():
            for model, build in MODELS.items():
                pred = _train(build(), tr, epochs, seed)
                for split, ds in (("own", tr), ("independent", ev)):
                    m = BacktestUseCase(pred).execute(BacktestCommand(dataset=ds, horizon=1.0))
                    w.writerow({"part": "a", "seed": seed, "scenario": sc, "J": "",
                                "model": model, "features": feats, "test": split,
                                "r2": m.r2, "mae": m.mae, "auroc": m.risk_auroc,
                                "n_samples": m.n_samples})
                    print(f"R23a | seed={seed} {sc} {model:<11} {feats} {split:<11} "
                          f"R2={m.r2:.3f} AUROC={m.risk_auroc:.3f}", flush=True)
                w.flush_file()


def part_b(seed, n_per_J, epochs, w):
    gen = GenerateDatasetUseCase(CorrelatorSimulator(), store=None)
    rng = np.random.default_rng(seed)
    for J in J_VALUES:
        ts = time.time()
        full = gen.execute(GenerateDatasetCommand(
            configs=_make_tfim_configs_fixed_J(n_per_J, J, rng), window_length=20,
            horizon=1.0, samples_per_trajectory=5, seed=seed,
            dataset_name=f"r23b_{seed}_{J}"))
        for feats, ds in (("base", base_features(full)), ("corr", full)):
            pred = _train(_transformer(), ds, epochs, seed)
            m = BacktestUseCase(pred).execute(BacktestCommand(dataset=ds, horizon=1.0))
            w.writerow({"part": "b", "seed": seed, "scenario": "E", "J": J,
                        "model": "transformer", "features": feats, "test": "own",
                        "r2": m.r2, "mae": m.mae, "auroc": m.risk_auroc,
                        "n_samples": m.n_samples})
            print(f"R23b | seed={seed} J={J:.1f} {feats} R2={m.r2:.3f} "
                  f"[{time.time() - ts:.0f}s]", flush=True)
        w.flush_file()


def part_c(seed, n, out_path):
    import dataclasses
    from src.domain.value_objects import DecoherenceCriterion, InteractionType
    sim = QuTipSimulator()
    rows = []
    for inter in (InteractionType.TFIM, InteractionType.XXZ):
        rng = np.random.default_rng(seed)
        for J in (0.2, 0.8, 1.5):
            for i, cfg in enumerate(_make_tfim_configs_fixed_J(n, J, rng, interaction=inter)):
                scaled = dataclasses.replace(cfg, dissipator=dataclasses.replace(
                    cfg.dissipator, gamma_coeffs=tuple(1.02 * c for c in cfg.dissipator.gamma_coeffs)))
                row = {"seed": seed, "system": inter.value, "J": J, "index": i}
                for crit, tag in ((DecoherenceCriterion.COHERENCE, "t2"),
                                  (DecoherenceCriterion.COHERENCE_ENERGY, "t2e")):
                    out = []
                    for c in (cfg, scaled):
                        np.random.seed(seed * 100000 + i)
                        traj = sim.simulate(dataclasses.replace(c, decoherence_criterion=crit))
                        out.append((traj.t_decoh, traj.censored))
                    row[tag], row[f"{tag}_scaled"] = out[0][0], out[1][0]
                    row[f"{tag}_censored"] = out[0][1] or out[1][1]
                rows.append(row)
            print(f"R23c | seed={seed} {inter.value} J={J} done", flush=True)
    with out_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


class _Writer(csv.DictWriter):
    def __init__(self, f):
        super().__init__(f, fieldnames=["part", "seed", "scenario", "J", "model",
                                        "features", "test", "r2", "mae", "auroc",
                                        "n_samples"])
        self._f = f

    def flush_file(self):
        self._f.flush()


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    p.add_argument("--part", choices=["a", "b", "ab", "c"], default="ab")
    p.add_argument("--fast", action="store_true")
    p.add_argument("--tag", type=str, default="")
    a = p.parse_args()
    n_per, n_eval, n_per_J, epochs = (30, 20, 20, 5) if a.fast else (500, 200, 200, 150)
    sfx = (f"_{a.tag}" if a.tag else "") + ("_fast" if a.fast else "")
    out = RESULTS_DIR / f"r23_two_body_correlators{sfx}.csv"
    print(f"R23 | seeds={a.seeds} part={a.part} epochs={epochs}", flush=True)
    if a.part == "c":
        part_c(a.seeds[0], 10 if a.fast else 100,
               RESULTS_DIR / f"r23c_target_sensitivity{sfx}.csv")
        return
    with out.open("w", newline="") as f:
        w = _Writer(f)
        w.writeheader()
        for seed in a.seeds:
            ts = time.time()
            if "a" in a.part:
                part_a(seed, n_per, n_eval, epochs, w)
            if "b" in a.part:
                part_b(seed, n_per_J, epochs, w)
            print(f"R23 | seed={seed} done [{time.time() - ts:.0f}s]", flush=True)


if __name__ == "__main__":
    main()
