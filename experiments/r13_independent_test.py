"""R13 — Paper 1 ablation scored on a large independent test set.

Why
---
E1/R7 score each model on the 10 test trajectories of a 100-trajectory
run; there even predicting the training mean gives R^2 between -2.0 and
0.0, so comparisons of a few hundredths are not resolved. This probe keeps
the E1 training setup and scores every model on 400 fresh trajectories per
scenario drawn from an independent generator seed, with 95% bootstrap
intervals obtained by resampling trajectories (not windows).

It also adds the missing ablation `physics_feature`: the same BiLSTM that
receives T_phys as an input scalar but predicts the whole remaining time
(no residual output). Comparing it with physics_lstm separates "T_phys is
a useful feature" from "the formula + correction output structure".

Writes ONLY to experiments/results/r13_independent_test.csv.

Usage
-----
    python -m experiments.r13_independent_test --fast
    python -m experiments.r13_independent_test --seeds 42 43 44
"""
from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from experiments._config import build_configs, trajectory_store
from src.application.generate_dataset import GenerateDatasetCommand, GenerateDatasetUseCase
from src.application.train_model import TrainModelCommand, TrainModelUseCase
from src.infrastructure.ml.lstm_predictor import LSTMPredictor
from src.infrastructure.ml.physics_only_predictor import PhysicsOnlyPredictor
from src.infrastructure.ml.stretched_exp_predictor import StretchedExpPredictor
from src.infrastructure.ml.transformer_predictor import TransformerPredictor
from src.infrastructure.quantum.qutip_simulator import QuTipSimulator

RESULTS = Path(__file__).parent / "results" / "r13_independent_test.csv"
VARIANTS = ["physics_only", "stretched_exp", "lstm_only", "physics_feature",
            "physics_lstm", "transformer"]


def _build(name: str):
    if name == "physics_only":
        return PhysicsOnlyPredictor(t_max=20.0)
    if name == "stretched_exp":
        return StretchedExpPredictor(t_max=20.0, dt=0.1)
    if name == "lstm_only":
        return LSTMPredictor(use_physics_prior=False)
    if name == "physics_feature":
        return LSTMPredictor(use_physics_prior=True, residual_output=False)
    if name == "physics_lstm":
        return LSTMPredictor(use_physics_prior=True)
    if name == "transformer":
        return TransformerPredictor()
    raise ValueError(name)


def _r2(y, p):
    return 1.0 - np.sum((y - p) ** 2) / np.sum((y - y.mean()) ** 2)


def _auroc(labels, scores):
    pos, neg = scores[labels > 0.5], scores[labels <= 0.5]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    diff = pos[:, None] - neg[None, :]
    return float((diff > 0).mean() + 0.5 * (diff == 0).mean())


def _predict(pred, ds):
    out_t, out_r = [], []
    for i in range(len(ds.sequences)):
        for attr, arr in (("inference_interaction_code", ds.interaction_type_codes),
                          ("inference_dissipator_code", ds.dissipator_type_codes)):
            if hasattr(pred, attr) and len(arr) == len(ds.sequences):
                setattr(pred, attr, float(arr[i]))
        r = pred.predict(ds.sequences[i], float(ds.t_obs[i]), 1.0)
        out_t.append(r.t_decoh_predicted)
        out_r.append(r.risk_score)
    return np.array(out_t), np.array(out_r)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    p.add_argument("--n-train", type=int, default=100)
    p.add_argument("--n-eval", type=int, default=400)
    p.add_argument("--n-boot", type=int, default=1000)
    p.add_argument("--fast", action="store_true")
    p.add_argument("--out", type=str, default=str(RESULTS))
    a = p.parse_args()
    if a.fast:
        a.n_train, a.n_eval, a.n_boot = 30, 30, 50
    epochs = 10 if a.fast else 100

    sim, store = QuTipSimulator(), trajectory_store()
    gen = GenerateDatasetUseCase(sim, store=store)
    fields = ["seed", "scenario", "variant", "r2", "r2_lo", "r2_hi", "mae",
              "auroc", "n_eval_traj", "n_eval_windows"]
    with open(a.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for seed in a.seeds:
            t0 = time.time()
            train_groups = build_configs(n_per_scenario=a.n_train, seed=seed)
            eval_groups = build_configs(n_per_scenario=a.n_eval, seed=seed + 1000)
            for sc in "ABC":
                train_ds = gen.execute(GenerateDatasetCommand(
                    configs=train_groups[sc], window_length=20, horizon=1.0,
                    samples_per_trajectory=5, seed=seed,
                    dataset_name=f"r13_train_{seed}_{sc}"))
                eval_ds = gen.execute(GenerateDatasetCommand(
                    configs=eval_groups[sc], window_length=20, horizon=1.0,
                    samples_per_trajectory=5, seed=seed + 1000,
                    train_ratio=0.0, val_ratio=0.0,
                    dataset_name=f"r13_eval_{seed}_{sc}"))
                y, lab = eval_ds.t_decoh_abs.astype(float), eval_ds.risk_labels
                # windows of one trajectory share T2: group by it for the bootstrap
                _, traj = np.unique(eval_ds.t_decoh_abs, return_inverse=True)
                n_traj = traj.max() + 1
                members = [np.where(traj == k)[0] for k in range(n_traj)]
                rng = np.random.default_rng(seed)
                boots = [np.concatenate([members[k] for k in rng.integers(0, n_traj, n_traj)])
                         for _ in range(a.n_boot)]
                for name in VARIANTS:
                    pred = _build(name)
                    if name not in ("physics_only", "stretched_exp"):
                        TrainModelUseCase(pred).execute(TrainModelCommand(
                            dataset=train_ds, n_epochs=epochs, batch_size=32,
                            learning_rate=1e-3, regression_loss="huber",
                            verbose=False, seed=seed))
                    pt, pr = _predict(pred, eval_ds)
                    bs = np.array([_r2(y[b], pt[b]) for b in boots])
                    row = {"seed": seed, "scenario": sc, "variant": name,
                           "r2": _r2(y, pt), "r2_lo": np.percentile(bs, 2.5),
                           "r2_hi": np.percentile(bs, 97.5),
                           "mae": float(np.mean(np.abs(y - pt))),
                           "auroc": _auroc(lab, pr), "n_eval_traj": int(n_traj),
                           "n_eval_windows": len(y)}
                    w.writerow(row)
                    f.flush()
                    print(f"R13 | seed={seed} {sc} {name:<16} R2={row['r2']:.3f} "
                          f"[{row['r2_lo']:.3f}, {row['r2_hi']:.3f}] "
                          f"MAE={row['mae']:.3f} AUROC={row['auroc']:.3f}", flush=True)
            print(f"R13 | seed {seed} done [{time.time() - t0:.0f}s]", flush=True)


if __name__ == "__main__":
    main()
