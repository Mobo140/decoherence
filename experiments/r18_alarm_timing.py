"""R18 — how early does the alarm fire, and how early is T2 predicted well?

Why
---
Both papers present an "early warning", but every alarm metric so far is
an average over windows whose end is drawn uniformly before T2, so it does
not say at which stage of the decay the prediction becomes reliable. This
probe follows each test trajectory step by step and reports accuracy as a
function of the elapsed fraction of its lifetime, p = t_obs / T2, together
with the time at which the alarm first fires.

Setup: models trained as in E1/R13 (100 trajectories per scenario, window
20, scenarios B and C, three seeds); 200 fresh test trajectories per
scenario from an independent generator seed. For each test trajectory a
window ends at every grid point from the first full window up to T2.

Per window: relative error of the predicted decoherence time,
|T2_hat - T2| / T2, binned in p.
Per trajectory: the alarm fires at the first t_obs where the predicted
remaining time is <= 1 (the score of Eq. alarm is then >= 0.5). With
lead = T2 - t_alarm:
    on time   0 < lead <= 2   (inside the horizon, with one unit of slack)
    early     lead > 2        (premature)
    missed    the alarm never fires before T2

Writes ONLY to experiments/results/r18_alarm_timing_bins.csv and
experiments/results/r18_alarm_timing_alarm.csv.

Usage
-----
    python -m experiments.r18_alarm_timing --fast
    python -m experiments.r18_alarm_timing --seeds 42 43 44
"""
from __future__ import annotations

import argparse
import csv
import sys
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

RES = Path(__file__).parent / "results"
BINS = [(0.0, 0.4), (0.4, 0.5), (0.5, 0.6), (0.6, 0.7), (0.7, 0.8), (0.8, 0.9), (0.9, 1.0)]
MODELS = {
    "physics_only": lambda: PhysicsOnlyPredictor(t_max=20.0),
    "stretched_exp": lambda: StretchedExpPredictor(t_max=20.0, dt=0.1),
    "lstm_only": lambda: LSTMPredictor(use_physics_prior=False),
    "physics_lstm": lambda: LSTMPredictor(use_physics_prior=True),
    "transformer": lambda: TransformerPredictor(),
}


def _predict(pred, ds, idx):
    out = np.empty(len(idx))
    for k, i in enumerate(idx):
        for attr, arr in (("inference_interaction_code", ds.interaction_type_codes),
                          ("inference_dissipator_code", ds.dissipator_type_codes)):
            if hasattr(pred, attr) and len(arr) == len(ds.sequences):
                setattr(pred, attr, float(arr[i]))
        out[k] = pred.predict(ds.sequences[i], float(ds.t_obs[i]), 1.0).t_decoh_predicted
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    p.add_argument("--n-eval", type=int, default=200)
    p.add_argument("--fast", action="store_true")
    p.add_argument("--tag", type=str, default="", help="suffix for the output files")
    a = p.parse_args()
    n_train, n_eval, epochs = (30, 20, 5) if a.fast else (100, a.n_eval, 100)
    gen = GenerateDatasetUseCase(QuTipSimulator(), store=trajectory_store())
    fb = (RES / f"r18_alarm_timing_bins{a.tag}.csv").open("w", newline="")
    fa = (RES / f"r18_alarm_timing_alarm{a.tag}.csv").open("w", newline="")
    wb = csv.DictWriter(fb, fieldnames=["seed", "scenario", "model", "p_lo", "p_hi",
                                        "n_windows", "median_rel_err", "frac_within_10pct"])
    wa = csv.DictWriter(fa, fieldnames=["seed", "scenario", "model", "n_traj", "on_time",
                                        "early", "missed", "median_lead", "median_alarm_p",
                                        "median_first_p"])
    wb.writeheader(); wa.writeheader()
    for seed in a.seeds:
        train_g = build_configs(n_per_scenario=n_train, seed=seed)
        eval_g = build_configs(n_per_scenario=n_eval, seed=seed + 2000)
        for sc in "BC":
            train_ds = gen.execute(GenerateDatasetCommand(
                configs=train_g[sc], window_length=20, horizon=1.0, samples_per_trajectory=5,
                seed=seed, dataset_name=f"r18_train_{seed}_{sc}"))
            # every admissible window end of every evaluation trajectory
            ev = gen.execute(GenerateDatasetCommand(
                configs=eval_g[sc], window_length=20, horizon=1.0, samples_per_trajectory=10_000,
                seed=seed + 2000, train_ratio=0.0, val_ratio=0.0,
                dataset_name=f"r18_eval_{seed}_{sc}"))
            idx = ev.test_idx
            T2 = ev.t_decoh_abs[idx].astype(float)
            tobs = ev.t_obs[idx].astype(float)
            _, traj = np.unique(T2, return_inverse=True)
            prog = tobs / T2
            for name, build in MODELS.items():
                pred = build()
                if name not in ("physics_only", "stretched_exp"):
                    TrainModelUseCase(pred).execute(TrainModelCommand(
                        dataset=train_ds, n_epochs=epochs, batch_size=32, learning_rate=1e-3,
                        regression_loss="huber", verbose=False, seed=seed))
                T2_hat = _predict(pred, ev, idx)
                rel = np.abs(T2_hat - T2) / T2
                for lo, hi in BINS:
                    m = (prog >= lo) & (prog < hi)
                    if m.sum() == 0:
                        continue
                    wb.writerow({"seed": seed, "scenario": sc, "model": name, "p_lo": lo,
                                 "p_hi": hi, "n_windows": int(m.sum()),
                                 "median_rel_err": float(np.median(rel[m])),
                                 "frac_within_10pct": float(np.mean(rel[m] < 0.10))})
                rem_hat = T2_hat - tobs
                on = early = missed = 0
                leads, alarm_p, first_p = [], [], []
                for k in range(traj.max() + 1):
                    sel = np.where(traj == k)[0]
                    sel = sel[np.argsort(tobs[sel])]
                    first_p.append(prog[sel[0]])
                    fired = sel[rem_hat[sel] <= 1.0]
                    if fired.size == 0:
                        missed += 1
                        continue
                    j = fired[0]
                    lead = T2[j] - tobs[j]
                    leads.append(lead); alarm_p.append(prog[j])
                    if lead > 2.0:
                        early += 1
                    else:
                        on += 1
                n = traj.max() + 1
                row = {"seed": seed, "scenario": sc, "model": name, "n_traj": int(n),
                       "on_time": on / n, "early": early / n, "missed": missed / n,
                       "median_lead": float(np.median(leads)) if leads else float("nan"),
                       "median_alarm_p": float(np.median(alarm_p)) if alarm_p else float("nan"),
                       "median_first_p": float(np.median(first_p))}
                wa.writerow(row); fa.flush(); fb.flush()
                print(f"R18 | seed={seed} {sc} {name:<13} on_time={row['on_time']:.2f} "
                      f"early={row['early']:.2f} missed={row['missed']:.2f} "
                      f"lead={row['median_lead']:.2f} first_p={row['median_first_p']:.2f}", flush=True)
    fb.close(); fa.close()


if __name__ == "__main__":
    main()
