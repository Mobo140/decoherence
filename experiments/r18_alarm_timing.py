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

Writes ONLY to experiments/results/r18_alarm_timing_bins[_<tag>][_fast].csv
and experiments/results/r18_alarm_timing_alarm[_<tag>][_fast].csv.

Usage
-----
    python -m experiments.r18_alarm_timing --fast
    python -m experiments.r18_alarm_timing --seeds 42 43 44
"""
from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from experiments import common
from src.application import generate_dataset
from src.application import train_model
from src.data import scenarios as scenario_lib
from src.evaluation import backtest
from src.models import registry
from src.simulation import simulator as simulator_lib

BIN_COLUMNS = (
    "seed", "scenario", "model", "p_lo", "p_hi", "n_windows",
    "median_rel_err", "frac_within_10pct")
ALARM_COLUMNS = (
    "seed", "scenario", "model", "n_traj", "on_time", "early", "missed",
    "median_lead", "median_alarm_p", "median_first_p")
BINS = ((0.0, 0.4), (0.4, 0.5), (0.5, 0.6), (0.6, 0.7), (0.7, 0.8),
        (0.8, 0.9), (0.9, 1.0))
MODELS = ("physics_only", "stretched_exp", "lstm_only", "physics_lstm",
          "transformer")
# The alarm fires when the predicted remaining time is at most this.
ALARM_REMAINING = 1.0
# Lead times above this count as a premature alarm.
EARLY_LEAD = 2.0


def _bin_rows(seed, scenario, model, progress, rel_err):
    """Yields the accuracy per bin of elapsed lifetime fraction."""
    for low, high in BINS:
        in_bin = (progress >= low) & (progress < high)
        if in_bin.sum() == 0:
            continue
        yield {
            "seed": seed, "scenario": scenario, "model": model,
            "p_lo": low, "p_hi": high, "n_windows": int(in_bin.sum()),
            "median_rel_err": float(np.median(rel_err[in_bin])),
            "frac_within_10pct": float(np.mean(rel_err[in_bin] < 0.10)),
        }


def _alarm_row(seed, scenario, model, trajectory, t2, t_obs, progress,
               remaining_hat):
    """Summarises when the alarm first fires on each test trajectory."""
    on_time = early = missed = 0
    leads, alarm_progress, first_progress = [], [], []
    for k in range(trajectory.max() + 1):
        windows = np.where(trajectory == k)[0]
        windows = windows[np.argsort(t_obs[windows])]
        first_progress.append(progress[windows[0]])
        fired = windows[remaining_hat[windows] <= ALARM_REMAINING]
        if fired.size == 0:
            missed += 1
            continue
        j = fired[0]
        lead = t2[j] - t_obs[j]
        leads.append(lead)
        alarm_progress.append(progress[j])
        if lead > EARLY_LEAD:
            early += 1
        else:
            on_time += 1
    n = trajectory.max() + 1
    return {
        "seed": seed, "scenario": scenario, "model": model,
        "n_traj": int(n),
        "on_time": on_time / n, "early": early / n, "missed": missed / n,
        "median_lead": float(np.median(leads)) if leads else float("nan"),
        "median_alarm_p": (
            float(np.median(alarm_progress)) if alarm_progress
            else float("nan")),
        "median_first_p": float(np.median(first_progress)),
    }


def _run_cell(generator, seed, scenario, train_groups, eval_groups, epochs,
              bins_out, alarm_out) -> None:
    """Trains and scores every model on one seed and scenario."""
    train_ds = generator.execute(generate_dataset.GenerateDatasetCommand(
        configs=train_groups[scenario], window_length=20, horizon=1.0,
        samples_per_trajectory=5, seed=seed,
        dataset_name=f"r18_train_{seed}_{scenario}"))
    # Every admissible window end of every evaluation trajectory.
    ev = generator.execute(generate_dataset.GenerateDatasetCommand(
        configs=eval_groups[scenario], window_length=20, horizon=1.0,
        samples_per_trajectory=10_000, seed=seed + 2000,
        train_ratio=0.0, val_ratio=0.0,
        dataset_name=f"r18_eval_{seed}_{scenario}"))
    idx = ev.test_idx
    t2 = ev.t_decoh_abs[idx].astype(float)
    t_obs = ev.t_obs[idx].astype(float)
    _, trajectory = np.unique(t2, return_inverse=True)
    progress = t_obs / t2
    for name in MODELS:
        predictor = registry.build_model(name)
        if name not in ("physics_only", "stretched_exp"):
            train_model.TrainModelUseCase(predictor).execute(
                train_model.TrainModelCommand(
                    dataset=train_ds, n_epochs=epochs, batch_size=32,
                    learning_rate=1e-3, regression_loss="huber",
                    verbose=False, seed=seed))
        t2_hat, _ = backtest.predict_windows(predictor, ev, idx)
        rel_err = np.abs(t2_hat - t2) / t2
        for row in _bin_rows(seed, scenario, name, progress, rel_err):
            bins_out.write(row)
        row = _alarm_row(seed, scenario, name, trajectory, t2, t_obs,
                         progress, t2_hat - t_obs)
        alarm_out.write(row)
        print(f"R18 | seed={seed} {scenario} {name:<13} "
              f"on_time={row['on_time']:.2f} early={row['early']:.2f} "
              f"missed={row['missed']:.2f} lead={row['median_lead']:.2f} "
              f"first_p={row['median_first_p']:.2f}", flush=True)


def _add_flags(parser) -> None:
    parser.add_argument("--n-eval", type=int, default=200)


def main(argv: Sequence[str] | None = None) -> None:
    """Scores accuracy and alarm timing and writes the two CSVs."""
    args = common.parse_args(
        __doc__, extra=_add_flags, argv=argv, out=False)
    n_train, n_eval, epochs = (
        (30, 20, 5) if args.fast else (100, args.n_eval, 100))
    config = {"n_train": n_train, "n_eval": n_eval, "epochs": epochs,
              "models": MODELS}
    generator = generate_dataset.GenerateDatasetUseCase(
        simulator_lib.QuTipSimulator(),
        store=scenario_lib.trajectory_store())
    bins_path = common.results_path("r18_alarm_timing_bins", args)
    alarm_path = common.results_path("r18_alarm_timing_alarm", args)
    with common.ResultsWriter(
            bins_path, BIN_COLUMNS, config=config, seeds=args.seeds
    ) as bins_out, common.ResultsWriter(
            alarm_path, ALARM_COLUMNS, config=config, seeds=args.seeds
    ) as alarm_out:
        for seed in args.seeds:
            train_groups = scenario_lib.build_configs(
                n_per_scenario=n_train, seed=seed)
            eval_groups = scenario_lib.build_configs(
                n_per_scenario=n_eval, seed=seed + 2000)
            for scenario in "BC":
                _run_cell(generator, seed, scenario, train_groups,
                          eval_groups, epochs, bins_out, alarm_out)


if __name__ == "__main__":
    main()
