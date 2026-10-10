"""R27 — permanence of entanglement sudden death on a large sample.

Why
---
R26 classified, from a window that ends while the pair is still
entangled, whether entanglement will die before t_max (well predictable)
and, given that it dies, whether it stays dead until t_max. The second
question was not resolved: on XXZ only about one death in six is
permanent, which left 6-19 positive test cases per seed and t_obs, and
the Brier score barely beat the base rate. (On TFIM every death is
followed by revival, so the question does not arise.)

Same protocol as R26 (windows ending at fixed t_obs in {2, ..., 4},
scored separately at each t_obs, gradient-boosted trees and logistic
regression, three input sets), XXZ only, with 8000 training and 4000
independent test trajectories per seed, about ten times the test
positives of R26. For each AUROC a 95% interval from 1000 bootstrap
resamples of test trajectories is recorded. Occurrence is re-scored on
the same sample.

Writes ONLY to experiments/results/r27_esd_permanence*.csv.
Each CSV is accompanied by <name>.csv.provenance.json (configuration,
seeds, software environment).

Usage
-----
    python -m experiments.r27_esd_permanence --fast --tag smoke
    python -m experiments.r27_esd_permanence --seeds 42 --tag s42
"""
from __future__ import annotations

import time

import numpy as np

from experiments import common
from experiments.r26_esd_occurrence import (
    T_OBS, classifiers, esd_configs, score, simulate, windows)
from src.data import features
from src.evaluation import metrics, stats

N_BOOT = 1000
SCENARIO = "D"
COLUMNS = ("seed", "task", "features", "model", "t_obs", "base_rate",
           "n_windows", "auroc", "brier", "bal_acc", "n_pos", "auroc_lo",
           "auroc_hi")


def boot_ci(labels, probability, owner, rng):
    """95% interval of the AUROC from resampling test trajectories.

    At a fixed t_obs each trajectory contributes at most one window, so
    resampling windows is resampling trajectories.

    Args:
        labels: Binary labels of the windows.
        probability: Predicted probability of the positive class.
        owner: Index of the trajectory each window comes from.
        rng: Source of the resamples.

    Returns:
        (lower, upper) bounds, ignoring resamples with one class only.

    Raises:
        ValueError: If a trajectory has more than one window.
    """
    if len(np.unique(owner)) != len(owner):
        raise ValueError("windows must come from distinct trajectories")
    resamples = stats.resample_indices(len(labels), N_BOOT, rng)
    values = stats.bootstrap_values(
        metrics.auroc, resamples, labels, probability)
    return stats.percentile_interval(values, ignore_nan=True)


def run(seed, n_train, n_test, out):
    """Trains and scores both tasks for every input set on XXZ.

    Args:
        seed: Seed of the training configurations and states; the test
            configurations are drawn with seed + 1000 and the test states
            with seed + 1.
        n_train: Training trajectories.
        n_test: Independent test trajectories.
        out: `common.ResultsWriter` receiving the rows.
    """
    started = time.time()
    train, train_died, train_perm = simulate(
        esd_configs(SCENARIO, n_train, seed), seed)
    test, test_died, test_perm = simulate(
        esd_configs(SCENARIO, n_test, seed + 1000), seed + 1)
    print(f"R27 | seed={seed} simulated [{time.time() - started:.0f}s] "
          f"died={train_died.mean():.2f} "
          f"permanent|died={train_perm[train_died].mean():.2f} "
          f"test permanent={test_perm.sum()}", flush=True)
    rng = np.random.default_rng(seed)
    for feature_set in features.FEATURE_SETS:
        x_train, owner_train, _ = windows(train, feature_set)
        x_test, owner_test, t_test = windows(test, feature_set)
        tasks = {"occurrence": (train_died, test_died, None),
                 "permanence": (train_perm, test_perm,
                                (train_died, test_died))}
        for task, (label_train, label_test, condition) in tasks.items():
            if condition is None:
                mask_train = np.ones(len(owner_train), bool)
                mask_test = np.ones(len(owner_test), bool)
            else:
                mask_train = condition[0][owner_train]
                mask_test = condition[1][owner_test]
            y_train = label_train[owner_train][mask_train].astype(int)
            y_test = label_test[owner_test][mask_test].astype(int)
            for name, classifier in classifiers().items():
                classifier.fit(x_train[mask_train], y_train)
                p = classifier.predict_proba(x_test[mask_test])[:, 1]
                for t_obs in T_OBS:
                    k = t_test[mask_test] == t_obs
                    row = score(y_test[k], p[k])
                    low, high = boot_ci(
                        y_test[k], p[k], owner_test[mask_test][k], rng)
                    out.write({
                        "seed": seed, "task": task, "features": feature_set,
                        "model": name, "t_obs": t_obs, **row,
                        "n_pos": int(y_test[k].sum()),
                        "auroc_lo": low, "auroc_hi": high})
                    print(f"R27 | seed={seed} {task:<10} {feature_set:<9} "
                          f"{name:<6} t={t_obs} AUROC={row['auroc']:.3f} "
                          f"[{low:.3f},{high:.3f}] "
                          f"pos={int(y_test[k].sum())}", flush=True)


def main() -> None:
    args = common.parse_args(__doc__)
    n_train, n_test = (300, 200) if args.fast else (8000, 4000)
    path = common.results_path("r27_esd_permanence", args)
    with common.ResultsWriter(
            path, COLUMNS, seeds=args.seeds,
            config=f"R27: n_train={n_train} n_test={n_test} "
                   f"scenario={SCENARIO}") as out:
        for seed in args.seeds:
            run(seed, n_train, n_test, out)


if __name__ == "__main__":
    main()
