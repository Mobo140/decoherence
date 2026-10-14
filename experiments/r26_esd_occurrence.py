"""R26 — will entanglement die, and will it stay dead?

Why
---
R24/R25 predict the time of entanglement sudden death given that it
happens, and drop trajectories still entangled at t_max (about half on
XXZ, a fifth on TFIM). R24 also found that entanglement revives after the
first death in 68-91% of XXZ and all TFIM trajectories. Two questions were
therefore left out, and both are classification problems:

  (a) occurrence: from a window that ends while the pair is still
      entangled, will the concurrence reach zero before t_max?
  (b) permanence: among trajectories that do die, will the concurrence stay
      at zero until t_max? Only XXZ has both outcomes (TFIM always revives).

Setup: scenarios D (XXZ) and E (TFIM) with the ESD target of R24; 2000
training and 400 independent test trajectories per scenario (generator
seeds seed and seed + 1000). Windows of 20 steps end at fixed times
t_obs in {2, 2.5, 3, 3.5, 4}, one per trajectory still entangled at that
time; the label is about what happens after t_obs. Fixing t_obs matters:
if windows of trajectories that never die could end anywhere while those
of dying ones must end before death, the classifier could read the label
off the window's position in time. The classifier is trained on all
t_obs pooled and scored separately at each t_obs, so a base rate that
changes with t_obs cannot inflate the AUROC. Classifiers: gradient-boosted
trees and logistic regression on the flattened window, with the three
input sets of R24 (base / + correlators / + correlators + concurrence). Metrics: AUROC,
Brier score and balanced accuracy at 0.5, with the class balance. Three
seeds.

Writes ONLY to experiments/results/r26_esd_occurrence*.csv.
Each CSV is accompanied by <name>.csv.provenance.json (configuration,
seeds, software environment).

Usage
-----
    python -m experiments.r26_esd_occurrence --fast --tag smoke
    python -m experiments.r26_esd_occurrence --seeds 42 --scenario D --tag D42
"""
from __future__ import annotations

import argparse
import time

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from experiments import common
from src import pipeline
from src import reproducibility
from src.data import features
from src.domain import value_objects
from src.evaluation import metrics
from src.simulation import variants

ESD = value_objects.DecoherenceCriterion.ENTANGLEMENT_DEATH.value
WINDOW = 20
T_OBS = (2.0, 2.5, 3.0, 3.5, 4.0)
# A concurrence above this after the first death counts as a revival.
REVIVAL_EPS = 1e-3
COLUMNS = ("seed", "scenario", "task", "features", "model", "t_obs",
           "base_rate", "n_windows", "auroc", "brier", "bal_acc")


def esd_configs(scenario: str, n: int, seed: int):
    """Draws the configurations of a scenario with the ESD target.

    Args:
        scenario: "D" or "E".
        n: Number of configurations.
        seed: Seed of the generator that draws them.

    Returns:
        The system configurations.
    """
    spec = pipeline.DataSpec(
        name="esd", scenarios=(scenario,), n_per_scenario=n, criterion=ESD)
    return pipeline.system_configs(spec, seed)


def simulate(configs, seed):
    """Simulates the configurations with the entanglement simulator.

    Args:
        configs: System configurations.
        seed: Seed of NumPy's global generator, which draws the initial
            states.

    Returns:
        The trajectories, and per trajectory whether entanglement died
        before t_max and whether it then stayed dead.
    """
    reproducibility.seed_numpy_global(seed)
    simulator = variants.EntanglementSimulator()
    trajectories, died, permanent = [], [], []
    for config in configs:
        trajectory = simulator.simulate(config)
        concurrence = trajectory.observables["concurrence"]
        after = (np.array([]) if trajectory.censored
                 else concurrence[trajectory.times > trajectory.t_decoh])
        trajectories.append(trajectory)
        died.append(not trajectory.censored)
        permanent.append(
            (not trajectory.censored) and not bool((after > REVIVAL_EPS).any()))
    return trajectories, np.array(died), np.array(permanent)


def windows(trajectories, feature_set):
    """Cuts one window per trajectory and t_obs that is still entangled.

    The window covers the WINDOW steps before t_obs, as everywhere else.
    Trajectories whose first death is at or before t_obs are skipped.

    Args:
        trajectories: Output of `simulate`.
        feature_set: Key of `features.FEATURE_SETS`.

    Returns:
        The flattened windows, the index of the trajectory each comes from
        and its t_obs.
    """
    inputs, owner, observed_at = [], [], []
    for i, trajectory in enumerate(trajectories):
        keep = [j for j, name in enumerate(trajectory.feature_names)
                if name not in features.FEATURE_SETS[feature_set]]
        matrix = trajectory.feature_matrix[:, keep]
        for t_obs in T_OBS:
            if trajectory.t_decoh <= t_obs:
                continue
            end = int(round(t_obs / trajectory.system_config.dt))
            inputs.append(matrix[end - WINDOW:end].ravel())
            owner.append(i)
            observed_at.append(t_obs)
    return (np.array(inputs, dtype=np.float32), np.array(owner),
            np.array(observed_at))


def classifiers():
    """Returns fresh gradient-boosted trees and logistic regression."""
    return {
        "gbt": HistGradientBoostingClassifier(
            max_iter=300, learning_rate=0.05, random_state=0),
        "logreg": make_pipeline(
            StandardScaler(), LogisticRegression(C=1.0, max_iter=3000)),
    }


def score(labels, probability):
    """Scores predicted probabilities against binary labels.

    Args:
        labels: Binary labels.
        probability: Predicted probability of the positive class.

    Returns:
        The base rate and window count, and, if both classes occur, the
        AUROC, Brier score and balanced accuracy at 0.5.
    """
    out = {"base_rate": float(labels.mean()), "n_windows": int(len(labels))}
    if 0 < labels.mean() < 1:
        out.update(auroc=metrics.auroc(labels, probability),
                   brier=metrics.brier(labels, probability),
                   bal_acc=metrics.balanced_accuracy(labels, probability))
    return out


def run(seed, scenario, n_train, n_test, out):
    """Trains and scores both tasks for every input set of one scenario.

    Args:
        seed: Seed of the training configurations and states; the test
            configurations are drawn with seed + 1000 and the test states
            with seed + 1.
        scenario: "D" or "E".
        n_train: Training trajectories.
        n_test: Independent test trajectories.
        out: `common.ResultsWriter` receiving the rows.
    """
    started = time.time()
    train, train_died, train_perm = simulate(
        esd_configs(scenario, n_train, seed), seed)
    test, test_died, test_perm = simulate(
        esd_configs(scenario, n_test, seed + 1000), seed + 1)
    print(f"R26 | seed={seed} {scenario} simulated "
          f"[{time.time() - started:.0f}s] died={train_died.mean():.2f} "
          f"permanent|died={train_perm[train_died].mean():.2f}", flush=True)
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
            t_sub = t_test[mask_test]
            key = {"seed": seed, "scenario": scenario, "task": task,
                   "features": feature_set}
            if y_train.min() == y_train.max():
                for t_obs in T_OBS:
                    k = t_sub == t_obs
                    out.write({**key, "model": "-", "t_obs": t_obs,
                               **score(y_test[k],
                                       np.full(k.sum(), y_train.mean()))})
                continue
            for name, classifier in classifiers().items():
                classifier.fit(x_train[mask_train], y_train)
                p = classifier.predict_proba(x_test[mask_test])[:, 1]
                for t_obs in T_OBS:
                    k = t_sub == t_obs
                    row = score(y_test[k], p[k])
                    out.write({**key, "model": name, "t_obs": t_obs, **row})
                    print(f"R26 | seed={seed} {scenario} {task:<10} "
                          f"{feature_set:<9} {name:<6} t={t_obs} "
                          f"AUROC={row.get('auroc', float('nan')):.3f} "
                          f"base={row['base_rate']:.2f} "
                          f"n={row['n_windows']}", flush=True)


def add_flags(parser: argparse.ArgumentParser) -> None:
    """Adds the choice of scenarios to the common flags."""
    parser.add_argument("--scenario", choices=["D", "E", "DE"],
                        default="DE")


def main() -> None:
    args = common.parse_args(__doc__, extra=add_flags)
    n_train, n_test = (60, 30) if args.fast else (2000, 400)
    path = common.results_path("r26_esd_occurrence", args)
    with common.ResultsWriter(
            path, COLUMNS, seeds=args.seeds,
            config=f"R26: n_train={n_train} n_test={n_test} "
                   f"scenarios={args.scenario}") as out:
        for seed in args.seeds:
            for scenario in args.scenario:
                run(seed, scenario, n_train, n_test, out)


if __name__ == "__main__":
    main()
