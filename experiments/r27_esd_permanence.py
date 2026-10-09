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

Usage
-----
    python -m experiments.r27_esd_permanence --fast
    python -m experiments.r27_esd_permanence --seeds 42 --tag s42
"""
from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from experiments._config import build_configs
from experiments.r24_entanglement_target import _esd
from experiments.r26_esd_occurrence import DROP, T_OBS, classifiers, score, simulate, windows

RESULTS_DIR = Path(__file__).parent / "results"
N_BOOT = 1000


def boot_ci(y, p, owner, rng):
    """95% interval of AUROC from resampling test trajectories.

    At a fixed t_obs each trajectory contributes at most one window, so
    resampling windows is resampling trajectories.
    """
    assert len(np.unique(owner)) == len(owner)
    vals = []
    for _ in range(N_BOOT):
        pick = rng.integers(0, len(y), len(y))
        if 0 < y[pick].mean() < 1:
            vals.append(roc_auc_score(y[pick], p[pick]))
    return np.percentile(vals, [2.5, 97.5])


def run(seed, n_train, n_test, w, f):
    ts = time.time()
    tr_cfg = _esd(build_configs(n_per_scenario=n_train, seed=seed))["D"]
    te_cfg = _esd(build_configs(n_per_scenario=n_test, seed=seed + 1000))["D"]
    tr, tr_died, tr_perm = simulate(tr_cfg, seed)
    te, te_died, te_perm = simulate(te_cfg, seed + 1)
    print(f"R27 | seed={seed} simulated [{time.time() - ts:.0f}s] died={tr_died.mean():.2f} "
          f"permanent|died={tr_perm[tr_died].mean():.2f} test permanent={te_perm.sum()}", flush=True)
    rng = np.random.default_rng(seed)
    for feats in DROP:
        Xtr, otr, _ = windows(tr, feats)
        Xte, ote, tte = windows(te, feats)
        tasks = {"occurrence": (tr_died, te_died, None),
                 "permanence": (tr_perm, te_perm, (tr_died, te_died))}
        for task, (ytr_t, yte_t, cond) in tasks.items():
            mtr = np.ones(len(otr), bool) if cond is None else cond[0][otr]
            mte = np.ones(len(ote), bool) if cond is None else cond[1][ote]
            ytr, yte = ytr_t[otr][mtr].astype(int), yte_t[ote][mte].astype(int)
            for name, clf in classifiers().items():
                clf.fit(Xtr[mtr], ytr)
                p = clf.predict_proba(Xte[mte])[:, 1]
                for t_obs in T_OBS:
                    k = tte[mte] == t_obs
                    row = score(yte[k], p[k])
                    lo, hi = boot_ci(yte[k], p[k], ote[mte][k], rng)
                    w.writerow({"seed": seed, "task": task, "features": feats, "model": name,
                                "t_obs": t_obs, **row, "n_pos": int(yte[k].sum()),
                                "auroc_lo": lo, "auroc_hi": hi})
                    print(f"R27 | seed={seed} {task:<10} {feats:<9} {name:<6} t={t_obs} "
                          f"AUROC={row['auroc']:.3f} [{lo:.3f},{hi:.3f}] pos={int(yte[k].sum())}",
                          flush=True)
                f.flush()


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    p.add_argument("--fast", action="store_true")
    p.add_argument("--tag", type=str, default="")
    a = p.parse_args()
    n_train, n_test = (300, 200) if a.fast else (8000, 4000)
    sfx = (f"_{a.tag}" if a.tag else "") + ("_fast" if a.fast else "")
    out = RESULTS_DIR / f"r27_esd_permanence{sfx}.csv"
    with out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["seed", "task", "features", "model", "t_obs", "base_rate",
                                          "n_windows", "auroc", "brier", "bal_acc", "n_pos",
                                          "auroc_lo", "auroc_hi"])
        w.writeheader()
        for seed in a.seeds:
            run(seed, n_train, n_test, w, f)


if __name__ == "__main__":
    main()
