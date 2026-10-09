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
changes with t_obs cannot inflate the AUROC. Classifiers: gradient-boosted trees and
logistic regression on the flattened window, with the three input sets of
R24 (base / + correlators / + correlators + concurrence). Metrics: AUROC,
Brier score and balanced accuracy at 0.5, with the class balance. Three
seeds.

Writes ONLY to experiments/results/r26_esd_occurrence*.csv.

Usage
-----
    python -m experiments.r26_esd_occurrence --fast
    python -m experiments.r26_esd_occurrence --seeds 42 --scenario D --tag D42
"""
from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, brier_score_loss, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from experiments._config import build_configs
from experiments.r23_two_body_correlators import CORR
from experiments.r24_entanglement_target import EntanglementSimulator, _esd

RESULTS_DIR = Path(__file__).parent / "results"
WINDOW = 20
T_OBS = [2.0, 2.5, 3.0, 3.5, 4.0]
REVIVAL_EPS = 1e-3
DROP = {"base": set(CORR) | {"concurrence"}, "corr": {"concurrence"}, "corr_conc": set()}


def simulate(cfgs, seed):
    """Trajectories plus, per trajectory, (died, permanent)."""
    np.random.seed(seed)
    sim = EntanglementSimulator()
    trajs, died, permanent = [], [], []
    for c in cfgs:
        t = sim.simulate(c)
        conc = t.observables["concurrence"]
        after = conc[t.times > t.t_decoh] if not t.censored else np.array([])
        trajs.append(t)
        died.append(not t.censored)
        permanent.append((not t.censored) and not bool((after > REVIVAL_EPS).any()))
    return trajs, np.array(died), np.array(permanent)


def windows(trajs, feats):
    """One window per (trajectory, t_obs) for trajectories still entangled at t_obs.

    The window covers the WINDOW steps before t_obs, as everywhere else.
    """
    X, owner, tob = [], [], []
    for i, t in enumerate(trajs):
        keep = [j for j, n in enumerate(t.feature_names) if n not in DROP[feats]]
        M = t.feature_matrix[:, keep]
        for t_obs in T_OBS:
            if t.t_decoh <= t_obs:          # already dead (first death) by t_obs
                continue
            e = int(round(t_obs / t.system_config.dt))
            X.append(M[e - WINDOW:e].ravel())
            owner.append(i)
            tob.append(t_obs)
    return np.array(X, dtype=np.float32), np.array(owner), np.array(tob)


def classifiers():
    return {
        "gbt": HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, random_state=0),
        "logreg": make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=3000)),
    }


def score(y, p):
    out = {"base_rate": float(y.mean()), "n_windows": int(len(y))}
    if 0 < y.mean() < 1:
        out.update(auroc=roc_auc_score(y, p), brier=brier_score_loss(y, p),
                   bal_acc=balanced_accuracy_score(y, p >= 0.5))
    return out


def run(seed, sc, n_train, n_test, w, f):
    ts = time.time()
    tr_cfg = _esd(build_configs(n_per_scenario=n_train, seed=seed))[sc]
    te_cfg = _esd(build_configs(n_per_scenario=n_test, seed=seed + 1000))[sc]
    tr, tr_died, tr_perm = simulate(tr_cfg, seed)
    te, te_died, te_perm = simulate(te_cfg, seed + 1)
    print(f"R26 | seed={seed} {sc} simulated [{time.time() - ts:.0f}s] "
          f"died={tr_died.mean():.2f} permanent|died={tr_perm[tr_died].mean():.2f}", flush=True)
    for feats in DROP:
        Xtr, otr, _ = windows(tr, feats)
        Xte, ote, tte = windows(te, feats)
        tasks = {"occurrence": (tr_died, te_died, None),
                 "permanence": (tr_perm, te_perm, (tr_died, te_died))}
        for task, (ytr_t, yte_t, cond) in tasks.items():
            mtr = np.ones(len(otr), bool) if cond is None else cond[0][otr]
            mte = np.ones(len(ote), bool) if cond is None else cond[1][ote]
            ytr, yte = ytr_t[otr][mtr].astype(int), yte_t[ote][mte].astype(int)
            t_sub = tte[mte]
            if ytr.min() == ytr.max():
                for t_obs in T_OBS:
                    k = t_sub == t_obs
                    w.writerow({"seed": seed, "scenario": sc, "task": task, "features": feats,
                                "model": "-", "t_obs": t_obs, **score(yte[k], np.full(k.sum(), ytr.mean()))})
                continue
            for name, clf in classifiers().items():
                clf.fit(Xtr[mtr], ytr)
                p = clf.predict_proba(Xte[mte])[:, 1]
                for t_obs in T_OBS:
                    k = t_sub == t_obs
                    row = score(yte[k], p[k])
                    w.writerow({"seed": seed, "scenario": sc, "task": task, "features": feats,
                                "model": name, "t_obs": t_obs, **row})
                    print(f"R26 | seed={seed} {sc} {task:<10} {feats:<9} {name:<6} t={t_obs} "
                          f"AUROC={row.get('auroc', float('nan')):.3f} base={row['base_rate']:.2f} "
                          f"n={row['n_windows']}", flush=True)
        f.flush()


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    p.add_argument("--scenario", choices=["D", "E", "DE"], default="DE")
    p.add_argument("--fast", action="store_true")
    p.add_argument("--tag", type=str, default="")
    a = p.parse_args()
    n_train, n_test = (60, 30) if a.fast else (2000, 400)
    sfx = (f"_{a.tag}" if a.tag else "") + ("_fast" if a.fast else "")
    out = RESULTS_DIR / f"r26_esd_occurrence{sfx}.csv"
    with out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["seed", "scenario", "task", "features", "model", "t_obs",
                                          "base_rate", "n_windows", "auroc", "brier", "bal_acc"])
        w.writeheader()
        for seed in a.seeds:
            for sc in a.scenario:
                run(seed, sc, n_train, n_test, w, f)


if __name__ == "__main__":
    main()
