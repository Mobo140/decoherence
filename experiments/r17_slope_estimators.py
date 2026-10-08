"""R17 — do smoothing-based slope estimates survive measurement noise?

Why
---
R15 showed that the slope-based estimate T2 = -1/s - t_obs collapses under
measurement noise. Before changing the model, this probe checks whether a
better estimate of the slope from the same window helps. No training: each
estimator is applied to the test windows of the R15 setup (scenarios B+C,
100 trajectories each, window 50, seeds 42-44) with Gaussian noise of
standard deviation sigma on the coherence channel.

Estimators:
  ols_log   OLS of log|C| on t (the estimator used in the models)
  wls_log   weighted OLS of log C, w = C^2 (Var[log C] ~ sigma^2 / C^2)
  exp_fit   least squares of C = A exp(s t) in linear space (Gauss-Newton)
  ssa_ols   SSA denoising (embedding 10, leading component), then ols_log
  holt      Holt double exponential smoothing of log C, final trend

Writes ONLY to experiments/results/r17_slope_estimators.csv.

Usage
-----
    python -m experiments.r17_slope_estimators
"""

import sys; sys.path.insert(0, ".")
import numpy as np
from experiments._config import build_configs, trajectory_store
from src.application.generate_dataset import GenerateDatasetUseCase, GenerateDatasetCommand
from src.infrastructure.quantum.qutip_simulator import QuTipSimulator

DT, TMAX = 0.1, 20.0
def r2(y, p): return 1 - np.sum((y-p)**2) / np.sum((y-y.mean())**2)

def ols(t, y, w=None):
    w = np.ones_like(t) if w is None else w
    tm, ym = np.average(t, weights=w), np.average(y, weights=w)
    return np.sum(w*(t-tm)*(y-ym)) / np.sum(w*(t-tm)**2)

def exp_fit(t, c):
    s = ols(t, np.log(np.maximum(np.abs(c), 1e-3)))
    a = np.exp(np.mean(np.log(np.maximum(np.abs(c), 1e-3)) - s*t))
    for _ in range(30):
        f = a*np.exp(s*t); J = np.stack([np.exp(s*t), a*t*np.exp(s*t)], 1)
        step, *_ = np.linalg.lstsq(J, c - f, rcond=None)
        a, s = a + step[0], s + step[1]
        if not np.isfinite(s): return 0.0
    return s

def ssa_denoise(x, L=10, k=1):
    N = len(x); K = N - L + 1
    X = np.stack([x[i:i+K] for i in range(L)])
    U, S, Vt = np.linalg.svd(X, full_matrices=False)
    Xr = (U[:, :k] * S[:k]) @ Vt[:k]
    out = np.zeros(N); cnt = np.zeros(N)
    for i in range(L):
        out[i:i+K] += Xr[i]; cnt[i:i+K] += 1
    return out / cnt

def holt(y, a=0.3, b=0.1):
    lvl, tr = y[0], (y[-1]-y[0])/(len(y)-1)
    for v in y[1:]:
        prev = lvl; lvl = a*v + (1-a)*(lvl+tr); tr = b*(lvl-prev) + (1-b)*tr
    return tr / DT

def t2_from(s, t_obs):
    rem = -1.0/s - t_obs if s < -1e-6 else TMAX
    return float(np.clip(t_obs + max(rem, 0.0), 0.0, TMAX))

gen = GenerateDatasetUseCase(QuTipSimulator(), store=trajectory_store())
res = {}
for seed in (42, 43, 44):
    g = build_configs(n_per_scenario=100, seed=seed)
    ds = gen.execute(GenerateDatasetCommand(configs=g["B"]+g["C"], window_length=50, horizon=1.0,
                     samples_per_trajectory=5, seed=seed, dataset_name=f"slopenoise_{seed}"))
    idx = ds.test_idx; ci = ds.feature_names.index("coherence_l1")
    y = ds.t_decoh_abs[idx]
    rng = np.random.default_rng(seed)
    base = rng.normal(0, 1, (len(idx), 50))
    t = np.arange(50) * DT
    for sigma in (0.0, 0.01, 0.02, 0.05, 0.10):
        preds = {k: [] for k in ("ols_log", "wls_log", "exp_fit", "ssa_ols", "holt")}
        for k, i in enumerate(idx):
            c = ds.sequences[i][:, ci].astype(float) + sigma*base[k]
            lc = np.log(np.maximum(np.abs(c), 1e-8)); to = float(ds.t_obs[i])
            preds["ols_log"].append(t2_from(ols(t, lc), to))
            preds["wls_log"].append(t2_from(ols(t, lc, np.maximum(c, 1e-3)**2), to))
            preds["exp_fit"].append(t2_from(exp_fit(t, c), to))
            cs = ssa_denoise(c)
            preds["ssa_ols"].append(t2_from(ols(t, np.log(np.maximum(np.abs(cs), 1e-8))), to))
            preds["holt"].append(t2_from(holt(lc), to))
        for kname, p in preds.items():
            res.setdefault((kname, sigma), []).append(r2(y, np.array(p)))
import csv
from pathlib import Path
out = Path(__file__).parent / "results" / "r17_slope_estimators.csv"
with out.open("w", newline="") as f:
    w = csv.writer(f); w.writerow(["estimator", "sigma", "seed", "r2"])
    for (kname, s), vals in res.items():
        for seed, v in zip((42, 43, 44), vals):
            w.writerow([kname, s, seed, v])
print(f"{'method':<9}" + "".join(f"{s:>16}" for s in (0.0, 0.01, 0.02, 0.05, 0.10)))
for kname in ("ols_log", "wls_log", "exp_fit", "ssa_ols", "holt"):
    print(f"{kname:<9}" + "".join(f"{np.mean(res[(kname,s)]):>9.3f}±{np.std(res[(kname,s)],ddof=1):.3f}" for s in (0.0, 0.01, 0.02, 0.05, 0.10)))
