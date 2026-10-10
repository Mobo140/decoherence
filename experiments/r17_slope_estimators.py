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

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from experiments._config import build_configs, trajectory_store
from src.application.generate_dataset import (
    GenerateDatasetCommand,
    GenerateDatasetUseCase,
)
from src.simulation.simulator import QuTipSimulator

DT = 0.1
TMAX = 20.0
WINDOW = 50
SEEDS = (42, 43, 44)
SIGMAS = (0.0, 0.01, 0.02, 0.05, 0.10)
ESTIMATORS = ("ols_log", "wls_log", "exp_fit", "ssa_ols", "holt")
OUTPUT_CSV = Path(__file__).parent / "results" / "r17_slope_estimators.csv"


def r2(y: np.ndarray, p: np.ndarray) -> float:
    """Returns the coefficient of determination of predictions ``p``."""
    return 1 - np.sum((y - p) ** 2) / np.sum((y - y.mean()) ** 2)


def ols(t: np.ndarray, y: np.ndarray, w: np.ndarray | None = None) -> float:
    """Fits the slope of ``y`` against ``t`` by weighted least squares."""
    w = np.ones_like(t) if w is None else w
    tm, ym = np.average(t, weights=w), np.average(y, weights=w)
    return np.sum(w * (t - tm) * (y - ym)) / np.sum(w * (t - tm) ** 2)


def exp_fit(t: np.ndarray, c: np.ndarray) -> float:
    """Fits C = A exp(s t) in linear space by Gauss-Newton; returns s.

    Starts from the log-space OLS estimate. Returns 0 if the iteration
    diverges.
    """
    log_c = np.log(np.maximum(np.abs(c), 1e-3))
    s = ols(t, log_c)
    a = np.exp(np.mean(log_c - s * t))
    for _ in range(30):
        f = a * np.exp(s * t)
        jac = np.stack([np.exp(s * t), a * t * np.exp(s * t)], 1)
        step, *_ = np.linalg.lstsq(jac, c - f, rcond=None)
        a, s = a + step[0], s + step[1]
        if not np.isfinite(s):
            return 0.0
    return s


def ssa_denoise(x: np.ndarray, window: int = 10, rank: int = 1) -> np.ndarray:
    """Denoises ``x`` with singular spectrum analysis.

    Args:
        x: Signal.
        window: Embedding dimension.
        rank: Number of leading components kept.

    Returns:
        The signal rebuilt from the leading components by diagonal
        averaging.
    """
    n = len(x)
    k = n - window + 1
    trajectory = np.stack([x[i:i + k] for i in range(window)])
    u, sv, vt = np.linalg.svd(trajectory, full_matrices=False)
    low_rank = (u[:, :rank] * sv[:rank]) @ vt[:rank]
    out = np.zeros(n)
    count = np.zeros(n)
    for i in range(window):
        out[i:i + k] += low_rank[i]
        count[i:i + k] += 1
    return out / count


def holt(y: np.ndarray, a: float = 0.3, b: float = 0.1) -> float:
    """Returns the final trend of Holt's double exponential smoothing."""
    level, trend = y[0], (y[-1] - y[0]) / (len(y) - 1)
    for v in y[1:]:
        prev = level
        level = a * v + (1 - a) * (level + trend)
        trend = b * (level - prev) + (1 - b) * trend
    return trend / DT


def t2_from(s: float, t_obs: float) -> float:
    """Converts a log-slope ``s`` at time ``t_obs`` to a T2 estimate."""
    rem = -1.0 / s - t_obs if s < -1e-6 else TMAX
    return float(np.clip(t_obs + max(rem, 0.0), 0.0, TMAX))


def estimate_t2(c: np.ndarray, t: np.ndarray, t_obs: float) -> dict:
    """Applies every estimator to one noisy coherence window."""
    log_c = np.log(np.maximum(np.abs(c), 1e-8))
    smooth = ssa_denoise(c)
    return {
        "ols_log": t2_from(ols(t, log_c), t_obs),
        "wls_log": t2_from(ols(t, log_c, np.maximum(c, 1e-3) ** 2), t_obs),
        "exp_fit": t2_from(exp_fit(t, c), t_obs),
        "ssa_ols": t2_from(
            ols(t, np.log(np.maximum(np.abs(smooth), 1e-8))), t_obs
        ),
        "holt": t2_from(holt(log_c), t_obs),
    }


def run() -> dict:
    """Scores every estimator for every seed and noise level.

    Returns:
        Dictionary mapping (estimator, sigma) to the list of R2 scores,
        one per seed.
    """
    gen = GenerateDatasetUseCase(QuTipSimulator(), store=trajectory_store())
    res = {}
    for seed in SEEDS:
        groups = build_configs(n_per_scenario=100, seed=seed)
        ds = gen.execute(GenerateDatasetCommand(
            configs=groups["B"] + groups["C"], window_length=WINDOW,
            horizon=1.0, samples_per_trajectory=5, seed=seed,
            dataset_name=f"slopenoise_{seed}"))
        idx = ds.test_idx
        coherence = ds.feature_names.index("coherence_l1")
        y = ds.t_decoh_abs[idx]
        rng = np.random.default_rng(seed)
        base = rng.normal(0, 1, (len(idx), WINDOW))
        t = np.arange(WINDOW) * DT
        for sigma in SIGMAS:
            preds = {name: [] for name in ESTIMATORS}
            for k, i in enumerate(idx):
                c = ds.sequences[i][:, coherence].astype(float)
                c = c + sigma * base[k]
                estimates = estimate_t2(c, t, float(ds.t_obs[i]))
                for name in ESTIMATORS:
                    preds[name].append(estimates[name])
            for name, p in preds.items():
                res.setdefault((name, sigma), []).append(r2(y, np.array(p)))
    return res


def write_csv(res: dict, path: Path) -> None:
    """Writes the scores as estimator, sigma, seed, r2 rows."""
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["estimator", "sigma", "seed", "r2"])
        for (name, sigma), vals in res.items():
            for seed, v in zip(SEEDS, vals):
                w.writerow([name, sigma, seed, v])


def print_table(res: dict) -> None:
    """Prints mean +- std over seeds for each estimator and noise level."""
    print(f"{'method':<9}" + "".join(f"{s:>16}" for s in SIGMAS))
    for name in ESTIMATORS:
        cells = "".join(
            f"{np.mean(res[(name, s)]):>9.3f}"
            f"±{np.std(res[(name, s)], ddof=1):.3f}"
            for s in SIGMAS
        )
        print(f"{name:<9}{cells}")


def main() -> None:
    res = run()
    write_csv(res, OUTPUT_CSV)
    print_table(res)


if __name__ == "__main__":
    main()
