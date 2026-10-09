"""R21 — correlated measurement noise and a stochastic dissipation rate.

Why
---
Two assumptions of the single-qubit results have not been tested.

1. Measurement noise is independent between time steps (R15, R16). Drifts
   in readout calibration make real noise correlated in time. For an OLS
   slope over n points with AR(1) errors of correlation phi, the variance
   is inflated by about (1 + phi) / (1 - phi) once n is larger than the
   correlation time: the window holds n_eff = n / (2 tau_int) effectively
   independent points, tau_int = (1 + phi) / (2 (1 - phi)). At equal
   marginal sigma, correlated noise should therefore hurt the slope-based
   estimate T_phys more than white noise. Noise-aware training (R16) used
   white noise only; whether it transfers is open.

2. gamma(t) is a smooth Bernstein polynomial. A real bath fluctuates. Here
   gamma(t) = gamma_B(t) * exp(s x(t) - s^2 / 2), where x is a stationary
   Ornstein-Uhlenbeck process with unit variance and correlation time tau.
   The factor has mean one, so the mean rate is unchanged; s sets the
   relative size of the fluctuations. Beyond tau the future rate is not
   predictable from the window, so some loss is unavoidable.

Setup
-----
Training as in R15/R16: scenarios B+C, 100 trajectories each, window 50,
three seeds. Arms:
    physics_only        no training
    lstm_only           clean training
    physics_lstm        clean training
    physics_lstm_aug    white-noise training (R16)
    transformer         clean training
    physics_lstm_sto    trained on stochastic gamma (s ~ U[0, 0.6], tau in {0.5, 2})
    transformer_sto     same
Evaluation
    noise   test windows of the training dataset + AR(1) noise,
            sigma in {0.02, 0.05}, phi in {0, 0.5, 0.9}. One standard-normal
            realisation per seed is filtered for every phi and scaled by sigma,
            so the comparison is paired; phi = 0 reproduces R15 exactly.
    gamma   independent B+C configurations (100 each, generator seed + 1000),
            simulated with s in {0, 0.3, 0.6}, tau in {0.5, 2.0}; all windows
            are test windows. Initial states and the OU driving noise are the
            same for every setting, so s = 0 is the paired control.

Writes ONLY to experiments/results/r21_stochastic_gamma_ar1*.csv.

Usage
-----
    python -m experiments.r21_stochastic_gamma_ar1 --fast
    python -m experiments.r21_stochastic_gamma_ar1 --seeds 42 --tag s42
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

from experiments._config import build_configs, trajectory_store
from experiments.r16_noise_aware_training import noise_augmented
from src.application.backtest import BacktestCommand, BacktestUseCase
from src.application.generate_dataset import GenerateDatasetCommand, GenerateDatasetUseCase
from src.application.train_model import TrainModelCommand, TrainModelUseCase
from src.infrastructure.ml.lstm_predictor import LSTMPredictor
from src.infrastructure.ml.physics_only_predictor import PhysicsOnlyPredictor
from src.infrastructure.ml.transformer_predictor import TransformerPredictor
from src.infrastructure.quantum.noise import ar1_filter as ar1
from src.infrastructure.quantum.qutip_simulator import QuTipSimulator

RESULTS_DIR = Path(__file__).parent / "results"
WINDOW = 50
SIGMAS = [0.02, 0.05]
PHIS = [0.0, 0.5, 0.9]
GAMMA_SETTINGS = [(0.0, 0.5), (0.3, 0.5), (0.6, 0.5), (0.3, 2.0), (0.6, 2.0)]
OU_GRID_DT = 0.01

# name -> (builder, training set: "clean" | "aug" | "sto" | None)
ARMS = {
    "physics_only": (lambda: PhysicsOnlyPredictor(t_max=20.0), None),
    "lstm_only": (lambda: LSTMPredictor(use_physics_prior=False), "clean"),
    "physics_lstm": (lambda: LSTMPredictor(use_physics_prior=True), "clean"),
    "physics_lstm_aug": (lambda: LSTMPredictor(use_physics_prior=True), "aug"),
    "transformer": (lambda: TransformerPredictor(), "clean"),
    "physics_lstm_sto": (lambda: LSTMPredictor(use_physics_prior=True), "sto"),
    "transformer_sto": (lambda: TransformerPredictor(), "sto"),
}


class StochasticGammaSimulator(QuTipSimulator):
    """QuTiP simulator with gamma(t) multiplied by a log-normal OU factor.

    One OU path is drawn per simulated trajectory from the simulator's own
    generator, in the order trajectories are simulated. Two instances with
    the same seed therefore drive the same configurations with the same
    standard-normal increments, whatever s and tau are.

    s, tau: fixed strength and correlation time; if s is None, each
    trajectory draws s ~ U[0, 0.6] and tau from {0.5, 2.0}.
    """

    def __init__(self, s: float | None, tau: float | None, seed: int):
        self.s, self.tau = s, tau
        self.rng = np.random.default_rng(seed)

    def _prepare_system(self, system, config):
        n = int(round(config.t_max / OU_GRID_DT)) + 1
        eta = self.rng.standard_normal(n)
        s, tau = self.s, self.tau
        if s is None:
            s = float(self.rng.uniform(0.0, 0.6))
            tau = float(self.rng.choice([0.5, 2.0]))
        if s == 0.0 or system.gamma_const is not None:
            return system
        a = np.exp(-OU_GRID_DT / tau)
        x = np.empty(n)
        x[0] = eta[0]
        for k in range(1, n):
            x[k] = a * x[k - 1] + np.sqrt(1.0 - a * a) * eta[k]
        grid = np.linspace(0.0, config.t_max, n)
        factor = np.exp(s * x - 0.5 * s * s)
        base = system.get_gamma
        system.get_gamma = lambda t: base(t) * float(np.interp(t, grid, factor))
        return system


def _with_sequences(ds, seqs):
    out = copy.copy(ds)
    out.sequences = seqs.astype(np.float32)
    return out


def _all_test(ds):
    out = copy.copy(ds)
    out.test_idx = np.arange(len(ds.sequences))
    out.train_idx = np.array([], dtype=np.int64)
    out.val_idx = np.array([], dtype=np.int64)
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    p.add_argument("--fast", action="store_true")
    p.add_argument("--tag", type=str, default="")
    a = p.parse_args()
    n_per, n_eval, epochs = (30, 20, 5) if a.fast else (100, 100, 100)
    sfx = (f"_{a.tag}" if a.tag else "") + ("_fast" if a.fast else "")
    out = RESULTS_DIR / f"r21_stochastic_gamma_ar1{sfx}.csv"
    print(f"R21 | seeds={a.seeds} n_per={n_per} epochs={epochs}", flush=True)

    gen_cached = GenerateDatasetUseCase(QuTipSimulator(), store=trajectory_store())
    with out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["seed", "arm", "test", "sigma", "phi", "s",
                                          "tau", "r2", "mae", "auroc", "n_samples"])
        w.writeheader()
        for seed in a.seeds:
            ts = time.time()
            groups = build_configs(n_per_scenario=n_per, seed=seed)
            train_cfgs = groups["B"] + groups["C"]
            cmd = dict(window_length=WINDOW, horizon=1.0, samples_per_trajectory=5, seed=seed)
            ds = gen_cached.execute(GenerateDatasetCommand(
                configs=train_cfgs, dataset_name=f"r21_{seed}", **cmd))
            ds_aug = noise_augmented(ds, 2, np.random.default_rng(seed + 7))
            ds_sto = GenerateDatasetUseCase(
                StochasticGammaSimulator(None, None, seed + 11), store=None
            ).execute(GenerateDatasetCommand(configs=train_cfgs, dataset_name=f"r21_sto_{seed}", **cmd))
            train_sets = {"clean": ds, "aug": ds_aug, "sto": ds_sto}

            eval_groups = build_configs(n_per_scenario=n_eval, seed=seed + 1000)
            eval_cfgs = eval_groups["B"] + eval_groups["C"]
            gamma_tests = {}
            for s, tau in GAMMA_SETTINGS:
                sim = StochasticGammaSimulator(s, tau, seed + 13)
                gamma_tests[(s, tau)] = _all_test(GenerateDatasetUseCase(sim, store=None).execute(
                    GenerateDatasetCommand(configs=eval_cfgs, train_ratio=0.0, val_ratio=0.0,
                                           dataset_name=f"r21_eval_{seed}", **{**cmd, "seed": seed + 1})))
            print(f"R21 | seed={seed} data ready [{time.time() - ts:.0f}s]", flush=True)

            eta = np.random.default_rng(seed).normal(0.0, 1.0, size=ds.sequences.shape)
            noise_tests = {(0.0, 0.0): ds}
            for phi in PHIS:
                e = ar1(eta, phi)
                for sigma in SIGMAS:
                    noise_tests[(sigma, phi)] = _with_sequences(ds, ds.sequences + sigma * e)

            for arm, (build, train) in ARMS.items():
                pred = build()
                if train is not None:
                    TrainModelUseCase(pred).execute(TrainModelCommand(
                        dataset=train_sets[train], n_epochs=epochs, batch_size=32,
                        learning_rate=1e-3, regression_loss="huber", verbose=False, seed=seed))
                rows = []
                for (sigma, phi), tds in noise_tests.items():
                    m = BacktestUseCase(pred).execute(BacktestCommand(dataset=tds, horizon=1.0))
                    rows.append(dict(test="noise", sigma=sigma, phi=phi, s="", tau="", m=m))
                for (s, tau), tds in gamma_tests.items():
                    m = BacktestUseCase(pred).execute(BacktestCommand(dataset=tds, horizon=1.0))
                    rows.append(dict(test="gamma", sigma="", phi="", s=s, tau=tau, m=m))
                for r in rows:
                    m = r.pop("m")
                    w.writerow({"seed": seed, "arm": arm, **r, "r2": m.r2, "mae": m.mae,
                                "auroc": m.risk_auroc, "n_samples": m.n_samples})
                    print(f"R21 | seed={seed} {arm:<17} {r['test']} sigma={r['sigma']} "
                          f"phi={r['phi']} s={r['s']} tau={r['tau']} R2={m.r2:.3f}", flush=True)
                f.flush()
            print(f"R21 | seed={seed} done [{time.time() - ts:.0f}s]", flush=True)


if __name__ == "__main__":
    main()
