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
    physics_lstm_sto    trained on stochastic gamma
                        (s ~ U[0, 0.6], tau in {0.5, 2})
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
Each CSV is accompanied by <name>.csv.provenance.json (configuration,
seeds, software environment).

Usage
-----
    python -m experiments.r21_stochastic_gamma_ar1 --fast --tag smoke
    python -m experiments.r21_stochastic_gamma_ar1 --seeds 42 --tag s42
"""
from __future__ import annotations

import dataclasses

from experiments import common
from src import pipeline

WINDOW = 50
SIGMAS = (0.02, 0.05)
PHIS = (0.0, 0.5, 0.9)
GAMMA_SETTINGS = ((0.0, 0.5), (0.3, 0.5), (0.6, 0.5), (0.3, 2.0), (0.6, 2.0))
COLUMNS = ("seed", "arm", "test", "sigma", "phi", "s", "tau", "r2", "mae",
           "auroc", "n_samples")

# arm -> (model alias, training set: "clean" | "aug" | "sto")
ARMS = {
    "physics_only": ("physics_only", "clean"),
    "lstm_only": ("lstm_only", "clean"),
    "physics_lstm": ("physics_lstm", "clean"),
    "physics_lstm_aug": ("physics_lstm", "aug"),
    "transformer": ("transformer", "clean"),
    "physics_lstm_sto": ("physics_lstm", "sto"),
    "transformer_sto": ("transformer", "sto"),
}


def noise_tests() -> list[pipeline.EvalSpec]:
    """Test windows of the training data plus AR(1) noise.

    The first test is the clean one. For the others one realisation of the
    noise is filtered for every phi, so the tests are paired; phi = 0
    reproduces R15.
    """
    clean = pipeline.EvalSpec(
        tags=(("test", "noise"), ("sigma", 0.0), ("phi", 0.0)))
    noisy = [
        pipeline.EvalSpec(
            "ar1", sigma=sigma, phi=phi,
            tags=(("test", "noise"), ("sigma", sigma), ("phi", phi)))
        for phi in PHIS for sigma in SIGMAS
    ]
    return [clean, *noisy]


def gamma_tests(
    data: pipeline.DataSpec, n_eval: int
) -> list[pipeline.EvalSpec]:
    """Independent configurations simulated with a stochastic rate.

    Initial states and the driving noise of the rate are the same for
    every setting, so s = 0 is the paired control.
    """
    independent = dataclasses.replace(
        data, name="r21_eval", n_per_scenario=n_eval, cache="none",
        train_ratio=0.0, val_ratio=0.0, config_seed_offset=1000,
        data_seed_offset=1)
    return [
        pipeline.EvalSpec(
            data=dataclasses.replace(
                independent, simulator=pipeline.SimulatorSpec(
                    "stochastic_gamma", (("s", s), ("tau", tau)), 13)),
            tags=(("test", "gamma"), ("s", s), ("tau", tau)))
        for s, tau in GAMMA_SETTINGS
    ]


def arms(
    n_per: int, n_eval: int, epochs: int
) -> list[pipeline.ExperimentConfig]:
    """Declares the arms: three training sets, noise and rate tests.

    Args:
        n_per: Trajectories per scenario.
        n_eval: Independent trajectories per scenario for the rate tests.
        epochs: Training epochs.

    Returns:
        The arms in the order of the result rows.
    """
    data = pipeline.DataSpec(
        name="r21", scenarios=("B", "C"), n_per_scenario=n_per,
        window_length=WINDOW, cache="disk")
    # gamma(t) drawn with s ~ U[0, 0.6] and tau in {0.5, 2} per trajectory.
    stochastic = dataclasses.replace(
        data, name="r21_sto", cache="none",
        simulator=pipeline.SimulatorSpec(
            "stochastic_gamma", (("s", None), ("tau", None)), 11))
    train_data = {"clean": None, "aug": None, "sto": stochastic}
    tests = tuple(noise_tests() + gamma_tests(data, n_eval))
    return [
        pipeline.ExperimentConfig(
            data=data, model=alias, epochs=epochs, tests=tests,
            train_data=train_data[train],
            augmentation=pipeline.Augmentation(2) if train == "aug" else None,
            tags=(("arm", arm),))
        for arm, (alias, train) in ARMS.items()
    ]


def main() -> None:
    args = common.parse_args(__doc__)
    n_per, n_eval, epochs = (30, 20, 5) if args.fast else (100, 100, 100)
    common.run_arms(
        "R21", arms(n_per, n_eval, epochs), args,
        common.results_path("r21_stochastic_gamma_ar1", args), COLUMNS)


if __name__ == "__main__":
    main()
