"""R16 — can noise-aware training make the residual model robust to noise?

Why
---
R15 showed that the residual model, best on clean data, collapses under
measurement noise (R^2 0.83 -> 0.31 at sigma = 0.05), because the slope of
log C over a short window and hence T_phys = -1/s - t_obs become unreliable,
while networks that do not use the estimate stay robust. A check without
training showed that smoothing the window (weighted or nonlinear fits, SSA,
exponential smoothing) does not fix the estimate itself.

During training the physics scalars are computed from clean windows only,
so the network never sees how far it can trust T_phys when the window is
noisy (the in-loop augmentation adds noise after the scalars are computed).
Here the training and validation sets are extended with noisy copies of
every window, and all inputs -- the window and the physics scalars derived
from it (slope, OLS R^2, T_phys) -- are computed from the noisy copy. The
standard error of the slope is a function of the slope and the OLS R^2, so
the network already has it.

Setup as in R15 (scenarios B+C, 100 trajectories each, window 50), three
seeds. Arms:
    physics_lstm        clean training (control, = R15)
    physics_lstm_aug    noise-aware training
    lstm_only_aug       noise-aware training, no physics estimate
    transformer         clean training
    transformer_aug     noise-aware training
Each training window gets two noisy copies with sigma drawn log-uniformly
from [0.005, 0.12]. Test windows are clean and get the same noise as in R15
(one realisation scaled by sigma, sigma in {0, 0.01, 0.02, 0.05, 0.10}).

Writes ONLY to experiments/results/r16_noise_aware_training*.csv.
Each CSV is accompanied by <name>.csv.provenance.json (configuration,
seeds, software environment).

Usage
-----
    python -m experiments.r16_noise_aware_training --fast --tag smoke
    python -m experiments.r16_noise_aware_training --seeds 42 43 44
"""
from __future__ import annotations

import argparse

from experiments import common
from src import pipeline

SIGMAS = (0.0, 0.01, 0.02, 0.05, 0.10)
# arm -> (model alias, trained on the noise-augmented set)
ARMS = {
    "physics_lstm": ("physics_lstm", False),
    "physics_lstm_aug": ("physics_lstm", True),
    "lstm_only_aug": ("lstm_only", True),
    "transformer": ("transformer", False),
    "transformer_aug": ("transformer", True),
}
COLUMNS = ("seed", "arm", "sigma", "r2", "mae", "auroc")


def arms(
    n_per: int, epochs: int, copies: int
) -> list[pipeline.ExperimentConfig]:
    """Declares the arms: clean and noise-aware training.

    Args:
        n_per: Trajectories per scenario.
        epochs: Training epochs.
        copies: Noisy copies of every training window.

    Returns:
        The arms in the order of the result rows.
    """
    data = pipeline.DataSpec(
        name="r16", scenarios=("B", "C"), n_per_scenario=n_per,
        cache="disk")
    tests = tuple(
        pipeline.EvalSpec("gaussian", sigma=sigma, tags=(("sigma", sigma),))
        for sigma in SIGMAS)
    return [
        pipeline.ExperimentConfig(
            data=data, model=alias, epochs=epochs, tests=tests,
            augmentation=pipeline.Augmentation(copies) if aug else None,
            tags=(("arm", arm),))
        for arm, (alias, aug) in ARMS.items()
    ]


def add_flags(parser: argparse.ArgumentParser) -> None:
    """Adds the number of noisy copies to the common flags."""
    parser.add_argument("--copies", type=int, default=2)


def main() -> None:
    """Trains the residual model with noise-aware augmentation.

    Writes experiments/results/r16_noise_aware_training*.csv.
    """
    args = common.parse_args(__doc__, extra=add_flags)
    n_per, epochs = (30, 5) if args.fast else (100, 100)
    common.run_arms(
        "R16", arms(n_per, epochs, args.copies), args,
        common.results_path("r16_noise_aware_training", args), COLUMNS)


if __name__ == "__main__":
    main()
