"""R22 — noise-aware training with time-correlated (AR(1)) noise.

Why
---
R21 showed that measurement noise correlated in time hurts more than white
noise of the same size: at phi = 0.9 the variance of the OLS slope over a
50-point window grows 9.6-fold, every trained network loses accuracy, and
the residual model trained with white noise (R16) loses its lead over the
Transformer. The noisy copies in R16 were white, so the network never saw a
window in which a slow drift of the noise imitates a trend. Here the noisy
copies are AR(1) with a lag-one correlation drawn per window from
U[0, phi_max]; everything else is as in R16.

Questions
  * Does correlated training noise restore the residual model's lead at
    phi = 0.9?
  * What does it cost on white noise and on clean data?
  * Do the networks without the slope estimate gain as much?

Setup as in R16/R21 (scenarios B+C, 100 trajectories each, window 50).
Arms:
    physics_lstm_white  white noisy copies (= R16 physics_lstm_aug)
    physics_lstm_ar1    AR(1) noisy copies, phi ~ U[0, 0.95]
    lstm_only_ar1       same, no physics estimate
    transformer         clean training (= R21)
    transformer_ar1     AR(1) noisy copies
Each training window gets two noisy copies, sigma log-uniform in
[0.005, 0.12] as in R16. Test windows get AR(1) noise with sigma in
{0, 0.01, 0.02, 0.05, 0.10} and phi in {0, 0.5, 0.9}; one standard-normal
realisation per seed is filtered for every phi and scaled by sigma, so all
comparisons are paired and phi = 0 is the R15/R16 test noise.

Writes ONLY to experiments/results/r22_ar1_noise_training*.csv.
Each CSV is accompanied by <name>.csv.provenance.json (configuration,
seeds, software environment).

Usage
-----
    python -m experiments.r22_ar1_noise_training --fast --tag smoke
    python -m experiments.r22_ar1_noise_training --seeds 42 --tag s42
"""
from __future__ import annotations

from experiments import common
from src import pipeline

SIGMAS = (0.0, 0.01, 0.02, 0.05, 0.10)
PHIS = (0.0, 0.5, 0.9)
PHI_MAX = 0.95
COLUMNS = ("seed", "arm", "sigma", "phi", "r2", "mae", "auroc", "n_samples")

# arm -> (model alias, training set)
ARMS = {
    "physics_lstm_white": ("physics_lstm", "white"),
    "physics_lstm_ar1": ("physics_lstm", "ar1"),
    "lstm_only_ar1": ("lstm_only", "ar1"),
    "transformer": ("transformer", "clean"),
    "transformer_ar1": ("transformer", "ar1"),
}
# Same generator seed as R16/R21, so the white copies are R16's.
AUGMENTATIONS = {
    "clean": None,
    "white": pipeline.Augmentation(2),
    "ar1": pipeline.Augmentation(2, phi_max=PHI_MAX),
}


def arms(n_per: int, epochs: int) -> list[pipeline.ExperimentConfig]:
    """Declares the arms: white and AR(1) noisy training copies.

    Every test applies the same realisation of AR(1) noise, filtered with
    each phi, to the clean test windows.

    Args:
        n_per: Trajectories per scenario.
        epochs: Training epochs.

    Returns:
        The arms in the order of the result rows.
    """
    data = pipeline.DataSpec(
        name="r22", scenarios=("B", "C"), n_per_scenario=n_per,
        cache="disk")
    tests = tuple(
        pipeline.EvalSpec(
            "ar1", sigma=sigma, phi=phi,
            tags=(("sigma", sigma), ("phi", phi)))
        for phi in PHIS for sigma in SIGMAS
        if sigma > 0.0 or phi == 0.0
    )
    return [
        pipeline.ExperimentConfig(
            data=data, model=alias, epochs=epochs, tests=tests,
            augmentation=AUGMENTATIONS[train], tags=(("arm", arm),))
        for arm, (alias, train) in ARMS.items()
    ]


def main() -> None:
    args = common.parse_args(__doc__)
    n_per, epochs = (30, 5) if args.fast else (100, 100)
    common.run_arms(
        "R22", arms(n_per, epochs), args,
        common.results_path("r22_ar1_noise_training", args), COLUMNS)


if __name__ == "__main__":
    main()
