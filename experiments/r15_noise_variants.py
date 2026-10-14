"""R15 — measurement-noise robustness of every model, across seeds.

Why
---
E3 measured only the residual model, in one run. With the BiLSTM training
fixed, the residual model degrades sharply under noise, and a diagnostic
shows the degradation comes from the physics scalars computed on the noisy
window (slope of log C, envelope switch, OLS R^2), not from the network:
fed the clean-window scalars, the same model is unaffected. To state this
as a comparison rather than a single curve, this probe trains each model on
clean data (E3 configuration: scenarios B+C, 100 trajectories each, window
50) and evaluates it on the test windows with Gaussian noise added to every
channel. The same noise realisation, scaled by sigma, is used at every
level, so the comparison across sigma is paired.

Writes ONLY to experiments/results/r15_noise_variants*.csv.
Each CSV is accompanied by <name>.csv.provenance.json (configuration,
seeds, software environment).

Usage
-----
    python -m experiments.r15_noise_variants --seeds 42 43 44
    python -m experiments.r15_noise_variants --fast --tag smoke
"""
from __future__ import annotations

from experiments import common
from src import pipeline

SIGMAS = (0.0, 0.01, 0.02, 0.05, 0.10)
MODELS = ("physics_only", "lstm_only", "physics_lstm", "transformer")
COLUMNS = ("seed", "variant", "sigma", "r2", "mae", "auroc")


def arms(n_per: int, epochs: int) -> list[pipeline.ExperimentConfig]:
    """Declares one arm per model, trained on clean data.

    Args:
        n_per: Trajectories per scenario.
        epochs: Training epochs.

    Returns:
        The arms in the order of the result rows.
    """
    data = pipeline.DataSpec(
        name="r15", scenarios=("B", "C"), n_per_scenario=n_per,
        cache="disk")
    tests = tuple(
        pipeline.EvalSpec("gaussian", sigma=sigma, tags=(("sigma", sigma),))
        for sigma in SIGMAS)
    return [
        pipeline.ExperimentConfig(
            data=data, model=name, epochs=epochs, tests=tests,
            tags=(("variant", name),))
        for name in MODELS
    ]


def main() -> None:
    """Evaluates every model under measurement noise.

    Writes experiments/results/r15_noise_variants*.csv.
    """
    args = common.parse_args(__doc__)
    n_per, epochs = (30, 10) if args.fast else (100, 100)
    common.run_arms(
        "R15", arms(n_per, epochs), args,
        common.results_path("r15_noise_variants", args), COLUMNS)


if __name__ == "__main__":
    main()
