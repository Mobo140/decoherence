"""R1 — multi-seed variance probe for the two-qubit ablation (E6a shape).

Why this exists
---------------
Every number in both papers comes from a single run with seed=42, so there are
no error bars anywhere.  Several conclusions rest on small gaps (e.g.
physics_lstm 0.636 vs lstm_only 0.673 on XXZ, a difference of 0.037), and the
E7b tau sweep swings by up to 0.46 between adjacent tau values, which is larger
than the effect it claims to measure.  This script re-runs the E6a-shaped
ablation across several seeds and reports mean +/- std so those gaps can be
judged against run-to-run spread.

Safety
------
Writes ONLY to
experiments/results/r1_multiseed_ablation[_<tag>][_fast].csv.  It never touches
the champion CSVs (e6_*_improved.csv etc.) that back the paper tables --
re-running e6_2qubit_improved.py directly would overwrite them, which is why
this is a separate entry point.

Caveat on interpretation
------------------------
Runs at a reduced size (default n_per=100, epochs=30) rather than the champion
configuration (500 / 150), so absolute values are NOT comparable to the paper
tables.  With less data and less training the spread here is, if anything, an
*upper bound* on the spread of the full configuration.  Use it to judge whether
a gap is inside the noise, not to restate headline numbers.

Each seed varies both the sampled physical configurations (build_configs) and
the training/split randomness (AblationCommand), i.e. it is a full experimental
replication rather than a training-only reseed.

Usage
-----
    python -m experiments.r1_multiseed_ablation --fast
    python -m experiments.r1_multiseed_ablation --seeds 42 43 44
    python -m experiments.r1_multiseed_ablation --seeds 42 43 44 \\
        --n-per 150 --epochs 40
"""
from __future__ import annotations

import time
from collections.abc import Sequence

from experiments import common
from experiments import e6_2qubit_improved
from src.application import run_ablation
from src.data import scenarios as scenario_lib
from src.domain import value_objects
from src.evaluation import stats
from src.simulation import simulator as simulator_lib

COLUMNS = ("seed", "scenario", "variant", "r2", "mae", "auroc", "n_samples")
VARIANTS = (
    value_objects.PredictorVariant.PHYSICS_ONLY,
    value_objects.PredictorVariant.LSTM_ONLY,
    value_objects.PredictorVariant.PHYSICS_LSTM,
    value_objects.PredictorVariant.TRANSFORMER,
)


def run_seed(
    simulator,
    seed: int,
    n_per: int,
    epochs: int,
    scenarios: Sequence[str],
    out: common.ResultsWriter,
) -> list[dict[str, object]]:
    """Runs E6a's ablation for one seed and writes each row as it arrives.

    The predictors are E6a's (`improved_ablation_predictor`); the base
    use case would build a weaker model than the paper reports.

    Args:
        simulator: The simulator.
        seed: Seed of the configurations, datasets and training.
        n_per: Trajectories per scenario.
        epochs: Training epochs.
        scenarios: Scenario labels to run.
        out: Writer of the results file.

    Returns:
        The rows written.
    """
    groups = scenario_lib.build_configs(n_per_scenario=n_per, seed=seed)
    use_case = run_ablation.AblationStudyUseCase(
        simulator,
        predictor_factory=e6_2qubit_improved.improved_ablation_predictor)
    rows = []
    for scenario in scenarios:
        results = use_case.execute(run_ablation.AblationCommand(
            configs=groups[scenario],
            variants=list(VARIANTS),
            window_length=20,
            horizon=1.0,
            samples_per_trajectory=5,
            n_epochs=epochs,
            batch_size=64,
            lr=5e-4,
            regression_loss="huber",
            seed=seed,
            verbose=False,
        ))
        for result in results:
            row = {
                "seed": seed,
                "scenario": scenario,
                "variant": result.spec.predictor_variant.value,
                "r2": result.r2,
                "mae": result.mae,
                "auroc": result.risk_auroc,
                "n_samples": result.n_samples,
            }
            rows.append(row)
            out.write(row)
            print(f"    seed={seed} {scenario} {row['variant']:14} "
                  f"R2={result.r2:8.3f} MAE={result.mae:7.3f} "
                  f"AUROC={result.risk_auroc:.3f}", flush=True)
    return rows


def summarise(rows: Sequence[dict[str, object]]) -> None:
    """Prints mean and standard deviation of R^2 and AUROC over the seeds.

    Args:
        rows: Result rows with the keys scenario, variant, r2 and auroc.
    """
    print(f"\n{'=' * 78}\nMEAN +/- STD ACROSS SEEDS\n{'=' * 78}")
    print(f"{'scen':5} {'variant':14} {'R2 mean':>9} {'R2 std':>8} "
          f"{'AUROC mean':>11} {'AUROC std':>10} {'n':>3}")
    for scenario in sorted({r["scenario"] for r in rows}):
        for variant in [v.value for v in VARIANTS]:
            sub = [r for r in rows
                   if r["scenario"] == scenario and r["variant"] == variant]
            if not sub:
                continue
            r2_mean, r2_std = stats.mean_std([r["r2"] for r in sub])
            au_mean, au_std = stats.mean_std([r["auroc"] for r in sub])
            print(f"{scenario:5} {variant:14} {r2_mean:9.3f} {r2_std:8.3f} "
                  f"{au_mean:11.3f} {au_std:10.3f} {len(sub):3d}")


def _add_flags(parser) -> None:
    parser.add_argument("--n-per", type=int, default=100)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--scenarios", nargs="+", default=["D", "E"])


def main(argv: Sequence[str] | None = None) -> None:
    """Runs the ablation for every seed and writes the CSV."""
    args = common.parse_args(__doc__, extra=_add_flags, argv=argv)
    n_per, epochs = (20, 3) if args.fast else (args.n_per, args.epochs)
    path = common.results_path("r1_multiseed_ablation", args)

    print(f"R1 multi-seed variance | seeds={args.seeds} n_per={n_per} "
          f"epochs={epochs} scenarios={args.scenarios}")
    print(f"writing -> {path}  (champion CSVs untouched)\n")

    simulator = simulator_lib.QuTipSimulator()
    all_rows = []
    start = time.time()
    with common.ResultsWriter(
            path, COLUMNS, seeds=args.seeds,
            config={"n_per": n_per, "epochs": epochs,
                    "scenarios": args.scenarios}) as out:
        for i, seed in enumerate(args.seeds, 1):
            print(f"--- seed {seed}  ({i}/{len(args.seeds)}) ---",
                  flush=True)
            seed_start = time.time()
            all_rows += run_seed(
                simulator, seed, n_per, epochs, args.scenarios, out)
            print(f"    [{time.time() - seed_start:.0f}s]", flush=True)

    summarise(all_rows)
    print(f"\ntotal {time.time() - start:.0f}s -> {path}")


if __name__ == "__main__":
    main()
