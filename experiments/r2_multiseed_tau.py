"""R2 — multi-seed variance probe for the adaptive-prior tau sweep (E7b shape).

Paper 2 concludes "the optimal threshold is tau=0.5 for XXZ" from a single
sweep in which adjacent tau values differ by up to 0.46 in R^2 -- larger than
the effect being claimed.  E7b also generates the dataset once per scenario and
trains every tau on it, while torch is never seeded, so consecutive tau points
differ in weight initialisation as well as in tau.  This re-runs the sweep
across seeds so the tau effect can be separated from run-to-run spread.

Writes ONLY to experiments/results/r2_multiseed_tau[_<tag>][_fast].csv; the
champion e7b_tau_sweep.csv is left untouched.

Reduced size (default 100 trajectories / 30 epochs) versus the champion
configuration, so absolute values are not comparable to the paper table -- the
spread, not the level, is the point.
"""
from __future__ import annotations

import time
from collections.abc import Sequence

from experiments import common
from experiments import e7_paper2_extended
from src.application import backtest
from src.application import generate_dataset
from src.application import train_model
from src.data import scenarios as scenario_lib
from src.evaluation import stats
from src.simulation import simulator as simulator_lib

COLUMNS = ("seed", "scenario", "tau", "r2", "mae", "auroc")
TAUS = e7_paper2_extended.TAUS


def summarise(rows: Sequence[dict[str, object]],
              scenarios: Sequence[str]) -> None:
    """Prints mean and standard deviation of R^2 and AUROC per tau.

    Args:
        rows: Result rows with the keys scenario, tau, r2 and auroc.
        scenarios: Scenarios to print, in order.
    """
    print(f"\n{'=' * 62}\nMEAN +/- STD PER TAU\n{'=' * 62}")
    print(f"{'scen':5} {'tau':>5} {'R2 mean':>9} {'R2 std':>8} "
          f"{'AUROC mean':>11} {'AUROC std':>10}")
    for scenario in scenarios:
        for tau in TAUS:
            sub = [r for r in rows
                   if r["scenario"] == scenario and r["tau"] == tau]
            r2_mean, r2_std = stats.mean_std([r["r2"] for r in sub])
            au_mean, au_std = stats.mean_std([r["auroc"] for r in sub])
            print(f"{scenario:5} {tau:5.1f} {r2_mean:9.3f} {r2_std:8.3f} "
                  f"{au_mean:11.3f} {au_std:10.3f}")


def _add_flags(parser) -> None:
    parser.add_argument("--n-per", type=int, default=100)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--scenarios", nargs="+", default=["D", "E"])


def main(argv: Sequence[str] | None = None) -> None:
    """Sweeps tau for every seed and scenario and writes the CSV."""
    args = common.parse_args(__doc__, extra=_add_flags, argv=argv)
    n_per, epochs = (20, 3) if args.fast else (args.n_per, args.epochs)
    path = common.results_path("r2_multiseed_tau", args)

    print(f"R2 tau sweep x seeds={args.seeds} n_per={n_per} "
          f"epochs={epochs}", flush=True)
    simulator = simulator_lib.QuTipSimulator()
    rows = []
    start = time.time()
    with common.ResultsWriter(
            path, COLUMNS, seeds=args.seeds,
            config={"n_per": n_per, "epochs": epochs, "taus": TAUS,
                    "scenarios": args.scenarios}) as out:
        for seed in args.seeds:
            groups = scenario_lib.build_configs(
                n_per_scenario=n_per, seed=seed)
            for scenario in args.scenarios:
                dataset = generate_dataset.GenerateDatasetUseCase(
                    simulator, store=scenario_lib.trajectory_store()
                ).execute(generate_dataset.GenerateDatasetCommand(
                    configs=groups[scenario], window_length=20,
                    horizon=1.0, samples_per_trajectory=5, seed=seed))
                for tau in TAUS:
                    predictor = e7_paper2_extended.tau_predictor(tau)
                    train_model.TrainModelUseCase(predictor).execute(
                        train_model.TrainModelCommand(
                            dataset=dataset, n_epochs=epochs,
                            batch_size=64, seed=seed, learning_rate=5e-4,
                            regression_loss="huber", verbose=False))
                    metrics = backtest.BacktestUseCase(predictor).execute(
                        backtest.BacktestCommand(
                            dataset=dataset, horizon=1.0))
                    row = {"seed": seed, "scenario": scenario, "tau": tau,
                           "r2": metrics.r2, "mae": metrics.mae,
                           "auroc": metrics.risk_auroc}
                    rows.append(row)
                    out.write(row)
                    print(f"  seed={seed} {scenario} tau={tau:.1f} "
                          f"R2={metrics.r2:8.3f} "
                          f"AUROC={metrics.risk_auroc:.3f}", flush=True)

    summarise(rows, args.scenarios)
    print(f"\ntotal {time.time() - start:.0f}s -> {path}")


if __name__ == "__main__":
    main()
