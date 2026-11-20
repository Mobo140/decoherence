"""Accuracy against the cost of collecting the data (R29 and R30).

The cost of one input window is the number of single-shot measurements it
needs: shots per basis N x measurement bases x time points of the window.
One qubit needs 3 bases (x, y, z); two qubits need 3 for local means (both
qubits in the same basis) and 9 for tomography. The accuracy is the best
R^2, over the models, of the mean over seeds; the table also gives the
spread over seeds (sample standard deviation) of that model and its name.

Reads experiments/results/r29_finite_shots_1q.csv and
r30_finite_shots_2q.csv; writes experiments/results/shot_cost_table.csv with
its provenance and prints the Markdown table.

Usage
-----
    python -m experiments.new.shot_cost_table
"""
from __future__ import annotations

import argparse

import pandas as pd

from experiments import common
from experiments.new import r29_finite_shots_1q as r29
from experiments.new import r30_finite_shots_2q as r30
from src.data import shots

COLUMNS = ("setting", "scenario", "target", "arm", "n_shots", "n_settings",
           "window_points", "cost", "best_model", "r2_mean", "r2_std",
           "n_seeds")


def best_model(frame: pd.DataFrame) -> dict[str, object]:
    """Picks the model with the highest mean R^2 over seeds.

    Args:
        frame: Rows of one table cell, with columns `model`, `seed`, `r2`.

    Returns:
        best_model, r2_mean, r2_std (sample std over seeds; nan for one
        seed) and n_seeds of that model.

    Raises:
        ValueError: If `frame` is empty.
    """
    if frame.empty:
        raise ValueError("There are no rows to pick the best model from.")
    per_model = frame.groupby("model")["r2"].agg(["mean", "std", "count"])
    name = per_model["mean"].idxmax()
    top = per_model.loc[name]
    return {"best_model": name, "r2_mean": top["mean"],
            "r2_std": top["std"], "n_seeds": int(top["count"])}


def cost(n_shots: int, n_bases: int, window_points: int) -> int:
    """Single-shot measurements needed for one input window.

    Args:
        n_shots: Shots per basis and time point.
        n_bases: Measurement bases per time point.
        window_points: Time points of the window.

    Returns:
        The product of the three.

    Raises:
        ValueError: If an argument is not a positive integer.
    """
    for name, value in (("n_shots", n_shots), ("n_bases", n_bases),
                        ("window_points", window_points)):
        if int(value) != value or value < 1:
            raise ValueError(
                f"{name} must be a positive integer, got {value!r}.")
    return int(n_shots) * int(n_bases) * int(window_points)


def table(one: pd.DataFrame, two: pd.DataFrame) -> pd.DataFrame:
    """Builds the cost table from the results of R29 and R30.

    Args:
        one: R29 rows.
        two: R30 rows.

    Returns:
        One row per setting, N, scenario, target and arm, with `COLUMNS`.
        Exact inputs have no cost (nan) and n_shots 0.
    """
    rows = []
    pooled = one[one["scenario"] == "B+C"]
    for (arm, n), cell in pooled.groupby(["arm", "n_shots"]):
        bases = shots.n_settings("local", 1) if n else None
        rows.append({
            "setting": "1q exact" if n == 0 else "1q local",
            "scenario": "B+C", "target": "t2", "arm": arm, "n_shots": n,
            "n_settings": bases, "window_points": r29.WINDOW_LENGTH,
            "cost": cost(n, bases, r29.WINDOW_LENGTH) if n else None,
            **best_model(cell)})
    names = {"i": "2q exact base", "ii": "2q exact local",
             "iii": "2q local", "iv": "2q tomography", "v": "2q tomography"}
    keys = ["scenario", "target", "input_set", "kind", "n_shots"]
    for (scenario, target, label, kind, n), cell in two.groupby(keys):
        bases = shots.n_settings(kind, 2)
        rows.append({
            "setting": names[label], "scenario": scenario, "target": target,
            "arm": "train_shots" if n else "train_exact", "n_shots": n,
            "n_settings": bases, "window_points": r30.WINDOW_LENGTH,
            "cost": cost(n, bases, r30.WINDOW_LENGTH) if n else None,
            **best_model(cell)})
    return pd.DataFrame(rows, columns=list(COLUMNS))


def markdown(frame: pd.DataFrame) -> str:
    """Renders the table as Markdown.

    Args:
        frame: Output of `table`.

    Returns:
        The Markdown text.
    """
    lines = ["| Настройка | Сценарий | Цель | Обучение | N | Базисов | "
             "Точек окна | Стоимость | Лучшая модель | R² |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    for r in frame.itertuples():
        cost_text = "—" if pd.isna(r.cost) else f"{int(r.cost):,}".replace(
            ",", " ")
        bases = "—" if pd.isna(r.n_settings) else int(r.n_settings)
        lines.append(
            f"| {r.setting} | {r.scenario} | {r.target} | {r.arm} | "
            f"{r.n_shots or 'точные'} | {bases} | {r.window_points} | "
            f"{cost_text} | {r.best_model} | "
            f"{r.r2_mean:.3f} ± {r.r2_std:.3f} |")
    return "\n".join(lines)


def main() -> None:
    """Writes shot_cost_table.csv and prints it as Markdown."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--r29", default="r29_finite_shots_1q.csv")
    parser.add_argument("--r30", default="r30_finite_shots_2q.csv")
    parser.add_argument("--out", default="shot_cost_table.csv")
    args = parser.parse_args()
    frame = table(pd.read_csv(common.RESULTS_DIR / args.r29),
                  pd.read_csv(common.RESULTS_DIR / args.r30))
    common.write_results(
        frame, common.RESULTS_DIR / args.out, columns=COLUMNS,
        config=f"cost table from {args.r29} and {args.r30}")
    print(markdown(frame))


if __name__ == "__main__":
    main()
