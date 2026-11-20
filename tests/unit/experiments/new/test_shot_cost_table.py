"""shot_cost_table: cost, best model and the table of both experiments."""
import pandas as pd
import pytest

from experiments.new import shot_cost_table as cost_table


@pytest.mark.parametrize("args", [(0, 3, 50), (100, 0, 50), (100, 3, -1),
                                  (1.5, 3, 50)])
def test_cost_rejects_non_positive_integers(args):
    with pytest.raises(ValueError, match="positive integer"):
        cost_table.cost(*args)


def test_cost_is_the_product():
    assert cost_table.cost(1000, 9, 20) == 180_000


def test_best_model_rejects_an_empty_frame():
    with pytest.raises(ValueError, match="no rows"):
        cost_table.best_model(pd.DataFrame(columns=["model", "seed", "r2"]))


def test_best_model_takes_the_highest_mean_over_seeds():
    frame = pd.DataFrame({
        "model": ["a", "a", "b", "b"], "seed": [1, 2, 1, 2],
        "r2": [0.9, 0.5, 0.7, 0.75]})
    best = cost_table.best_model(frame)
    assert best["best_model"] == "b" and best["n_seeds"] == 2
    assert best["r2_mean"] == pytest.approx(0.725)
    assert best["r2_std"] == pytest.approx(0.0353553, rel=1e-4)


def test_table_costs_and_markdown():
    one = pd.DataFrame({
        "seed": [1, 1, 1], "arm": ["train_exact", "train_exact", "train_shots"],
        "model": ["m"] * 3, "n_shots": [0, 100, 100],
        "scenario": ["B+C"] * 3, "r2": [0.9, 0.8, 0.7]})
    two = pd.DataFrame({
        "seed": [1, 1], "scenario": ["D", "D"], "target": ["t2e", "t2e"],
        "input_set": ["iv", "iii"], "kind": ["tomography", "local"],
        "n_shots": [1000, 1000], "model": ["m", "m"], "r2": [0.6, 0.5]})
    frame = cost_table.table(one, two)
    costs = dict(zip(frame["setting"], frame["cost"]))
    assert costs["1q local"] == 100 * 3 * 50
    assert costs["2q tomography"] == 1000 * 9 * 20
    assert costs["2q local"] == 1000 * 3 * 20
    assert pd.isna(costs["1q exact"])
    text = cost_table.markdown(frame)
    assert "180 000" in text and "точные" in text
