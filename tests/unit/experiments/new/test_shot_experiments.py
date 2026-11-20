"""R29 and R30: harness guards and the fast path of both scripts."""
import dataclasses

import numpy as np
import pytest

from experiments.new import r29_finite_shots_1q as r29
from experiments.new import r30_finite_shots_2q as r30
from experiments.new import shot_harness as harness
from src import pipeline
from src.data import shots

WINDOW = dict(
    window_length=10, horizon=1.0, samples_per_trajectory=2,
    train_ratio=0.6, val_ratio=0.2)


@pytest.fixture(scope="module")
def dataset():
    spec = pipeline.DataSpec(
        name="harness", scenarios=("B",), n_per_scenario=10)
    runs = shots.simulate(pipeline.system_configs(spec, 1), 1)
    return shots.windowed(
        runs, shots.InputSet("local", 100), seed=1, stream="test", **WINDOW)


def test_restrict_test_rejects_a_mask_of_the_wrong_length(dataset):
    with pytest.raises(ValueError, match="entries for"):
        harness.restrict_test(dataset, np.ones(3, dtype=bool))


def test_restrict_test_keeps_only_masked_test_windows(dataset):
    mask = np.zeros(len(dataset.sequences), dtype=bool)
    mask[dataset.test_idx[:1]] = True
    assert list(harness.restrict_test(dataset, mask).test_idx) == list(
        dataset.test_idx[:1])


def test_score_rejects_an_empty_test_split(dataset):
    empty = dataclasses.replace(
        dataset, test_idx=np.array([], dtype=np.int64))
    with pytest.raises(ValueError, match="no window"):
        harness.score(None, empty, 1.0)


def test_with_tail_tells_the_input_sets_apart(dataset):
    # One qubit has the tail even with shots; two qubits only with
    # tomography or exact inputs.
    assert harness.with_tail(dataset)
    spec = pipeline.DataSpec(
        name="harness", scenarios=("D",), n_per_scenario=10)
    runs = shots.simulate(pipeline.system_configs(spec, 1), 1)
    for kind, n_shots, expected in (
            ("local", 100, False), ("exact_local", None, False),
            ("exact", None, True), ("tomography", 100, True)):
        data = shots.windowed(
            runs, shots.InputSet(kind, n_shots), seed=1, stream="test",
            **WINDOW)
        assert harness.with_tail(data) is expected, kind


def test_r30_physics_models_run_only_where_the_tail_exists():
    rows = list(r30.rows_for_seed(
        1, 20, 1, ("physics_only", "transformer"),
        r30.INPUT_SETS, scenarios=("D",), targets=("t2e",)))
    seen = {(r["input_set"], r["model"]) for r in rows}
    assert seen == {
        ("i", "physics_only"), ("i", "transformer"),
        ("ii", "transformer"), ("iii", "transformer"),
        ("iv", "physics_only"), ("iv", "transformer")}
    assert {r["input_set"]: r["n_features"] for r in rows} == {
        "i": 8, "ii": 6, "iii": 6, "iv": 17}
    assert all(set(r30.COLUMNS) <= set(r) for r in rows)


def test_r29_fast_path_gives_both_arms_for_pooled_and_each_scenario():
    rows = list(r29.rows_for_seed(1, 60, 1, ("physics_only",), (100,)))
    keys = {(r["arm"], r["n_shots"], r["scenario"]) for r in rows}
    assert keys == {
        (arm, n, scenario)
        for arm, n in (("train_exact", 0), ("train_exact", 100),
                       ("train_shots", 100))
        for scenario in ("B+C", "B", "C")}
    assert all(set(r29.COLUMNS) <= set(r) for r in rows)
