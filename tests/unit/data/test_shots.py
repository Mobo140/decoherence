"""shots: input sets, shot-noise streams and paired windowed datasets."""
import numpy as np
import pytest

from src import pipeline
from src.data import scenarios, shots

WINDOW = dict(
    window_length=10, horizon=1.0, samples_per_trajectory=2,
    train_ratio=0.6, val_ratio=0.2)


def test_input_set_rejects_an_unknown_kind():
    with pytest.raises(ValueError, match="kind must be one of"):
        shots.InputSet("full")


def test_input_set_rejects_shots_on_an_exact_kind():
    with pytest.raises(ValueError, match="takes no n_shots"):
        shots.InputSet("exact", 100)


@pytest.mark.parametrize("n_shots", [None, 0, -5, 2.5])
def test_input_set_needs_a_positive_integer_for_shots(n_shots):
    with pytest.raises(ValueError, match="positive integer"):
        shots.InputSet("tomography", n_shots)


def test_n_settings_rejects_unknown_arguments():
    with pytest.raises(ValueError):
        shots.n_settings("full", 2)
    with pytest.raises(ValueError):
        shots.n_settings("local", 3)


def test_n_settings_counts_bases():
    assert shots.n_settings("local", 2) == 3
    assert shots.n_settings("tomography", 2) == 9
    assert shots.n_settings("local", 1) == 3
    assert shots.n_settings("tomography", 1) == 3
    assert shots.n_settings("exact", 2) is None


def test_shot_rng_rejects_an_unknown_stream():
    with pytest.raises(KeyError, match="Unknown stream"):
        shots.shot_rng(0, "val", 100, 0)


def test_shot_rng_is_reproducible_and_streams_differ():
    a = shots.shot_rng(1, "train", 100, 3).random(4)
    assert np.array_equal(a, shots.shot_rng(1, "train", 100, 3).random(4))
    for other in (shots.shot_rng(1, "test", 100, 3),
                  shots.shot_rng(1, "train", 1000, 3),
                  shots.shot_rng(1, "train", 100, 4),
                  shots.shot_rng(2, "train", 100, 3)):
        assert not np.array_equal(a, other.random(4))


def test_windowed_needs_states():
    runs = shots.simulate(_configs("D", 2), 0)
    stripped = [r.__class__(**{**r.__dict__, "states": None}) for r in runs]
    with pytest.raises(ValueError, match="needs runs with states"):
        shots.windowed(
            stripped, shots.InputSet("exact"), seed=0, stream="test",
            **WINDOW)


def _configs(label, n):
    return scenarios.build_configs(n_per_scenario=n, seed=0)[label]


@pytest.fixture(scope="module")
def runs_two():
    return shots.simulate(_configs("D", 8), 5)


@pytest.fixture(scope="module")
def runs_one():
    return shots.simulate(_configs("B", 8), 5)


def _two(kind, n_shots, runs, stream="test"):
    return shots.windowed(
        runs, shots.InputSet(kind, n_shots), seed=5, stream=stream, **WINDOW)


def test_feature_columns_of_each_two_qubit_input_set(runs_two):
    exact = _two("exact", None, runs_two)
    assert exact.feature_names == [
        "sigma_x1", "sigma_y1", "sigma_z1", "sigma_x2", "sigma_y2",
        "sigma_z2", "purity", "coherence_l1"]
    assert _two("exact_local", None, runs_two).feature_names == (
        exact.feature_names[:6])
    assert _two("local", 200, runs_two).feature_names == (
        exact.feature_names[:6])
    tomography = _two("tomography", 200, runs_two).feature_names
    assert len(tomography) == 17 and tomography[-2:] == [
        "purity", "coherence_l1"]


def test_input_sets_share_windows_splits_and_labels(runs_two):
    reference = _two("exact", None, runs_two)
    for kind, n_shots in (("exact_local", None), ("local", 200),
                          ("tomography", 200)):
        other = _two(kind, n_shots, runs_two)
        for name in ("train_idx", "val_idx", "test_idx", "t_obs",
                     "t_decoh_abs", "remaining_time", "risk_labels"):
            assert np.array_equal(
                getattr(reference, name), getattr(other, name)), name


def test_streams_give_different_noise_on_the_same_windows(runs_one):
    a = shots.windowed(
        runs_one, shots.InputSet("local", 100), seed=5, stream="train",
        **WINDOW)
    b = shots.windowed(
        runs_one, shots.InputSet("local", 100), seed=5, stream="test",
        **WINDOW)
    assert np.array_equal(a.test_idx, b.test_idx)
    assert not np.array_equal(a.sequences, b.sequences)
    again = shots.windowed(
        runs_one, shots.InputSet("local", 100), seed=5, stream="train",
        **WINDOW)
    assert np.array_equal(a.sequences, again.sequences)


def test_shot_noise_shrinks_with_the_number_of_shots(runs_one):
    exact = _two("exact", None, runs_one).sequences
    errors = [
        np.abs(shots.windowed(
            runs_one, shots.InputSet("local", n), seed=5, stream="test",
            **WINDOW).sequences[..., :3] - exact[..., :3]).mean()
        for n in (100, 10000)]
    assert errors[1] < errors[0] / 5


def test_exact_set_equals_the_pipeline_dataset():
    spec = pipeline.DataSpec(
        name="shots_check", scenarios=("B",), n_per_scenario=6,
        window_length=10, samples_per_trajectory=2, train_ratio=0.6,
        val_ratio=0.2)
    reference = pipeline.generate(spec, 7, {})
    runs = shots.simulate(pipeline.system_configs(spec, 7), 7)
    ours = shots.windowed(
        runs, shots.InputSet("exact"), seed=7, stream="test",
        window_length=10, horizon=1.0, samples_per_trajectory=2,
        train_ratio=0.6, val_ratio=0.2)
    assert np.array_equal(ours.sequences, reference.sequences)
    assert np.array_equal(ours.remaining_time, reference.remaining_time)
    assert np.array_equal(ours.test_idx, reference.test_idx)
