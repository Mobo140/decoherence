"""Window cutting: trajectory-level splits, window ends and context codes."""
import dataclasses

import numpy as np
import pytest

from src.data import windows
from src.domain import value_objects
from tests import support

NAN = float("nan")


def _rng(seed: int = 0) -> np.random.Generator:
    return np.random.default_rng(seed)


def _trajectories(n: int = 6, first_t2: float = 3.0) -> list:
    """Scripted trajectories whose t_decoh identifies the trajectory."""
    return [support.scripted_trajectory(first_t2 + 0.2 * i) for i in range(n)]


def _build(trajectories=None, **overrides):
    arguments = dict(
        window_length=5, horizon=1.0, samples_per_trajectory=3,
        train_ratio=0.6, val_ratio=0.2, min_window_gap=0,
        include_censored=True, seed=0, rng=_rng())
    arguments.update(overrides)
    return windows.build_dataset(
        _trajectories() if trajectories is None else trajectories,
        **arguments)


def _censored(t_decoh: float = 3.0):
    return dataclasses.replace(
        support.scripted_trajectory(t_decoh), censored=True)


# --- assign_splits ---------------------------------------------------------

def test_assign_splits_is_a_trajectory_level_partition():
    split = windows.assign_splits(20, 0.8, 0.1, np.random.default_rng(1))
    assert split.count("train") == 16 and split.count("val") == 2
    assert split.count("test") == 2
    assert split == windows.assign_splits(
        20, 0.8, 0.1, np.random.default_rng(1))


def test_assign_splits_other_generators_give_other_assignments():
    a = windows.assign_splits(30, 0.5, 0.2, _rng(1))
    b = windows.assign_splits(30, 0.5, 0.2, _rng(2))
    assert a != b and sorted(a) == sorted(b)


def test_assign_splits_truncates_the_group_sizes():
    split = windows.assign_splits(7, 0.5, 0.3, _rng())
    assert split.count("train") == 3 and split.count("val") == 2
    assert split.count("test") == 2


@pytest.mark.parametrize("train,val,expected", [
    (1.0, 0.0, {"train": 10}), (0.0, 0.0, {"test": 10}),
    (0.0, 1.0, {"val": 10}), (0.9, 0.1, {"train": 9, "val": 1})])
def test_assign_splits_boundary_ratios(train, val, expected):
    split = windows.assign_splits(10, train, val, _rng())
    assert {k: split.count(k) for k in set(split)} == expected


def test_assign_splits_of_no_trajectories_is_empty():
    assert windows.assign_splits(0, 0.8, 0.1, _rng()) == []


# --- valid_window_ends and sample_window_ends ------------------------------

def test_sample_window_ends_respects_the_minimum_gap():
    ends = np.arange(20, 200)
    chosen = windows.sample_window_ends(ends, 5, 10, np.random.default_rng(2))
    assert len(chosen) == 5
    gaps = np.abs(np.subtract.outer(chosen, chosen))[np.triu_indices(5, 1)]
    assert gaps.min() >= 10
    times = np.arange(30) * 0.1
    valid = windows.valid_window_ends(times, 2.0, 5)
    assert valid.min() == 5 and times[valid].max() < 2.0


def test_valid_window_ends_exclude_the_decoherence_time_itself():
    times = np.arange(10.0)
    assert list(windows.valid_window_ends(times, 6.0, 2)) == [2, 3, 4, 5]
    assert list(windows.valid_window_ends(times, 6.5, 2)) == [2, 3, 4, 5, 6]


def test_valid_window_ends_window_of_the_full_prefix_is_allowed():
    times = np.arange(10.0)
    assert list(windows.valid_window_ends(times, 100.0, 9)) == [9]


def test_sample_window_ends_without_replacement_up_to_what_exists():
    ends = np.arange(5, 9)
    chosen = windows.sample_window_ends(ends, 10, 0, _rng())
    assert sorted(chosen) == [5, 6, 7, 8]


def test_sample_window_ends_returns_nothing_for_zero_samples():
    for gap in (0, 5):
        assert len(windows.sample_window_ends(
            np.arange(10, 40), 0, gap, _rng())) == 0


def test_sample_window_ends_with_a_gap_may_find_fewer_than_asked():
    chosen = windows.sample_window_ends(np.arange(20, 25), 4, 4, _rng(3))
    assert 1 <= len(chosen) < 4
    gaps = np.abs(np.subtract.outer(chosen, chosen))[
        np.triu_indices(len(chosen), 1)]
    assert gaps.size == 0 or gaps.min() >= 4


def test_sample_window_ends_gap_larger_than_the_range_gives_one_end():
    chosen = windows.sample_window_ends(np.arange(20, 40), 5, 1000, _rng())
    assert len(chosen) == 1


def test_sample_window_ends_single_candidate_ignores_the_gap():
    assert list(windows.sample_window_ends(
        np.array([7]), 3, 5, _rng())) == [7]


def test_sample_window_ends_without_candidates_is_empty():
    for gap in (0, 5):
        out = windows.sample_window_ends(
            np.array([], dtype=np.int64), 3, gap, _rng())
        assert len(out) == 0 and out.dtype == np.int64


def test_sample_window_ends_is_reproducible():
    ends = np.arange(20, 200)
    a = windows.sample_window_ends(ends, 6, 8, _rng(5))
    b = windows.sample_window_ends(ends, 6, 8, _rng(5))
    np.testing.assert_array_equal(a, b)


# --- pad_features ----------------------------------------------------------

def test_pad_features_inserts_zeros_before_the_last_two_columns():
    short = np.arange(12, dtype=np.float32).reshape(3, 4)
    full = np.ones((3, 6), dtype=np.float32)
    out = windows.pad_features([short, full])
    assert out[0].shape == (3, 6)
    np.testing.assert_array_equal(out[0][:, :2], short[:, :2])
    assert not out[0][:, 2:4].any()
    np.testing.assert_array_equal(out[0][:, 4:], short[:, 2:])


def test_pad_features_equal_widths_return_the_windows_unchanged():
    same = [np.ones((3, 4)), np.zeros((3, 4))]
    assert windows.pad_features(same) is same


def test_pad_features_keeps_the_dtype_and_the_widest_window():
    short = np.ones((2, 3), dtype=np.float32)
    full = np.ones((2, 5), dtype=np.float32)
    out = windows.pad_features([short, full, short])
    assert [w.shape[-1] for w in out] == [5, 5, 5]
    assert out[1] is full and out[0].dtype == np.float32


def test_pad_features_pads_a_window_of_exactly_the_physics_columns():
    out = windows.pad_features([np.ones((2, 2)), np.ones((2, 4))])
    assert out[0].shape == (2, 4) and not out[0][:, :2].any()
    assert out[0][:, 2:].all()


# --- context_codes ---------------------------------------------------------

def test_context_codes_distinguish_the_four_system_types(scenario_configs):
    g = {sc: [cfg] for sc, cfg in scenario_configs.items()}
    codes = {sc: windows.context_codes(g[sc][0]) for sc in "ABCDE"}
    assert [codes[sc][0] for sc in "ABCDE"] == [0.0, 0.0, 1.0, 2.0, 3.0]
    assert [codes[sc][1] for sc in "ABCDE"] == [0.0, 0.0, 1.0, 0.0, 0.0]
    assert codes["A"][2] == 0.0 and codes["D"][2] == g["D"][0].J


def test_context_codes_of_a_constant_sigma_z_qubit():
    config = dataclasses.replace(
        support.make_system_config(),
        dissipator=value_objects.DissipatorConfig.constant(
            0.1, value_objects.DissipatorType.SIGMA_Z))
    assert windows.context_codes(config) == (1.0, 1.0, 0.0)


def test_context_codes_of_two_qubits_with_sigma_z_dissipation():
    config = dataclasses.replace(
        support.make_system_config(), n_qubits=value_objects.QubitCount.TWO,
        J=0.7, dissipator=value_objects.DissipatorConfig.constant(
            0.1, value_objects.DissipatorType.SIGMA_Z))
    assert windows.context_codes(config) == (2.0, 1.0, 0.7)


# --- build_dataset ---------------------------------------------------------

def test_build_dataset_windows_end_before_decoherence_with_exact_labels():
    ds = _build()
    assert ds.sequences.shape == (18, 5, 3)
    assert ds.feature_names == ["sigma_x", "coherence_l1", "purity"]
    assert (ds.remaining_time > 0).all()
    np.testing.assert_allclose(
        ds.remaining_time, ds.t_decoh_abs - ds.t_obs, atol=1e-5)
    np.testing.assert_array_equal(
        ds.risk_labels, (ds.remaining_time <= 1.0).astype(np.float32))
    assert ds.sequences.dtype == np.float32


def test_build_dataset_windows_are_slices_of_the_trajectory_features():
    trajectories = _trajectories(2)
    ds = _build(trajectories, train_ratio=1.0, val_ratio=0.0,
                samples_per_trajectory=2)
    by_t2 = {round(t.t_decoh, 3): t for t in trajectories}
    for i in range(len(ds.sequences)):
        traj = by_t2[round(float(ds.t_decoh_abs[i]), 3)]
        end = int(round(float(ds.t_obs[i]) / 0.1))
        np.testing.assert_array_equal(
            ds.sequences[i], traj.feature_matrix[end - 5:end])


def test_build_dataset_splits_whole_trajectories():
    ds = _build()
    split_of_t2 = {}
    for name, idx in (("train", ds.train_idx), ("val", ds.val_idx),
                      ("test", ds.test_idx)):
        for t2 in np.unique(ds.t_decoh_abs[idx]):
            split_of_t2.setdefault(round(float(t2), 3), set()).add(name)
    assert all(len(v) == 1 for v in split_of_t2.values())
    assert len(ds.train_idx) + len(ds.val_idx) + len(ds.test_idx) == 18
    assert len(ds.train_idx) == 9 and len(ds.val_idx) == 3


def test_build_dataset_metadata_records_the_construction():
    ds = _build(seed=11)
    assert ds.metadata == {
        "n_trajectories": 6, "n_skipped": 0, "n_censored": 0,
        "include_censored": True, "n_censored_windows": 0,
        "window_length": 5, "horizon": 1.0, "seed": 11, "n_train": 9,
        "n_val": 3, "n_test": 6, "dt": pytest.approx(0.1)}


def test_build_dataset_is_reproducible_for_equal_generators():
    a = _build(rng=_rng(4))
    b = _build(rng=_rng(4))
    np.testing.assert_array_equal(a.sequences, b.sequences)
    np.testing.assert_array_equal(a.train_idx, b.train_idx)


def test_build_dataset_gap_keeps_windows_of_one_trajectory_apart():
    ds = _build(_trajectories(2), train_ratio=1.0, val_ratio=0.0,
                min_window_gap=10)
    for t2 in np.unique(ds.t_decoh_abs):
        ends = np.round(ds.t_obs[ds.t_decoh_abs == t2] / 0.1)
        gaps = np.abs(np.subtract.outer(ends, ends))[
            np.triu_indices(len(ends), 1)]
        assert gaps.size == 0 or gaps.min() >= 10


def test_build_dataset_gap_larger_than_the_range_leaves_one_window_each():
    ds = _build(min_window_gap=10_000)
    assert len(ds.sequences) == 6


def test_build_dataset_window_may_use_the_whole_prefix_of_a_trajectory():
    # t_decoh 3.0 on a 0.1 grid: the last valid end is index 29.
    ds = _build([support.scripted_trajectory(3.0)], window_length=29,
                samples_per_trajectory=5, train_ratio=1.0, val_ratio=0.0)
    assert len(ds.sequences) == 1 and ds.sequences.shape[1] == 29


def test_build_dataset_all_windows_to_train_leaves_val_and_test_empty():
    ds = _build(train_ratio=1.0, val_ratio=0.0)
    assert len(ds.val_idx) == 0 and len(ds.test_idx) == 0
    assert len(ds.train_idx) == 18


def test_build_dataset_censored_windows_get_label_zero_and_are_flagged():
    mixed = _trajectories(3) + [_censored(2.5), _censored(2.5)]
    ds = _build(mixed, horizon=100.0)
    assert ds.metadata["n_censored"] == 2
    assert ds.metadata["n_censored_windows"] == int(ds.censored.sum()) == 6
    assert (ds.risk_labels[ds.censored] == 0).all()
    assert (ds.risk_labels[~ds.censored] == 1).all()


def test_build_dataset_can_leave_censored_trajectories_out():
    ds = _build(_trajectories(3) + [_censored(2.5)], include_censored=False)
    assert not ds.censored.any() and len(ds.sequences) == 9
    assert ds.metadata["n_censored"] == 1
    assert ds.metadata["include_censored"] is False


def test_build_dataset_skips_trajectories_shorter_than_the_window():
    short = support.scripted_trajectory(2.0, t_max=0.4)
    ds = _build(_trajectories(3) + [short], window_length=5)
    assert len(ds.sequences) == 9 and ds.metadata["n_skipped"] == 0


def test_build_dataset_counts_trajectories_that_decohere_within_a_window():
    fast = support.scripted_trajectory(0.3)
    ds = _build(_trajectories(3) + [fast], window_length=5)
    assert ds.metadata["n_skipped"] == 1 and len(ds.sequences) == 9


def test_build_dataset_time_step_is_taken_from_the_first_trajectory():
    ds = _build([support.scripted_trajectory(3.0, t_max=6.0, dt=0.2)
                 for _ in range(3)])
    assert ds.metadata["dt"] == pytest.approx(0.2)


def test_build_dataset_time_step_falls_back_to_a_tenth_for_a_one_point_start():
    one_point = dataclasses.replace(
        support.scripted_trajectory(3.0, t_max=6.0, dt=0.2),
        times=np.zeros(1), observables={
            k: np.zeros(1, dtype=np.float32)
            for k in ("sigma_x", "coherence_l1", "purity")})
    ds = _build([one_point] + [
        support.scripted_trajectory(3.0, t_max=6.0, dt=0.2)
        for _ in range(3)])
    assert ds.metadata["dt"] == 0.1


def test_build_dataset_context_codes_follow_each_trajectory(
        scenario_configs):
    one = support.decay_trajectory(support.make_system_config(
        gamma=0.4, t_max=20.0, dt=0.1))
    two = dataclasses.replace(
        one, system_config=scenario_configs["D"])
    ds = _build([one, two], window_length=5, train_ratio=1.0, val_ratio=0.0,
                samples_per_trajectory=2)
    assert set(ds.context["interaction_type"]) == {0.0, 2.0}
    assert set(ds.context["J"]) == {0.0, np.float32(scenario_configs["D"].J)}
    assert len(ds.context["dissipator_type"]) == len(ds.sequences)


def test_build_dataset_pads_one_qubit_windows_next_to_two_qubit_ones():
    one = support.decay_trajectory(support.make_system_config(t_max=8.0))
    wide = dataclasses.replace(
        one, t_decoh=6.0, observables={
            "a": np.ones(81, dtype=np.float32),
            "b": np.ones(81, dtype=np.float32),
            **one.observables})
    ds = _build([one, wide], window_length=5, samples_per_trajectory=2,
                train_ratio=1.0, val_ratio=0.0)
    assert ds.sequences.shape[2] == 7
    assert ds.feature_names == [
        "a", "b", "sigma_x", "sigma_y", "sigma_z", "purity", "coherence_l1"]
    narrow = ds.sequences[ds.t_decoh_abs == np.float32(one.t_decoh)]
    full = ds.sequences[ds.t_decoh_abs == np.float32(6.0)]
    assert len(narrow) == len(full) == 2
    # The zero columns sit right before purity and coherence_l1.
    assert not narrow[:, :, 3:5].any() and narrow[:, :, :3].any()
    assert full[:, :, :2].all()


# --- Negative cases ---


@pytest.mark.parametrize("train,val", [
    (-0.1, 0.5), (0.5, -0.1), (1.1, 0.0), (0.0, 1.1), (NAN, 0.1),
    (0.1, NAN)])
def test_assign_splits_ratio_outside_zero_one_raises_value_error(train, val):
    with pytest.raises(ValueError, match="ratio must be in"):
        windows.assign_splits(10, train, val, _rng())


@pytest.mark.parametrize("train,val", [(0.8, 0.5), (1.0, 0.01), (0.6, 0.6)])
def test_assign_splits_ratios_summing_above_one_raise_value_error(train, val):
    with pytest.raises(ValueError, match="must not exceed 1"):
        windows.assign_splits(10, train, val, _rng())


def test_assign_splits_negative_count_raises_value_error():
    with pytest.raises(ValueError, match="n_trajectories"):
        windows.assign_splits(-1, 0.8, 0.1, _rng())


def test_assign_splits_does_not_consume_the_generator_on_invalid_ratios():
    rng = _rng(3)
    with pytest.raises(ValueError):
        windows.assign_splits(10, 0.9, 0.9, rng)
    assert rng.random() == _rng(3).random()


@pytest.mark.parametrize("window_length", [0, -1, -50])
def test_valid_window_ends_window_below_one_raises_value_error(window_length):
    with pytest.raises(ValueError, match="window_length must be at least 1"):
        windows.valid_window_ends(np.arange(10.0), 5.0, window_length)


def test_valid_window_ends_of_a_short_t2_or_long_window_is_empty():
    times = np.arange(30) * 0.1
    assert len(windows.valid_window_ends(times, 0.3, 5)) == 0
    assert len(windows.valid_window_ends(times, 2.0, 50)) == 0
    assert len(windows.valid_window_ends(times[:0], 2.0, 5)) == 0
    assert len(windows.valid_window_ends(times, NAN, 5)) == 0
    assert len(windows.valid_window_ends(times, -1.0, 5)) == 0


@pytest.mark.parametrize("n_samples", [-1, -7])
@pytest.mark.parametrize("gap", [0, 5])
def test_sample_window_ends_negative_sample_count_raises_value_error(
        n_samples, gap):
    with pytest.raises(ValueError, match="n_samples must be non-negative"):
        windows.sample_window_ends(np.arange(20), n_samples, gap, _rng())


@pytest.mark.parametrize("gap", [-1, -10])
def test_sample_window_ends_negative_gap_raises_value_error(gap):
    with pytest.raises(ValueError, match="min_gap must be non-negative"):
        windows.sample_window_ends(np.arange(20), 3, gap, _rng())


def test_pad_features_without_windows_raises_value_error():
    with pytest.raises(ValueError, match="no windows to pad"):
        windows.pad_features([])


@pytest.mark.parametrize("sequences", [
    [np.zeros(3), np.zeros(5)], [np.zeros((2, 2, 2)), np.zeros((2, 3))]])
def test_pad_features_windows_that_are_not_two_dimensional_raise(sequences):
    with pytest.raises(ValueError, match="shape \\(length, n_features\\)"):
        windows.pad_features(sequences)


def test_pad_features_window_narrower_than_the_physics_columns_raises():
    with pytest.raises(ValueError, match="at least 2 columns, got 1"):
        windows.pad_features([np.zeros((3, 1)), np.zeros((3, 5))])


@pytest.mark.parametrize("window_length", [0, -3])
def test_build_dataset_window_length_below_one_raises_value_error(
        window_length):
    with pytest.raises(ValueError, match="window_length must be at least 1"):
        _build(window_length=window_length)


@pytest.mark.parametrize("samples", [0, -2])
def test_build_dataset_non_positive_samples_per_trajectory_raises(samples):
    with pytest.raises(ValueError, match="samples_per_trajectory"):
        _build(samples_per_trajectory=samples)


@pytest.mark.parametrize("horizon", [0.0, -1.0, NAN])
def test_build_dataset_non_positive_horizon_raises_value_error(horizon):
    with pytest.raises(ValueError, match="horizon must be positive"):
        _build(horizon=horizon)


@pytest.mark.parametrize("gap", [-1, -10])
def test_build_dataset_negative_min_window_gap_raises_value_error(gap):
    with pytest.raises(ValueError, match="min_window_gap"):
        _build(min_window_gap=gap)


@pytest.mark.parametrize("train,val", [(0.8, 0.5), (-0.1, 0.2), (0.5, 1.5)])
def test_build_dataset_invalid_split_ratios_raise_value_error(train, val):
    with pytest.raises(ValueError, match="ratio"):
        _build(train_ratio=train, val_ratio=val)


def test_build_dataset_validates_before_drawing_random_numbers():
    rng = _rng(9)
    with pytest.raises(ValueError):
        _build(window_length=0, rng=rng)
    assert rng.random() == _rng(9).random()


def test_build_dataset_without_trajectories_raises_runtime_error():
    with pytest.raises(RuntimeError, match="No valid pre-decoherence"):
        _build([])


def test_build_dataset_window_as_long_as_every_trajectory_raises():
    n_points = len(_trajectories(1)[0].times)
    for window_length in (n_points, n_points + 50):
        with pytest.raises(RuntimeError, match="No valid pre-decoherence"):
            _build(window_length=window_length)


def test_build_dataset_t2_shorter_than_the_window_raises_runtime_error():
    with pytest.raises(RuntimeError, match="Reduce window_length"):
        _build([support.scripted_trajectory(0.2)] * 3, window_length=5)


def test_build_dataset_all_trajectories_censored_and_excluded_raises():
    with pytest.raises(RuntimeError, match="No valid pre-decoherence"):
        _build([_censored()] * 4, include_censored=False)


def test_build_dataset_all_trajectories_censored_but_included_still_works():
    ds = _build([_censored()] * 4)
    assert ds.censored.all() and (ds.risk_labels == 0).all()
    assert ds.metadata["n_censored"] == 4
