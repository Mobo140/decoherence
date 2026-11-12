"""src.data: Dataset operations, scenarios and window building."""
import numpy as np
import pytest

from src.data import scenarios, windows
from src.data.dataset import Dataset
from src.domain import value_objects


def _dataset(n=6, length=4, names=("a", "b", "corr_xx", "concurrence")):
    rng = np.random.default_rng(0)
    return Dataset(
        sequences=rng.normal(size=(n, length, len(names))).astype(np.float32),
        t_obs=np.ones(n, dtype=np.float32),
        t_decoh_abs=np.full(n, 5.0, dtype=np.float32),
        remaining_time=np.full(n, 4.0, dtype=np.float32),
        risk_labels=np.zeros(n, dtype=np.float32),
        train_idx=np.array([0, 1, 2]),
        val_idx=np.array([3]),
        test_idx=np.array([4, 5]),
        feature_names=list(names),
        metadata={},
    )


def test_with_sequences_replaces_only_the_windows():
    ds = _dataset()
    new = ds.with_sequences(np.zeros(ds.sequences.shape))
    assert new.sequences.dtype == np.float32 and not new.sequences.any()
    assert ds.sequences.any()
    assert new.test_idx is ds.test_idx and new.t_obs is ds.t_obs
    with pytest.raises(ValueError):
        ds.with_sequences(ds.sequences[:2])


def test_select_and_drop_features_keep_dataset_order():
    ds = _dataset()
    kept = ds.select_features(["concurrence", "a"])
    assert kept.feature_names == ["a", "concurrence"]
    np.testing.assert_array_equal(kept.sequences, ds.sequences[:, :, [0, 3]])
    dropped = ds.drop_features(["corr_xx", "not_a_column"])
    assert dropped.feature_names == ["a", "b", "concurrence"]
    assert ds.sequences.shape[-1] == 4
    with pytest.raises(ValueError):
        ds.select_features(["not_a_column"])


def test_as_test_only_moves_every_window_to_test():
    ds = _dataset()
    out = ds.as_test_only()
    assert list(out.test_idx) == list(range(6))
    assert len(out.train_idx) == 0 and len(out.val_idx) == 0
    assert out.train_idx.dtype == np.int64
    assert list(ds.test_idx) == [4, 5]


def test_assign_splits_is_a_trajectory_level_partition():
    split = windows.assign_splits(20, 0.8, 0.1, np.random.default_rng(1))
    assert split.count("train") == 16 and split.count("val") == 2
    assert split.count("test") == 2
    assert split == windows.assign_splits(
        20, 0.8, 0.1, np.random.default_rng(1))


def test_sample_window_ends_respects_the_minimum_gap():
    ends = np.arange(20, 200)
    chosen = windows.sample_window_ends(ends, 5, 10, np.random.default_rng(2))
    assert len(chosen) == 5
    gaps = np.abs(np.subtract.outer(chosen, chosen))[np.triu_indices(5, 1)]
    assert gaps.min() >= 10
    times = np.arange(30) * 0.1
    valid = windows.valid_window_ends(times, 2.0, 5)
    assert valid.min() == 5 and times[valid].max() < 2.0


def test_pad_features_inserts_zeros_before_the_last_two_columns():
    short = np.arange(12, dtype=np.float32).reshape(3, 4)
    full = np.ones((3, 6), dtype=np.float32)
    out = windows.pad_features([short, full])
    assert out[0].shape == (3, 6)
    np.testing.assert_array_equal(out[0][:, :2], short[:, :2])
    assert not out[0][:, 2:4].any()
    np.testing.assert_array_equal(out[0][:, 4:], short[:, 2:])


def test_context_codes_distinguish_the_four_system_types():
    g = scenarios.build_configs(n_per_scenario=1, seed=0)
    codes = {sc: windows.context_codes(g[sc][0]) for sc in "ABCDE"}
    assert [codes[sc][0] for sc in "ABCDE"] == [0.0, 0.0, 1.0, 2.0, 3.0]
    assert [codes[sc][1] for sc in "ABCDE"] == [0.0, 0.0, 1.0, 0.0, 0.0]
    assert codes["A"][2] == 0.0 and codes["D"][2] == g["D"][0].J


def test_fixed_coupling_configs_share_the_coupling():
    cfgs = scenarios.fixed_coupling_configs(
        7, 1.3, np.random.default_rng(0),
        interaction=value_objects.InteractionType.XXZ)
    assert len(cfgs) == 7 and {c.J for c in cfgs} == {1.3}
    assert {c.interaction_type for c in cfgs} == {
        value_objects.InteractionType.XXZ}
