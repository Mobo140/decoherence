"""Dataset: copy-on-write operations on the windows and the splits."""
import numpy as np
import pytest


def test_with_sequences_replaces_only_the_windows(small_dataset):
    ds = small_dataset
    new = ds.with_sequences(np.zeros(ds.sequences.shape))
    assert new.sequences.dtype == np.float32 and not new.sequences.any()
    assert ds.sequences.any()
    assert new.test_idx is ds.test_idx and new.t_obs is ds.t_obs
    with pytest.raises(ValueError):
        ds.with_sequences(ds.sequences[:2])


def test_select_and_drop_features_keep_dataset_order(small_dataset):
    ds = small_dataset
    kept = ds.select_features(["concurrence", "a"])
    assert kept.feature_names == ["a", "concurrence"]
    np.testing.assert_array_equal(kept.sequences, ds.sequences[:, :, [0, 3]])
    dropped = ds.drop_features(["corr_xx", "not_a_column"])
    assert dropped.feature_names == ["a", "b", "concurrence"]
    assert ds.sequences.shape[-1] == 4
    with pytest.raises(ValueError):
        ds.select_features(["not_a_column"])


def test_as_test_only_moves_every_window_to_test(small_dataset):
    ds = small_dataset
    out = ds.as_test_only()
    assert list(out.test_idx) == list(range(6))
    assert len(out.train_idx) == 0 and len(out.val_idx) == 0
    assert out.train_idx.dtype == np.int64
    assert list(ds.test_idx) == [4, 5]
