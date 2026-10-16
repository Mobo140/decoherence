"""Dataset: invariants and copy-on-write operations on windows and splits."""
import dataclasses

import numpy as np
import pytest

from src.data import dataset as dataset_lib


def _empty(ds: dataset_lib.Dataset) -> dataset_lib.Dataset:
    """The dataset with every window removed."""
    return dataclasses.replace(
        ds, sequences=ds.sequences[:0], t_obs=ds.t_obs[:0],
        t_decoh_abs=ds.t_decoh_abs[:0], remaining_time=ds.remaining_time[:0],
        risk_labels=ds.risk_labels[:0], train_idx=np.array([], dtype=int),
        val_idx=np.array([], dtype=int), test_idx=np.array([], dtype=int))


# --- construction ----------------------------------------------------------

def test_dataset_optional_per_window_arrays_default_to_empty(small_dataset):
    ds = small_dataset
    assert len(ds.interaction_type_codes) == 0 and len(ds.censored) == 0
    assert len(ds.dissipator_type_codes) == 0 and len(ds.J_values) == 0
    assert ds.censored.dtype == bool


def test_dataset_accepts_complete_optional_arrays(small_dataset):
    ds = dataclasses.replace(
        small_dataset, censored=np.zeros(6, dtype=bool),
        J_values=np.zeros(6, dtype=np.float32))
    assert len(ds.censored) == 6 and len(ds.J_values) == 6


def test_dataset_accepts_split_indices_up_to_the_last_window(small_dataset):
    ds = dataclasses.replace(small_dataset, test_idx=np.array([0, 5]))
    assert list(ds.test_idx) == [0, 5]


def test_dataset_may_have_no_windows(small_dataset):
    ds = _empty(small_dataset)
    assert ds.sequences.shape == (0, 4, 4) and len(ds.test_idx) == 0


# --- with_sequences --------------------------------------------------------

def test_with_sequences_replaces_only_the_windows(small_dataset):
    ds = small_dataset
    new = ds.with_sequences(np.zeros(ds.sequences.shape))
    assert new.sequences.dtype == np.float32 and not new.sequences.any()
    assert ds.sequences.any()
    assert new.test_idx is ds.test_idx and new.t_obs is ds.t_obs
    with pytest.raises(ValueError):
        ds.with_sequences(ds.sequences[:2])


def test_with_sequences_converts_lists_to_float32(small_dataset):
    ds = small_dataset
    new = ds.with_sequences(np.zeros((6, 4, 4), dtype=np.float64).tolist())
    assert new.sequences.dtype == np.float32


def test_with_sequences_may_change_the_window_length(small_dataset):
    new = small_dataset.with_sequences(np.zeros((6, 7, 4)))
    assert new.sequences.shape == (6, 7, 4)


# --- select_features and drop_features -------------------------------------

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


def test_select_features_accepts_any_collection_of_names(small_dataset):
    ds = small_dataset
    for names in (("a", "b"), {"a", "b"}, iter(["a", "b"])):
        assert ds.select_features(names).feature_names == ["a", "b"]


def test_select_features_of_every_column_keeps_the_data(small_dataset):
    ds = small_dataset
    same = ds.select_features(ds.feature_names)
    np.testing.assert_array_equal(same.sequences, ds.sequences)
    assert same.feature_names == ds.feature_names


def test_select_features_returns_a_contiguous_copy_of_the_columns(
        small_dataset):
    kept = small_dataset.select_features(["b"])
    assert kept.sequences.flags["C_CONTIGUOUS"]
    assert not np.shares_memory(kept.sequences, small_dataset.sequences)
    assert kept.t_obs is small_dataset.t_obs


def test_drop_features_without_names_changes_nothing(small_dataset):
    ds = small_dataset
    out = ds.drop_features([])
    assert out.feature_names == ds.feature_names
    np.testing.assert_array_equal(out.sequences, ds.sequences)


def test_drop_features_accepts_a_frozenset_and_leaves_one_column(
        small_dataset):
    out = small_dataset.drop_features(frozenset({"a", "b", "corr_xx"}))
    assert out.feature_names == ["concurrence"]


def test_select_features_on_an_empty_dataset_keeps_the_window_axes(
        small_dataset):
    out = _empty(small_dataset).select_features(["a"])
    assert out.sequences.shape == (0, 4, 1)


# --- as_test_only ----------------------------------------------------------

def test_as_test_only_moves_every_window_to_test(small_dataset):
    ds = small_dataset
    out = ds.as_test_only()
    assert list(out.test_idx) == list(range(6))
    assert len(out.train_idx) == 0 and len(out.val_idx) == 0
    assert out.train_idx.dtype == np.int64
    assert list(ds.test_idx) == [4, 5]


def test_as_test_only_shares_the_windows(small_dataset):
    out = small_dataset.as_test_only()
    assert out.sequences is small_dataset.sequences


# --- Negative cases ---


@pytest.mark.parametrize("shape", [(6, 4), (6,), (6, 4, 4, 1)])
def test_dataset_sequences_that_are_not_three_dimensional_raise(
        small_dataset, shape):
    with pytest.raises(ValueError, match="shape \\(N, window_length"):
        dataclasses.replace(small_dataset, sequences=np.zeros(shape))


def test_dataset_feature_names_must_match_the_last_axis(small_dataset):
    with pytest.raises(ValueError, match="4 features but feature_names has 3"):
        dataclasses.replace(small_dataset, feature_names=["a", "b", "c"])
    with pytest.raises(ValueError, match="feature_names has 5"):
        dataclasses.replace(
            small_dataset, feature_names=["a", "b", "c", "d", "e"])


@pytest.mark.parametrize(
    "field", ["t_obs", "t_decoh_abs", "remaining_time", "risk_labels"])
def test_dataset_per_window_array_of_the_wrong_length_raises(
        small_dataset, field):
    with pytest.raises(ValueError, match=f"{field} has 5 entries for 6"):
        dataclasses.replace(small_dataset, **{field: np.zeros(5)})


@pytest.mark.parametrize("field", [
    "interaction_type_codes", "dissipator_type_codes", "J_values",
    "censored"])
def test_dataset_partial_optional_array_raises(small_dataset, field):
    with pytest.raises(ValueError, match=f"{field} has 3 entries for 6"):
        dataclasses.replace(small_dataset, **{field: np.zeros(3)})


@pytest.mark.parametrize("field", ["train_idx", "val_idx", "test_idx"])
@pytest.mark.parametrize("index", [6, 100, -1])
def test_dataset_split_index_outside_the_windows_raises(
        small_dataset, field, index):
    with pytest.raises(ValueError, match=f"{field} contains an index"):
        dataclasses.replace(
            small_dataset, **{field: np.array([0, index])})


def test_with_sequences_wrong_number_of_windows_raises_value_error(
        small_dataset):
    with pytest.raises(ValueError, match="Expected 6 windows, got 2"):
        small_dataset.with_sequences(small_dataset.sequences[:2])
    with pytest.raises(ValueError, match="Expected 6 windows, got 0"):
        small_dataset.with_sequences(np.zeros((0, 4, 4)))


def test_with_sequences_wrong_number_of_features_raises_value_error(
        small_dataset):
    with pytest.raises(ValueError, match="2 features but feature_names has 4"):
        small_dataset.with_sequences(np.zeros((6, 4, 2)))


def test_with_sequences_two_dimensional_windows_raise_value_error(
        small_dataset):
    with pytest.raises(ValueError, match="window_length, n_features"):
        small_dataset.with_sequences(np.zeros((6, 4)))


@pytest.mark.parametrize("value", [None, 5])
def test_with_sequences_without_an_array_raises_type_error(
        small_dataset, value):
    with pytest.raises(TypeError, match="has no len"):
        small_dataset.with_sequences(value)


def test_select_features_unknown_name_raises_and_lists_the_unknown_ones(
        small_dataset):
    with pytest.raises(ValueError, match="Unknown features: .*zzz"):
        small_dataset.select_features(["a", "zzz", "yyy"])


def test_select_features_without_names_raises_value_error(small_dataset):
    with pytest.raises(ValueError, match="at least one feature column"):
        small_dataset.select_features([])


@pytest.mark.parametrize("names", ["a", "ab", "concurrence"])
def test_select_features_bare_string_raises_type_error(small_dataset, names):
    with pytest.raises(TypeError, match="collection of feature names"):
        small_dataset.select_features(names)


def test_drop_features_every_column_raises_value_error(small_dataset):
    with pytest.raises(ValueError, match="at least one feature column"):
        small_dataset.drop_features(small_dataset.feature_names)


@pytest.mark.parametrize("names", ["a", "corr_xx"])
def test_drop_features_bare_string_raises_type_error(small_dataset, names):
    with pytest.raises(TypeError, match="collection of feature names"):
        small_dataset.drop_features(names)


def test_drop_features_unknown_names_are_ignored_not_an_error(small_dataset):
    # Documented: feature sets name columns a dataset may not have.
    out = small_dataset.drop_features(["nope"])
    assert out.feature_names == small_dataset.feature_names


def test_as_test_only_of_an_empty_dataset_gives_empty_splits(small_dataset):
    # Debatable, kept: nothing to move is not an error.
    out = _empty(small_dataset).as_test_only()
    assert len(out.test_idx) == 0 and out.test_idx.dtype == np.int64

