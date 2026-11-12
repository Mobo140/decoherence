"""Feature sets: which columns each alias keeps."""
import pytest

from src.data import dataset as dataset_lib
from src.data import features
from src.simulation import variants


def test_select_drops_the_documented_columns_per_feature_set(small_dataset):
    ds = small_dataset
    assert features.select(ds, "base").feature_names == ["a", "b"]
    assert features.select(ds, "corr").feature_names == ["a", "b", "corr_xx"]
    assert features.select(ds, "corr_conc").feature_names == ds.feature_names
    with pytest.raises(KeyError):
        features.select(ds, "unknown")


def test_feature_sets_name_the_columns_each_alias_removes():
    assert set(features.FEATURE_SETS) == {"base", "corr", "corr_conc"}
    assert features.FEATURE_SETS["corr_conc"] == frozenset()
    assert features.FEATURE_SETS["corr"] == {"concurrence"}
    assert features.FEATURE_SETS["base"] == (
        set(variants.CORR) | {"concurrence"})


def test_select_ignores_columns_the_dataset_does_not_have(small_dataset):
    plain = small_dataset.select_features(["a", "b"])
    assert features.select(plain, "base").feature_names == ["a", "b"]
    assert features.select(plain, "corr_conc").feature_names == ["a", "b"]


def test_select_does_not_modify_its_input(small_dataset):
    features.select(small_dataset, "base")
    assert small_dataset.feature_names == ["a", "b", "corr_xx", "concurrence"]
    assert small_dataset.sequences.shape[-1] == 4


# --- Negative cases ---


def test_select_unknown_alias_raises_key_error_listing_valid_ones(
        small_dataset):
    with pytest.raises(KeyError) as excinfo:
        features.select(small_dataset, "local")
    message = str(excinfo.value)
    assert "'local'" in message
    for alias in features.FEATURE_SETS:
        assert alias in message


@pytest.mark.parametrize("alias", ["Base", "BASE", "", None, 1])
def test_select_alias_is_matched_exactly(small_dataset, alias):
    with pytest.raises(KeyError, match="Unknown feature set"):
        features.select(small_dataset, alias)


def test_select_of_a_set_that_would_remove_every_column_raises(small_dataset):
    only_extras = small_dataset.select_features(["corr_xx", "concurrence"])
    with pytest.raises(ValueError, match="at least one feature column"):
        features.select(only_extras, "base")


def test_feature_sets_cannot_be_modified():
    with pytest.raises(TypeError):
        features.FEATURE_SETS["extra"] = frozenset()


def test_select_returns_a_dataset_of_the_same_type(small_dataset):
    assert isinstance(
        features.select(small_dataset, "corr"), dataset_lib.Dataset)
