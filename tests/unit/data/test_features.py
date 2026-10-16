"""Feature sets: which columns each alias keeps."""
import pytest

from src.data import features


def test_select_drops_the_documented_columns_per_feature_set(small_dataset):
    ds = small_dataset
    assert features.select(ds, "base").feature_names == ["a", "b"]
    assert features.select(ds, "corr").feature_names == ["a", "b", "corr_xx"]
    assert features.select(ds, "corr_conc").feature_names == ds.feature_names
    with pytest.raises(KeyError):
        features.select(ds, "unknown")
