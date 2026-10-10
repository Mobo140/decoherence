"""src.data: observation models and feature sets."""
import numpy as np
import pytest

from src.data import features, observation
from src.data.dataset import Dataset


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


def test_feature_sets_drop_the_documented_columns():
    ds = _dataset()
    assert features.select(ds, "base").feature_names == ["a", "b"]
    assert features.select(ds, "corr").feature_names == ["a", "b", "corr_xx"]
    assert features.select(ds, "corr_conc").feature_names == ds.feature_names
    with pytest.raises(KeyError):
        features.select(ds, "unknown")


def test_observation_registry_aliases():
    assert set(observation.OBSERVATIONS) == {"exact", "gaussian", "ar1"}
    ds = _dataset()
    rng = np.random.default_rng
    exact = observation.OBSERVATIONS["exact"]
    assert exact(ds, sigma=0.0, phi=0.0, rng=rng(0)) is ds
    with pytest.raises(ValueError):
        exact(ds, sigma=0.1, rng=rng(0))

    gauss = observation.OBSERVATIONS["gaussian"]
    assert gauss(ds, sigma=0.0, rng=rng(0)) is ds
    noisy = gauss(ds, sigma=0.1, rng=rng(0))
    expected = ds.sequences + rng(0).normal(
        0.0, 0.1, size=ds.sequences.shape).astype(np.float32)
    np.testing.assert_array_equal(noisy.sequences, expected)
    with pytest.raises(ValueError):
        gauss(ds, sigma=0.1, phi=0.5, rng=rng(0))

    ar1 = observation.OBSERVATIONS["ar1"]
    white = ar1(ds, sigma=0.1, phi=0.0, rng=rng(0))
    eta = rng(0).normal(0.0, 1.0, size=ds.sequences.shape)
    np.testing.assert_array_equal(
        white.sequences, (ds.sequences + 0.1 * eta).astype(np.float32))
    correlated = ar1(ds, sigma=0.1, phi=0.9, rng=rng(0))
    assert not np.array_equal(correlated.sequences, white.sequences)


def test_noise_augmented_appends_copies_to_train_and_val_only():
    ds = _dataset()
    out = observation.noise_augmented(ds, 2, np.random.default_rng(0))
    assert len(out.sequences) == 6 + 2 * 4
    assert list(out.test_idx) == [4, 5]
    assert len(out.train_idx) == 3 + 2 * 3 and len(out.val_idx) == 1 + 2
    np.testing.assert_array_equal(out.sequences[:6], ds.sequences)
