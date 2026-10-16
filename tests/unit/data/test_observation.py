"""Observation models: registry aliases, noise augmentation and AR(1) noise."""
import numpy as np
import pytest

from src.data import observation
from src.data.observation import ar1_filter


def test_observation_registry_aliases(small_dataset):
    assert set(observation.OBSERVATIONS) == {"exact", "gaussian", "ar1"}
    ds = small_dataset
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


def test_noise_augmented_appends_copies_to_train_and_val_only(small_dataset):
    ds = small_dataset
    out = observation.noise_augmented(ds, 2, np.random.default_rng(0))
    assert len(out.sequences) == 6 + 2 * 4
    assert list(out.test_idx) == [4, 5]
    assert len(out.train_idx) == 3 + 2 * 3 and len(out.val_idx) == 1 + 2
    np.testing.assert_array_equal(out.sequences[:6], ds.sequences)


def test_ar1_is_stationary_with_unit_variance_and_lag_one_correlation_phi():
    eta = np.random.default_rng(0).standard_normal((4000, 200))
    for phi in (0.0, 0.5, 0.9):
        e = ar1_filter(eta, phi)
        assert abs(e.var() - 1.0) < 0.03
        assert abs(e[:, 0].var() - 1.0) < 0.08          # stationary from t = 0
        r1 = np.mean(e[:, 1:] * e[:, :-1])
        assert abs(r1 - phi) < 0.02


def test_ar1_accepts_one_correlation_per_window():
    eta = np.random.default_rng(1).standard_normal((3000, 100, 2))
    phi = np.repeat([0.0, 0.9], 1500)
    e = ar1_filter(eta, phi)
    np.testing.assert_array_equal(e[:1500], eta[:1500])
    r1 = np.mean(e[1500:, 1:] * e[1500:, :-1])
    assert abs(r1 - 0.9) < 0.02 and abs(e[1500:].var() - 1.0) < 0.05
