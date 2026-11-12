"""Observation models: registry aliases, noise augmentation and AR(1) noise."""
import dataclasses

import numpy as np
import pytest

from src.data import observation
from src.data.observation import ar1_filter

NAN = float("nan")
INF = float("inf")


def _rng(seed: int = 0) -> np.random.Generator:
    return np.random.default_rng(seed)


def _with_codes(ds):
    """The dataset with every optional per-window array filled in."""
    n = len(ds.sequences)
    return dataclasses.replace(
        ds, context={
            "interaction_type": np.arange(n, dtype=np.float32),
            "dissipator_type": np.zeros(n, dtype=np.float32),
            "J": np.linspace(0, 1, n, dtype=np.float32)},
        censored=np.arange(n) % 2 == 0)


# --- registry and the three models -----------------------------------------

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


def test_exact_observation_ignores_the_generator(small_dataset):
    assert observation.exact(small_dataset) is small_dataset


def test_gaussian_noise_is_reproducible_and_leaves_the_input_alone(
        small_dataset):
    before = small_dataset.sequences.copy()
    a = observation.gaussian(small_dataset, sigma=0.2, rng=_rng(3))
    b = observation.gaussian(small_dataset, sigma=0.2, rng=_rng(3))
    c = observation.gaussian(small_dataset, sigma=0.2, rng=_rng(4))
    np.testing.assert_array_equal(a.sequences, b.sequences)
    assert not np.array_equal(a.sequences, c.sequences)
    np.testing.assert_array_equal(small_dataset.sequences, before)
    assert a.t_obs is small_dataset.t_obs


def test_gaussian_noise_scales_with_sigma(small_dataset):
    small = observation.gaussian(small_dataset, sigma=0.01, rng=_rng(0))
    large = observation.gaussian(small_dataset, sigma=1.0, rng=_rng(0))
    ratio = (np.abs(large.sequences - small_dataset.sequences).mean()
             / np.abs(small.sequences - small_dataset.sequences).mean())
    assert ratio == pytest.approx(100.0, rel=0.05)


def test_gaussian_with_zero_sigma_does_not_need_a_generator(small_dataset):
    assert observation.gaussian(
        small_dataset, sigma=0.0, rng=None) is small_dataset


def test_ar1_models_with_equal_generators_share_one_realisation(
        small_dataset):
    low = observation.ar1(small_dataset, sigma=0.1, phi=0.0, rng=_rng(5))
    high = observation.ar1(small_dataset, sigma=0.2, phi=0.0, rng=_rng(5))
    np.testing.assert_allclose(
        high.sequences - small_dataset.sequences,
        2 * (low.sequences - small_dataset.sequences), atol=1e-6)


def test_ar1_with_zero_sigma_still_draws_and_returns_the_same_windows(
        small_dataset):
    rng = _rng(1)
    out = observation.ar1(small_dataset, sigma=0.0, phi=0.5, rng=rng)
    np.testing.assert_array_equal(out.sequences, small_dataset.sequences)
    # The stream moved on, so a following draw is not the first one.
    assert rng.normal() != _rng(1).normal()


def test_observations_of_an_empty_dataset_are_empty(small_dataset):
    empty = dataclasses.replace(
        small_dataset, sequences=small_dataset.sequences[:0],
        t_obs=small_dataset.t_obs[:0],
        t_decoh_abs=small_dataset.t_decoh_abs[:0],
        remaining_time=small_dataset.remaining_time[:0],
        risk_labels=small_dataset.risk_labels[:0],
        train_idx=np.array([], dtype=int), val_idx=np.array([], dtype=int),
        test_idx=np.array([], dtype=int))
    for model in ("gaussian", "ar1"):
        out = observation.OBSERVATIONS[model](
            empty, sigma=0.1, phi=0.0, rng=_rng())
        assert out.sequences.shape == (0, 4, 4)


# --- noise_augmented -------------------------------------------------------

def test_noise_augmented_appends_copies_to_train_and_val_only(small_dataset):
    ds = small_dataset
    out = observation.noise_augmented(ds, 2, np.random.default_rng(0))
    assert len(out.sequences) == 6 + 2 * 4
    assert list(out.test_idx) == [4, 5]
    assert len(out.train_idx) == 3 + 2 * 3 and len(out.val_idx) == 1 + 2
    np.testing.assert_array_equal(out.sequences[:6], ds.sequences)


def test_noise_augmented_without_copies_changes_nothing(small_dataset):
    out = observation.noise_augmented(small_dataset, 0, _rng())
    np.testing.assert_array_equal(out.sequences, small_dataset.sequences)
    assert list(out.train_idx) == [0, 1, 2] and list(out.val_idx) == [3]


def test_noise_augmented_copies_keep_the_labels_of_their_source(
        small_dataset):
    ds = _with_codes(small_dataset)
    out = observation.noise_augmented(ds, 1, _rng())
    source = np.concatenate([ds.train_idx, ds.val_idx])
    np.testing.assert_array_equal(out.t_obs[6:], ds.t_obs[source])
    np.testing.assert_array_equal(out.remaining_time[6:],
                                  ds.remaining_time[source])
    for key, values in ds.context.items():
        np.testing.assert_array_equal(out.context[key][:6], values)
        np.testing.assert_array_equal(out.context[key][6:], values[source])
    np.testing.assert_array_equal(out.censored[6:], ds.censored[source])
    assert list(out.train_idx) == [0, 1, 2, 6, 7, 8]
    assert list(out.val_idx) == [3, 9]


def test_noise_augmented_leaves_empty_optional_arrays_empty(small_dataset):
    out = observation.noise_augmented(small_dataset, 2, _rng())
    assert out.context == {} and len(out.censored) == 0


def test_noise_augmented_noise_is_within_the_sigma_range(small_dataset):
    # Long windows make the sample standard deviation of each copy accurate.
    long = dataclasses.replace(
        small_dataset, sequences=np.zeros((6, 2000, 4), dtype=np.float32))
    out = observation.noise_augmented(long, 20, _rng())
    noise = out.sequences[6:]
    per_window_std = np.sqrt((noise ** 2).mean(axis=(1, 2)))
    assert 0.0049 < per_window_std.min() and per_window_std.max() < 0.1225
    assert per_window_std.max() > 5 * per_window_std.min()


def test_noise_augmented_is_reproducible_and_leaves_the_input_alone(
        small_dataset):
    before = small_dataset.sequences.copy()
    a = observation.noise_augmented(small_dataset, 2, _rng(8))
    b = observation.noise_augmented(small_dataset, 2, _rng(8))
    np.testing.assert_array_equal(a.sequences, b.sequences)
    np.testing.assert_array_equal(small_dataset.sequences, before)


def test_noise_augmented_correlated_noise_differs_from_white_noise(
        small_dataset):
    white = observation.noise_augmented(small_dataset, 1, _rng(2))
    correlated = observation.noise_augmented(
        small_dataset, 1, _rng(2), phi_max=0.9)
    assert not np.array_equal(white.sequences[6:], correlated.sequences[6:])
    np.testing.assert_array_equal(white.sequences[:6],
                                  correlated.sequences[:6])


def test_noise_augmented_accepts_the_boundary_phi_max(small_dataset):
    out = observation.noise_augmented(small_dataset, 1, _rng(), phi_max=1.0)
    assert np.isfinite(out.sequences).all()


def test_noise_augmented_of_a_test_only_dataset_adds_nothing_to_train(
        small_dataset):
    # Debatable, kept: there are no train/val windows to copy, so the
    # augmentation is a no-op instead of an error.
    out = observation.noise_augmented(
        small_dataset.as_test_only(), 2, _rng())
    assert len(out.sequences) == 6 and len(out.train_idx) == 0


# --- ar1_filter ------------------------------------------------------------

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


def test_ar1_zero_phi_returns_the_input_unchanged():
    eta = np.random.default_rng(0).standard_normal((5, 6))
    assert ar1_filter(eta, 0.0) is eta
    assert ar1_filter(eta, np.zeros(5)) is eta


def test_ar1_follows_the_recursion_and_keeps_the_first_step():
    eta = np.random.default_rng(2).standard_normal((3, 5))
    out = ar1_filter(eta, 0.6)
    np.testing.assert_array_equal(out[:, 0], eta[:, 0])
    expected = 0.6 * out[:, 1] + np.sqrt(1 - 0.36) * eta[:, 2]
    np.testing.assert_allclose(out[:, 2], expected)


def test_ar1_negative_correlation_alternates_the_sign():
    eta = np.random.default_rng(3).standard_normal((4000, 50))
    out = ar1_filter(eta, -0.7)
    assert np.mean(out[:, 1:] * out[:, :-1]) == pytest.approx(-0.7, abs=0.03)


def test_ar1_keeps_the_dtype_of_the_noise():
    eta = np.random.default_rng(0).standard_normal((4, 6)).astype(np.float32)
    assert ar1_filter(eta, 0.5).dtype == np.float32


def test_ar1_of_no_windows_is_empty():
    assert ar1_filter(np.zeros((0, 5)), 0.3).shape == (0, 5)


# --- Negative cases ---


@pytest.mark.parametrize("phi", [1.0, -1.0, 1.5, -2.0, NAN, INF])
def test_ar1_filter_non_stationary_phi_raises_value_error(phi):
    with pytest.raises(ValueError, match=r"\|phi\| < 1"):
        ar1_filter(np.ones((4, 5)), phi)


def test_ar1_filter_one_phi_of_an_array_outside_the_range_raises():
    with pytest.raises(ValueError, match=r"\|phi\| < 1"):
        ar1_filter(np.ones((3, 5)), np.array([0.2, 1.0, 0.1]))


@pytest.mark.parametrize("phi", [np.array([0.1, 0.2]), np.zeros(5),
                                 np.array([0.3])])
def test_ar1_filter_phi_with_the_wrong_number_of_windows_raises(phi):
    with pytest.raises(ValueError, match="one value per window"):
        ar1_filter(np.ones((4, 5)), phi)


def test_ar1_filter_two_dimensional_phi_raises_value_error():
    with pytest.raises(ValueError, match="one value per window"):
        ar1_filter(np.ones((4, 5)), np.full((2, 2), 0.3))


@pytest.mark.parametrize("shape", [(5,), (4, 0), ()])
def test_ar1_filter_noise_without_windows_and_steps_raises(shape):
    with pytest.raises(ValueError, match="at least one step"):
        ar1_filter(np.zeros(shape), 0.3)


def test_ar1_filter_checks_phi_even_when_it_would_return_the_noise():
    with pytest.raises(ValueError, match="one value per window"):
        ar1_filter(np.ones((4, 5)), np.zeros(2))


@pytest.mark.parametrize("sigma,phi", [(0.1, 0.0), (-0.1, 0.0), (NAN, 0.0),
                                       (0.0, 0.1), (0.0, NAN)])
def test_exact_observation_with_noise_or_nan_raises_value_error(
        small_dataset, sigma, phi):
    with pytest.raises(ValueError, match="takes no noise"):
        observation.exact(small_dataset, sigma=sigma, phi=phi)


@pytest.mark.parametrize("sigma", [-0.1, NAN, INF, -INF])
def test_gaussian_invalid_sigma_raises_value_error(small_dataset, sigma):
    with pytest.raises(ValueError, match="sigma must be finite"):
        observation.gaussian(small_dataset, sigma=sigma, rng=_rng())


@pytest.mark.parametrize("phi", [0.1, -0.5, NAN])
def test_gaussian_correlated_noise_raises_and_points_to_ar1(
        small_dataset, phi):
    with pytest.raises(ValueError, match="use 'ar1'"):
        observation.gaussian(small_dataset, sigma=0.1, phi=phi, rng=_rng())


@pytest.mark.parametrize("rng", [None, 3, np.random.RandomState(0), "seed"])
def test_gaussian_without_a_generator_raises_type_error(small_dataset, rng):
    with pytest.raises(TypeError, match="numpy.random.Generator"):
        observation.gaussian(small_dataset, sigma=0.1, rng=rng)


def test_gaussian_missing_rng_argument_raises_type_error(small_dataset):
    with pytest.raises(TypeError, match="rng"):
        observation.gaussian(small_dataset, sigma=0.1)


@pytest.mark.parametrize("sigma", [-0.1, NAN, INF])
def test_ar1_observation_invalid_sigma_raises_value_error(
        small_dataset, sigma):
    with pytest.raises(ValueError, match="sigma must be finite"):
        observation.ar1(small_dataset, sigma=sigma, phi=0.3, rng=_rng())


@pytest.mark.parametrize("phi", [1.0, -1.0, 1.5, NAN])
def test_ar1_observation_non_stationary_phi_raises_value_error(
        small_dataset, phi):
    with pytest.raises(ValueError, match=r"\|phi\| < 1"):
        observation.ar1(small_dataset, sigma=0.1, phi=phi, rng=_rng())


@pytest.mark.parametrize("rng", [None, 3, np.random.RandomState(0)])
def test_ar1_observation_without_a_generator_raises_type_error(
        small_dataset, rng):
    with pytest.raises(TypeError, match="numpy.random.Generator"):
        observation.ar1(small_dataset, sigma=0.0, phi=0.3, rng=rng)


def test_ar1_observation_missing_rng_argument_raises_type_error(
        small_dataset):
    with pytest.raises(TypeError, match="rng"):
        observation.ar1(small_dataset, sigma=0.1, phi=0.3)


def test_observation_registry_unknown_alias_is_a_plain_key_error():
    # The registry is a plain mapping; callers (pipeline.EvalSpec) check the
    # alias, so the lookup itself only reports the missing key.
    with pytest.raises(KeyError, match="shots"):
        observation.OBSERVATIONS["shots"]


def test_observation_registry_cannot_be_modified():
    with pytest.raises(TypeError):
        observation.OBSERVATIONS["shots"] = observation.exact


@pytest.mark.parametrize("copies", [-1, -10])
def test_noise_augmented_negative_copies_raise_value_error(
        small_dataset, copies):
    with pytest.raises(ValueError, match="copies must be non-negative"):
        observation.noise_augmented(small_dataset, copies, _rng())


def test_noise_augmented_fractional_copies_raise_type_error(small_dataset):
    with pytest.raises(TypeError, match="integer"):
        observation.noise_augmented(small_dataset, 1.5, _rng())


@pytest.mark.parametrize("phi_max", [-0.1, 1.01, 2.0, NAN])
def test_noise_augmented_phi_max_outside_zero_one_raises_value_error(
        small_dataset, phi_max):
    with pytest.raises(ValueError, match=r"phi_max must be in \[0, 1\]"):
        observation.noise_augmented(small_dataset, 1, _rng(), phi_max=phi_max)


@pytest.mark.parametrize("rng", [None, 0, np.random.RandomState(0)])
def test_noise_augmented_without_a_generator_raises_type_error(
        small_dataset, rng):
    with pytest.raises(TypeError, match="numpy.random.Generator"):
        observation.noise_augmented(small_dataset, 1, rng)
