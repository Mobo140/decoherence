"""Domain value objects validate themselves on construction."""
from __future__ import annotations

import dataclasses
import math

import numpy as np
import pytest

from src import domain
from src.domain import value_objects
from src.domain.value_objects import (
    AblationResult,
    BacktestMetrics,
    DecoherenceCriterion,
    DissipatorConfig,
    DissipatorType,
    ExperimentSpec,
    InteractionType,
    NoiseConfig,
    PredictionResult,
    PredictorVariant,
    QubitCount,
    SystemConfig,
    TrajectoryResult,
)

NAN = float("nan")
INF = float("inf")


def _config(**overrides) -> SystemConfig:
    """A valid one-qubit configuration with the given fields replaced."""
    fields = dict(
        n_qubits=QubitCount.ONE, omega=1.0,
        dissipator=DissipatorConfig.constant(0.1))
    fields.update(overrides)
    return SystemConfig(**fields)


def _trajectory(**observables) -> TrajectoryResult:
    """A trajectory of three time points with the given observables."""
    return TrajectoryResult(
        times=np.arange(3.0), observables=observables, t_decoh=1.0,
        system_config=_config())


# --- enumerations ----------------------------------------------------------

def test_enum_values_are_the_strings_and_numbers_stored_in_the_cache():
    assert [q.value for q in QubitCount] == [1, 2]
    assert {d.value for d in DissipatorType} == {"sigma_minus", "sigma_z"}
    assert {i.value for i in InteractionType} == {"XXZ", "TFIM"}
    assert {c.value for c in DecoherenceCriterion} == {
        "coherence", "purity", "coherence_energy", "concurrence",
        "entanglement_death"}
    assert value_objects.ONE_OVER_E == pytest.approx(math.exp(-1.0))


def test_enums_are_built_from_their_values():
    assert QubitCount(2) is QubitCount.TWO
    assert DissipatorType("sigma_z") is DissipatorType.SIGMA_Z
    assert InteractionType("TFIM") is InteractionType.TFIM
    assert PredictorVariant("lstm_only") is PredictorVariant.LSTM_ONLY


def test_domain_package_exports_resolve():
    for name in domain.__all__:
        assert hasattr(domain, name)


# --- DissipatorConfig ------------------------------------------------------

def test_dissipator_constant_builds_a_constant_rate():
    config = DissipatorConfig.constant(0.3, DissipatorType.SIGMA_Z)
    assert config.gamma_const == 0.3 and config.gamma_coeffs is None
    assert config.operator_type is DissipatorType.SIGMA_Z


def test_dissipator_time_dependent_builds_a_bernstein_rate():
    config = DissipatorConfig.time_dependent((0.1, 0.2, 0.3))
    assert config.gamma_coeffs == (0.1, 0.2, 0.3)
    assert config.gamma_const is None
    assert config.operator_type is DissipatorType.SIGMA_MINUS


def test_dissipator_accepts_a_zero_rate_and_zero_coefficients():
    assert DissipatorConfig.constant(0.0).gamma_const == 0.0
    assert DissipatorConfig.time_dependent((0.0, 0.0)).gamma_coeffs == (
        0.0, 0.0)


def test_dissipator_is_immutable_and_hashable():
    config = DissipatorConfig.time_dependent((0.1, 0.2))
    with pytest.raises(dataclasses.FrozenInstanceError):
        config.gamma_const = 1.0
    assert hash(config) == hash(DissipatorConfig.time_dependent((0.1, 0.2)))


# --- SystemConfig ----------------------------------------------------------

def test_system_config_defaults_match_the_published_setup():
    config = _config()
    assert config.interaction_type is InteractionType.XXZ
    assert config.J == 0.5 and config.t_max == 10.0 and config.dt == 0.1
    assert config.decoherence_criterion is DecoherenceCriterion.COHERENCE
    assert config.decoherence_threshold == value_objects.ONE_OVER_E


def test_system_config_one_qubit_accepts_the_single_system_criteria():
    for criterion in (DecoherenceCriterion.COHERENCE,
                      DecoherenceCriterion.PURITY):
        assert _config(decoherence_criterion=criterion).decoherence_criterion \
            is criterion


def test_system_config_two_qubits_accept_every_criterion():
    for criterion in DecoherenceCriterion:
        config = _config(
            n_qubits=QubitCount.TWO, decoherence_criterion=criterion)
        assert config.decoherence_criterion is criterion


def test_system_config_dt_just_below_t_max_is_valid():
    assert _config(t_max=1.0, dt=0.999).dt == 0.999


def test_system_config_is_immutable():
    with pytest.raises(dataclasses.FrozenInstanceError):
        _config().omega = 2.0


def test_two_qubit_criteria_are_the_ones_one_qubit_cannot_define():
    assert value_objects.TWO_QUBIT_CRITERIA == {
        DecoherenceCriterion.COHERENCE_ENERGY,
        DecoherenceCriterion.CONCURRENCE,
        DecoherenceCriterion.ENTANGLEMENT_DEATH}


# --- TrajectoryResult ------------------------------------------------------

def test_trajectory_feature_names_skip_the_density_matrices():
    traj = _trajectory(
        sigma_x=np.zeros(3), density_matrices=["raw"], purity=np.ones(3))
    assert traj.feature_names == ["sigma_x", "purity"]


def test_trajectory_feature_matrix_is_float32_with_columns_in_order():
    traj = _trajectory(a=np.array([1.0, 2.0, 3.0]), b=np.array([4, 5, 6]))
    matrix = traj.feature_matrix
    assert matrix.dtype == np.float32 and matrix.shape == (3, 2)
    np.testing.assert_array_equal(matrix[:, 0], [1.0, 2.0, 3.0])
    np.testing.assert_array_equal(matrix[:, 1], [4.0, 5.0, 6.0])


def test_trajectory_density_matrices_need_not_match_the_time_axis():
    traj = _trajectory(a=np.zeros(3), density_matrices=[np.eye(2)])
    assert traj.feature_matrix.shape == (3, 1)


def test_trajectory_defaults_to_not_censored():
    assert _trajectory(a=np.zeros(3)).censored is False


# --- PredictionResult, NoiseConfig, ExperimentSpec -------------------------

def test_prediction_result_accepts_the_risk_bounds():
    assert PredictionResult(1.0, 0.0, 1.0).risk_score == 0.0
    assert PredictionResult(1.0, 1.0, 1.0).risk_score == 1.0


def test_prediction_result_is_immutable():
    with pytest.raises(dataclasses.FrozenInstanceError):
        PredictionResult(1.0, 0.5, 1.0).risk_score = 0.1


def test_noise_config_defaults_to_no_noise_without_a_seed():
    config = NoiseConfig()
    assert config.sigma == 0.0 and config.seed is None
    assert NoiseConfig(sigma=0.02, seed=3).seed == 3


def test_experiment_spec_accepts_valid_arguments():
    spec = ExperimentSpec(
        name="test", predictor_variant=PredictorVariant.PHYSICS_LSTM,
        window_fraction=0.20,
    )
    assert spec.name == "test"


def test_experiment_spec_accepts_the_boundary_values():
    spec = ExperimentSpec(name="edge", window_fraction=1.0, n_trajectories=10)
    assert spec.window_fraction == 1.0 and spec.n_trajectories == 10


def test_experiment_spec_defaults():
    spec = ExperimentSpec(name="d")
    assert spec.predictor_variant is PredictorVariant.PHYSICS_LSTM
    assert spec.noise == NoiseConfig() and spec.seed == 42
    assert spec.n_trajectories == 500 and spec.window_fraction == 0.20


def test_experiment_spec_is_hashable_as_the_cache_key_needs():
    a = ExperimentSpec(name="a")
    assert hash(a) == hash(ExperimentSpec(name="a"))
    assert a != ExperimentSpec(name="b")


# --- result containers -----------------------------------------------------

def test_ablation_result_as_dict_is_a_flat_csv_row():
    spec = ExperimentSpec(
        name="arm", predictor_variant=PredictorVariant.LSTM_ONLY,
        window_fraction=0.3, noise=NoiseConfig(sigma=0.05))
    row = AblationResult(spec, 1.0, 2.0, 0.5, 10.0, 0.9, 40).as_dict()
    assert row == {
        "name": "arm", "variant": "lstm_only", "window_fraction": 0.3,
        "noise_sigma": 0.05, "mae": 1.0, "rmse": 2.0, "r2": 0.5,
        "mape": 10.0, "auroc": 0.9, "n_samples": 40}


def test_backtest_metrics_summary_formats_one_line():
    metrics = BacktestMetrics(
        mae=0.5, rmse=0.75, r2=0.9, mape=12.5, risk_auroc=0.95,
        n_samples=7, predictions=np.zeros(7), actuals=np.zeros(7))
    assert metrics.summary() == (
        "BacktestMetrics(n=7  MAE=0.5000  RMSE=0.7500  "
        "R²=0.9000  MAPE=12.50%  AUROC=0.9500)")
    assert metrics.extra == {}


def test_backtest_metrics_extra_is_not_shared_between_instances():
    a = BacktestMetrics(1, 1, 1, 1, 1, 1, np.zeros(1), np.zeros(1))
    b = BacktestMetrics(1, 1, 1, 1, 1, 1, np.zeros(1), np.zeros(1))
    a.extra["x"] = 1.0
    assert b.extra == {}


# --- Negative cases ---


@pytest.mark.parametrize("value", [0, -0.5, NAN])
def test_system_config_non_positive_or_nan_omega_raises_value_error(value):
    with pytest.raises(ValueError, match="omega"):
        _config(omega=value)


@pytest.mark.parametrize("value", [0.0, -1.0])
def test_system_config_non_positive_t_max_raises_value_error(value):
    with pytest.raises(ValueError, match="t_max"):
        _config(t_max=value, dt=0.1)


@pytest.mark.parametrize("dt", [0.0, -0.1, 10.0, 11.0])
def test_system_config_dt_outside_zero_t_max_raises_value_error(dt):
    with pytest.raises(ValueError, match="0 < dt < t_max"):
        _config(t_max=10.0, dt=dt)


@pytest.mark.parametrize("value", [0.0, -1.0])
def test_system_config_non_positive_coupling_raises_value_error(value):
    with pytest.raises(ValueError, match="J must be positive"):
        _config(J=value)


@pytest.mark.parametrize("field", ["omega", "t_max", "dt", "J"])
@pytest.mark.parametrize("value", [NAN, INF])
def test_system_config_non_finite_value_raises_value_error(field, value):
    with pytest.raises(ValueError, match=field):
        _config(**{field: value})


@pytest.mark.parametrize("threshold", [0.0, 1.0, 1.5, -0.1, NAN])
def test_system_config_threshold_outside_open_unit_interval_raises(threshold):
    with pytest.raises(ValueError, match="decoherence_threshold"):
        _config(decoherence_threshold=threshold)


@pytest.mark.parametrize("criterion", [
    DecoherenceCriterion.COHERENCE_ENERGY,
    DecoherenceCriterion.CONCURRENCE,
    DecoherenceCriterion.ENTANGLEMENT_DEATH])
def test_system_config_one_qubit_with_two_qubit_criterion_raises(criterion):
    with pytest.raises(ValueError, match=criterion.value):
        _config(n_qubits=QubitCount.ONE, decoherence_criterion=criterion)


def test_system_config_replace_to_a_two_qubit_criterion_on_one_qubit_raises():
    with pytest.raises(ValueError, match="two qubits only"):
        dataclasses.replace(
            _config(), decoherence_criterion=DecoherenceCriterion.CONCURRENCE)


@pytest.mark.parametrize("field,value", [
    ("n_qubits", 1), ("n_qubits", 3),
    ("dissipator", None), ("dissipator", 0.1),
    ("interaction_type", "XXZ"),
    ("decoherence_criterion", "coherence")])
def test_system_config_field_of_the_wrong_type_raises_type_error(
        field, value):
    with pytest.raises(TypeError, match=field):
        _config(**{field: value})


@pytest.mark.parametrize("enum,value", [
    (QubitCount, 3), (QubitCount, 0), (DissipatorType, "sigma_x"),
    (InteractionType, "XYZ"), (DecoherenceCriterion, "nope"),
    (PredictorVariant, "gradient_boosting")])
def test_enum_unknown_value_raises_value_error(enum, value):
    with pytest.raises(ValueError, match="is not a valid"):
        enum(value)


def test_dissipator_neither_rate_raises_value_error():
    with pytest.raises(ValueError, match="either gamma_const or gamma_coeffs"):
        DissipatorConfig()


def test_dissipator_both_rates_raise_value_error():
    with pytest.raises(ValueError, match="not both"):
        DissipatorConfig(gamma_const=0.1, gamma_coeffs=(0.1, 0.2))


def test_dissipator_negative_constant_rate_raises_value_error():
    with pytest.raises(ValueError, match="gamma_const"):
        DissipatorConfig.constant(-0.1)


@pytest.mark.parametrize("value", [NAN, INF, -INF])
def test_dissipator_non_finite_constant_rate_raises_value_error(value):
    with pytest.raises(ValueError, match="gamma_const"):
        DissipatorConfig.constant(value)


def test_dissipator_negative_coefficient_raises_value_error():
    with pytest.raises(ValueError, match="Bernstein coefficients"):
        DissipatorConfig.time_dependent((0.1, -0.2, 0.3))


@pytest.mark.parametrize("value", [NAN, INF])
def test_dissipator_non_finite_coefficient_raises_value_error(value):
    with pytest.raises(ValueError, match="Bernstein coefficients"):
        DissipatorConfig.time_dependent((0.1, value))


def test_dissipator_empty_coefficient_tuple_raises_value_error():
    with pytest.raises(ValueError, match="must not be empty"):
        DissipatorConfig.time_dependent(())


@pytest.mark.parametrize("operator_type", ["sigma_minus", None, 1])
def test_dissipator_operator_type_that_is_not_an_enum_raises_type_error(
        operator_type):
    with pytest.raises(TypeError, match="operator_type"):
        DissipatorConfig(operator_type=operator_type, gamma_const=0.1)


@pytest.mark.parametrize("risk", [-0.01, 1.01, 2.0, NAN, INF])
def test_prediction_result_risk_outside_unit_interval_raises_value_error(
        risk):
    with pytest.raises(ValueError, match=r"risk_score must be in \[0, 1\]"):
        PredictionResult(1.0, risk, 1.0)


@pytest.mark.parametrize("sigma", [-0.1, NAN, INF])
def test_noise_config_negative_or_non_finite_sigma_raises_value_error(sigma):
    with pytest.raises(ValueError, match="sigma"):
        NoiseConfig(sigma=sigma)


@pytest.mark.parametrize("fraction", [0.0, -0.1, 1.01, NAN])
def test_experiment_spec_window_fraction_outside_zero_one_raises(fraction):
    with pytest.raises(ValueError, match="window_fraction"):
        ExperimentSpec(name="bad", window_fraction=fraction)


@pytest.mark.parametrize("n", [9, 5, 0, -1])
def test_experiment_spec_too_few_trajectories_raises_value_error(n):
    with pytest.raises(ValueError, match="at least 10"):
        ExperimentSpec(name="bad", n_trajectories=n, window_fraction=0.2)


def test_trajectory_observable_longer_than_the_time_axis_raises():
    with pytest.raises(ValueError, match="'b' has 5 values for 3"):
        _trajectory(a=np.zeros(3), b=np.zeros(5))


def test_trajectory_observable_shorter_than_the_time_axis_raises():
    with pytest.raises(ValueError, match="'a' has 2 values for 3"):
        _trajectory(a=np.zeros(2))


def test_trajectory_without_observables_has_no_feature_matrix():
    traj = _trajectory()
    assert traj.feature_names == []
    with pytest.raises(ValueError, match="at least one array"):
        traj.feature_matrix


# Debatable behaviours, kept as they are and pinned here so a change is
# deliberate.

def test_prediction_result_accepts_nan_time_and_non_positive_horizon():
    assert math.isnan(PredictionResult(NAN, 0.5, 1.0).t_decoh_predicted)
    assert PredictionResult(1.0, 0.5, -1.0).horizon == -1.0


def test_experiment_spec_accepts_an_empty_name_and_a_string_variant():
    assert ExperimentSpec(name="").name == ""
    spec = ExperimentSpec(name="a", predictor_variant="physics_only")
    assert spec.predictor_variant == "physics_only"


def test_trajectory_accepts_a_negative_or_nan_decoherence_time():
    traj = TrajectoryResult(
        times=np.arange(3.0), observables={"a": np.zeros(3)}, t_decoh=-1.0,
        system_config=_config())
    assert traj.t_decoh == -1.0
    traj.t_decoh = NAN
    assert math.isnan(traj.t_decoh)
