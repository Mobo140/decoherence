"""Train/val/test separation and target-leakage invariants.

These guard the properties that make the reported metrics mean anything:
a window must never share a trajectory with a window in another split, a
window must never see data from at or after the decoherence it predicts,
and normalisation must be fitted on training rows only.
"""
from __future__ import annotations

import numpy as np
import pytest

from src.application.generate_dataset import (
    Dataset,
    GenerateDatasetCommand,
    GenerateDatasetUseCase,
)
from src.domain.value_objects import (
    DissipatorConfig,
    QubitCount,
    SystemConfig,
    TrajectoryResult,
)


def _cfg(t_max: float = 12.0, dt: float = 0.1) -> SystemConfig:
    return SystemConfig(
        n_qubits=QubitCount.ONE,
        omega=1.0,
        dissipator=DissipatorConfig.constant(gamma=0.4),
        t_max=t_max,
        dt=dt,
    )


def _traj(t_decoh: float, t_max: float = 12.0, dt: float = 0.1) -> TrajectoryResult:
    """Trajectory whose T2 is unique, so it can be identified in the dataset."""
    n = int(round(t_max / dt)) + 1
    times = np.linspace(0.0, t_max, n)
    return TrajectoryResult(
        times=times,
        observables={
            "sigma_x": np.cos(times).astype(np.float32),
            "coherence_l1": np.exp(-times / max(t_decoh, 1e-6)).astype(np.float32),
            "purity": np.ones(n, dtype=np.float32),
        },
        t_decoh=t_decoh,
        system_config=_cfg(t_max, dt),
    )


class _ScriptedSimulator:
    """Hands out prepared trajectories in order, ignoring the config."""

    def __init__(self, trajectories):
        self._trajs = list(trajectories)
        self._i = 0

    def simulate(self, config):
        traj = self._trajs[self._i]
        self._i += 1
        return traj


def _build(n_traj: int = 12, **kwargs) -> Dataset:
    # Distinct T2 per trajectory, so t_decoh_abs identifies the source.
    trajs = [_traj(2.0 + 0.5 * i) for i in range(n_traj)]
    uc = GenerateDatasetUseCase(_ScriptedSimulator(trajs), store=None)
    params = dict(
        configs=[_cfg() for _ in range(n_traj)],
        window_length=10,
        horizon=1.0,
        samples_per_trajectory=4,
        seed=7,
        train_ratio=0.6,
        val_ratio=0.2,
    )
    params.update(kwargs)
    return uc.execute(GenerateDatasetCommand(**params))


def _split_of(ds: Dataset) -> dict:
    """Map every sample index to the split it belongs to."""
    where = {}
    for name, idx in (("train", ds.train_idx), ("val", ds.val_idx), ("test", ds.test_idx)):
        for i in idx.tolist():
            where[i] = name
    return where


def test_splits_partition_every_sample_exactly_once():
    ds = _build()
    idx = np.concatenate([ds.train_idx, ds.val_idx, ds.test_idx])
    assert sorted(idx.tolist()) == list(range(len(ds.sequences)))
    assert len(set(idx.tolist())) == len(idx), "a sample appears in two splits"


def test_windows_of_one_trajectory_never_span_two_splits():
    """The split must be trajectory-level.

    Several windows are cut from each trajectory and they share its T2, so
    splitting per window would put near-duplicate targets on both sides and
    inflate every metric.
    """
    ds = _build()
    where = _split_of(ds)
    by_traj = {}
    for i, t2 in enumerate(ds.t_decoh_abs.tolist()):
        by_traj.setdefault(round(t2, 9), set()).add(where[i])
    assert by_traj, "no samples produced"
    offenders = {t2: s for t2, s in by_traj.items() if len(s) > 1}
    assert not offenders, f"trajectories split across sets: {offenders}"


def test_every_trajectory_contributes_more_than_one_window():
    """Otherwise the test above would hold vacuously."""
    ds = _build()
    counts = {}
    for t2 in ds.t_decoh_abs.tolist():
        counts[round(t2, 9)] = counts.get(round(t2, 9), 0) + 1
    assert max(counts.values()) > 1


def test_no_window_reaches_its_own_decoherence_time():
    """Target leakage: the input must end strictly before T2."""
    ds = _build()
    assert np.all(ds.t_obs < ds.t_decoh_abs)


def test_regression_target_is_strictly_positive():
    ds = _build()
    assert np.all(ds.remaining_time > 0.0)
    assert np.allclose(ds.remaining_time, ds.t_decoh_abs - ds.t_obs)


def test_risk_label_matches_horizon():
    ds = _build(horizon=1.5)
    expected = (ds.remaining_time <= 1.5).astype(int)
    assert np.array_equal(ds.risk_labels, expected)


def test_ratios_are_respected_at_trajectory_level():
    ds = _build(n_traj=10, train_ratio=0.6, val_ratio=0.2)
    where = _split_of(ds)
    trajs = {}
    for i, t2 in enumerate(ds.t_decoh_abs.tolist()):
        trajs[round(t2, 9)] = where[i]
    counts = {"train": 0, "val": 0, "test": 0}
    for split in trajs.values():
        counts[split] += 1
    assert counts["train"] == 6
    assert counts["val"] == 2
    assert counts["test"] == 2


def test_zero_ratios_send_everything_to_test():
    """The evaluation datasets in E11 rely on this."""
    ds = _build(train_ratio=0.0, val_ratio=0.0)
    assert len(ds.train_idx) == 0
    assert len(ds.val_idx) == 0
    assert len(ds.test_idx) == len(ds.sequences)


def test_min_window_gap_is_enforced_within_a_trajectory():
    """Overlapping windows are near-duplicates; min_window_gap bounds that."""
    gap = 15
    ds = _build(min_window_gap=gap, samples_per_trajectory=3)
    by_traj = {}
    for t2, t_obs in zip(ds.t_decoh_abs.tolist(), ds.t_obs.tolist()):
        by_traj.setdefault(round(t2, 9), []).append(t_obs)
    dt = 0.1
    for t2, obs in by_traj.items():
        obs = sorted(obs)
        for a, b in zip(obs, obs[1:]):
            steps = round((b - a) / dt)
            assert steps >= gap, f"T2={t2}: windows {steps} steps apart, need {gap}"


def test_normalisation_statistics_come_from_training_rows_only():
    """Fitting the scaler on all rows would leak test information."""
    torch = pytest.importorskip("torch")
    from src.infrastructure.ml.lstm_predictor import LSTMPredictor

    ds = _build(n_traj=12)
    predictor = LSTMPredictor(hidden_size=8, num_layers=1, dropout=0.0)
    predictor.train(ds, n_epochs=1, batch_size=8, verbose=False, seed=0)
    fitted_mean = np.array(predictor._feature_mean, dtype=np.float64).copy()

    # Corrupt only the rows the model must not have looked at.
    poisoned = Dataset(**{**ds.__dict__})
    poisoned.sequences = ds.sequences.copy()
    poisoned.sequences[ds.test_idx] += 1000.0

    predictor2 = LSTMPredictor(hidden_size=8, num_layers=1, dropout=0.0)
    predictor2.train(poisoned, n_epochs=1, batch_size=8, verbose=False, seed=0)
    again = np.array(predictor2._feature_mean, dtype=np.float64)

    assert np.allclose(fitted_mean, again), (
        "feature normalisation changed when only test rows changed"
    )
