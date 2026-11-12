"""GenerateDatasetUseCase: splits, leakage, censoring and seeding.

These guard the properties that make the reported metrics meaningful:
a window must never share a trajectory with a window in another split, a
window must never see data from at or after the decoherence it predicts,
censored trajectories stay out unless asked for, and the same seed must give
the same data. Trajectories come from scripted or random test doubles, so
no QuTiP is involved.
"""
from __future__ import annotations

import numpy as np

from src.application.generate_dataset import (
    GenerateDatasetCommand,
    GenerateDatasetUseCase,
)
from src.data.dataset import Dataset
from src.domain.value_objects import TrajectoryResult
from tests import support


def _cfg(t_max: float = 12.0, dt: float = 0.1):
    return support.make_system_config(gamma=0.4, t_max=t_max, dt=dt)


def _build(n_traj: int = 12, **kwargs) -> Dataset:
    # Distinct T2 per trajectory, so t_decoh_abs identifies the source.
    trajs = [support.scripted_trajectory(2.0 + 0.5 * i)
             for i in range(n_traj)]
    uc = GenerateDatasetUseCase(support.ScriptedSimulator(trajs), store=None)
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
    for name, idx in (("train", ds.train_idx), ("val", ds.val_idx),
                      ("test", ds.test_idx)):
        for i in idx.tolist():
            where[i] = name
    return where


def test_execute_splits_partition_every_sample_exactly_once():
    ds = _build()
    idx = np.concatenate([ds.train_idx, ds.val_idx, ds.test_idx])
    assert sorted(idx.tolist()) == list(range(len(ds.sequences)))
    assert len(set(idx.tolist())) == len(idx), "a sample appears in two splits"


def test_execute_never_spans_one_trajectory_across_two_splits():
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


def test_execute_cuts_more_than_one_window_per_trajectory():
    """Otherwise the test above would hold vacuously."""
    ds = _build()
    counts = {}
    for t2 in ds.t_decoh_abs.tolist():
        counts[round(t2, 9)] = counts.get(round(t2, 9), 0) + 1
    assert max(counts.values()) > 1


def test_execute_windows_end_before_their_own_decoherence_time():
    """Target leakage: the input must end strictly before T2."""
    ds = _build()
    assert np.all(ds.t_obs < ds.t_decoh_abs)


def test_execute_remaining_time_is_positive_and_t2_minus_t_obs():
    ds = _build()
    assert np.all(ds.remaining_time > 0.0)
    assert np.allclose(ds.remaining_time, ds.t_decoh_abs - ds.t_obs)


def test_execute_risk_label_is_remaining_time_within_horizon():
    ds = _build(horizon=1.5)
    expected = (ds.remaining_time <= 1.5).astype(int)
    assert np.array_equal(ds.risk_labels, expected)


def test_execute_applies_split_ratios_to_trajectories():
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


def test_execute_zero_ratios_send_every_window_to_test():
    """The evaluation datasets in E11 rely on this."""
    ds = _build(train_ratio=0.0, val_ratio=0.0)
    assert len(ds.train_idx) == 0
    assert len(ds.val_idx) == 0
    assert len(ds.test_idx) == len(ds.sequences)


def test_execute_enforces_min_window_gap_within_a_trajectory():
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
            assert steps >= gap, (
                f"T2={t2}: windows {steps} steps apart, need {gap}")


# --- censoring ---------------------------------------------------------------

def _censoring_trajectory(
    t_decoh: float, censored: bool, t_max: float = 10.0, dt: float = 0.1
) -> TrajectoryResult:
    n = int(round(t_max / dt)) + 1
    times = np.linspace(0.0, t_max, n)
    obs = {
        "sigma_x": np.ones(n, dtype=np.float32),
        "sigma_y": np.zeros(n, dtype=np.float32),
        "sigma_z": np.zeros(n, dtype=np.float32),
        "coherence_l1": np.exp(-times / max(t_decoh, 1e-3)).astype(np.float32),
        "purity": np.ones(n, dtype=np.float32),
    }
    return TrajectoryResult(
        times=times,
        observables=obs,
        t_decoh=t_decoh,
        system_config=support.make_system_config(0.4, t_max=t_max, dt=dt),
        censored=censored,
    )


def _nine_uncensored_and_one_censored() -> list[TrajectoryResult]:
    trajs = [_censoring_trajectory(4.0, censored=False) for _ in range(9)]
    trajs.append(_censoring_trajectory(10.0, censored=True))
    return trajs


def test_execute_excludes_censored_trajectories_by_default():
    trajs = _nine_uncensored_and_one_censored()
    uc = GenerateDatasetUseCase(support.ScriptedSimulator(trajs))
    ds = uc.execute(GenerateDatasetCommand(
        configs=[t.system_config for t in trajs],
        window_length=10,
        samples_per_trajectory=2,
        seed=0,
    ))

    assert ds.metadata["n_censored"] == 1
    assert ds.metadata["n_trajectories"] == 10
    # Every remaining window must come from an uncensored trajectory.
    assert np.all(ds.t_decoh_abs < 10.0 - 1e-6)
    assert len(ds.sequences) > 0
    assert ds.metadata["include_censored"] is False
    assert int(ds.censored.sum()) == 0


def test_execute_include_censored_keeps_their_windows():
    trajs = _nine_uncensored_and_one_censored()
    uc = GenerateDatasetUseCase(support.ScriptedSimulator(trajs))
    ds = uc.execute(GenerateDatasetCommand(
        configs=[t.system_config for t in trajs],
        window_length=10,
        samples_per_trajectory=2,
        seed=0,
        include_censored=True,
    ))
    assert ds.metadata["n_censored"] == 1
    assert ds.metadata["n_censored_windows"] >= 1
    assert bool(ds.censored.any())


# --- seeding -----------------------------------------------------------------

def _seeded_dataset(seed: int) -> Dataset:
    uc = GenerateDatasetUseCase(support.GlobalRngSimulator(), store=None)
    return uc.execute(GenerateDatasetCommand(
        configs=[_cfg(t_max=10.0) for _ in range(6)],
        window_length=10,
        horizon=1.0,
        samples_per_trajectory=3,
        seed=seed,
        train_ratio=0.5,
        val_ratio=0.25,
    ))


def test_execute_same_seed_gives_an_identical_dataset():
    a, b = _seeded_dataset(11), _seeded_dataset(11)
    assert np.array_equal(a.sequences, b.sequences)
    assert np.array_equal(a.t_decoh_abs, b.t_decoh_abs)
    assert np.array_equal(a.train_idx, b.train_idx)


def test_execute_seeds_the_simulators_initial_state_draw_too():
    """The simulator's own randomness must be covered, not just the split.

    This is the defect that made two runs with the same seed differ: the
    split was reproducible while the trajectories were not.
    """
    np.random.seed(999)
    a = _seeded_dataset(11)
    np.random.seed(12345)            # different global state before the run
    b = _seeded_dataset(11)
    assert np.array_equal(a.t_decoh_abs, b.t_decoh_abs), (
        "trajectory content depended on RNG state outside the command's seed"
    )


def test_execute_different_seeds_give_different_data():
    a, b = _seeded_dataset(11), _seeded_dataset(12)
    assert not np.array_equal(a.t_decoh_abs, b.t_decoh_abs)
