"""The trajectory cache must survive an interrupted run.

Cache keys are deterministic, so a half-written entry is not a one-off
annoyance: every later run computes the same key and hits the same broken
file. A killed run once made a key permanently unusable and took two
re-run attempts down with it.
"""
from __future__ import annotations

import json

import numpy as np
import pytest

from src.application.generate_dataset import (
    GenerateDatasetCommand,
    GenerateDatasetUseCase,
    trajectory_cache_key,
)
from src.domain.value_objects import (
    DissipatorConfig,
    QubitCount,
    SystemConfig,
    TrajectoryResult,
)
from src.infrastructure.persistence.stores import NumpyTrajectoryStore


def _cfg() -> SystemConfig:
    return SystemConfig(
        n_qubits=QubitCount.ONE,
        omega=1.0,
        dissipator=DissipatorConfig.constant(gamma=0.4),
        t_max=6.0,
        dt=0.2,
    )


def _traj(t_decoh: float = 3.0) -> TrajectoryResult:
    cfg = _cfg()
    n = int(round(cfg.t_max / cfg.dt)) + 1
    times = np.linspace(0.0, cfg.t_max, n)
    return TrajectoryResult(
        times=times,
        observables={
            "sigma_x": np.ones(n, dtype=np.float32),
            "coherence_l1": np.exp(-times / 3.0).astype(np.float32),
            "purity": np.ones(n, dtype=np.float32),
        },
        t_decoh=t_decoh,
        system_config=cfg,
    )


class _CountingSim:
    def __init__(self, traj):
        self.traj = traj
        self.calls = 0

    def simulate(self, config):
        self.calls += 1
        return self.traj


def test_round_trip(tmp_path):
    store = NumpyTrajectoryStore(root=tmp_path)
    store.save([_traj(4.25)], "k")
    (loaded,) = store.load("k")
    assert loaded.t_decoh == pytest.approx(4.25)


def test_missing_entry_is_a_miss(tmp_path):
    with pytest.raises(FileNotFoundError):
        NumpyTrajectoryStore(root=tmp_path).load("absent")


def test_garbage_archive_reads_as_a_miss(tmp_path):
    (tmp_path / "k.npz").write_bytes(b"this is not an npz")
    (tmp_path / "k_meta.json").write_text("{}")
    with pytest.raises(FileNotFoundError):
        NumpyTrajectoryStore(root=tmp_path).load("k")


def test_empty_metadata_reads_as_a_miss(tmp_path):
    store = NumpyTrajectoryStore(root=tmp_path)
    store.save([_traj()], "k")
    (tmp_path / "k_meta.json").write_text("")      # interrupted before the JSON landed
    with pytest.raises(FileNotFoundError):
        store.load("k")


def test_truncated_archive_reads_as_a_miss(tmp_path):
    """np.load is lazy: the CRC error only fires when an array is touched,
    so the guard has to cover the whole reconstruction, not just the open."""
    store = NumpyTrajectoryStore(root=tmp_path)
    store.save([_traj()], "k")
    raw = (tmp_path / "k.npz").read_bytes()
    (tmp_path / "k.npz").write_bytes(raw[: len(raw) // 2])
    with pytest.raises(FileNotFoundError):
        store.load("k")


def test_metadata_without_configs_reads_as_a_miss(tmp_path):
    store = NumpyTrajectoryStore(root=tmp_path)
    store.save([_traj()], "k")
    (tmp_path / "k_meta.json").write_text(json.dumps({"n": 1}))
    with pytest.raises(FileNotFoundError):
        store.load("k")


def test_save_leaves_no_temporary_files(tmp_path):
    store = NumpyTrajectoryStore(root=tmp_path)
    store.save([_traj()], "k")
    assert not list(tmp_path.glob("*.tmp")), "temporary file left behind"
    assert not list(tmp_path.glob("*.tmp.npz")), "temporary archive left behind"


def test_overwriting_keeps_the_entry_readable(tmp_path):
    store = NumpyTrajectoryStore(root=tmp_path)
    store.save([_traj(1.0)], "k")
    store.save([_traj(2.0)], "k")
    (loaded,) = store.load("k")
    assert loaded.t_decoh == pytest.approx(2.0)


def test_a_damaged_entry_is_re_simulated_and_repaired(tmp_path):
    """End to end: the use case must not fail on a poisoned cache."""
    traj = _traj()
    sim = _CountingSim(traj)
    store = NumpyTrajectoryStore(root=tmp_path)
    cmd = GenerateDatasetCommand(
        configs=[_cfg()],
        window_length=5,
        horizon=1.0,
        samples_per_trajectory=2,
        seed=3,
        train_ratio=0.0,
        val_ratio=0.0,
    )
    uc = GenerateDatasetUseCase(sim, store=store)

    uc.execute(cmd)
    assert sim.calls == 1                     # cold cache: simulated once
    uc.execute(cmd)
    assert sim.calls == 1                     # warm cache: served from disk

    key = trajectory_cache_key(_cfg(), 3, 0)
    (tmp_path / f"{key}_meta.json").write_text("")   # poison it

    uc.execute(cmd)                           # must not raise
    assert sim.calls == 2, "damaged entry was not re-simulated"

    uc.execute(cmd)
    assert sim.calls == 2, "the repaired entry was not reused"
