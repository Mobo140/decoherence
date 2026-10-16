"""NumpyTrajectoryStore must survive an interrupted run.

Cache keys are deterministic, so a half-written entry is not a one-off
annoyance: every later run computes the same key and hits the same broken
file. A killed run once made a key permanently unusable and took two
re-run attempts down with it.
"""
from __future__ import annotations

import json

import numpy as np
import pytest

from src.domain.value_objects import TrajectoryResult
from src.persistence.stores import NumpyTrajectoryStore
from tests import support


def _traj(t_decoh: float = 3.0, censored: bool = False) -> TrajectoryResult:
    cfg = support.make_system_config(gamma=0.4, t_max=6.0, dt=0.2)
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
        censored=censored,
    )


def test_load_after_save_returns_the_trajectory(tmp_path):
    store = NumpyTrajectoryStore(root=tmp_path)
    store.save([_traj(4.25)], "k")
    (loaded,) = store.load("k")
    assert loaded.t_decoh == pytest.approx(4.25)


def test_save_leaves_no_temporary_files(tmp_path):
    store = NumpyTrajectoryStore(root=tmp_path)
    store.save([_traj()], "k")
    assert not list(tmp_path.glob("*.tmp")), "temporary file left behind"
    assert not list(tmp_path.glob("*.tmp.npz")), "temporary archive left behind"


def test_save_over_an_entry_keeps_it_readable(tmp_path):
    store = NumpyTrajectoryStore(root=tmp_path)
    store.save([_traj(1.0)], "k")
    store.save([_traj(2.0)], "k")
    (loaded,) = store.load("k")
    assert loaded.t_decoh == pytest.approx(2.0)


def test_save_persists_the_censored_flag(tmp_path):
    store = NumpyTrajectoryStore(root=tmp_path)
    store.save([_traj(censored=True)], "one")
    loaded = store.load("one")
    assert loaded[0].censored is True


# --- Negative cases ---


def test_load_missing_entry_raises_file_not_found(tmp_path):
    with pytest.raises(FileNotFoundError):
        NumpyTrajectoryStore(root=tmp_path).load("absent")


def test_load_garbage_archive_reads_as_a_miss(tmp_path):
    (tmp_path / "k.npz").write_bytes(b"this is not an npz")
    (tmp_path / "k_meta.json").write_text("{}")
    with pytest.raises(FileNotFoundError):
        NumpyTrajectoryStore(root=tmp_path).load("k")


def test_load_empty_metadata_reads_as_a_miss(tmp_path):
    store = NumpyTrajectoryStore(root=tmp_path)
    store.save([_traj()], "k")
    # Interrupted before the JSON landed.
    (tmp_path / "k_meta.json").write_text("")
    with pytest.raises(FileNotFoundError):
        store.load("k")


def test_load_truncated_archive_reads_as_a_miss(tmp_path):
    """np.load is lazy: the CRC error only fires when an array is touched,
    so the guard has to cover the whole reconstruction, not just the open."""
    store = NumpyTrajectoryStore(root=tmp_path)
    store.save([_traj()], "k")
    raw = (tmp_path / "k.npz").read_bytes()
    (tmp_path / "k.npz").write_bytes(raw[: len(raw) // 2])
    with pytest.raises(FileNotFoundError):
        store.load("k")


def test_load_metadata_without_configs_reads_as_a_miss(tmp_path):
    store = NumpyTrajectoryStore(root=tmp_path)
    store.save([_traj()], "k")
    (tmp_path / "k_meta.json").write_text(json.dumps({"n": 1}))
    with pytest.raises(FileNotFoundError):
        store.load("k")
