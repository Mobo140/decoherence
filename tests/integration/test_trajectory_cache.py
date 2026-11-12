"""The trajectory cache between GenerateDatasetUseCase and the file store.

A run must not fail on a poisoned entry, and a warm cache must not simulate
again.
"""
from __future__ import annotations

from src.application.generate_dataset import (
    GenerateDatasetCommand,
    GenerateDatasetUseCase,
    trajectory_cache_key,
)
from tests import support


def test_execute_a_damaged_entry_is_re_simulated_and_repaired(
        tmp_path, trajectory_store, counting_simulator, small_config):
    """End to end: the use case must not fail on a poisoned cache."""
    sim = counting_simulator
    cmd = GenerateDatasetCommand(
        configs=[small_config],
        window_length=5,
        horizon=1.0,
        samples_per_trajectory=2,
        seed=3,
        train_ratio=0.0,
        val_ratio=0.0,
    )
    uc = GenerateDatasetUseCase(sim, store=trajectory_store)

    uc.execute(cmd)
    assert sim.calls == 1                     # cold cache: simulated once
    uc.execute(cmd)
    assert sim.calls == 1                     # warm cache: served from disk

    key = trajectory_cache_key(small_config, 3, 0)
    root = tmp_path / "trajectories"
    (root / f"{key}_meta.json").write_text("")   # poison it

    uc.execute(cmd)                           # must not raise
    assert sim.calls == 2, "damaged entry was not re-simulated"

    uc.execute(cmd)
    assert sim.calls == 2, "the repaired entry was not reused"


def test_execute_warm_cache_skips_the_second_simulate(
        tmp_path, trajectory_store, counting_simulator, make_system_config):
    cfg = make_system_config(gamma=0.4, t_max=8.0, dt=0.2)
    uc = GenerateDatasetUseCase(counting_simulator, store=trajectory_store)
    cmd = GenerateDatasetCommand(
        configs=[cfg],
        window_length=8,
        samples_per_trajectory=2,
        seed=7,
        dataset_name="bulk",
    )
    ds1 = uc.execute(cmd)
    assert counting_simulator.calls == 1
    assert len(ds1.sequences) > 0

    sim2 = support.CountingSimulator()
    ds2 = GenerateDatasetUseCase(sim2, store=trajectory_store).execute(cmd)
    assert sim2.calls == 0
    assert len(ds2.sequences) == len(ds1.sequences)
    key = trajectory_cache_key(cfg, 7, 0)
    assert (tmp_path / "trajectories" / f"{key}.npz").exists()
