"""Fixtures for the integration tests."""
from __future__ import annotations

from collections.abc import Iterator

import pytest

from tests import support


@pytest.fixture(autouse=True)
def protected_dirs_unchanged() -> Iterator[None]:
    """Fails a test that writes to the published results or the caches.

    Guards `experiments/results/`, `data/trajectories/` and `checkpoints/`;
    tests that need to write use `tmp_path` or the `results_dir`,
    `trajectory_store` and `isolated_cwd` fixtures.
    """
    before = support.snapshot_protected_dirs()
    yield
    after = support.snapshot_protected_dirs()
    changed = [name for name in before if before[name] != after[name]]
    assert not changed, f"the test modified {changed}"


@pytest.fixture(autouse=True)
def _one_torch_thread(single_thread: None) -> None:
    """Pins every integration test to one torch thread."""
