"""MemoryTrajectoryStore: a trajectory store that lives in the process."""
import pytest

from src.persistence import stores


def test_memory_store_hits_after_a_save_and_misses_before():
    store = stores.MemoryTrajectoryStore()
    with pytest.raises(FileNotFoundError):
        store.load("a")
    store.save([1, 2], "a")
    assert store.load("a") == [1, 2]
    assert store.list_available() == ["a"]
