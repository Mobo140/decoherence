"""Fixtures shared by the unit and integration tests.

Plain builders and test doubles are in `tests/support.py`; the fixtures here
only hand them out. Directory-specific fixtures live in the conftest next to
the tests that need them.
"""
from __future__ import annotations

import pathlib
from collections.abc import Callable, Iterator

import pytest
import torch

from experiments import common
from src import reproducibility
from src.data import dataset as dataset_lib
from src.data import scenarios
from src.domain import value_objects
from src.persistence import stores
from tests import support

INTEGRATION_DIR = pathlib.Path(__file__).parent / "integration"


def pytest_collection_modifyitems(
    config: pytest.Config, items: list[pytest.Item],
) -> None:
    """Marks every test under tests/integration with `integration`."""
    for item in items:
        if INTEGRATION_DIR in item.path.parents:
            item.add_marker(pytest.mark.integration)


# --- datasets --------------------------------------------------------------

@pytest.fixture
def toy_dataset() -> dataset_lib.Dataset:
    """24 windows of exact exponential decay: 16 train, 4 val, 4 test."""
    return support.toy_dataset()


@pytest.fixture
def make_toy_dataset() -> Callable[..., dataset_lib.Dataset]:
    """Factory `make_toy_dataset(n=24, window=10)` for other sizes."""
    return support.toy_dataset


@pytest.fixture
def small_dataset() -> dataset_lib.Dataset:
    """Six noise windows with features a, b, corr_xx, concurrence."""
    return support.small_dataset()


# --- system configurations -------------------------------------------------

@pytest.fixture
def make_system_config() -> Callable[..., value_objects.SystemConfig]:
    """Factory `make_system_config(gamma, t_max, dt, omega)`, one qubit."""
    return support.make_system_config


@pytest.fixture
def small_config() -> value_objects.SystemConfig:
    """One qubit, gamma 0.4 (T2 = 5), 6 time units in steps of 0.2."""
    return support.make_system_config(gamma=0.4, t_max=6.0, dt=0.2)


@pytest.fixture(scope="session")
def scenario_configs() -> dict[str, value_objects.SystemConfig]:
    """One configuration per scenario A-E (seed 0), keyed by the letter."""
    groups = scenarios.build_configs(n_per_scenario=1, seed=0)
    return {letter: groups[letter][0] for letter in scenarios.SCENARIOS}


# --- simulators ------------------------------------------------------------

@pytest.fixture
def fake_simulator() -> support.ExponentialDecaySimulator:
    """Analytic exponential decay behind the `ISimulator` port."""
    return support.ExponentialDecaySimulator()


@pytest.fixture
def counting_simulator() -> support.CountingSimulator:
    """The fake simulator wrapped so that calls can be counted."""
    return support.CountingSimulator()


# --- global state ----------------------------------------------------------

@pytest.fixture
def single_thread() -> Iterator[None]:
    """Runs the test with one torch thread, then restores the old count.

    One thread makes training bit-exact between runs, as the golden outputs
    and the same-seed tests need.
    """
    before = torch.get_num_threads()
    reproducibility.set_num_threads(1)
    try:
        yield
    finally:
        reproducibility.set_num_threads(before)


# --- storage ---------------------------------------------------------------

@pytest.fixture
def trajectory_store(tmp_path: pathlib.Path) -> stores.NumpyTrajectoryStore:
    """A trajectory cache in a temporary directory."""
    return stores.NumpyTrajectoryStore(root=tmp_path / "trajectories")


@pytest.fixture
def results_dir(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch,
) -> pathlib.Path:
    """A temporary results directory that `experiments.common` writes to."""
    path = tmp_path / "results"
    path.mkdir()
    monkeypatch.setattr(common, "RESULTS_DIR", path)
    return path


@pytest.fixture
def isolated_cwd(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch,
) -> pathlib.Path:
    """Runs the test in an empty directory.

    Code that opens `data/trajectories` or `checkpoints/` by a relative
    path then writes under `tmp_path` instead of the repository.
    """
    monkeypatch.chdir(tmp_path)
    return tmp_path
