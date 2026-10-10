"""Seeding helpers do exactly what the direct calls do."""
from __future__ import annotations

import numpy as np
import torch

from src import reproducibility


def test_seed_numpy_global_matches_np_random_seed():
    reproducibility.seed_numpy_global(5)
    a = np.random.random(3)
    np.random.seed(5)
    assert np.array_equal(a, np.random.random(3))


def test_seed_torch_matches_manual_seed():
    reproducibility.seed_torch(5)
    a = torch.rand(3)
    torch.manual_seed(5)
    assert torch.equal(a, torch.rand(3))


def test_set_num_threads_is_reflected_by_torch():
    before = torch.get_num_threads()
    try:
        reproducibility.set_num_threads(1)
        assert torch.get_num_threads() == 1
    finally:
        torch.set_num_threads(before)


def test_environment_info_has_versions_and_commit():
    info = reproducibility.environment_info()
    assert set(info) == {"python", "numpy", "scipy", "qutip", "torch",
                         "git_commit", "torch_threads"}
    assert info["numpy"] == np.__version__
    assert info["torch_threads"] == torch.get_num_threads()
    commit = info["git_commit"]
    assert commit is None or len(commit) == 40
