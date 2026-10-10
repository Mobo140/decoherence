"""Seeding, thread count and provenance of a run.

These are thin wrappers around the calls the code base makes anyway. They
add no random draws and decide neither where nor when seeding happens;
callers keep seeding at the same points as before. Importing this module
does not import torch.
"""
from __future__ import annotations

import pathlib
import platform
import subprocess
from importlib import metadata

import numpy as np

_REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
_PACKAGES = ("numpy", "scipy", "qutip", "torch")


def seed_numpy_global(seed: int) -> None:
    """Seeds NumPy's global generator.

    The Lindblad simulators draw from this generator (stochastic gamma and
    the like), so seeding it before a simulation makes the trajectory
    reproducible.

    Args:
        seed: Seed for `np.random.seed`.
    """
    np.random.seed(seed)


def seed_torch(seed: int) -> None:
    """Seeds torch's global generator.

    Weight initialisation, dropout and batch shuffling all draw from it, so
    this must run before the network is built.

    Args:
        seed: Seed for `torch.manual_seed`.
    """
    import torch
    torch.manual_seed(seed)


def set_num_threads(n: int) -> None:
    """Sets the number of CPU threads torch uses for intra-op parallelism.

    One thread makes training bit-exact between runs on a machine; more
    threads can change the summation order.

    Args:
        n: Number of threads.
    """
    import torch
    torch.set_num_threads(n)


def _git_commit() -> str | None:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=_REPO_ROOT, check=True,
            capture_output=True, text=True, timeout=5)
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout.strip() or None


def _package_version(name: str) -> str | None:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None


def environment_info() -> dict[str, str | int | None]:
    """Describes the software environment, to store next to results.

    Returns:
        Versions of python, numpy, scipy, qutip and torch (None for a
        package that is not installed), the git commit of the repository
        (None outside a git checkout) and the number of torch threads
        (None if torch is not installed).
    """
    info: dict[str, str | int | None] = {
        "python": platform.python_version(),
    }
    for name in _PACKAGES:
        info[name] = _package_version(name)
    info["git_commit"] = _git_commit()
    try:
        import torch
    except ImportError:
        info["torch_threads"] = None
    else:
        info["torch_threads"] = torch.get_num_threads()
    return info
