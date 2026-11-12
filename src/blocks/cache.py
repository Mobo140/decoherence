"""A content-addressed disk cache of ``States``.

The key is a sha1 of everything that determines the integration input: the
resolved system, channel and initial blocks, the time grid and the identity
of the trajectory's seed, plus the bytes of the actual mesolve input
(Hamiltonian, jump operators, rates sampled on the grid, initial state).
The last part makes the key honest for blocks whose behaviour depends on a
random stream, such as a shared generator in legacy mode. States are stored
as complex128 in ``.npz`` files, so a hit returns exactly the bits of a
miss.
"""
from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Sequence
from pathlib import Path

import numpy as np

from src.blocks.channels import Channel
from src.blocks.evolve import States
from src.blocks.system import System


def _digest_arrays(hasher, arrays: Sequence[np.ndarray]) -> None:
    """Feeds the dtype, shape and bytes of every array to the hasher."""
    for array in arrays:
        array = np.ascontiguousarray(array)
        hasher.update(f"{array.dtype}{array.shape}".encode())
        hasher.update(array.tobytes())


def states_key(
    description: dict,
    system: System,
    channels: Sequence[Channel],
    rho0: np.ndarray,
    times: np.ndarray,
) -> str:
    """Computes the cache key of one integration.

    Args:
        description: JSON-serialisable description of the recipe part and
            seed identity; floats are written with their exact repr.
        system: The system.
        channels: The channels as built for this trajectory.
        rho0: Initial density matrix as an array.
        times: The time grid.

    Returns:
        The sha1 hex digest.
    """
    hasher = hashlib.sha1()
    hasher.update(json.dumps(description, sort_keys=True).encode())
    arrays = [system.H.full(), rho0, times]
    for channel in channels:
        arrays.append(channel.operator.full())
        if callable(channel.rate):
            arrays.append(np.array([channel.rate(t) for t in times]))
        else:
            arrays.append(np.array([channel.rate]))
    _digest_arrays(hasher, arrays)
    return hasher.hexdigest()


def _path(cache_dir: Path, key: str) -> Path:
    return Path(cache_dir) / f"{key}.npz"


def load(cache_dir: Path, key: str) -> States | None:
    """Reads cached states.

    Args:
        cache_dir: Cache directory.
        key: Key from `states_key`.

    Returns:
        The states, or None if the key is not in the cache.
    """
    path = _path(cache_dir, key)
    if not path.exists():
        return None
    with np.load(path) as data:
        return States(
            times=data["times"],
            rho=[np.array(r, dtype=np.complex128) for r in data["rho"]])


def store(cache_dir: Path, key: str, states: States) -> None:
    """Writes states to the cache, atomically.

    Args:
        cache_dir: Cache directory; created if missing.
        key: Key from `states_key`.
        states: The states to store.
    """
    directory = Path(cache_dir)
    directory.mkdir(parents=True, exist_ok=True)
    target = _path(directory, key)
    temporary = target.with_name(f"{key}.{os.getpid()}.tmp")
    with open(temporary, "wb") as handle:
        np.savez(
            handle, times=np.asarray(states.times, dtype=np.float64),
            rho=np.asarray(states.rho, dtype=np.complex128))
    os.replace(temporary, target)
