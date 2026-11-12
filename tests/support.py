"""Builders and test doubles shared by the test suite.

Plain functions and classes live here; `tests/conftest.py` wraps them in
fixtures. Nothing in this module touches `data/` or `experiments/results/`.
"""
from __future__ import annotations

import math
import pathlib
from collections.abc import Sequence

import numpy as np

from src.data import dataset as dataset_lib
from src.domain import ports
from src.domain import value_objects

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
RESULTS_DIR = REPO_ROOT / "experiments" / "results"
TRAJECTORY_DIR = REPO_ROOT / "data" / "trajectories"
CHECKPOINT_DIR = REPO_ROOT / "checkpoints"


def make_system_config(
    gamma: float = 0.4,
    t_max: float = 10.0,
    dt: float = 0.1,
    omega: float = 1.0,
) -> value_objects.SystemConfig:
    """Builds a one-qubit system with a constant damping rate.

    Args:
        gamma: Constant sigma_minus rate.
        t_max: End of the simulated time axis.
        dt: Time step.
        omega: Qubit frequency.

    Returns:
        The configuration; its analytic T2 is 2 / gamma.
    """
    return value_objects.SystemConfig(
        n_qubits=value_objects.QubitCount.ONE,
        omega=omega,
        dissipator=value_objects.DissipatorConfig.constant(gamma=gamma),
        t_max=t_max,
        dt=dt,
    )


def toy_dataset(n: int = 24, window: int = 10) -> dataset_lib.Dataset:
    """Builds windows of exact exponential decay with random rates.

    Args:
        n: Number of windows; 16 train, 4 validation, the rest test.
        window: Window length.

    Returns:
        A dataset with the features coherence_l1 and purity.
    """
    rng = np.random.default_rng(0)
    t = np.arange(window) * 0.1
    rates = rng.uniform(0.3, 1.0, n)
    seq = np.exp(-rates[:, None] * t[None, :])[:, :, None]
    seq = np.concatenate([seq, np.ones_like(seq)], axis=-1).astype(np.float32)
    t_obs = np.full(n, t[-1], dtype=np.float32)
    remaining = (2.0 / rates - t_obs).astype(np.float32)
    idx = np.arange(n)
    return dataset_lib.Dataset(
        sequences=seq, t_obs=t_obs, t_decoh_abs=t_obs + remaining,
        remaining_time=remaining,
        risk_labels=(remaining <= 1.0).astype(np.float32),
        train_idx=idx[:16], val_idx=idx[16:20], test_idx=idx[20:],
        feature_names=["coherence_l1", "purity"], metadata={"dt": 0.1},
    )


def small_dataset(
    n: int = 6,
    length: int = 4,
    names: Sequence[str] = ("a", "b", "corr_xx", "concurrence"),
) -> dataset_lib.Dataset:
    """Builds a random dataset with named feature columns.

    Args:
        n: Number of windows; the first three train, then one validation,
            the rest test (for the default n = 6).
        length: Window length.
        names: Feature names, one per column.

    Returns:
        A dataset whose targets are constant and whose windows are noise.
    """
    rng = np.random.default_rng(0)
    return dataset_lib.Dataset(
        sequences=rng.normal(size=(n, length, len(names))).astype(np.float32),
        t_obs=np.ones(n, dtype=np.float32),
        t_decoh_abs=np.full(n, 5.0, dtype=np.float32),
        remaining_time=np.full(n, 4.0, dtype=np.float32),
        risk_labels=np.zeros(n, dtype=np.float32),
        train_idx=np.array([0, 1, 2]),
        val_idx=np.array([3]),
        test_idx=np.array([4, 5]),
        feature_names=list(names),
        metadata={},
    )


def _damping_rate(config: value_objects.SystemConfig) -> float:
    """Constant rate, or the mean of the Bernstein coefficients."""
    dissipator = config.dissipator
    if dissipator.gamma_const is not None:
        return float(dissipator.gamma_const)
    return float(np.mean(dissipator.gamma_coeffs))


def decay_trajectory(
    config: value_objects.SystemConfig,
) -> value_objects.TrajectoryResult:
    """Builds the analytic trajectory C(t) = exp(-gamma t / 2).

    The coherence crosses the 1/e threshold at T2 = 2 / gamma (for the
    default threshold). Time-dependent rates use the mean coefficient.
    Both qubit counts get the same observables, with `coherence_l1` last as
    the physics prior expects.

    Args:
        config: System whose rate and time axis are used.

    Returns:
        The trajectory; censored when T2 lies beyond `t_max`, in which case
        `t_decoh` is `t_max` like the real simulator's.
    """
    gamma = _damping_rate(config)
    n_steps = int(round(config.t_max / config.dt)) + 1
    times = np.linspace(0.0, config.t_max, n_steps)
    coherence = np.exp(-gamma * times / 2.0)
    t2 = 2.0 * -math.log(config.decoherence_threshold) / gamma
    censored = t2 > config.t_max
    return value_objects.TrajectoryResult(
        times=times,
        observables={
            "sigma_x": coherence.astype(np.float32),
            "sigma_y": np.zeros(n_steps, dtype=np.float32),
            "sigma_z": np.zeros(n_steps, dtype=np.float32),
            "purity": ((1.0 + coherence ** 2) / 2.0).astype(np.float32),
            "coherence_l1": coherence.astype(np.float32),
        },
        t_decoh=config.t_max if censored else t2,
        system_config=config,
        censored=censored,
    )


def scripted_trajectory(
    t_decoh: float, t_max: float = 12.0, dt: float = 0.1,
) -> value_objects.TrajectoryResult:
    """Builds a trajectory with a prescribed T2 and a recognisable shape.

    Give each trajectory of a dataset its own `t_decoh` and the windows can
    be traced back to their source through `t_decoh_abs`.

    Args:
        t_decoh: Decoherence time stored on the result; the coherence decays
            as exp(-t / t_decoh).
        t_max: End of the time axis.
        dt: Time step.

    Returns:
        An uncensored trajectory with sigma_x, coherence_l1 and purity.
    """
    config = make_system_config(gamma=0.4, t_max=t_max, dt=dt)
    n = int(round(t_max / dt)) + 1
    times = np.linspace(0.0, t_max, n)
    return value_objects.TrajectoryResult(
        times=times,
        observables={
            "sigma_x": np.cos(times).astype(np.float32),
            "coherence_l1": np.exp(
                -times / max(t_decoh, 1e-6)).astype(np.float32),
            "purity": np.ones(n, dtype=np.float32),
        },
        t_decoh=t_decoh,
        system_config=config,
    )


class ExponentialDecaySimulator(ports.ISimulator):
    """Deterministic, instantaneous stand-in for the QuTiP simulator."""

    def simulate(
        self, config: value_objects.SystemConfig,
    ) -> value_objects.TrajectoryResult:
        """Returns `decay_trajectory(config)`."""
        return decay_trajectory(config)


class CountingSimulator(ports.ISimulator):
    """Forwards to another simulator and counts the calls."""

    def __init__(self, inner: ports.ISimulator | None = None) -> None:
        self.inner = inner or ExponentialDecaySimulator()
        self.calls = 0

    def simulate(
        self, config: value_objects.SystemConfig,
    ) -> value_objects.TrajectoryResult:
        """Counts the call and simulates with the wrapped simulator."""
        self.calls += 1
        return self.inner.simulate(config)


class GlobalRngSimulator(ports.ISimulator):
    """Draws its trajectories from numpy's global generator.

    Like the real simulator it takes the initial state from the global
    stream, so datasets built from it are reproducible only if the caller
    seeds that stream.
    """

    def simulate(
        self, config: value_objects.SystemConfig,
    ) -> value_objects.TrajectoryResult:
        """Returns a trajectory with a random T2 in [2, 6) and noisy sigma_x."""
        n = int(round(config.t_max / config.dt)) + 1
        times = np.linspace(0.0, config.t_max, n)
        t2 = 2.0 + 4.0 * float(np.random.rand())
        return value_objects.TrajectoryResult(
            times=times,
            observables={
                "sigma_x": np.random.rand(n).astype(np.float32),
                "coherence_l1": np.exp(-times / t2).astype(np.float32),
                "purity": np.ones(n, dtype=np.float32),
            },
            t_decoh=t2,
            system_config=config,
        )


class ScriptedSimulator(ports.ISimulator):
    """Hands out prepared trajectories in order, ignoring the config."""

    def __init__(
        self, trajectories: Sequence[value_objects.TrajectoryResult],
    ) -> None:
        self._trajectories = list(trajectories)
        self._next = 0

    def simulate(
        self, config: value_objects.SystemConfig,
    ) -> value_objects.TrajectoryResult:
        """Returns the next prepared trajectory."""
        trajectory = self._trajectories[self._next]
        self._next += 1
        return trajectory


def snapshot_protected_dirs() -> dict[str, object]:
    """Records the state of the directories tests must not write to.

    `data/trajectories` holds ~10^5 files, so it is compared through the
    directory's own modification time and size (any added or removed entry
    changes them) instead of by listing it.

    Returns:
        A comparable description of `experiments/results`,
        `data/trajectories` and `checkpoints`.
    """
    def listing(path: pathlib.Path) -> object:
        if not path.exists():
            return None
        return sorted(
            (p.name, p.stat().st_mtime_ns, p.stat().st_size)
            for p in path.iterdir())

    def directory_stamp(path: pathlib.Path) -> object:
        if not path.exists():
            return None
        stat = path.stat()
        return (stat.st_mtime_ns, stat.st_size)

    return {
        "experiments/results": listing(RESULTS_DIR),
        "data/trajectories": directory_stamp(TRAJECTORY_DIR),
        "checkpoints": listing(CHECKPOINT_DIR),
    }
