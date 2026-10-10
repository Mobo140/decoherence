"""Use-case: generate a labelled ML dataset from a list of SystemConfigs."""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import List, Optional

import numpy as np

from src.data import dataset as dataset_lib
from src.data import windows
from src.domain.ports import ISimulator, ITrajectoryStore
from src.domain.value_objects import SystemConfig, TrajectoryResult


def trajectory_cache_key(config: SystemConfig, seed: Optional[int], index: int) -> str:
    """Stable per-trajectory name for ITrajectoryStore (content-addressed)."""
    d = config.dissipator
    payload = {
        "nq": config.n_qubits.value,
        "omega": config.omega,
        "op": d.operator_type.value,
        "g": d.gamma_const,
        "c": list(d.gamma_coeffs) if d.gamma_coeffs else None,
        "it": config.interaction_type.value,
        "J": config.J,
        "tmax": config.t_max,
        "dt": config.dt,
        "crit": config.decoherence_criterion.value,
        "thr": config.decoherence_threshold,
        "seed": seed,
        "i": index,
    }
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return "t_" + hashlib.sha1(raw.encode()).hexdigest()[:16]


# ---------------------------------------------------------------------------
# Command + Use-case
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class GenerateDatasetCommand:
    """Input for the GenerateDatasetUseCase."""

    configs: List[SystemConfig]
    window_length: int = 50
    horizon: float = 1.0
    samples_per_trajectory: int = 5
    train_ratio: float = 0.8
    val_ratio: float = 0.1
    seed: Optional[int] = 42
    dataset_name: str = "dataset"
    # Minimum gap between sampled window end-indices (in timesteps).
    # 0 = off.  Positive values reduce within-trajectory overlap,
    # improving generalisation.
    min_window_gap: int = 0
    # False (default): drop right-censored trajectories.
    # True: keep them; remaining_time is a lower bound, Dataset.censored=True.
    include_censored: bool = False


class GenerateDatasetUseCase:
    """Simulate multiple systems and produce a windowed ML dataset.

    Sliding-window approach:
    - For each trajectory, sample `samples_per_trajectory` random windows.
    - Each window of length `window_length` ends at time t_obs.
    - Label: risk = 1 if t_obs < t_decoh ≤ t_obs + horizon, else 0.
    - Target: remaining_time = t_decoh - t_obs (strictly positive for pre-decoherence windows).

    Train/val/test split is done at the *trajectory* level to prevent
    data leakage across time windows from the same trajectory.
    """

    def __init__(
        self,
        simulator: ISimulator,
        store: Optional[ITrajectoryStore] = None,
    ) -> None:
        self._simulator = simulator
        self._store = store

    def execute(self, command: GenerateDatasetCommand) -> dataset_lib.Dataset:
        rng = np.random.default_rng(command.seed)

        # QuTipSimulator.simulate() draws the initial pure state from numpy's
        # *global* RNG (sample_random_pure_state is called without a seed), so
        # without this the same command simulates different physical states on
        # every run and the dataset is not reproducible.  Cached trajectories
        # hide this; a cold cache does not.
        if command.seed is not None:
            np.random.seed(command.seed)

        trajectories: List[TrajectoryResult] = []
        for i, cfg in enumerate(command.configs):
            trajectories.append(self._get_or_simulate(cfg, command.seed, i))

        if self._store is not None:
            self._store.save(trajectories, command.dataset_name)

        return windows.build_dataset(
            trajectories,
            window_length=command.window_length,
            horizon=command.horizon,
            samples_per_trajectory=command.samples_per_trajectory,
            train_ratio=command.train_ratio,
            val_ratio=command.val_ratio,
            min_window_gap=command.min_window_gap,
            include_censored=command.include_censored,
            seed=command.seed,
            rng=rng,
        )

    def _get_or_simulate(
        self,
        cfg: SystemConfig,
        seed: Optional[int],
        index: int,
    ) -> TrajectoryResult:
        if self._store is None:
            return self._simulator.simulate(cfg)
        key = trajectory_cache_key(cfg, seed, index)
        try:
            return self._store.load(key)[0]
        except FileNotFoundError:
            traj = self._simulator.simulate(cfg)
            self._store.save([traj], key)
            return traj
