"""Use case: generate a labelled ML dataset from a list of SystemConfigs."""
from __future__ import annotations

import dataclasses
import hashlib
import json

import numpy as np

from src import reproducibility
from src.data import dataset as dataset_lib
from src.data import windows
from src.domain import ports
from src.domain import value_objects


def trajectory_cache_key(
    config: value_objects.SystemConfig,
    seed: int | None,
    index: int,
) -> str:
    """Returns the content-addressed name of a trajectory in the store.

    Args:
        config: The simulated system.
        seed: Seed of the dataset command.
        index: Position of the configuration in the command.

    Returns:
        A stable name, `t_` followed by 16 hex digits of a SHA-1.
    """
    dissipator = config.dissipator
    payload = {
        "nq": config.n_qubits.value,
        "omega": config.omega,
        "op": dissipator.operator_type.value,
        "g": dissipator.gamma_const,
        "c": (list(dissipator.gamma_coeffs)
              if dissipator.gamma_coeffs else None),
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


@dataclasses.dataclass(frozen=True)
class GenerateDatasetCommand:
    """Input of `GenerateDatasetUseCase`.

    Attributes:
        configs: Systems to simulate.
        window_length: Observation window in time steps.
        horizon: Risk horizon.
        samples_per_trajectory: Windows drawn from each trajectory.
        train_ratio: Share of trajectories used for training.
        val_ratio: Share of trajectories used for validation.
        seed: Seed of the window sampling and of the initial states;
            `None` leaves both unseeded.
        dataset_name: Name under which the trajectories are saved.
        min_window_gap: Minimum gap between sampled window ends, in time
            steps; 0 is off. A positive value reduces the overlap
            within a trajectory.
        include_censored: False drops right-censored trajectories; True
            keeps them, with remaining time as a lower bound and
            `Dataset.censored` set.
    """

    configs: list[value_objects.SystemConfig]
    window_length: int = 50
    horizon: float = 1.0
    samples_per_trajectory: int = 5
    train_ratio: float = 0.8
    val_ratio: float = 0.1
    seed: int | None = 42
    dataset_name: str = "dataset"
    min_window_gap: int = 0
    include_censored: bool = False


class GenerateDatasetUseCase:
    """Simulates several systems and produces a windowed ML dataset.

    Each trajectory yields `samples_per_trajectory` random windows of
    length `window_length` ending at `t_obs`. The label is risk = 1 if
    `t_obs < t_decoh <= t_obs + horizon`; the target is
    `t_decoh - t_obs`. The train/val/test split is made per trajectory,
    so windows of one trajectory never leak across splits.
    """

    def __init__(
        self,
        simulator: ports.ISimulator,
        store: ports.ITrajectoryStore | None = None,
    ) -> None:
        """Wires the use case.

        Args:
            simulator: Simulates the configurations.
            store: Trajectory cache; `None` disables caching.
        """
        self._simulator = simulator
        self._store = store

    def execute(
        self, command: GenerateDatasetCommand
    ) -> dataset_lib.Dataset:
        """Simulates (or loads) the trajectories and cuts the windows.

        Args:
            command: What to generate.

        Returns:
            The dataset with train, validation and test indices.
        """
        rng = np.random.default_rng(command.seed)

        # `QuTipSimulator.simulate` draws the initial pure state from
        # numpy's global RNG, so without this the same command simulates
        # different states on every run. Cached trajectories hide the
        # problem; a cold cache does not.
        if command.seed is not None:
            reproducibility.seed_numpy_global(command.seed)

        trajectories = [
            self._get_or_simulate(config, command.seed, index)
            for index, config in enumerate(command.configs)
        ]
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
        config: value_objects.SystemConfig,
        seed: int | None,
        index: int,
    ) -> value_objects.TrajectoryResult:
        """Loads a trajectory from the store or simulates and saves it."""
        if self._store is None:
            return self._simulator.simulate(config)
        key = trajectory_cache_key(config, seed, index)
        try:
            return self._store.load(key)[0]
        except FileNotFoundError:
            trajectory = self._simulator.simulate(config)
            self._store.save([trajectory], key)
            return trajectory
