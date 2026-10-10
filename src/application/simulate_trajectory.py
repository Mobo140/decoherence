"""Use-case: simulate a single quantum trajectory."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from src import reproducibility
from src.domain.ports import ISimulator
from src.domain.value_objects import SystemConfig, TrajectoryResult


@dataclass(frozen=True)
class SimulateTrajectoryCommand:
    """Input for the SimulateTrajectoryUseCase."""

    config: SystemConfig
    seed: Optional[int] = None


class SimulateTrajectoryUseCase:
    """Simulate one Lindblad trajectory given a SystemConfig.

    Dependencies are injected via the constructor so the use-case
    remains independent of any concrete framework.
    """

    def __init__(self, simulator: ISimulator) -> None:
        self._simulator = simulator

    def execute(self, command: SimulateTrajectoryCommand) -> TrajectoryResult:
        if command.seed is not None:
            reproducibility.seed_numpy_global(command.seed)

        return self._simulator.simulate(command.config)
