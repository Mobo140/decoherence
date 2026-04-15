"""Use-case: simulate a single quantum trajectory."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from ..domain.ports import ISimulator
from ..domain.value_objects import SystemConfig, TrajectoryResult


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
            np.random.seed(command.seed)

        return self._simulator.simulate(command.config)
