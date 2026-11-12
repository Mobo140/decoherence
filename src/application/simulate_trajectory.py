"""Use case: simulate a single quantum trajectory."""
from __future__ import annotations

import dataclasses

from src import reproducibility
from src.domain import ports
from src.domain import value_objects


@dataclasses.dataclass(frozen=True)
class SimulateTrajectoryCommand:
    """Input of `SimulateTrajectoryUseCase`.

    Attributes:
        config: The system to simulate.
        seed: Seeds numpy's global generator, which draws the initial
            state; `None` leaves it alone.
    """

    config: value_objects.SystemConfig
    seed: int | None = None


class SimulateTrajectoryUseCase:
    """Simulates one Lindblad trajectory of a SystemConfig."""

    def __init__(self, simulator: ports.ISimulator) -> None:
        """Wires the use case.

        Args:
            simulator: Simulates the configuration.
        """
        self._simulator = simulator

    def execute(
        self, command: SimulateTrajectoryCommand
    ) -> value_objects.TrajectoryResult:
        """Simulates the configuration.

        Args:
            command: What to simulate.

        Returns:
            The trajectory.
        """
        if command.seed is not None:
            reproducibility.seed_numpy_global(command.seed)
        return self._simulator.simulate(command.config)
