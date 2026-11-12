"""QuTiP adapter for the domain ISimulator port.

The physics lives in ``src.blocks`` (system, channels, initial state,
evolution) and ``src.physics`` (measures). This module draws the initial
state, runs a system and turns the resulting density matrices into a
``TrajectoryResult`` with a decoherence time. The time is the first
crossing of a threshold by the chosen criterion, linearly interpolated, or
the end of the window with ``censored=True`` if it is never crossed.
"""
from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import qutip as qt

from src.blocks import channels, evolve, initial, observe, target
from src.blocks import system as blocks_system
# The crossing helpers moved to blocks.target; they stay importable here.
from src.blocks.target import (  # noqa: F401
    first_crossing,
    interpolate_crossing,
    relative_level,
    sudden_death_time,
)
from src.domain import ports, value_objects


def sample_initial_state(config: value_objects.SystemConfig) -> qt.Qobj:
    """Draws the random pure initial state of a trajectory.

    Uses NumPy's global generator, so ``np.random.seed`` makes it
    reproducible.

    Args:
        config: System specification; only the qubit count is used.

    Returns:
        Density matrix of a Haar-random pure state.
    """
    return initial.random_pure(build_system(config))


def compute_t_decoh(
    config: value_objects.SystemConfig,
    times: np.ndarray,
    observables: dict,
) -> tuple[float, bool]:
    """Computes the decoherence time from the coherence or purity curve.

    The coherence criterion uses a level relative to the initial l1
    coherence; the purity criterion measures the decay from the initial
    purity towards the maximally mixed value 1 / 2^n.

    Args:
        config: System specification with criterion and threshold.
        times: Time points.
        observables: Observable trajectories with ``coherence_l1`` and
            ``purity``.

    Returns:
        ``(t_decoh, censored)``. If the level is never crossed in
        [0, t_max], ``t_decoh`` is ``times[-1]`` and ``censored`` is True.

    Raises:
        ValueError: If the criterion is not COHERENCE or PURITY, or the
            curve is empty or contains NaN.
        KeyError: If `observables` lacks the curve the criterion needs.
    """
    criterion = config.decoherence_criterion
    threshold = config.decoherence_threshold

    if criterion == value_objects.DecoherenceCriterion.COHERENCE:
        coherence = _curve(observables, "coherence_l1", criterion)
        return first_crossing(
            times, coherence, relative_level(coherence, threshold)
        )
    if criterion == value_objects.DecoherenceCriterion.PURITY:
        purity = _curve(observables, "purity", criterion)
        initial = float(purity[0]) if len(purity) > 0 else 1.0
        p_min = 1.0 / (2 ** config.n_qubits.value)
        return first_crossing(
            times, purity, initial - threshold * (initial - p_min)
        )
    raise ValueError(
        f"compute_t_decoh handles the criteria 'coherence' and 'purity', "
        f"not {criterion.value!r}.")


def _curve(
    observables: dict,
    name: str,
    criterion: value_objects.DecoherenceCriterion,
) -> Sequence[float]:
    """Returns observables[name]; KeyError names the criterion if absent."""
    if name not in observables:
        raise KeyError(
            f"The criterion {criterion.value!r} needs the observable "
            f"{name!r}; got {sorted(observables)}.")
    return observables[name]


def energy_basis_t_decoh(
    config: value_objects.SystemConfig,
    times: np.ndarray,
    hamiltonian: np.ndarray,
    density_matrices: Sequence[np.ndarray],
) -> tuple[float, bool]:
    """Computes the decoherence time of the l1 coherence in the energy basis.

    Only the target changes: the observables fed to the model stay in the
    computational basis.

    Args:
        config: System specification with the threshold.
        times: Time points.
        hamiltonian: Hamiltonian as a dense array.
        density_matrices: Density matrix at each time point.

    Returns:
        ``(t_decoh, censored)`` as in ``first_crossing``.

    Raises:
        ValueError: If `hamiltonian` is not a Hermitian square matrix, or
            as `first_crossing`.
    """
    values = target.energy_basis_coherence(hamiltonian, density_matrices)
    return first_crossing(
        times, values, relative_level(values, config.decoherence_threshold)
    )


def compute_target(
    config: value_objects.SystemConfig,
    states: evolve.States,
    system: blocks_system.System,
) -> target.Target:
    """Computes the decoherence time of a config from the states.

    Args:
        config: System specification with criterion and threshold.
        states: The states of the trajectory.
        system: The system the states live on.

    Returns:
        The target block that the criterion of the config selects.

    Raises:
        ValueError: As the target blocks.
    """
    criterion = config.decoherence_criterion
    threshold = config.decoherence_threshold
    kinds = value_objects.DecoherenceCriterion
    if criterion == kinds.COHERENCE_ENERGY:
        return target.energy_basis_crossing(states, system, threshold)
    if criterion == kinds.ENTANGLEMENT_DEATH:
        return target.sudden_death(states, system)
    return target.coherence_crossing(
        states, system, threshold, criterion.value)


def build_system(
    config: value_objects.SystemConfig,
) -> blocks_system.System:
    """Builds the Hamiltonian block of a config.

    Args:
        config: System specification.

    Returns:
        The one- or two-qubit system.
    """
    if config.n_qubits == value_objects.QubitCount.ONE:
        return blocks_system.single_qubit(config.omega)
    return blocks_system.two_qubit(config.J, config.interaction_type.value)


def build_channels(
    system: blocks_system.System, config: value_objects.SystemConfig
) -> list[channels.Channel]:
    """Builds the phenomenological jump channels of a config.

    Args:
        system: The system built from `config`.
        config: System specification with the dissipator.

    Returns:
        One jump channel per qubit, with the constant rate or the Bernstein
        rate gamma(t) of the dissipator.
    """
    dissipator = config.dissipator
    op_type = dissipator.operator_type.value
    if dissipator.gamma_const is not None:
        return channels.jump(system, op_type, dissipator.gamma_const)
    return channels.bernstein_profile(
        channels.jump(system, op_type, 1.0),
        np.array(dissipator.gamma_coeffs), config.t_max)


class QuTipSimulator(ports.ISimulator):
    """Simulates the Lindblad master equation with QuTiP.

    Subclasses can change a run through three hooks, each the published
    behaviour by default: ``_initial_state``, ``_channels`` and
    ``_extra_observables``. The run itself is built from ``src.blocks``.
    """

    def simulate(
        self, config: value_objects.SystemConfig
    ) -> value_objects.TrajectoryResult:
        """Simulates one trajectory.

        The initial state is drawn from NumPy's global generator before
        anything else, so ``np.random.seed`` fixes the trajectory.

        Args:
            config: Full system specification.

        Returns:
            Observable trajectories (without density matrices) and the
            decoherence time.

        Raises:
            TypeError: If `config` is not a `SystemConfig`.
        """
        if not isinstance(config, value_objects.SystemConfig):
            raise TypeError(
                f"config must be a SystemConfig, got {config!r}.")
        n_steps = int(round(config.t_max / config.dt)) + 1
        times = np.linspace(0.0, config.t_max, n_steps)
        initial_state = self._initial_state(config)
        system = build_system(config)
        states = evolve.master_equation(
            system, self._channels(system, config), initial_state, times)

        parts = [observe.pauli_means(states)]
        if config.n_qubits == value_objects.QubitCount.TWO:
            parts += self._extra_observables(states)
        parts += [observe.purity(states), observe.coherence_l1(states)]
        observables = observe.model_layout(observe.combine(*parts)).series
        result_target = compute_target(config, states, system)
        t_decoh, censored = result_target.time, result_target.censored

        return value_objects.TrajectoryResult(
            times=times,
            observables=observables,
            t_decoh=t_decoh,
            system_config=config,
            censored=censored,
        )

    def _initial_state(self, config: value_objects.SystemConfig) -> qt.Qobj:
        """Draws the initial pure state.

        Override to constrain the preparation (the parity-sector experiment
        does); the default is what every other run used.
        """
        return sample_initial_state(config)

    def _channels(
        self,
        system: blocks_system.System,
        config: value_objects.SystemConfig,
    ) -> list[channels.Channel]:
        """Builds the dissipation channels of the run.

        Override to change them (the stochastic-rate and Kondo variants do);
        the default is the jump operators and rate of the config.
        """
        return build_channels(system, config)

    def _extra_observables(
        self, states: evolve.States
    ) -> list[observe.Observations]:
        """Computes additional two-qubit observations from rho(t).

        They are inserted before purity and coherence_l1. By default
        nothing is added.
        """
        return []
