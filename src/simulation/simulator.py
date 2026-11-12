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
from pathlib import Path

import numpy as np
import qutip as qt

from src.blocks import initial, legacy, runner, target
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


class QuTipSimulator(ports.ISimulator):
    """Simulates one trajectory per config by running its recipe.

    A thin facade over ``src.blocks``: `legacy.recipe_for` turns the config
    and the simulator alias into a `Recipe` with legacy seeds, and
    `runner.run_with_generator` runs it. The initial state is drawn from
    NumPy's global generator, so ``np.random.seed`` fixes the trajectory;
    the random blocks of one simulator object (the stochastic-gamma path)
    share one generator that advances from call to call.

    Attributes:
        alias: Which published physics is simulated, a key of
            `legacy.ALIAS_PARAMS`: ``plain`` (default), ``correlators``
            (also the nine two-body Pauli correlators, R23),
            ``entanglement`` (correlators and the Wootters concurrence,
            R24), ``stochastic_gamma`` (``s``, ``tau``, ``seed``: gamma(t)
            times a log-normal Ornstein-Uhlenbeck factor, R21),
            ``parity_sector`` (``sector``: two-qubit initial states in one
            parity sector, R6) and ``kondo`` (``rho_J``, ``kT``, ``kF_d``,
            ``dim``: the Kondo bath of R28 replaces the jump operators).
        params: The arguments of the alias.
        cache_dir: Directory of the content-addressed states cache, or
            None for no disk cache. The key covers the whole physics, so it
            works for every alias.
    """

    def __init__(
        self,
        alias: str = "plain",
        cache_dir: str | Path | None = None,
        **params: object,
    ) -> None:
        """Initialises the simulator.

        Args:
            alias: A key of `legacy.ALIAS_PARAMS`.
            cache_dir: Directory of the states cache, or None.
            **params: Arguments of the alias, e.g. ``s``, ``tau`` and
                ``seed`` for ``stochastic_gamma``.

        Raises:
            KeyError: If `alias` is unknown; the message lists the valid
                aliases.
            TypeError: If an argument is missing or unexpected.
            ValueError: If an argument is out of range.
        """
        legacy.validate_simulator(alias, params)
        self.alias = alias
        self.params = dict(params)
        self.cache_dir = None if cache_dir is None else Path(cache_dir)
        seed = self.params.get("seed")
        self._rng = np.random.default_rng(seed)
        self._seed = 0 if seed is None else seed
        self._calls = 0

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
        recipe = legacy.recipe_for(config, self.alias, **self.params)
        result = runner.run_with_generator(
            recipe, self._rng, index=self._calls, seed=self._seed,
            cache_dir=self.cache_dir)
        self._calls += 1
        return value_objects.TrajectoryResult(
            times=result.observations.times,
            observables=result.observations.series,
            t_decoh=result.target.time,
            system_config=config,
            censored=result.target.censored,
        )
