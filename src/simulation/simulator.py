"""QuTiP adapter for the domain ISimulator port.

The physics itself (systems, master equation, measures) lives in
``src.physics``. This module draws the initial state, runs a system and
turns the resulting density matrices into a ``TrajectoryResult`` with a
decoherence time. The time is the first crossing of a threshold by the
chosen criterion, linearly interpolated, or the end of the window with
``censored=True`` if it is never crossed.
"""
from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import qutip as qt

from src.domain import ports, value_objects
from src.physics import lindblad_systems, measures, simulate_qutip

# Below this the initial value is treated as zero and the threshold is
# absolute instead of relative to it.
_MIN_INITIAL_VALUE = 1e-10
# Regularises the interpolation when two neighbouring values coincide.
_INTERPOLATION_EPS = 1e-12


def sample_initial_state(config: value_objects.SystemConfig) -> qt.Qobj:
    """Draws the random pure initial state of a trajectory.

    Uses NumPy's global generator, so ``np.random.seed`` makes it
    reproducible.

    Args:
        config: System specification; only the qubit count is used.

    Returns:
        Density matrix of a Haar-random pure state.
    """
    return lindblad_systems.sample_random_pure_state(
        n_qubits=config.n_qubits.value
    )


def interpolate_crossing(
    times: np.ndarray,
    values: Sequence[float],
    index: int,
    level: float,
) -> float:
    """Interpolates linearly where a curve crosses a level.

    Args:
        times: Time points.
        values: Curve values at ``times``.
        index: Index of the first value that is past the level.
        level: The level that is crossed between ``index - 1`` and ``index``.

    Returns:
        The crossing time; ``times[0]`` if the curve starts past the level.
    """
    if index == 0:
        return float(times[0])
    prev_val = float(values[index - 1])
    val = float(values[index])
    frac = (prev_val - level) / (prev_val - val + _INTERPOLATION_EPS)
    frac = float(np.clip(frac, 0.0, 1.0))
    return float(times[index - 1] + frac * (times[index] - times[index - 1]))


def first_crossing(
    times: np.ndarray,
    values: Sequence[float],
    level: float,
    inclusive: bool = False,
) -> tuple[float, bool]:
    """Finds the first time a decaying curve falls below a level.

    Every decoherence criterion reduces to this function.

    Args:
        times: Time points.
        values: Curve values at ``times``.
        level: The level to fall below.
        inclusive: If True, a value equal to ``level`` already counts.

    Returns:
        ``(time, censored)``. ``time`` is the interpolated crossing time.
        If the level is never reached, ``time`` is ``times[-1]``, a lower
        bound, and ``censored`` is True.
    """
    for i, value in enumerate(values):
        crossed = value <= level if inclusive else value < level
        if crossed:
            return interpolate_crossing(times, values, i, level), False
    return float(times[-1]), True


def relative_level(values: Sequence[float], threshold: float) -> float:
    """Returns ``threshold`` times the initial value of a curve.

    Args:
        values: Curve values; the first one is the initial value.
        threshold: Fraction of the initial value, e.g. 1/e.

    Returns:
        The level, or ``threshold`` itself if the curve starts at zero.
    """
    initial = float(values[0]) if len(values) > 0 else 1.0
    return threshold * initial if initial > _MIN_INITIAL_VALUE else threshold


def sudden_death_time(
    times: np.ndarray, lam: Sequence[float]
) -> tuple[float, bool]:
    """Finds the time of entanglement sudden death.

    The first zero of the concurrence, interpolated on the unclipped
    Wootters lambda. A state that starts separable gives 0 (no usable
    windows); one that is still entangled at the end is censored.

    Args:
        times: Time points.
        lam: Unclipped Wootters lambda at ``times``.

    Returns:
        ``(time, censored)`` as in ``first_crossing``.
    """
    return first_crossing(times, lam, 0.0, inclusive=True)


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
        [0, t_max] (or the criterion is not handled here), ``t_decoh`` is
        ``times[-1]`` and ``censored`` is True.
    """
    criterion = config.decoherence_criterion
    threshold = config.decoherence_threshold
    coherence = observables.get("coherence_l1")
    purity = observables.get("purity")

    if (criterion == value_objects.DecoherenceCriterion.COHERENCE
            and coherence is not None):
        return first_crossing(
            times, coherence, relative_level(coherence, threshold)
        )
    if (criterion == value_objects.DecoherenceCriterion.PURITY
            and purity is not None):
        initial = float(purity[0]) if len(purity) > 0 else 1.0
        p_min = 1.0 / (2 ** config.n_qubits.value)
        return first_crossing(
            times, purity, initial - threshold * (initial - p_min)
        )
    return float(times[-1]), True


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
    """
    eigenvectors = np.linalg.eigh(hamiltonian)[1]
    values = [
        measures.energy_basis_coherence_l1(
            r.full() if hasattr(r, "full") else np.asarray(r), eigenvectors
        )
        for r in density_matrices
    ]
    return first_crossing(
        times, values, relative_level(values, config.decoherence_threshold)
    )


def _two_qubit_t_decoh(
    config: value_objects.SystemConfig,
    times: np.ndarray,
    hamiltonian: np.ndarray,
    observables: dict,
) -> tuple[float, bool]:
    """Dispatches the decoherence criterion of a two-qubit trajectory."""
    criterion = config.decoherence_criterion
    rhos = observables["density_matrices"]
    if criterion == value_objects.DecoherenceCriterion.COHERENCE_ENERGY:
        return energy_basis_t_decoh(config, times, hamiltonian, rhos)
    if criterion == value_objects.DecoherenceCriterion.CONCURRENCE:
        values = [measures.concurrence(r) for r in rhos]
        return first_crossing(
            times, values, relative_level(values, config.decoherence_threshold)
        )
    if criterion == value_objects.DecoherenceCriterion.ENTANGLEMENT_DEATH:
        return sudden_death_time(
            times, [measures.wootters_lambda(r) for r in rhos]
        )
    return compute_t_decoh(config, times, observables)


def _gamma_coeffs(config: value_objects.SystemConfig) -> np.ndarray | None:
    """Returns the Bernstein coefficients of the config as an array."""
    coeffs = config.dissipator.gamma_coeffs
    return np.array(coeffs) if coeffs is not None else None


def _single_qubit_system(
    config: value_objects.SystemConfig,
) -> lindblad_systems.SingleQubitSystem:
    """Builds the single-qubit Lindblad system of a config."""
    return lindblad_systems.SingleQubitSystem(
        omega=config.omega,
        gamma_const=config.dissipator.gamma_const,
        gamma_coeffs=_gamma_coeffs(config),
        jump_operators=[config.dissipator.operator_type.value],
        gamma_t_max=config.t_max,
    )


def _two_qubit_system(
    config: value_objects.SystemConfig,
) -> lindblad_systems.TwoQubitSystem:
    """Builds the two-qubit Lindblad system of a config."""
    return lindblad_systems.TwoQubitSystem(
        J=config.J,
        interaction_type=config.interaction_type.value,
        gamma_const=config.dissipator.gamma_const,
        gamma_coeffs=_gamma_coeffs(config),
        gamma_t_max=config.t_max,
        jump_operators=[config.dissipator.operator_type.value],
    )


class QuTipSimulator(ports.ISimulator):
    """Simulates the Lindblad master equation with QuTiP.

    Subclasses can change a run through three hooks, each a no-op or the
    published behaviour by default: ``_initial_state``, ``_prepare_system``
    and ``_extra_observables``.
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
        """
        n_steps = int(round(config.t_max / config.dt)) + 1
        times = np.linspace(0.0, config.t_max, n_steps)
        initial_state = self._initial_state(config)

        if config.n_qubits == value_objects.QubitCount.ONE:
            system = self._prepare_system(_single_qubit_system(config), config)
            observables = simulate_qutip.simulate_single_qubit(
                system, initial_state, times
            )
            t_decoh, censored = compute_t_decoh(config, times, observables)
        else:
            system = self._prepare_system(_two_qubit_system(config), config)
            observables = simulate_qutip.simulate_two_qubit(
                system, initial_state, times
            )
            observables = self._with_extra_observables(observables)
            t_decoh, censored = _two_qubit_t_decoh(
                config, times, system.H.full(), observables
            )

        # The raw density matrices are large and are not model features.
        observables.pop("density_matrices", None)
        return value_objects.TrajectoryResult(
            times=times,
            observables=observables,
            t_decoh=t_decoh,
            system_config=config,
            censored=censored,
        )

    def _with_extra_observables(self, observables: dict) -> dict:
        """Inserts the hook's observables before purity and coherence_l1."""
        extra = self._extra_observables(observables["density_matrices"])
        if not extra:
            return observables
        tail = {
            key: observables.pop(key)
            for key in ("purity", "coherence_l1", "density_matrices")
        }
        return {**observables, **extra, **tail}

    def _initial_state(self, config: value_objects.SystemConfig) -> qt.Qobj:
        """Draws the initial pure state.

        Override to constrain the preparation (the parity-sector experiment
        does); the default is what every other run used.
        """
        return sample_initial_state(config)

    def _prepare_system(self, system, config: value_objects.SystemConfig):
        """Adjusts the Lindblad system before integration.

        Override to modify it (the stochastic-rate variant does); the
        default returns the system unchanged.
        """
        return system

    def _extra_observables(self, density_matrices) -> dict:
        """Computes additional two-qubit observables from rho(t).

        They are inserted before purity and coherence_l1, which must stay
        the last two feature columns. By default nothing is added.
        """
        return {}
