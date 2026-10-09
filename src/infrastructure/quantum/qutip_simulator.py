"""QuTiP adapter: implements ISimulator using the existing physics layer.

This adapter is the bridge between the domain port (ISimulator) and the
concrete QuTiP-based physics code in src/physics/.  No physics logic lives
here – it is delegated entirely to the existing physics modules.
"""
from __future__ import annotations

import numpy as np
import qutip as qt

from ...domain.ports import ISimulator
from ...domain.value_objects import (
    DecoherenceCriterion,
    DissipatorType,
    InteractionType,
    QubitCount,
    SystemConfig,
    TrajectoryResult,
)
from ...physics.lindblad_systems import (
    SingleQubitSystem,
    TwoQubitSystem,
    sample_random_pure_state,
)
from ...physics.simulate_qutip import simulate_single_qubit, simulate_two_qubit
from ...physics.decoherence_time import calculate_decoherence_time


class QuTipSimulator(ISimulator):
    """Simulate Lindblad master equation using QuTiP.

    Adapts the existing physics layer to the ISimulator port so that no
    application-layer code needs to import QuTiP directly.
    """

    def simulate(self, config: SystemConfig) -> TrajectoryResult:
        n_steps = int(round(config.t_max / config.dt)) + 1
        times = np.linspace(0.0, config.t_max, n_steps)

        initial_state = self._initial_state(config)

        if config.n_qubits == QubitCount.ONE:
            return self._simulate_single(config, initial_state, times)
        else:
            return self._simulate_two(config, initial_state, times)

    def _initial_state(self, config: SystemConfig) -> qt.Qobj:
        """Draw the initial pure state. Override to constrain its preparation.

        R6 overrides this to project onto one parity sector; nothing else
        does, so the default behaviour is what every published run used.
        """
        return sample_random_pure_state(n_qubits=config.n_qubits.value)

    def _prepare_system(self, system, config: SystemConfig):
        """Adjust the Lindblad system before integration. Override to modify it.

        R21 overrides this to multiply gamma(t) by a random process; nothing
        else does, so the default behaviour is what every published run used.
        """
        return system

    def _extra_observables(self, density_matrices) -> dict:
        """Additional observables per time step, computed from rho(t).

        They are inserted before purity and coherence_l1, which must stay the
        last two feature columns. R23 overrides this to add the two-body
        Pauli correlators; by default nothing is added.
        """
        return {}

    # ------------------------------------------------------------------
    # Single-qubit path
    # ------------------------------------------------------------------

    def _simulate_single(
        self,
        config: SystemConfig,
        initial_state: qt.Qobj,
        times: np.ndarray,
    ) -> TrajectoryResult:
        gamma_const, gamma_coeffs = self._unpack_gamma(config)
        jump_ops = [config.dissipator.operator_type.value]

        system = SingleQubitSystem(
            omega=config.omega,
            gamma_const=gamma_const,
            gamma_coeffs=(
                np.array(gamma_coeffs) if gamma_coeffs is not None else None
            ),
            jump_operators=jump_ops,
            gamma_t_max=config.t_max,
        )
        system = self._prepare_system(system, config)

        observables_keys = ["sigma_x", "sigma_y", "sigma_z"]
        obs_dict = simulate_single_qubit(
            system, initial_state, times, observables_keys
        )

        t_decoh, censored = self._compute_t_decoh(config, times, obs_dict)

        # Drop raw density matrices from the feature dict (too large, not needed)
        obs_dict.pop("density_matrices", None)

        return TrajectoryResult(
            times=times,
            observables=obs_dict,
            t_decoh=t_decoh,
            system_config=config,
            censored=censored,
        )

    # ------------------------------------------------------------------
    # Two-qubit path
    # ------------------------------------------------------------------

    def _simulate_two(
        self,
        config: SystemConfig,
        initial_state: qt.Qobj,
        times: np.ndarray,
    ) -> TrajectoryResult:
        gamma_const, gamma_coeffs = self._unpack_gamma(config)
        jump_ops = [config.dissipator.operator_type.value]

        system = TwoQubitSystem(
            J=config.J,
            interaction_type=config.interaction_type.value,
            gamma_const=gamma_const,
            gamma_coeffs=(
                np.array(gamma_coeffs) if gamma_coeffs is not None else None
            ),
            gamma_t_max=config.t_max,
            jump_operators=jump_ops,
        )
        system = self._prepare_system(system, config)

        observables_keys = [
            "sigma_x1", "sigma_y1", "sigma_z1",
            "sigma_x2", "sigma_y2", "sigma_z2",
        ]
        obs_dict = simulate_two_qubit(
            system, initial_state, times, observables_keys
        )

        extra = self._extra_observables(obs_dict["density_matrices"])
        if extra:
            tail = {k: obs_dict.pop(k) for k in ("purity", "coherence_l1", "density_matrices")}
            obs_dict = {**obs_dict, **extra, **tail}

        if config.decoherence_criterion == DecoherenceCriterion.COHERENCE_ENERGY:
            t_decoh, censored = self._energy_basis_t_decoh(
                config, times, system.H.full(), obs_dict["density_matrices"])
        else:
            t_decoh, censored = self._compute_t_decoh(config, times, obs_dict)
        obs_dict.pop("density_matrices", None)

        return TrajectoryResult(
            times=times,
            observables=obs_dict,
            t_decoh=t_decoh,
            system_config=config,
            censored=censored,
        )

    @staticmethod
    def _energy_basis_t_decoh(config, times, H, rhos) -> tuple[float, bool]:
        """1/e crossing of the l1-coherence in the eigenbasis of H.

        Unitary evolution only rotates the phases of rho_ij in that basis, so
        the crossing is set by dissipation alone. Only the target changes:
        the observables fed to the model stay in the computational basis.
        """
        U = np.linalg.eigh(H)[1]
        vals = []
        for r in rhos:
            m = r.full() if hasattr(r, "full") else np.asarray(r)
            e = U.conj().T @ m @ U
            vals.append(float(np.abs(e).sum() - np.abs(np.diag(e)).sum()))
        initial = vals[0]
        thr = config.decoherence_threshold
        abs_threshold = thr * initial if initial > 1e-10 else thr
        for i, v in enumerate(vals):
            if v < abs_threshold:
                return QuTipSimulator._interpolate_crossing(
                    times, vals, i, abs_threshold), False
        return float(times[-1]), True

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _unpack_gamma(
        config: SystemConfig,
    ) -> tuple[float | None, tuple[float, ...] | None]:
        d = config.dissipator
        gamma_const = d.gamma_const
        gamma_coeffs = list(d.gamma_coeffs) if d.gamma_coeffs is not None else None
        return gamma_const, gamma_coeffs

    @staticmethod
    def _interpolate_crossing(
        times: np.ndarray,
        values,
        index: int,
        abs_threshold: float,
    ) -> float:
        """Linear interpolation of the threshold crossing between index-1 and index."""
        if index == 0:
            return float(times[0])
        prev_val = float(values[index - 1])
        val = float(values[index])
        frac = (prev_val - abs_threshold) / (prev_val - val + 1e-12)
        frac = float(np.clip(frac, 0.0, 1.0))
        return float(times[index - 1] + frac * (times[index] - times[index - 1]))

    @staticmethod
    def _compute_t_decoh(
        config: SystemConfig,
        times: np.ndarray,
        obs_dict: dict,
    ) -> tuple[float, bool]:
        """Compute decoherence time from observable trajectories.

        Returns (t_decoh, censored).  ``censored=True`` means the 1/e
        threshold was never crossed inside [0, t_max]; ``t_decoh`` is then
        only a lower bound (equal to t_max).

        Crossing time is linearly interpolated between the last point above
        the threshold and the first point below it.
        """
        coherence_vals = obs_dict.get("coherence_l1")
        purity_vals = obs_dict.get("purity")

        criterion = config.decoherence_criterion
        threshold = config.decoherence_threshold

        if criterion == DecoherenceCriterion.COHERENCE and coherence_vals is not None:
            # Use relative threshold: decay to threshold * initial_coherence
            initial = float(coherence_vals[0]) if len(coherence_vals) > 0 else 1.0
            abs_threshold = threshold * initial if initial > 1e-10 else threshold
            for i, val in enumerate(coherence_vals):
                if val < abs_threshold:
                    t_cross = QuTipSimulator._interpolate_crossing(
                        times, coherence_vals, i, abs_threshold
                    )
                    return t_cross, False

        elif criterion == DecoherenceCriterion.PURITY and purity_vals is not None:
            initial = float(purity_vals[0]) if len(purity_vals) > 0 else 1.0
            p_min = 1.0 / (2 ** config.n_qubits.value)
            abs_threshold = initial - threshold * (initial - p_min)
            for i, val in enumerate(purity_vals):
                if val < abs_threshold:
                    t_cross = QuTipSimulator._interpolate_crossing(
                        times, purity_vals, i, abs_threshold
                    )
                    return t_cross, False

        # Threshold never crossed: t_decoh is a right-censored lower bound.
        return float(times[-1]), True
