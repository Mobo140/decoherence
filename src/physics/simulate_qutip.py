"""QuTiP simulation wrappers for Lindblad master equation."""
from typing import Dict, List, Optional

import numpy as np
import qutip as qt

from .decoherence_time import (
    compute_observables_single_qubit,
    compute_observables_two_qubit,
)
from .lindblad_systems import SingleQubitSystem, TwoQubitSystem


def simulate_single_qubit(
    system: SingleQubitSystem,
    initial_state: qt.Qobj,
    times: np.ndarray,
    observables: Optional[List[str]] = None,
) -> Dict[str, np.ndarray]:
    """Simulate single qubit evolution with time-dependent dissipation.
    
    Args:
        system: Single qubit system
        initial_state: Initial density matrix
        times: Time points for evolution
        observables: Observable names to compute; defaults to
            ['sigma_x', 'sigma_y', 'sigma_z'].

    Returns:
        Dictionary with trajectories of observables and density matrices
    """
    if observables is None:
        observables = ['sigma_x', 'sigma_y', 'sigma_z']
    # Handle constant vs time-dependent gamma
    if system.gamma_const is not None:
        # Constant gamma: simple collapse operators list
        L_and_rates = system.get_lindblad_operators(0.0)
        collapse_ops = [L for L, rate in L_and_rates]
        rates = [rate for L, rate in L_and_rates]
        
        # For constant rate, multiply operators by sqrt(rate)
        c_ops = [np.sqrt(rate) * L for L, rate in zip(collapse_ops, rates)]
    else:
        # Time-dependent gamma: need to use list of functions
        # QuTiP expects sqrt(gamma(t)) for collapse operators
        def gamma_func(t, args):
            gamma_t = system.get_gamma(t)
            return np.sqrt(gamma_t) if gamma_t > 0 else 0.0
        
        # Create time-dependent collapse operators
        c_ops = []
        for L in system.L_list:
            # Format: [operator, coefficient function]
            c_ops.append([L, gamma_func])
    
    # Solve master equation
    result = qt.mesolve(
        system.H,
        initial_state,
        times,
        c_ops,
        e_ops=[],
        options={"nsteps": 10000},
    )

    # Extract density matrices
    density_matrices = [state.full() for state in result.states]

    # Compute observables
    obs_trajectories = {obs: [] for obs in observables}
    obs_trajectories['purity'] = []
    obs_trajectories['coherence_l1'] = []

    for rho in density_matrices:
        obs_dict = compute_observables_single_qubit(rho)
        for key in observables:
            if key in obs_dict:
                obs_trajectories[key].append(obs_dict[key])
        obs_trajectories['purity'].append(obs_dict['purity'])
        obs_trajectories['coherence_l1'].append(obs_dict['coherence_l1'])
    
    # Convert to numpy arrays
    for key in obs_trajectories:
        obs_trajectories[key] = np.array(obs_trajectories[key])
    
    obs_trajectories['density_matrices'] = density_matrices
    
    return obs_trajectories


def simulate_two_qubit(
    system: TwoQubitSystem,
    initial_state: qt.Qobj,
    times: np.ndarray,
    observables: Optional[List[str]] = None,
) -> Dict[str, np.ndarray]:
    """Simulate two-qubit evolution with time-dependent dissipation.
    
    Args:
        system: Two-qubit system
        initial_state: Initial density matrix
        times: Time points for evolution
        observables: Observable names to compute; defaults to the Pauli
            expectations of both qubits (sigma_{x,y,z}1 and sigma_{x,y,z}2).

    Returns:
        Dictionary with trajectories of observables and density matrices
    """
    if observables is None:
        observables = [
            'sigma_x1', 'sigma_y1', 'sigma_z1',
            'sigma_x2', 'sigma_y2', 'sigma_z2',
        ]
    # Handle constant vs time-dependent gamma
    if system.gamma_const is not None:
        # Constant gamma
        L_and_rates = system.get_lindblad_operators(0.0)
        collapse_ops = [L for L, rate in L_and_rates]
        rates = [rate for L, rate in L_and_rates]
        c_ops = [np.sqrt(rate) * L for L, rate in zip(collapse_ops, rates)]
    else:
        # Time-dependent gamma
        # QuTiP expects sqrt(gamma(t)) for collapse operators
        def gamma_func(t, args):
            gamma_t = system.get_gamma(t)
            return np.sqrt(gamma_t) if gamma_t > 0 else 0.0
        
        c_ops = []
        for L in system.L_list:
            c_ops.append([L, gamma_func])
    
    # Solve master equation
    result = qt.mesolve(
        system.H,
        initial_state,
        times,
        c_ops,
        e_ops=[],
        options={"nsteps": 10000},
    )

    # Extract density matrices
    density_matrices = [state.full() for state in result.states]

    # Compute observables
    obs_trajectories = {obs: [] for obs in observables}
    obs_trajectories['purity'] = []
    obs_trajectories['coherence_l1'] = []

    for rho in density_matrices:
        obs_dict = compute_observables_two_qubit(rho)
        for key in observables:
            if key in obs_dict:
                obs_trajectories[key].append(obs_dict[key])
        obs_trajectories['purity'].append(obs_dict['purity'])
        obs_trajectories['coherence_l1'].append(obs_dict['coherence_l1'])
    
    # Convert to numpy arrays
    for key in obs_trajectories:
        obs_trajectories[key] = np.array(obs_trajectories[key])
    
    obs_trajectories['density_matrices'] = density_matrices
    
    return obs_trajectories
