"""QuTiP simulation wrappers for Lindblad master equation."""
import numpy as np
import qutip as qt
from typing import List, Dict, Optional, Tuple
from .lindblad_systems import SingleQubitSystem, TwoQubitSystem
from .decoherence_time import (
    compute_observables_single_qubit,
    compute_observables_two_qubit,
    calculate_decoherence_time
)


def simulate_single_qubit(
    system: SingleQubitSystem,
    initial_state: qt.Qobj,
    times: np.ndarray,
    observables: List[str] = ['sigma_x', 'sigma_y', 'sigma_z']
) -> Dict[str, np.ndarray]:
    """Simulate single qubit evolution with time-dependent dissipation.
    
    Args:
        system: Single qubit system
        initial_state: Initial density matrix
        times: Time points for evolution
        observables: List of observables to compute
        
    Returns:
        Dictionary with trajectories of observables and density matrices
    """
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
        [],
        options=qt.Options(nsteps=10000)
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
    observables: List[str] = ['sigma_x1', 'sigma_y1', 'sigma_z1', 'sigma_x2', 'sigma_y2', 'sigma_z2']
) -> Dict[str, np.ndarray]:
    """Simulate two-qubit evolution with time-dependent dissipation.
    
    Args:
        system: Two-qubit system
        initial_state: Initial density matrix
        times: Time points for evolution
        observables: List of observables to compute
        
    Returns:
        Dictionary with trajectories of observables and density matrices
    """
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
        [],
        options=qt.Options(nsteps=10000)
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


def simulate_trajectory(
    system_type: str,
    system_params: dict,
    initial_state: qt.Qobj,
    times: np.ndarray,
    observables: List[str],
    decoherence_criterion: str = 'coherence',
    decoherence_threshold: float = 0.01
) -> Dict:
    """Simulate quantum trajectory and compute decoherence time.
    
    Args:
        system_type: 'single_qubit' or 'two_qubit'
        system_params: Parameters for system initialization
        initial_state: Initial density matrix
        times: Time points
        observables: List of observables to track
        decoherence_criterion: 'coherence' or 'purity'
        decoherence_threshold: Threshold for decoherence
        
    Returns:
        Dictionary with observables, density matrices, and decoherence time
    """
    # Create system
    if system_type == 'single_qubit':
        system = SingleQubitSystem(**system_params)
        obs_dict = simulate_single_qubit(system, initial_state, times, observables)
    elif system_type == 'two_qubit':
        system = TwoQubitSystem(**system_params)
        obs_dict = simulate_two_qubit(system, initial_state, times, observables)
    else:
        raise ValueError(f"Unknown system type: {system_type}")
    
    # Compute decoherence time
    density_matrices = obs_dict['density_matrices']
    
    # Get initial values for relative thresholds
    if decoherence_criterion == 'coherence':
        initial_value = obs_dict['coherence_l1'][0] if len(obs_dict['coherence_l1']) > 0 else None
    else:
        initial_value = obs_dict['purity'][0] if len(obs_dict['purity']) > 0 else None
    
    t_decoh = calculate_decoherence_time(
        times,
        density_matrices,
        criterion=decoherence_criterion,
        threshold=decoherence_threshold,
        initial_coherence=initial_value if decoherence_criterion == 'coherence' else None,
        initial_purity=initial_value if decoherence_criterion == 'purity' else None
    )
    
    # Store system parameters
    if system.gamma_const is not None:
        gamma_value = system.gamma_const
    elif system.gamma_coeffs is not None:
        gamma_value = system.gamma_coeffs  # Store coefficients
    else:
        gamma_value = 0.0
    
    result = {
        'observables': obs_dict,
        'times': times,
        't_decoh': t_decoh if t_decoh is not None else times[-1],  # Use max time if not reached
        'gamma': gamma_value,
        'system_params': system_params
    }
    
    return result
