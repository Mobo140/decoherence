"""Baseline control actions for quantum systems."""
import numpy as np
import qutip as qt
from typing import Dict, List, Optional, Tuple, Callable
from .simulate_qutip import simulate_trajectory
from .lindblad_systems import SingleQubitSystem, TwoQubitSystem
from .decoherence_time import calculate_decoherence_time


class ControlAction:
    """Base class for control actions."""
    
    def apply(
        self,
        system,
        current_state: qt.Qobj,
        t: float,
        **kwargs
    ) -> qt.Qobj:
        """Apply control action to current state.
        
        Args:
            system: Quantum system
            current_state: Current density matrix
            t: Current time
            **kwargs: Additional parameters
            
        Returns:
            Modified state after control
        """
        raise NotImplementedError


class PulseControl(ControlAction):
    """Apply a pulse (instantaneous unitary) to the system."""
    
    def __init__(self, pulse_operator: qt.Qobj, strength: float = 0.1):
        """Initialize pulse control.
        
        Args:
            pulse_operator: Operator for pulse (e.g., sigma_x, sigma_y, sigma_z)
            strength: Pulse strength (rotation angle)
        """
        self.pulse_operator = pulse_operator
        self.strength = strength
    
    def apply(
        self,
        system,
        current_state: qt.Qobj,
        t: float,
        **kwargs
    ) -> qt.Qobj:
        """Apply pulse."""
        # Generate unitary from operator
        U = (-1j * self.strength * self.pulse_operator).expm()
        
        # Apply unitary: rho' = U * rho * U^dagger
        new_state = U * current_state * U.dag()
        
        return new_state


class ParameterModification(ControlAction):
    """Modify system parameters (e.g., reduce gamma temporarily)."""
    
    def __init__(self, param_name: str, modification_factor: float, duration: float):
        """Initialize parameter modification.
        
        Args:
            param_name: Parameter to modify (e.g., 'gamma_const')
            modification_factor: Factor to multiply parameter by
            duration: Duration of modification
        """
        self.param_name = param_name
        self.modification_factor = modification_factor
        self.duration = duration
    
    def apply(
        self,
        system,
        current_state: qt.Qobj,
        t: float,
        **kwargs
    ) -> qt.Qobj:
        """Apply parameter modification.
        
        Note: This returns the state unchanged, but modifies the system.
        Actual modification should be handled by the simulation loop.
        """
        # Modify system parameter
        if hasattr(system, self.param_name):
            original_value = getattr(system, self.param_name)
            setattr(system, self.param_name, original_value * self.modification_factor)
        
        return current_state


class ContinuousHamiltonianControl(ControlAction):
    """Add a control Hamiltonian for a duration."""
    
    def __init__(self, control_hamiltonian: qt.Qobj, strength: float, duration: float):
        """Initialize continuous control.
        
        Args:
            control_hamiltonian: Control Hamiltonian operator
            strength: Control strength
            duration: Duration of control
        """
        self.H_control = control_hamiltonian
        self.strength = strength
        self.duration = duration
    
    def apply(
        self,
        system,
        current_state: qt.Qobj,
        t: float,
        **kwargs
    ) -> qt.Qobj:
        """Apply continuous control.
        
        Note: Actual continuous control requires re-simulation with modified Hamiltonian.
        This is a simplified version that applies an effective unitary.
        """
        # Approximate continuous control as a unitary
        U = (-1j * self.strength * self.duration * self.H_control).expm()
        new_state = U * current_state * U.dag()
        
        return new_state


def simulate_with_trigger_control(
    system_type: str,
    system_params: dict,
    initial_state: qt.Qobj,
    times: np.ndarray,
    observables: List[str],
    decoherence_criterion: str,
    decoherence_threshold: float,
    risk_model: Optional[Callable] = None,
    risk_threshold: float = 0.5,
    control_action: Optional[ControlAction] = None,
    window_length: int = 50,
    check_interval: int = 5
) -> Dict:
    """Simulate trajectory with triggered control based on risk predictions.
    
    Args:
        system_type: 'single_qubit' or 'two_qubit'
        system_params: System parameters
        initial_state: Initial density matrix
        times: Time points for simulation
        observables: List of observables to track
        decoherence_criterion: 'coherence' or 'purity'
        decoherence_threshold: Threshold for decoherence
        risk_model: Function that predicts risk from observable window
                    Should have signature: risk_model(window) -> probability
        risk_threshold: Threshold for triggering control
        control_action: Control action to apply when risk exceeds threshold
        window_length: Length of observation window for risk prediction
        check_interval: How often to check risk (in timesteps)
        
    Returns:
        Dictionary with simulation results including control interventions
    """
    # Create system
    if system_type == 'single_qubit':
        system = SingleQubitSystem(**system_params)
    elif system_type == 'two_qubit':
        system = TwoQubitSystem(**system_params)
    else:
        raise ValueError(f"Unknown system type: {system_type}")
    
    # Storage for results
    density_matrices = [initial_state.full()]
    obs_trajectories = {obs: [] for obs in observables}
    obs_trajectories['purity'] = []
    obs_trajectories['coherence_l1'] = []
    
    risk_history = []
    control_times = []
    
    current_state = initial_state
    dt = times[1] - times[0]
    
    # Simulate step by step
    for i, t in enumerate(times[:-1]):
        # Check if we should evaluate risk
        if risk_model is not None and i >= window_length and i % check_interval == 0:
            # Extract observation window
            # Build feature matrix from recent observables
            window_features = []
            for obs_name in observables:
                if obs_name in obs_trajectories and len(obs_trajectories[obs_name]) >= window_length:
                    window_features.append(obs_trajectories[obs_name][-window_length:])
            
            if len(window_features) > 0:
                window = np.array(window_features).T  # (window_length, n_features)
                
                # Predict risk
                risk = risk_model(window)
                risk_history.append({'time': t, 'risk': risk})
                
                # Trigger control if risk exceeds threshold
                if risk >= risk_threshold and control_action is not None:
                    # Apply control
                    current_state_qobj = qt.Qobj(current_state, dims=initial_state.dims)
                    current_state_qobj = control_action.apply(system, current_state_qobj, t)
                    current_state = current_state_qobj.full()
                    control_times.append(t)
        
        # Evolve for one timestep
        t_segment = np.array([t, t + dt])
        current_state_qobj = qt.Qobj(current_state, dims=initial_state.dims)
        
        # Get Lindblad operators
        if system.gamma_const is not None:
            L_and_rates = system.get_lindblad_operators(t)
            collapse_ops = [L for L, rate in L_and_rates]
            rates = [rate for L, rate in L_and_rates]
            c_ops = [np.sqrt(rate) * L for L, rate in zip(collapse_ops, rates)]
        else:
            # Time-dependent gamma
            def gamma_func(t_val, args):
                gamma_t = system.get_gamma(t_val)
                return np.sqrt(gamma_t) if gamma_t > 0 else 0.0
            
            c_ops = []
            for L in system.L_list:
                c_ops.append([L, gamma_func])
        
        # Solve for one step
        result = qt.mesolve(
            system.H,
            current_state_qobj,
            t_segment,
            c_ops,
            [],
            options=qt.Options(nsteps=10000)
        )
        
        current_state = result.states[-1].full()
        density_matrices.append(current_state)
        
        # Compute observables
        from .decoherence_time import compute_observables_single_qubit, compute_observables_two_qubit
        
        if system_type == 'single_qubit':
            obs_dict = compute_observables_single_qubit(current_state)
        else:
            obs_dict = compute_observables_two_qubit(current_state)
        
        for key in observables:
            if key in obs_dict:
                obs_trajectories[key].append(obs_dict[key])
        obs_trajectories['purity'].append(obs_dict['purity'])
        obs_trajectories['coherence_l1'].append(obs_dict['coherence_l1'])
    
    # Convert to numpy arrays
    for key in obs_trajectories:
        obs_trajectories[key] = np.array(obs_trajectories[key])
    
    # Compute decoherence time
    if decoherence_criterion == 'coherence':
        initial_value = obs_trajectories['coherence_l1'][0] if len(obs_trajectories['coherence_l1']) > 0 else None
    else:
        initial_value = obs_trajectories['purity'][0] if len(obs_trajectories['purity']) > 0 else None
    
    t_decoh = calculate_decoherence_time(
        times,
        density_matrices,
        criterion=decoherence_criterion,
        threshold=decoherence_threshold,
        initial_coherence=initial_value if decoherence_criterion == 'coherence' else None,
        initial_purity=initial_value if decoherence_criterion == 'purity' else None
    )
    
    result = {
        'observables': obs_trajectories,
        'density_matrices': density_matrices,
        'times': times,
        't_decoh': t_decoh if t_decoh is not None else times[-1],
        'control_times': control_times,
        'risk_history': risk_history,
        'n_interventions': len(control_times)
    }
    
    return result


def compare_with_without_control(
    system_type: str,
    system_params: dict,
    initial_state: qt.Qobj,
    times: np.ndarray,
    observables: List[str],
    decoherence_criterion: str,
    decoherence_threshold: float,
    risk_model: Callable,
    risk_threshold: float,
    control_action: ControlAction,
    n_trials: int = 10,
    seed: Optional[int] = None
) -> Dict:
    """Compare system behavior with and without control.
    
    Args:
        system_type: System type
        system_params: System parameters
        initial_state: Initial state
        times: Time points
        observables: Observables to track
        decoherence_criterion: Criterion for decoherence
        decoherence_threshold: Threshold for decoherence
        risk_model: Risk prediction model
        risk_threshold: Threshold for triggering control
        control_action: Control action to apply
        n_trials: Number of trials (for different random parameters)
        seed: Random seed
        
    Returns:
        Dictionary with comparison results
    """
    if seed is not None:
        np.random.seed(seed)
    
    results_without_control = []
    results_with_control = []
    
    for trial in range(n_trials):
        # Randomize initial state slightly
        if trial > 0:
            # Add small perturbation
            dim = initial_state.shape[0]
            psi = qt.rand_ket(dim)
            initial_state = psi * psi.dag()
        
        # Simulate without control
        result_no_control = simulate_trajectory(
            system_type=system_type,
            system_params=system_params,
            initial_state=initial_state,
            times=times,
            observables=observables,
            decoherence_criterion=decoherence_criterion,
            decoherence_threshold=decoherence_threshold
        )
        results_without_control.append(result_no_control)
        
        # Simulate with control
        result_with_control = simulate_with_trigger_control(
            system_type=system_type,
            system_params=system_params,
            initial_state=initial_state,
            times=times,
            observables=observables,
            decoherence_criterion=decoherence_criterion,
            decoherence_threshold=decoherence_threshold,
            risk_model=risk_model,
            risk_threshold=risk_threshold,
            control_action=control_action
        )
        results_with_control.append(result_with_control)
    
    # Compute statistics
    t_decoh_no_control = [r['t_decoh'] for r in results_without_control]
    t_decoh_with_control = [r['t_decoh'] for r in results_with_control]
    
    # Purity and coherence at final time
    purity_no_control = [r['observables']['purity'][-1] for r in results_without_control]
    purity_with_control = [r['observables']['purity'][-1] for r in results_with_control]
    
    coherence_no_control = [r['observables']['coherence_l1'][-1] for r in results_without_control]
    coherence_with_control = [r['observables']['coherence_l1'][-1] for r in results_with_control]
    
    # Number of interventions
    n_interventions = [r['n_interventions'] for r in results_with_control]
    
    comparison = {
        't_decoh_no_control': {
            'mean': np.mean(t_decoh_no_control),
            'std': np.std(t_decoh_no_control),
            'values': t_decoh_no_control
        },
        't_decoh_with_control': {
            'mean': np.mean(t_decoh_with_control),
            'std': np.std(t_decoh_with_control),
            'values': t_decoh_with_control
        },
        'improvement': {
            'mean_delta_t': np.mean(t_decoh_with_control) - np.mean(t_decoh_no_control),
            'relative_improvement': (np.mean(t_decoh_with_control) - np.mean(t_decoh_no_control)) / np.mean(t_decoh_no_control) * 100
        },
        'purity_no_control': {
            'mean': np.mean(purity_no_control),
            'std': np.std(purity_no_control)
        },
        'purity_with_control': {
            'mean': np.mean(purity_with_control),
            'std': np.std(purity_with_control)
        },
        'coherence_no_control': {
            'mean': np.mean(coherence_no_control),
            'std': np.std(coherence_no_control)
        },
        'coherence_with_control': {
            'mean': np.mean(coherence_with_control),
            'std': np.std(coherence_with_control)
        },
        'n_interventions': {
            'mean': np.mean(n_interventions),
            'std': np.std(n_interventions),
            'values': n_interventions
        }
    }
    
    return comparison
