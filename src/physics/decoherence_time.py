"""Decoherence time calculation from density matrix trajectories."""
import numpy as np
from typing import List, Optional


def coherence_l1(rho: np.ndarray) -> float:
    """Calculate L1 coherence measure: sum of off-diagonal elements.
    
    Args:
        rho: Density matrix (2D array)
        
    Returns:
        L1 coherence value
    """
    return np.sum(np.abs(rho - np.diag(np.diag(rho))))


def purity(rho: np.ndarray) -> float:
    """Calculate purity: Tr(rho^2).
    
    Args:
        rho: Density matrix (2D array)
        
    Returns:
        Purity value
    """
    return np.real(np.trace(rho @ rho))


def calculate_decoherence_time(
    times: np.ndarray,
    density_matrices: List[np.ndarray],
    criterion: str = 'coherence',
    threshold: float = 0.01,
    initial_coherence: Optional[float] = None,
    initial_purity: Optional[float] = None
) -> Optional[float]:
    """Calculate decoherence time based on chosen criterion.
    
    Args:
        times: Array of time points
        density_matrices: List of density matrices at each time point
        criterion: 'coherence' or 'purity'
        threshold: Threshold value for decoherence (eps for coherence, P_thr for purity)
        initial_coherence: Initial coherence value (for relative threshold, optional)
        initial_purity: Initial purity value (for relative threshold, optional)
        
    Returns:
        Decoherence time, or None if threshold not reached
    """
    if criterion == 'coherence':
        values = [coherence_l1(rho) for rho in density_matrices]
        
        # Use absolute threshold or relative to initial
        if initial_coherence is not None:
            thresh = threshold * initial_coherence
        else:
            thresh = threshold
        
        # Find first time when coherence < threshold
        for i, val in enumerate(values):
            if val < thresh:
                return times[i]
        
        # If threshold not reached, return None or last time
        return None
    
    elif criterion == 'purity':
        values = [purity(rho) for rho in density_matrices]
        
        # Use absolute threshold or relative to initial
        if initial_purity is not None:
            thresh = initial_purity - threshold * (initial_purity - 1.0 / len(density_matrices[0]))
        else:
            thresh = threshold
        
        # Find first time when purity < threshold
        for i, val in enumerate(values):
            if val < thresh:
                return times[i]
        
        return None
    
    else:
        raise ValueError(f"Unknown criterion: {criterion}")


def compute_observables_single_qubit(rho: np.ndarray) -> dict:
    """Compute observables for single qubit.
    
    Args:
        rho: Density matrix (2x2)
        
    Returns:
        Dictionary with observables
    """
    # Pauli matrices
    sigma_x = np.array([[0, 1], [1, 0]], dtype=complex)
    sigma_y = np.array([[0, -1j], [1j, 0]], dtype=complex)
    sigma_z = np.array([[1, 0], [0, -1]], dtype=complex)
    
    obs_x = np.real(np.trace(rho @ sigma_x))
    obs_y = np.real(np.trace(rho @ sigma_y))
    obs_z = np.real(np.trace(rho @ sigma_z))
    
    return {
        'sigma_x': obs_x,
        'sigma_y': obs_y,
        'sigma_z': obs_z,
        'purity': purity(rho),
        'coherence_l1': coherence_l1(rho)
    }


def compute_observables_two_qubit(rho: np.ndarray) -> dict:
    """Compute observables for two qubits.
    
    Args:
        rho: Density matrix (4x4)
        
    Returns:
        Dictionary with observables
    """
    # Single-qubit Pauli operators
    sigma_x = np.array([[0, 1], [1, 0]], dtype=complex)
    sigma_y = np.array([[0, -1j], [1j, 0]], dtype=complex)
    sigma_z = np.array([[1, 0], [0, -1]], dtype=complex)
    I = np.eye(2, dtype=complex)
    
    # Tensor products
    sigma_x1 = np.kron(sigma_x, I)
    sigma_y1 = np.kron(sigma_y, I)
    sigma_z1 = np.kron(sigma_z, I)
    
    sigma_x2 = np.kron(I, sigma_x)
    sigma_y2 = np.kron(I, sigma_y)
    sigma_z2 = np.kron(I, sigma_z)
    
    obs_x1 = np.real(np.trace(rho @ sigma_x1))
    obs_y1 = np.real(np.trace(rho @ sigma_y1))
    obs_z1 = np.real(np.trace(rho @ sigma_z1))
    
    obs_x2 = np.real(np.trace(rho @ sigma_x2))
    obs_y2 = np.real(np.trace(rho @ sigma_y2))
    obs_z2 = np.real(np.trace(rho @ sigma_z2))
    
    return {
        'sigma_x1': obs_x1,
        'sigma_y1': obs_y1,
        'sigma_z1': obs_z1,
        'sigma_x2': obs_x2,
        'sigma_y2': obs_y2,
        'sigma_z2': obs_z2,
        'purity': purity(rho),
        'coherence_l1': coherence_l1(rho)
    }
