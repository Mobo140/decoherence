"""Coherence, purity and Pauli observables of density matrices."""
import numpy as np


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
