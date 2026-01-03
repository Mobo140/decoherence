"""Lindblad system definitions and parameterization."""
import numpy as np
import qutip as qt
from typing import List, Tuple, Optional, Callable
from scipy.special import comb


def bernstein_polynomial(degree: int, k: int, t: float) -> float:
    """Evaluate Bernstein polynomial B_{k,n}(t).
    
    Args:
        degree: Degree n of polynomial
        k: Index k (0 <= k <= n)
        t: Evaluation point in [0, 1]
        
    Returns:
        Value of Bernstein polynomial
    """
    if k < 0 or k > degree:
        return 0.0
    return comb(degree, k) * (t ** k) * ((1 - t) ** (degree - k))


def bernstein_gamma(t: float, coeffs: np.ndarray, t_max: float) -> float:
    """Evaluate time-dependent gamma(t) using Bernstein polynomials.
    
    Args:
        t: Time point
        coeffs: Bernstein coefficients (must be non-negative)
        t_max: Maximum time (normalization)
        
    Returns:
        Gamma value at time t (guaranteed >= 0)
    """
    t_norm = np.clip(t / t_max, 0.0, 1.0)
    degree = len(coeffs) - 1
    gamma_val = sum(coeffs[k] * bernstein_polynomial(degree, k, t_norm) for k in range(len(coeffs)))
    return max(0.0, gamma_val)  # Ensure non-negative


class SingleQubitSystem:
    """Single qubit system with Lindblad dissipation."""
    
    def __init__(
        self,
        omega: float = 1.0,
        gamma_const: Optional[float] = None,
        gamma_coeffs: Optional[np.ndarray] = None,
        jump_operators: List[str] = ['sigma_minus'],  # 'sigma_minus' or 'sigma_z'
        gamma_t_max: float = 10.0
    ):
        """Initialize single qubit system.
        
        Args:
            omega: Frequency for H = omega * sigma_z
            gamma_const: Constant dissipation rate (if None, use time-dependent)
            gamma_coeffs: Bernstein coefficients for time-dependent gamma(t)
            jump_operators: List of jump operator types
            gamma_t_max: Maximum time for gamma(t) normalization
        """
        self.omega = omega
        self.gamma_const = gamma_const
        self.gamma_coeffs = gamma_coeffs
        self.jump_operators = jump_operators
        self.gamma_t_max = gamma_t_max
        
        # Build Hamiltonian: H = omega * sigma_z
        self.H = omega * qt.sigmaz()
        
        # Build jump operators
        self.L_list = []
        for op_type in jump_operators:
            if op_type == 'sigma_minus':
                self.L_list.append(qt.destroy(2))  # sigma_-
            elif op_type == 'sigma_z':
                self.L_list.append(qt.sigmaz())
            else:
                raise ValueError(f"Unknown jump operator: {op_type}")
    
    def get_gamma(self, t: float) -> float:
        """Get dissipation rate at time t.
        
        Args:
            t: Time point
            
        Returns:
            Dissipation rate gamma(t)
        """
        if self.gamma_const is not None:
            return self.gamma_const
        elif self.gamma_coeffs is not None:
            return bernstein_gamma(t, self.gamma_coeffs, self.gamma_t_max)
        else:
            return 0.0
    
    def get_lindblad_operators(self, t: float) -> List[Tuple[qt.Qobj, float]]:
        """Get Lindblad operators with time-dependent rates.
        
        Args:
            t: Time point
            
        Returns:
            List of (operator, rate) tuples
        """
        gamma = self.get_gamma(t)
        return [(L, gamma) for L in self.L_list]


class TwoQubitSystem:
    """Two-qubit system with interaction and dissipation."""
    
    def __init__(
        self,
        J: float = 0.5,  # Interaction strength
        interaction_type: str = 'XXZ',  # 'XXZ' or 'TFIM'
        gamma_const: Optional[float] = None,
        gamma_coeffs: Optional[np.ndarray] = None,
        gamma_t_max: float = 10.0
    ):
        """Initialize two-qubit system.
        
        Args:
            J: Interaction strength
            interaction_type: Type of interaction ('XXZ' or 'TFIM')
            gamma_const: Constant dissipation rate per qubit
            gamma_coeffs: Bernstein coefficients for time-dependent gamma(t)
            gamma_t_max: Maximum time for gamma(t) normalization
        """
        self.J = J
        self.interaction_type = interaction_type
        self.gamma_const = gamma_const
        self.gamma_coeffs = gamma_coeffs
        self.gamma_t_max = gamma_t_max
        
        # Pauli operators for two qubits
        sx1 = qt.tensor(qt.sigmax(), qt.qeye(2))
        sy1 = qt.tensor(qt.sigmay(), qt.qeye(2))
        sz1 = qt.tensor(qt.sigmaz(), qt.qeye(2))
        
        sx2 = qt.tensor(qt.qeye(2), qt.sigmax())
        sy2 = qt.tensor(qt.qeye(2), qt.sigmay())
        sz2 = qt.tensor(qt.qeye(2), qt.sigmaz())
        
        # Build Hamiltonian
        if interaction_type == 'XXZ':
            # H = J * (sx1*sx2 + sy1*sy2 + delta*sz1*sz2), delta=1 for simplicity
            self.H = J * (sx1 * sx2 + sy1 * sy2 + sz1 * sz2)
        elif interaction_type == 'TFIM':
            # Transverse field Ising model
            h_field = 1.0  # Transverse field strength
            self.H = J * sx1 * sx2 + h_field * (sz1 + sz2)
        else:
            raise ValueError(f"Unknown interaction type: {interaction_type}")
        
        # Jump operators: sigma_- on each qubit
        self.L_list = [
            qt.tensor(qt.destroy(2), qt.qeye(2)),  # sigma_- on qubit 1
            qt.tensor(qt.qeye(2), qt.destroy(2))   # sigma_- on qubit 2
        ]
    
    def get_gamma(self, t: float) -> float:
        """Get dissipation rate at time t.
        
        Args:
            t: Time point
            
        Returns:
            Dissipation rate gamma(t)
        """
        if self.gamma_const is not None:
            return self.gamma_const
        elif self.gamma_coeffs is not None:
            return bernstein_gamma(t, self.gamma_coeffs, self.gamma_t_max)
        else:
            return 0.0
    
    def get_lindblad_operators(self, t: float) -> List[Tuple[qt.Qobj, float]]:
        """Get Lindblad operators with time-dependent rates.
        
        Args:
            t: Time point
            
        Returns:
            List of (operator, rate) tuples
        """
        gamma = self.get_gamma(t)
        return [(L, gamma) for L in self.L_list]


def sample_random_pure_state(n_qubits: int = 1, seed: Optional[int] = None) -> qt.Qobj:
    """Sample a random pure state from Haar measure.
    
    Args:
        n_qubits: Number of qubits
        seed: Random seed
        
    Returns:
        Random pure state density matrix
    """
    if seed is not None:
        np.random.seed(seed)
    
    dim = 2 ** n_qubits
    
    # Generate random complex vector
    vec = np.random.randn(dim) + 1j * np.random.randn(dim)
    vec = vec / np.linalg.norm(vec)
    
    # Create density matrix: |psi><psi|
    # For single qubit: vec is (2,), need ket vector (2,1)
    if n_qubits == 1:
        vec_ket = vec.reshape(-1, 1)  # (2, 1)
        vec_bra = vec.reshape(1, -1).conj()  # (1, 2)
        rho = vec_ket @ vec_bra  # (2, 2)
        return qt.Qobj(rho, dims=[[2], [2]])
    else:
        vec_ket = vec.reshape(-1, 1)
        vec_bra = vec.reshape(1, -1).conj()
        rho = vec_ket @ vec_bra
        return qt.Qobj(rho, dims=[[2] * n_qubits, [2] * n_qubits])
