"""Lindblad systems: Hamiltonian, jump operators and the rate gamma(t)."""
from __future__ import annotations

import numpy as np
import qutip as qt
from scipy.special import comb


def bernstein_polynomial(degree: int, k: int, t: float) -> float:
    """Evaluates the Bernstein basis polynomial B_{k,n}(t).

    Args:
        degree: Degree n of the polynomial.
        k: Index k, 0 <= k <= n.
        t: Evaluation point in [0, 1].

    Returns:
        The value of the polynomial; 0 if ``k`` is out of range.
    """
    if k < 0 or k > degree:
        return 0.0
    return comb(degree, k) * (t ** k) * ((1 - t) ** (degree - k))


def bernstein_gamma(t: float, coeffs: np.ndarray, t_max: float) -> float:
    """Evaluates the time-dependent rate gamma(t) in the Bernstein basis.

    Args:
        t: Time point.
        coeffs: Bernstein coefficients.
        t_max: Time that is mapped to 1 in the polynomial argument.

    Returns:
        The rate at ``t``, clipped at 0 so that it is never negative.
    """
    t_norm = np.clip(t / t_max, 0.0, 1.0)
    degree = len(coeffs) - 1
    gamma_val = sum(
        coeffs[k] * bernstein_polynomial(degree, k, t_norm)
        for k in range(len(coeffs))
    )
    return max(0.0, gamma_val)


class _LindbladSystem:
    """Base class: a Hamiltonian ``H``, jump operators and one shared rate.

    Subclasses set ``H`` and ``L_list`` in their constructor and call
    ``_init_rate``. Every jump operator decays with the same rate gamma(t),
    either a constant or a Bernstein polynomial.

    Attributes:
        H: Hamiltonian.
        L_list: Jump operators.
        gamma_const: Constant rate, or None if the rate is time dependent.
        gamma_coeffs: Bernstein coefficients of gamma(t), or None.
        gamma_t_max: Time that normalises gamma(t) in the Bernstein basis.
    """

    def _init_rate(
        self,
        gamma_const: float | None,
        gamma_coeffs: np.ndarray | None,
        gamma_t_max: float,
    ) -> None:
        self.gamma_const = gamma_const
        self.gamma_coeffs = gamma_coeffs
        self.gamma_t_max = gamma_t_max

    def get_gamma(self, t: float) -> float:
        """Returns the dissipation rate gamma(t).

        Args:
            t: Time point.

        Returns:
            ``gamma_const`` if set, else the Bernstein polynomial of
            ``gamma_coeffs``, else 0.
        """
        if self.gamma_const is not None:
            return self.gamma_const
        if self.gamma_coeffs is not None:
            return bernstein_gamma(t, self.gamma_coeffs, self.gamma_t_max)
        return 0.0

    def get_lindblad_operators(self, t: float) -> list[tuple[qt.Qobj, float]]:
        """Returns the jump operators with their rate at time ``t``.

        Args:
            t: Time point.

        Returns:
            List of (operator, rate) tuples.
        """
        gamma = self.get_gamma(t)
        return [(L, gamma) for L in self.L_list]


class SingleQubitSystem(_LindbladSystem):
    """Single qubit, H = omega * sigma_z, with Lindblad dissipation.

    Attributes:
        omega: Frequency in H.
        jump_operators: Names of the jump operators, 'sigma_minus' or
            'sigma_z'.
    """

    def __init__(
        self,
        omega: float = 1.0,
        gamma_const: float | None = None,
        gamma_coeffs: np.ndarray | None = None,
        jump_operators: list[str] | None = None,
        gamma_t_max: float = 10.0,
    ):
        """Initialises the system.

        Args:
            omega: Frequency in H = omega * sigma_z.
            gamma_const: Constant rate; if None, gamma_coeffs is used.
            gamma_coeffs: Bernstein coefficients of gamma(t).
            jump_operators: Jump operator names, 'sigma_minus' (default)
                or 'sigma_z'.
            gamma_t_max: Time that normalises gamma(t).

        Raises:
            ValueError: If a jump operator name is unknown.
        """
        self._init_rate(gamma_const, gamma_coeffs, gamma_t_max)
        self.omega = omega
        self.jump_operators = jump_operators
        self.H = omega * qt.sigmaz()

        if self.jump_operators is None:
            self.jump_operators = ["sigma_minus"]

        self.L_list = []
        for op_type in self.jump_operators:
            if op_type == "sigma_minus":
                self.L_list.append(qt.sigmam())
            elif op_type == "sigma_z":
                self.L_list.append(qt.sigmaz())
            else:
                raise ValueError(f"Unknown jump operator: {op_type}")


def _two_qubit_jump_operator(op_type: str, qubit: int) -> qt.Qobj:
    """Builds a single-qubit jump operator acting on one of two qubits.

    Args:
        op_type: 'sigma_minus' or 'sigma_z'.
        qubit: 0 for the first qubit, 1 for the second.

    Returns:
        The 4x4 operator.

    Raises:
        ValueError: If ``op_type`` is unknown.
    """
    eye = qt.qeye(2)
    if op_type == "sigma_minus":
        local_op = qt.destroy(2)
    elif op_type == "sigma_z":
        local_op = qt.sigmaz()
    else:
        raise ValueError(f"Unknown jump operator type: {op_type}")
    if qubit == 0:
        return qt.tensor(local_op, eye)
    return qt.tensor(eye, local_op)


class TwoQubitSystem(_LindbladSystem):
    """Two qubits with an exchange or Ising coupling and dissipation.

    Attributes:
        J: Coupling strength.
        interaction_type: 'XXZ' or 'TFIM'.
    """

    def __init__(
        self,
        J: float = 0.5,
        interaction_type: str = "XXZ",
        gamma_const: float | None = None,
        gamma_coeffs: np.ndarray | None = None,
        gamma_t_max: float = 10.0,
        jump_operators: list[str] | None = None,
    ):
        """Initialises the system.

        Args:
            J: Coupling strength.
            interaction_type: 'XXZ' for J (XX + YY + ZZ), 'TFIM' for
                J XX + (Z1 + Z2) with unit transverse field.
            gamma_const: Constant rate per jump operator; if None,
                gamma_coeffs is used.
            gamma_coeffs: Bernstein coefficients of gamma(t).
            gamma_t_max: Time that normalises gamma(t).
            jump_operators: Jump operator names applied to each qubit,
                'sigma_minus' (default) or 'sigma_z'.

        Raises:
            ValueError: If the interaction type or a jump operator name is
                unknown.
        """
        self._init_rate(gamma_const, gamma_coeffs, gamma_t_max)
        self.J = J
        self.interaction_type = interaction_type

        sx1 = qt.tensor(qt.sigmax(), qt.qeye(2))
        sy1 = qt.tensor(qt.sigmay(), qt.qeye(2))
        sz1 = qt.tensor(qt.sigmaz(), qt.qeye(2))
        sx2 = qt.tensor(qt.qeye(2), qt.sigmax())
        sy2 = qt.tensor(qt.qeye(2), qt.sigmay())
        sz2 = qt.tensor(qt.qeye(2), qt.sigmaz())

        if interaction_type == "XXZ":
            # Isotropic: the anisotropy delta is fixed to 1.
            self.H = J * (sx1 * sx2 + sy1 * sy2 + sz1 * sz2)
        elif interaction_type == "TFIM":
            h_field = 1.0
            self.H = J * sx1 * sx2 + h_field * (sz1 + sz2)
        else:
            raise ValueError(f"Unknown interaction type: {interaction_type}")

        if jump_operators is None:
            jump_operators = ["sigma_minus"]

        self.L_list = []
        for op_type in jump_operators:
            self.L_list.append(_two_qubit_jump_operator(op_type, 0))
            self.L_list.append(_two_qubit_jump_operator(op_type, 1))


def sample_random_pure_state(
    n_qubits: int = 1, seed: int | None = None
) -> qt.Qobj:
    """Samples a Haar-random pure state as a density matrix.

    With ``seed=None`` the legacy global NumPy generator is used, so
    ``np.random.seed`` makes the draw reproducible (the simulator relies
    on this).

    Args:
        n_qubits: Number of qubits.
        seed: Seed of a private generator, or None for the global one.

    Returns:
        The density matrix |psi><psi| with dims [[2]*n, [2]*n].
    """
    dim = 2 ** n_qubits
    if seed is not None:
        rng = np.random.default_rng(seed)
        vec = rng.standard_normal(dim) + 1j * rng.standard_normal(dim)
    else:
        vec = np.random.randn(dim) + 1j * np.random.randn(dim)

    vec = vec / np.linalg.norm(vec)

    vec_ket = vec.reshape(-1, 1)
    vec_bra = vec.reshape(1, -1).conj()
    rho = vec_ket @ vec_bra
    return qt.Qobj(rho, dims=[[2] * n_qubits, [2] * n_qubits])
