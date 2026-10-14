"""Domain value objects: immutable, self-validating descriptors."""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum

import numpy as np

# Standard 1/e decoherence threshold: coherence decays to C(0)/e, defining T₂.
ONE_OVER_E: float = 1.0 / math.e  # ≈ 0.3679


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------

class QubitCount(int, Enum):
    """Number of qubits of a system."""

    ONE = 1
    TWO = 2


class DissipatorType(str, Enum):
    """Jump operator of the dissipator."""

    SIGMA_MINUS = "sigma_minus"
    SIGMA_Z = "sigma_z"


class InteractionType(str, Enum):
    """Two-qubit interaction Hamiltonian."""

    XXZ = "XXZ"
    TFIM = "TFIM"


class DecoherenceCriterion(str, Enum):
    """Quantity whose decay defines the decoherence time T2."""

    COHERENCE = "coherence"
    PURITY = "purity"
    # l1-coherence in the eigenbasis of H (two qubits): insensitive to the
    # unitary modulation that the computational-basis measure picks up when
    # H is not diagonal there (see experiments/r9_target_modulation.py).
    COHERENCE_ENERGY = "coherence_energy"
    # Wootters concurrence (two qubits): an entanglement measure, invariant
    # under local unitaries (see experiments/r24_entanglement_target.py).
    CONCURRENCE = "concurrence"
    # Entanglement sudden death: first time the concurrence reaches zero.
    ENTANGLEMENT_DEATH = "entanglement_death"


# ---------------------------------------------------------------------------
# Value objects
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class DissipatorConfig:
    """Configuration for the dissipative (Lindblad) term.

    Either ``gamma_const`` (constant rate) or ``gamma_coeffs`` (Bernstein
    polynomial coefficients for a time-dependent rate) must be supplied,
    but not both.
    """

    operator_type: DissipatorType = DissipatorType.SIGMA_MINUS
    gamma_const: float | None = None
    gamma_coeffs: tuple[float, ...] | None = None  # frozen → tuple, not list

    def __post_init__(self) -> None:
        if self.gamma_const is None and self.gamma_coeffs is None:
            raise ValueError(
                "DissipatorConfig requires either gamma_const or gamma_coeffs."
            )
        if self.gamma_const is not None and self.gamma_const < 0:
            raise ValueError("gamma_const must be non-negative.")
        if self.gamma_coeffs is not None and any(
                c < 0 for c in self.gamma_coeffs):
            raise ValueError(
                "All Bernstein coefficients must be non-negative.")

    # convenience constructors ------------------------------------------------

    @classmethod
    def constant(
        cls,
        gamma: float,
        operator_type: DissipatorType = DissipatorType.SIGMA_MINUS,
    ) -> "DissipatorConfig":
        """Builds a dissipator with a constant rate.

        Args:
            gamma: The rate, non-negative.
            operator_type: The jump operator.

        Returns:
            The configuration.
        """
        return cls(operator_type=operator_type, gamma_const=gamma)

    @classmethod
    def time_dependent(
        cls,
        coeffs: tuple[float, ...],
        operator_type: DissipatorType = DissipatorType.SIGMA_MINUS,
    ) -> "DissipatorConfig":
        """Builds a dissipator with a Bernstein-polynomial rate.

        Args:
            coeffs: Non-negative Bernstein coefficients of gamma(t).
            operator_type: The jump operator.

        Returns:
            The configuration.
        """
        return cls(operator_type=operator_type, gamma_coeffs=coeffs)


@dataclass(frozen=True)
class SystemConfig:
    """Complete specification of a quantum system to simulate.

    Single-qubit: H = ω·σ_z, one dissipator.
    Two-qubit:    H = J·(interaction), one dissipator per qubit (same γ).
    """

    n_qubits: QubitCount
    omega: float                      # qubit transition frequency
    dissipator: DissipatorConfig
    interaction_type: InteractionType = InteractionType.XXZ  # two-qubit only
    J: float = 0.5                    # two-qubit coupling strength
    t_max: float = 10.0
    dt: float = 0.1
    decoherence_criterion: DecoherenceCriterion = DecoherenceCriterion.COHERENCE
    # 1/e threshold: t_decoh is defined as the time when coherence (or purity)
    # drops to 1/e of its initial value — the standard T₂ definition.
    decoherence_threshold: float = ONE_OVER_E

    def __post_init__(self) -> None:
        if self.omega <= 0:
            raise ValueError("omega must be positive.")
        if self.t_max <= 0:
            raise ValueError("t_max must be positive.")
        if not (0 < self.dt < self.t_max):
            raise ValueError("dt must satisfy 0 < dt < t_max.")
        if self.J <= 0:
            raise ValueError("J must be positive.")


@dataclass
class TrajectoryResult:
    """Output of a single quantum trajectory simulation.

    ``observables`` maps observable name → 1-D array of length
    ``len(times)``.  The ``density_matrices`` key (if present) is excluded
    from ``feature_matrix`` because it contains raw complex arrays.
    """

    times: np.ndarray
    observables: dict[str, np.ndarray]
    t_decoh: float
    system_config: SystemConfig
    # True: 1/e threshold was not crossed by t_max; t_decoh is a lower bound.
    censored: bool = False

    # ------------------------------------------------------------------
    # Derived views
    # ------------------------------------------------------------------

    @property
    def feature_names(self) -> list[str]:
        """Observable names used as ML features (excludes density matrices)."""
        return [k for k in self.observables if k != "density_matrices"]

    @property
    def feature_matrix(self) -> np.ndarray:
        """Shape: (n_timesteps, n_features) float32 matrix."""
        return np.stack(
            [self.observables[k] for k in self.feature_names], axis=-1
        ).astype(np.float32)


@dataclass(frozen=True)
class PredictionResult:
    """Inference output for a single observation window."""

    t_decoh_predicted: float       # predicted decoherence time
    risk_score: float              # P(decoherence within `horizon` steps)
    horizon: float                 # time horizon used for risk_score

    def __post_init__(self) -> None:
        if not (0.0 <= self.risk_score <= 1.0):
            raise ValueError("risk_score must be in [0, 1].")


@dataclass(frozen=True)
class NoiseConfig:
    """Measurement noise specification.

    Models Gaussian additive noise on each observable independently.
    ``sigma`` is the standard deviation expressed in the *same units* as
    the observable (Pauli expectations are in [−1, 1]).

    Example::
        NoiseConfig(sigma=0.02)   # 2 % noise — realistic for SC qubits
    """

    sigma: float = 0.0
    seed: int | None = None

    def __post_init__(self) -> None:
        if self.sigma < 0:
            raise ValueError("NoiseConfig.sigma must be non-negative.")


class PredictorVariant(str, Enum):
    """Ablation study: which predictor variant to evaluate."""

    PHYSICS_ONLY = "physics_only"          # OLS-slope formula, no ML
    LSTM_ONLY = "lstm_only"                # BiLSTM without physics prior
    PHYSICS_LSTM = "physics_lstm"          # Residual: physics + LSTM (default)
    # Pure Transformer (2505.06928 baseline).
    TRANSFORMER = "transformer"
    STRETCHED_EXP = "stretched_exp"        # curve_fit baseline, no ML


@dataclass(frozen=True)
class ExperimentSpec:
    """Fully describes one experiment run (used as a hashable cache key).

    Combines dataset generation parameters with predictor variant and noise
    so that results can be stored / retrieved by spec without ambiguity.
    """

    name: str
    predictor_variant: PredictorVariant = PredictorVariant.PHYSICS_LSTM
    window_fraction: float = 0.20          # t_obs / T₂ target
    noise: NoiseConfig = field(default_factory=NoiseConfig)
    n_trajectories: int = 500
    seed: int = 42

    def __post_init__(self) -> None:
        if not (0 < self.window_fraction <= 1.0):
            raise ValueError("window_fraction must be in (0, 1].")
        if self.n_trajectories < 10:
            raise ValueError("n_trajectories must be at least 10.")


@dataclass
class AblationResult:
    """Aggregated metrics for one experiment spec."""

    spec: ExperimentSpec
    mae: float
    rmse: float
    r2: float
    mape: float
    risk_auroc: float
    n_samples: int

    def as_dict(self) -> dict:
        """Returns the result as a flat row of the results CSVs."""
        return {
            "name": self.spec.name,
            "variant": self.spec.predictor_variant.value,
            "window_fraction": self.spec.window_fraction,
            "noise_sigma": self.spec.noise.sigma,
            "mae": self.mae,
            "rmse": self.rmse,
            "r2": self.r2,
            "mape": self.mape,
            "auroc": self.risk_auroc,
            "n_samples": self.n_samples,
        }


@dataclass
class BacktestMetrics:
    """Aggregate evaluation metrics produced by the backtest use-case.

    Attributes:
        mae: Mean absolute error of the predicted decoherence time.
        rmse: Root mean squared error of the same.
        r2: Coefficient of determination (0.0 for constant targets).
        mape: Mean absolute percentage error, nan if no target > 1e-8.
        risk_auroc: AUROC of the risk score, nan if one class only.
        n_samples: Number of scored windows.
        predictions: Predicted decoherence times.
        actuals: True decoherence times.
        extra: Additional metrics requested from `evaluation.backtest`,
            by alias.
    """

    mae: float
    rmse: float
    r2: float
    mape: float
    risk_auroc: float
    n_samples: int
    predictions: np.ndarray
    actuals: np.ndarray
    extra: dict[str, float] = field(default_factory=dict)

    def summary(self) -> str:
        """Returns the metrics as one line of text."""
        return (
            f"BacktestMetrics(n={self.n_samples}  "
            f"MAE={self.mae:.4f}  RMSE={self.rmse:.4f}  "
            f"R²={self.r2:.4f}  MAPE={self.mape:.2f}%  "
            f"AUROC={self.risk_auroc:.4f})"
        )
