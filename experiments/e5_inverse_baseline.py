"""Experiment E5 — Direct prediction vs. inverse-problem baseline.

Goal
----
Show that at short observation windows a direct T₂ predictor beats the
two-step inverse route (reconstruct γ → integrate Lindblad).

Claim
-----
C3: at f ≤ 0.20, direct physics+LSTM T₂ prediction outperforms inverse
    (fit γ coefficients with a Transformer, then forward-integrate).

Scenarios
---------
B (1q σ₋ time-dep γ) — the Paper 1 regime where γ(t) is unknown.

Metrics
-------
R², MAE, RMSE, AUROC vs window fraction. Success: direct R² ≫ inverse R²
at the same f (Paper 1: 0.92 vs −0.86).

Output
------
experiments/results/e5_inverse_baseline[_<tag>][_fast].csv

Usage
-----
    python -m experiments.e5_inverse_baseline --fast
    python -m experiments.e5_inverse_baseline
"""
from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import torch
from torch import nn
from torch.utils import data as torch_data

from experiments import common
from src.application import backtest
from src.application import generate_dataset
from src.application import train_model
from src.data import scenarios as scenario_lib
from src.domain import value_objects
from src.evaluation import metrics
from src.models import registry
from src.simulation import simulator as simulator_lib

COLUMNS = ("method", "window_fraction", "mae", "rmse", "r2", "auroc")
FRACTIONS = (0.10, 0.15, 0.20, 0.30, 0.50)
N_COEFFS = 5


class GammaReconstructorPredictor:
    """Two-step baseline: reconstruct γ coefficients, then integrate.

    Step 1: a small MLP is trained (supervised) to map a window to
    Bernstein coefficients. The targets are the true γ coefficients stored
    in the SystemConfig of each trajectory.

    Step 2: given the reconstructed coefficients, the Lindblad equation is
    forward-simulated with QuTiP and T₂ is read off.

    This is the fair analogue of arXiv 2505.06928 applied to the T₂ task.
    Its weakness at short windows is that the reconstruction is
    ill-conditioned: many γ(t) curves fit a short partial observation,
    which gives a high T₂ prediction variance.
    """

    def __init__(self, simulator: simulator_lib.QuTipSimulator) -> None:
        """Creates an untrained reconstructor.

        Args:
            simulator: Forward-simulates the reconstructed systems.
        """
        self._simulator = simulator
        self._mlp = None
        self._mean = None
        self._std = None

    def train_gamma_reconstruction(
        self,
        sequences: np.ndarray,
        gamma_targets: np.ndarray,
        n_epochs: int = 100,
        lr: float = 1e-3,
        verbose: bool = True,
    ) -> None:
        """Trains the MLP window -> γ Bernstein coefficients.

        Args:
            sequences: Windows, shape (N, T, F).
            gamma_targets: True coefficients, shape (N, n_coeffs).
            n_epochs: Training epochs.
            lr: AdamW learning rate.
            verbose: Whether to print the loss every 20 epochs.
        """
        n_windows = sequences.shape[0]
        flat = sequences.reshape(n_windows, -1).astype(np.float32)
        self._mean = flat.mean(axis=0)
        self._std = flat.std(axis=0) + 1e-8
        normed = (flat - self._mean) / self._std

        self._mlp = nn.Sequential(
            nn.Linear(normed.shape[1], 256), nn.GELU(), nn.Dropout(0.1),
            nn.Linear(256, 128), nn.GELU(), nn.Dropout(0.1),
            # Softplus keeps the coefficients non-negative.
            nn.Linear(128, gamma_targets.shape[1]), nn.Softplus(),
        )
        optimizer = torch.optim.AdamW(
            self._mlp.parameters(), lr=lr, weight_decay=1e-3)
        loader = torch_data.DataLoader(
            torch_data.TensorDataset(
                torch.tensor(normed, dtype=torch.float32),
                torch.tensor(gamma_targets, dtype=torch.float32)),
            batch_size=64, shuffle=True)
        criterion = nn.HuberLoss(delta=0.5)

        for epoch in range(1, n_epochs + 1):
            self._mlp.train()
            total = 0.0
            for batch_x, batch_y in loader:
                loss = criterion(self._mlp(batch_x), batch_y)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total += loss.item()
            if verbose and epoch % 20 == 0:
                print(f"  [GammaReconstructor] Epoch {epoch}/{n_epochs}  "
                      f"loss={total / len(loader):.4f}")

    def predict_t2(
        self,
        window: np.ndarray,
        system_config: value_objects.SystemConfig,
    ) -> float:
        """Reconstructs γ(t), forward-simulates and returns T₂.

        Args:
            window: One observation window, shape (T, F).
            system_config: The system whose dissipator is replaced by the
                reconstructed one.

        Returns:
            The decoherence time of the simulated system.

        Raises:
            RuntimeError: If the MLP has not been trained.
        """
        if self._mlp is None:
            raise RuntimeError("Not trained.")
        normed = (window.flatten()[np.newaxis] - self._mean) / self._std
        self._mlp.eval()
        with torch.no_grad():
            coeffs = self._mlp(
                torch.tensor(normed, dtype=torch.float32)).numpy()[0]

        dissipator = value_objects.DissipatorConfig.time_dependent(
            tuple(float(c) for c in coeffs),
            system_config.dissipator.operator_type,
        )
        new_config = value_objects.SystemConfig(
            n_qubits=system_config.n_qubits,
            omega=system_config.omega,
            dissipator=dissipator,
            interaction_type=system_config.interaction_type,
            J=system_config.J,
            t_max=system_config.t_max,
            dt=system_config.dt,
            decoherence_criterion=system_config.decoherence_criterion,
            decoherence_threshold=system_config.decoherence_threshold,
        )
        return self._simulator.simulate(new_config).t_decoh


def _extract_gamma_targets(
    trajectories: Sequence[value_objects.TrajectoryResult],
    n_coeffs: int = N_COEFFS,
) -> np.ndarray:
    """Reads the Bernstein coefficients from the trajectories' configs.

    A constant γ is replicated to `n_coeffs` values; a shorter coefficient
    list is padded with its last value.
    """
    targets = []
    for trajectory in trajectories:
        dissipator = trajectory.system_config.dissipator
        if dissipator.gamma_const is not None:
            targets.append([dissipator.gamma_const] * n_coeffs)
        elif dissipator.gamma_coeffs is not None:
            coeffs = list(dissipator.gamma_coeffs)
            if len(coeffs) >= n_coeffs:
                targets.append(coeffs[:n_coeffs])
            else:
                targets.append(
                    coeffs + [coeffs[-1]] * (n_coeffs - len(coeffs)))
        else:
            targets.append([0.0] * n_coeffs)
    return np.array(targets, dtype=np.float32)


def _direct_row(dataset, fraction: float, seed: int, n_epochs: int) -> dict:
    """Trains physics + LSTM on the windows and scores it."""
    direct = registry.build_model("physics_lstm")
    train_model.TrainModelUseCase(direct).execute(
        train_model.TrainModelCommand(
            dataset=dataset, n_epochs=n_epochs, batch_size=32,
            verbose=True, seed=seed))
    result = backtest.BacktestUseCase(direct).execute(
        backtest.BacktestCommand(dataset=dataset, horizon=1.0))
    print(f"  Direct: {result.summary()}")
    return {
        "method": "direct_physics_lstm",
        "window_fraction": fraction,
        "mae": result.mae,
        "rmse": result.rmse,
        "r2": result.r2,
        "auroc": result.risk_auroc,
    }


def _inverse_row(simulator, configs, dataset, fraction: float,
                 median_t2: float, n_epochs: int) -> dict:
    """Reconstructs γ, forward-simulates and scores the T₂ estimates."""
    print("  Training γ reconstructor...")
    trajectories = [simulator.simulate(cfg) for cfg in configs]
    gamma_targets = _extract_gamma_targets(trajectories)

    test_seqs = dataset.sequences[dataset.test_idx]
    test_t_decoh = dataset.t_decoh_abs[dataset.test_idx]
    train_gamma = np.array([
        gamma_targets[i % len(gamma_targets)] for i in dataset.train_idx])
    reconstructor = GammaReconstructorPredictor(simulator)
    reconstructor.train_gamma_reconstruction(
        dataset.sequences[dataset.train_idx], train_gamma,
        n_epochs=n_epochs, verbose=True)

    predictions = []
    for i, sequence in enumerate(test_seqs):
        try:
            t2_pred = reconstructor.predict_t2(
                sequence, configs[i % len(configs)])
        except Exception:  # pylint: disable=broad-except
            # A reconstructed γ can make the solver fail; score such
            # windows as the pilot median instead of dropping them.
            t2_pred = float(median_t2)
        predictions.append(t2_pred)

    predictions = np.array(predictions)
    actuals = test_t_decoh.astype(np.float64)
    row = {
        "method": "inverse_gamma_reconstruct",
        "window_fraction": fraction,
        "mae": metrics.mae(actuals, predictions),
        "rmse": metrics.rmse(actuals, predictions),
        "r2": metrics.r2(actuals, predictions),
        "auroc": float("nan"),
    }
    print(f"  Inverse: MAE={row['mae']:.4f}  RMSE={row['rmse']:.4f}  "
          f"R²={row['r2']:.4f}")
    return row


def run(
    *,
    seed: int = 42,
    n_per: int = 80,
    n_epochs: int = 100,
    recon_epochs: int = 80,
) -> list[dict[str, object]]:
    """Compares the direct and the inverse predictor on scenario B.

    Args:
        seed: Seed of the configurations, dataset and training.
        n_per: Trajectories of scenario B.
        n_epochs: Training epochs of the direct predictor.
        recon_epochs: Training epochs of the γ reconstructor.

    Returns:
        Two rows (direct, inverse) per window fraction, with the keys of
        `COLUMNS`.
    """
    simulator = simulator_lib.QuTipSimulator()
    # Time-dependent γ only: the hardest case for the inverse approach.
    configs = scenario_lib.build_configs(
        n_per_scenario=n_per, seed=seed)["B"]

    pilot_t2 = [simulator.simulate(cfg).t_decoh for cfg in configs[:20]]
    median_t2 = float(np.median(pilot_t2))
    dt = float(configs[0].dt)
    print(f"Median T₂ (pilot): {median_t2:.2f},  dt={dt}")

    generator = generate_dataset.GenerateDatasetUseCase(
        simulator, store=scenario_lib.trajectory_store())
    rows = []
    for fraction in FRACTIONS:
        window_length = max(5, int(round(fraction * median_t2 / dt)))
        print(f"\n{'=' * 60}\nWindow fraction f={fraction:.2f}  "
              f"window_length={window_length}\n{'=' * 60}")
        dataset = generator.execute(generate_dataset.GenerateDatasetCommand(
            configs=configs, window_length=window_length, horizon=1.0,
            samples_per_trajectory=5, seed=seed))
        rows.append(_direct_row(dataset, fraction, seed, n_epochs))
        rows.append(_inverse_row(
            simulator, configs, dataset, fraction, median_t2, recon_epochs))
    return rows


def main(argv: Sequence[str] | None = None) -> None:
    """Runs E5 and writes its CSV."""
    args = common.parse_args(__doc__, argv=argv, seeds=None)
    sizes = (
        {"n_per": 20, "n_epochs": 10, "recon_epochs": 10} if args.fast
        else {})
    rows = run(**sizes)
    path = common.results_path("e5_inverse_baseline", args)
    common.write_results(
        rows, path, columns=COLUMNS, seeds=[42],
        config={"fractions": FRACTIONS, **sizes})
    print(f"\nResults saved to {path}")
    print(f"\n{'Method':<28} {'f':>5} {'R²':>8} {'MAE':>8}")
    print("-" * 54)
    for row in rows:
        print(f"{row['method']:<28} {row['window_fraction']:>5.2f} "
              f"{row['r2']:>8.4f} {row['mae']:>8.4f}")


if __name__ == "__main__":
    main()
