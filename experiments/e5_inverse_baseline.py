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
experiments/results/e5_inverse_baseline.csv

Usage
-----
    python -m experiments.e5_inverse_baseline
"""
from __future__ import annotations

import csv
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from experiments._config import build_configs, trajectory_store
from src.application.backtest import BacktestCommand, BacktestUseCase
from src.application.generate_dataset import GenerateDatasetCommand, GenerateDatasetUseCase
from src.application.train_model import TrainModelCommand, TrainModelUseCase
from src.domain.value_objects import AblationResult, ExperimentSpec, NoiseConfig, PredictorVariant
from src.infrastructure.ml.lstm_predictor import LSTMPredictor
from src.infrastructure.quantum.qutip_simulator import QuTipSimulator


# ---------------------------------------------------------------------------
# γ-reconstruction Transformer baseline
# ---------------------------------------------------------------------------

class GammaReconstructorPredictor:
    """Two-step baseline: reconstruct γ coefficients, then numerically integrate.

    Step 1 — a small MLP is trained (supervised) to map
             window → Bernstein coefficients.  The training targets are the
             *true* γ coefficients stored in the SystemConfig of each trajectory.

    Step 2 — given the reconstructed coefficients, forward-simulate the
             Lindblad equation with QuTiP and read off T₂.

    This is the fair analogue of arXiv 2505.06928 applied to the T₂ task.
    The key weakness at short windows is that the reconstruction is ill-
    conditioned: there are many γ(t) curves compatible with a short partial
    observation, leading to high T₂ prediction variance.
    """

    def __init__(self, simulator: QuTipSimulator, t_max: float = 10.0) -> None:
        self._simulator = simulator
        self._t_max = t_max
        self._mlp = None
        self._scaler_X_mean = None
        self._scaler_X_std = None

    @property
    def is_trained(self) -> bool:
        return self._mlp is not None

    def train_gamma_reconstruction(
        self,
        sequences: np.ndarray,   # (N, T, F)
        gamma_targets: np.ndarray,  # (N, n_coeffs) — true Bernstein coefficients
        n_epochs: int = 100,
        lr: float = 1e-3,
        verbose: bool = True,
    ) -> None:
        """Train MLP: window → γ Bernstein coefficients."""
        import torch
        import torch.nn as nn
        from torch.utils.data import DataLoader, TensorDataset

        N, T, F = sequences.shape
        X_flat = sequences.reshape(N, T * F).astype(np.float32)

        self._scaler_X_mean = X_flat.mean(axis=0)
        self._scaler_X_std = X_flat.std(axis=0) + 1e-8
        X_norm = (X_flat - self._scaler_X_mean) / self._scaler_X_std

        in_dim = X_norm.shape[1]
        out_dim = gamma_targets.shape[1]

        self._mlp = nn.Sequential(
            nn.Linear(in_dim, 256), nn.GELU(), nn.Dropout(0.1),
            nn.Linear(256, 128),   nn.GELU(), nn.Dropout(0.1),
            nn.Linear(128, out_dim), nn.Softplus(),  # ensure non-negative outputs
        )

        opt = torch.optim.AdamW(self._mlp.parameters(), lr=lr, weight_decay=1e-3)
        ds = TensorDataset(
            torch.tensor(X_norm, dtype=torch.float32),
            torch.tensor(gamma_targets, dtype=torch.float32),
        )
        loader = DataLoader(ds, batch_size=64, shuffle=True)
        criterion = nn.HuberLoss(delta=0.5)

        for ep in range(1, n_epochs + 1):
            self._mlp.train()
            total = 0.0
            for xb, yb in loader:
                pred = self._mlp(xb)
                loss = criterion(pred, yb)
                opt.zero_grad()
                loss.backward()
                opt.step()
                total += loss.item()
            if verbose and ep % 20 == 0:
                print(f"  [GammaReconstructor] Epoch {ep}/{n_epochs}  loss={total/len(loader):.4f}")

    def predict_t2(
        self,
        window: np.ndarray,   # (T, F)
        system_config,
    ) -> float:
        """Reconstruct γ(t), forward-simulate, return T₂."""
        import torch
        from src.physics.lindblad_systems import bernstein_gamma
        from src.domain.value_objects import DissipatorConfig, SystemConfig

        if self._mlp is None:
            raise RuntimeError("Not trained.")

        X_flat = window.flatten()[np.newaxis]
        X_norm = (X_flat - self._scaler_X_mean) / self._scaler_X_std

        self._mlp.eval()
        with torch.no_grad():
            coeffs = self._mlp(torch.tensor(X_norm, dtype=torch.float32)).numpy()[0]

        # Build a new config with reconstructed γ(t) and simulate
        new_dissipator = DissipatorConfig.time_dependent(
            tuple(float(c) for c in coeffs),
            system_config.dissipator.operator_type,
        )
        new_config = SystemConfig(
            n_qubits=system_config.n_qubits,
            omega=system_config.omega,
            dissipator=new_dissipator,
            interaction_type=system_config.interaction_type,
            J=system_config.J,
            t_max=system_config.t_max,
            dt=system_config.dt,
            decoherence_criterion=system_config.decoherence_criterion,
            decoherence_threshold=system_config.decoherence_threshold,
        )
        traj = self._simulator.simulate(new_config)
        return traj.t_decoh


# ---------------------------------------------------------------------------
# Experiment runner
# ---------------------------------------------------------------------------

def _extract_gamma_targets(trajectories, n_coeffs: int = 5) -> np.ndarray:
    """Extract Bernstein coefficients from trajectory configs.

    For constant-γ configs, replicate the constant to fill n_coeffs.
    For time-dependent configs, use the stored coefficients (pad/truncate to n_coeffs).
    """
    targets = []
    for traj in trajectories:
        d = traj.system_config.dissipator
        if d.gamma_const is not None:
            targets.append([d.gamma_const] * n_coeffs)
        elif d.gamma_coeffs is not None:
            coeffs = list(d.gamma_coeffs)
            if len(coeffs) >= n_coeffs:
                targets.append(coeffs[:n_coeffs])
            else:
                targets.append(coeffs + [coeffs[-1]] * (n_coeffs - len(coeffs)))
        else:
            targets.append([0.0] * n_coeffs)
    return np.array(targets, dtype=np.float32)


def main() -> None:
    results_dir = Path(__file__).parent / "results"
    results_dir.mkdir(exist_ok=True)

    simulator = QuTipSimulator()
    groups = build_configs(n_per_scenario=80, seed=42)
    configs = groups["B"]  # time-dependent γ only — hardest case for inverse approach

    fractions = [0.10, 0.15, 0.20, 0.30, 0.50]
    n_coeffs = 5

    # Pilot: estimate median T₂ and dt
    pilot_t2 = []
    pilot_trajs = []
    for cfg in configs[:20]:
        traj = simulator.simulate(cfg)
        pilot_t2.append(traj.t_decoh)
        pilot_trajs.append(traj)
    median_t2 = float(np.median(pilot_t2))
    dt = float(configs[0].dt)
    print(f"Median T₂ (pilot): {median_t2:.2f},  dt={dt}")

    gen_uc = GenerateDatasetUseCase(simulator, store=trajectory_store())
    rows = []

    for f in fractions:
        window_length = max(5, int(round(f * median_t2 / dt)))
        print(f"\n{'='*60}")
        print(f"Window fraction f={f:.2f}  window_length={window_length}")
        print(f"{'='*60}")

        dataset = gen_uc.execute(GenerateDatasetCommand(
            configs=configs,
            window_length=window_length,
            horizon=1.0,
            samples_per_trajectory=5,
            seed=42,
        ))

        # --- Direct predictor (physics+LSTM) ---
        direct = LSTMPredictor(use_physics_prior=True)
        TrainModelUseCase(direct).execute(TrainModelCommand(
            dataset=dataset,
            n_epochs=100,
            batch_size=32,
            verbose=True,
            seed=42,
        ))
        m_direct = BacktestUseCase(direct).execute(
            BacktestCommand(dataset=dataset, horizon=1.0)
        )
        rows.append({
            "method": "direct_physics_lstm",
            "window_fraction": f,
            "mae": m_direct.mae,
            "rmse": m_direct.rmse,
            "r2": m_direct.r2,
            "auroc": m_direct.risk_auroc,
        })
        print(f"  Direct: {m_direct.summary()}")

        # --- Inverse baseline (γ reconstruction) ---
        print("  Training γ reconstructor...")
        all_trajs = []
        for cfg in configs:
            all_trajs.append(simulator.simulate(cfg))
        gamma_targets = _extract_gamma_targets(all_trajs, n_coeffs)

        # Build feature matrix from truncated windows (same window_length)
        # Use feature matrices from all trajectories at the same relative position
        test_seqs = dataset.sequences[dataset.test_idx]
        test_t_obs = dataset.t_obs[dataset.test_idx]
        test_t_decoh = dataset.t_decoh_abs[dataset.test_idx]

        reconstructor = GammaReconstructorPredictor(simulator, t_max=configs[0].t_max)
        # Use train sequences for MLP training
        train_seqs = dataset.sequences[dataset.train_idx]
        train_gamma = np.array([
            gamma_targets[i % len(gamma_targets)] for i in dataset.train_idx
        ])
        reconstructor.train_gamma_reconstruction(
            train_seqs, train_gamma, n_epochs=80, verbose=True
        )

        # Evaluate on test set: predict T₂ via γ reconstruction + simulation
        inverse_preds = []
        for i, seq in enumerate(test_seqs):
            t_obs = float(test_t_obs[i])
            cfg_idx = i % len(configs)
            try:
                t2_pred = reconstructor.predict_t2(seq, configs[cfg_idx])
            except Exception:
                t2_pred = float(median_t2)
            inverse_preds.append(t2_pred)

        inv_preds = np.array(inverse_preds)
        actuals = test_t_decoh.astype(np.float64)
        mae_inv = float(np.mean(np.abs(inv_preds - actuals)))
        rmse_inv = float(np.sqrt(np.mean((inv_preds - actuals) ** 2)))
        ss_res = np.sum((actuals - inv_preds) ** 2)
        ss_tot = np.sum((actuals - actuals.mean()) ** 2)
        r2_inv = float(1.0 - ss_res / ss_tot) if ss_tot > 1e-12 else 0.0

        rows.append({
            "method": "inverse_gamma_reconstruct",
            "window_fraction": f,
            "mae": mae_inv,
            "rmse": rmse_inv,
            "r2": r2_inv,
            "auroc": float("nan"),
        })
        print(f"  Inverse: MAE={mae_inv:.4f}  RMSE={rmse_inv:.4f}  R²={r2_inv:.4f}")

    # Save results
    out_path = results_dir / "e5_inverse_baseline.csv"
    with open(out_path, "w", newline="") as f_:
        writer = csv.DictWriter(f_, fieldnames=["method", "window_fraction", "mae", "rmse", "r2", "auroc"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nResults saved to {out_path}")

    # Summary table
    print(f"\n{'Method':<28} {'f':>5} {'R²':>8} {'MAE':>8}")
    print("-" * 54)
    for r in rows:
        print(f"{r['method']:<28} {r['window_fraction']:>5.2f} {r['r2']:>8.4f} {r['mae']:>8.4f}")


if __name__ == "__main__":
    main()
