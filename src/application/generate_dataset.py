"""Use-case: generate a labelled ML dataset from a list of SystemConfigs."""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

from ..domain.ports import ISimulator, ITrajectoryStore
from ..domain.value_objects import DissipatorType, InteractionType, QubitCount, SystemConfig, TrajectoryResult


# ---------------------------------------------------------------------------
# Dataset DTO
# ---------------------------------------------------------------------------

@dataclass
class Dataset:
    """Labelled dataset ready for training / backtesting.

    Arrays:
        sequences       (N, window_length, n_features)  – input time windows
        t_obs           (N,)   – observation time at the end of each window
        t_decoh_abs     (N,)   – absolute decoherence time T₂ (for evaluation)
        remaining_time  (N,)   – t_decoh_abs - t_obs  (regression target; always > 0)
        risk_labels     (N,)   – binary: 1 if remaining_time ≤ horizon
        train_idx       (k,)   – indices into the first axis of sequences
        val_idx         (m,)
        test_idx        (p,)
        feature_names        – names of observables used as features
        metadata             – provenance dict

    Only pre-decoherence windows are included (t_obs < t_decoh_abs), so
    remaining_time is guaranteed to be strictly positive everywhere.
    """

    sequences: np.ndarray
    t_obs: np.ndarray
    t_decoh_abs: np.ndarray     # absolute T₂ — used for backtest evaluation
    remaining_time: np.ndarray  # regression target: t_decoh_abs - t_obs
    risk_labels: np.ndarray
    train_idx: np.ndarray
    val_idx: np.ndarray
    test_idx: np.ndarray
    feature_names: List[str]
    metadata: Dict
    # Context codes injected as physics scalars (0.0 = default/unknown).
    # interaction_type_codes (4-class system type):
    #   0.0 = 1-qubit σ₋,  1.0 = 1-qubit σ_z,  2.0 = 2-qubit XXZ,  3.0 = 2-qubit TFIM
    # dissipator_type_codes:
    #   0.0 = σ₋ (amplitude damping),  1.0 = σ_z (dephasing)
    interaction_type_codes: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.float32))
    dissipator_type_codes: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.float32))
    # Coupling J per window. 1-qubit → 0.0; 2-qubit → SystemConfig.J.
    J_values: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.float32))
    # True: t_decoh_abs is a lower bound (threshold not crossed by t_max).
    censored: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=bool))

    # keep backward-compat alias
    @property
    def t_decoh(self) -> np.ndarray:
        """Alias for t_decoh_abs (absolute decoherence time)."""
        return self.t_decoh_abs

    # convenience splits
    def train_sequences(self) -> np.ndarray:
        return self.sequences[self.train_idx]

    def val_sequences(self) -> np.ndarray:
        return self.sequences[self.val_idx]

    def test_sequences(self) -> np.ndarray:
        return self.sequences[self.test_idx]

    def train_targets(self) -> np.ndarray:
        return self.remaining_time[self.train_idx]

    def val_targets(self) -> np.ndarray:
        return self.remaining_time[self.val_idx]

    def test_targets(self) -> np.ndarray:
        return self.remaining_time[self.test_idx]

    def train_risk(self) -> np.ndarray:
        return self.risk_labels[self.train_idx]

    def val_risk(self) -> np.ndarray:
        return self.risk_labels[self.val_idx]

    def test_risk(self) -> np.ndarray:
        return self.risk_labels[self.test_idx]


def trajectory_cache_key(config: SystemConfig, seed: Optional[int], index: int) -> str:
    """Stable per-trajectory name for ITrajectoryStore (content-addressed)."""
    d = config.dissipator
    payload = {
        "nq": config.n_qubits.value,
        "omega": config.omega,
        "op": d.operator_type.value,
        "g": d.gamma_const,
        "c": list(d.gamma_coeffs) if d.gamma_coeffs else None,
        "it": config.interaction_type.value,
        "J": config.J,
        "tmax": config.t_max,
        "dt": config.dt,
        "crit": config.decoherence_criterion.value,
        "thr": config.decoherence_threshold,
        "seed": seed,
        "i": index,
    }
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return "t_" + hashlib.sha1(raw.encode()).hexdigest()[:16]


# ---------------------------------------------------------------------------
# Command + Use-case
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class GenerateDatasetCommand:
    """Input for the GenerateDatasetUseCase."""

    configs: List[SystemConfig]
    window_length: int = 50
    horizon: float = 1.0
    samples_per_trajectory: int = 5
    train_ratio: float = 0.8
    val_ratio: float = 0.1
    seed: Optional[int] = 42
    dataset_name: str = "dataset"
    # Minimum gap between sampled window end-indices (in timesteps).
    # 0 = off (default, backward-compatible).  Positive values reduce
    # within-trajectory overlap, improving generalisation.
    min_window_gap: int = 0
    # False (default): drop right-censored trajectories (task 06).
    # True: keep them; remaining_time is a lower bound, Dataset.censored=True.
    include_censored: bool = False


class GenerateDatasetUseCase:
    """Simulate multiple systems and produce a windowed ML dataset.

    Sliding-window approach:
    - For each trajectory, sample `samples_per_trajectory` random windows.
    - Each window of length `window_length` ends at time t_obs.
    - Label: risk = 1 if t_obs < t_decoh ≤ t_obs + horizon, else 0.
    - Target: remaining_time = t_decoh - t_obs (strictly positive for pre-decoherence windows).

    Train/val/test split is done at the *trajectory* level to prevent
    data leakage across time windows from the same trajectory.
    """

    def __init__(
        self,
        simulator: ISimulator,
        store: Optional[ITrajectoryStore] = None,
    ) -> None:
        self._simulator = simulator
        self._store = store

    def execute(self, command: GenerateDatasetCommand) -> Dataset:
        rng = np.random.default_rng(command.seed)

        trajectories: List[TrajectoryResult] = []
        for i, cfg in enumerate(command.configs):
            trajectories.append(self._get_or_simulate(cfg, command.seed, i))

        if self._store is not None:
            self._store.save(trajectories, command.dataset_name)

        return self._build_dataset(trajectories, command, rng)

    def _get_or_simulate(
        self,
        cfg: SystemConfig,
        seed: Optional[int],
        index: int,
    ) -> TrajectoryResult:
        if self._store is None:
            return self._simulator.simulate(cfg)
        key = trajectory_cache_key(cfg, seed, index)
        try:
            return self._store.load(key)[0]
        except FileNotFoundError:
            traj = self._simulator.simulate(cfg)
            self._store.save([traj], key)
            return traj

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _build_dataset(
        trajectories: List[TrajectoryResult],
        command: GenerateDatasetCommand,
        rng: np.random.Generator,
    ) -> Dataset:
        # Trajectory-level split indices
        n_trajs = len(trajectories)
        traj_idx = rng.permutation(n_trajs)
        n_train = int(n_trajs * command.train_ratio)
        n_val = int(n_trajs * command.val_ratio)

        train_trajs = set(traj_idx[:n_train].tolist())
        val_trajs = set(traj_idx[n_train : n_train + n_val].tolist())

        all_sequences: List[np.ndarray] = []
        all_t_obs: List[float] = []
        all_t_decoh_abs: List[float] = []
        all_remaining: List[float] = []
        all_risk: List[int] = []
        all_split: List[str] = []  # 'train' | 'val' | 'test'
        all_interaction_codes: List[float] = []
        all_dissipator_codes: List[float] = []
        all_J: List[float] = []
        all_censored: List[bool] = []

        L = command.window_length
        n_skipped = 0
        n_censored = 0

        for traj_i, traj in enumerate(trajectories):
            is_censored = bool(getattr(traj, "censored", False))
            if is_censored:
                n_censored += 1
                if not command.include_censored:
                    continue

            feat = traj.feature_matrix           # (T, F)
            times = traj.times
            n_steps = len(times)

            # Context codes: encode Hamiltonian type and jump-operator type
            # so the model knows which physics regime it is in.
            # 4-class unified system type code (stored in interaction_type_codes):
            #   0 = 1-qubit, σ₋   1 = 1-qubit, σ_z
            #   2 = 2-qubit XXZ   3 = 2-qubit TFIM
            # Using distinct codes for all 4 regimes avoids the ambiguity where
            # XXZ (binary=0) and 1-qubit-σ₋ (binary=0) appeared identical to the model.
            cfg = traj.system_config
            if cfg.n_qubits == QubitCount.TWO:
                _int_code = 3.0 if cfg.interaction_type == InteractionType.TFIM else 2.0
            else:
                _int_code = 1.0 if cfg.dissipator.operator_type == DissipatorType.SIGMA_Z else 0.0
            _dis_code = 1.0 if (cfg.dissipator.operator_type == DissipatorType.SIGMA_Z) else 0.0
            _J = float(cfg.J) if cfg.n_qubits == QubitCount.TWO else 0.0

            if n_steps <= L:
                continue  # trajectory too short

            # Only keep windows that END strictly before decoherence.
            # This guarantees remaining_time > 0 everywhere and keeps the
            # regression target physically meaningful.
            pre_decoh_mask = times < traj.t_decoh
            valid_end_indices = np.where(pre_decoh_mask)[0]
            valid_end_indices = valid_end_indices[valid_end_indices >= L]

            if len(valid_end_indices) == 0:
                n_skipped += 1
                continue  # T₂ shorter than window — skip trajectory

            n_sample = min(command.samples_per_trajectory, len(valid_end_indices))
            if command.min_window_gap > 0 and len(valid_end_indices) > 1:
                # Greedy selection with minimum gap between chosen end-indices.
                shuffled = rng.permutation(valid_end_indices)
                selected: list = []
                for idx in shuffled:
                    if all(abs(int(idx) - int(s)) >= command.min_window_gap for s in selected):
                        selected.append(idx)
                    if len(selected) == n_sample:
                        break
                chosen = np.array(selected)
            else:
                chosen = rng.choice(valid_end_indices, size=n_sample, replace=False)

            split = (
                "train" if traj_i in train_trajs
                else "val" if traj_i in val_trajs
                else "test"
            )

            for end_idx in chosen:
                window = feat[end_idx - L : end_idx]      # (L, F)
                t_obs = float(times[end_idx])
                remaining = traj.t_decoh - t_obs          # > 0; lower bound if censored
                # Censored + remaining ≤ horizon: event may or may not be imminent — label 0,
                # training will mask this sample out of the risk BCE.
                risk = 0 if is_censored else int(remaining <= command.horizon)

                all_sequences.append(window)
                all_t_obs.append(t_obs)
                all_t_decoh_abs.append(traj.t_decoh)
                all_remaining.append(remaining)
                all_risk.append(risk)
                all_split.append(split)
                all_interaction_codes.append(_int_code)
                all_dissipator_codes.append(_dis_code)
                all_J.append(_J)
                all_censored.append(is_censored)

        if not all_sequences:
            raise RuntimeError(
                "No valid pre-decoherence windows found. "
                "Reduce window_length or increase t_max / T₂ (lower gamma)."
            )

        # Pad to uniform feature dimension when mixing 1-qubit and 2-qubit configs.
        # coherence_l1 and purity are always the last 2 columns; insert zero-padding
        # before them so that the physics features stay at a fixed position.
        max_f = max(s.shape[-1] for s in all_sequences)
        if max_f > min(s.shape[-1] for s in all_sequences):
            padded = []
            for s in all_sequences:
                if s.shape[-1] < max_f:
                    n_pad = max_f - s.shape[-1]
                    s = np.concatenate(
                        [s[:, :-2],
                         np.zeros((s.shape[0], n_pad), dtype=s.dtype),
                         s[:, -2:]],
                        axis=-1,
                    )
                padded.append(s)
            all_sequences = padded

        sequences = np.array(all_sequences, dtype=np.float32)
        t_obs_arr = np.array(all_t_obs, dtype=np.float32)
        t_decoh_abs_arr = np.array(all_t_decoh_abs, dtype=np.float32)
        remaining_arr = np.array(all_remaining, dtype=np.float32)
        risk_arr = np.array(all_risk, dtype=np.float32)
        split_arr = np.array(all_split)
        interaction_codes_arr = np.array(all_interaction_codes, dtype=np.float32)
        dissipator_codes_arr = np.array(all_dissipator_codes, dtype=np.float32)
        J_arr = np.array(all_J, dtype=np.float32)
        censored_arr = np.array(all_censored, dtype=bool)

        train_idx = np.where(split_arr == "train")[0]
        val_idx = np.where(split_arr == "val")[0]
        test_idx = np.where(split_arr == "test")[0]

        # Use feature names from the trajectory with the most features so that
        # index lookups (coherence_l1, purity) remain valid after zero-padding.
        if trajectories:
            feature_names = max(
                (t.feature_names for t in trajectories),
                key=len,
                default=[],
            )
        else:
            feature_names = []

        # Extract dt from the first trajectory's time array (robust to linspace vs arange)
        _dt = float(trajectories[0].times[1] - trajectories[0].times[0]) if trajectories and len(trajectories[0].times) > 1 else 0.1

        return Dataset(
            sequences=sequences,
            t_obs=t_obs_arr,
            t_decoh_abs=t_decoh_abs_arr,
            remaining_time=remaining_arr,
            risk_labels=risk_arr,
            train_idx=train_idx,
            val_idx=val_idx,
            test_idx=test_idx,
            feature_names=feature_names,
            interaction_type_codes=interaction_codes_arr,
            dissipator_type_codes=dissipator_codes_arr,
            J_values=J_arr,
            censored=censored_arr,
            metadata={
                "n_trajectories": n_trajs,
                "n_skipped": n_skipped,
                "n_censored": n_censored,
                "include_censored": command.include_censored,
                "n_censored_windows": int(censored_arr.sum()),
                "window_length": L,
                "horizon": command.horizon,
                "seed": command.seed,
                "n_train": len(train_idx),
                "n_val": len(val_idx),
                "n_test": len(test_idx),
                "dt": _dt,
            },
        )
