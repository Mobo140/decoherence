"""File-based adapters for ITrajectoryStore and IModelStore.

NumpyTrajectoryStore  – saves/loads trajectory collections as .npz archives.
PickleModelStore      – saves/loads any IPredictor via state_bytes + kind sidecar.
"""
from __future__ import annotations

import json
import zipfile
from pathlib import Path
from typing import Dict, List, Type

import numpy as np

from ...domain.ports import IModelStore, ITrajectoryStore, IPredictor
from ...domain.value_objects import (
    DecoherenceCriterion,
    DissipatorConfig,
    DissipatorType,
    InteractionType,
    QubitCount,
    SystemConfig,
    TrajectoryResult,
)


class NumpyTrajectoryStore(ITrajectoryStore):
    """Persist trajectory lists as compressed NumPy archives.

    Layout::

        <root>/<name>.npz          – arrays
        <root>/<name>_meta.json    – metadata (system configs, feature names)
    """

    def __init__(self, root: str | Path = "data/trajectories") -> None:
        self._root = Path(root)
        self._root.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # ITrajectoryStore
    # ------------------------------------------------------------------

    def save(self, trajectories: List[TrajectoryResult], name: str) -> None:
        arrays: Dict[str, np.ndarray] = {}
        meta: Dict = {"n": len(trajectories), "configs": [], "feature_names": []}

        for i, traj in enumerate(trajectories):
            arrays[f"times_{i}"] = traj.times
            arrays[f"t_decoh_{i}"] = np.array([traj.t_decoh])
            arrays[f"censored_{i}"] = np.array([int(getattr(traj, "censored", False))])
            for obs_key, obs_arr in traj.observables.items():
                if obs_key != "density_matrices":
                    arrays[f"obs_{i}_{obs_key}"] = obs_arr

            meta["configs"].append(_config_to_dict(traj.system_config))
            if i == 0:
                meta["feature_names"] = traj.feature_names

        # Written through temporary files and renamed into place: a run that
        # is interrupted mid-write would otherwise leave a truncated .npz or
        # an empty _meta.json behind, and because the cache key is
        # deterministic every later run would hit that same broken entry.
        npz_path = self._root / f"{name}.npz"
        # The temporary name must itself end in .npz: savez_compressed
        # appends the extension when it is missing, which would leave the
        # bytes somewhere other than where the rename looks for them.
        npz_tmp = self._root / f"{name}.tmp.npz"
        np.savez_compressed(npz_tmp, **arrays)
        npz_tmp.replace(npz_path)

        meta_path = self._root / f"{name}_meta.json"
        meta_tmp = meta_path.with_suffix(".json.tmp")
        with open(meta_tmp, "w") as f:
            json.dump(meta, f, indent=2)
        meta_tmp.replace(meta_path)

    def load(self, name: str) -> List[TrajectoryResult]:
        npz_path = self._root / f"{name}.npz"
        meta_path = self._root / f"{name}_meta.json"

        if not npz_path.exists():
            raise FileNotFoundError(f"No trajectory store found at {npz_path}")

        # A damaged entry is reported as a miss rather than an error, so the
        # caller re-simulates and overwrites it instead of failing. Callers
        # treat FileNotFoundError as "not cached"; anything else propagates.
        # The whole reconstruction sits inside the guard because np.load is
        # lazy: a truncated archive raises its CRC error when an array is
        # first touched, not when the file is opened.
        try:
            data = np.load(npz_path, allow_pickle=False)
            with open(meta_path) as f:
                meta = json.load(f)

            trajectories: List[TrajectoryResult] = []
            feature_names: List[str] = meta.get("feature_names", [])

            for i, cfg_dict in enumerate(meta["configs"]):
                times = data[f"times_{i}"]
                t_decoh = float(data[f"t_decoh_{i}"][0])
                censored = bool(int(data[f"censored_{i}"][0])) if f"censored_{i}" in data else False
                observables: Dict[str, np.ndarray] = {}
                for key in feature_names:
                    arr_key = f"obs_{i}_{key}"
                    if arr_key in data:
                        observables[key] = data[arr_key]

                config = _config_from_dict(cfg_dict)
                trajectories.append(
                    TrajectoryResult(
                        times=times,
                        observables=observables,
                        t_decoh=t_decoh,
                        system_config=config,
                        censored=censored,
                    )
                )
        except (OSError, ValueError, KeyError, EOFError,
                json.JSONDecodeError, zipfile.BadZipFile) as exc:
            raise FileNotFoundError(
                f"Unreadable trajectory cache entry {name}: {exc}"
            ) from exc

        return trajectories

    def list_available(self) -> List[str]:
        return [p.stem for p in self._root.glob("*.npz")]


class BestModelRegistry:
    """Stores one best model per qubit count, compared by backtest R².

    Files on disk:
        <root>/best_1qubit.pt         – weights for 1-qubit champion
        <root>/best_1qubit_meta.json  – metrics for 1-qubit champion
        <root>/best_2qubit.pt / meta  – same for 2-qubit
    """

    def __init__(self, root: str | Path = "checkpoints") -> None:
        self._root = Path(root)
        self._root.mkdir(parents=True, exist_ok=True)
        self._store = PickleModelStore(root=root)

    def _name(self, n_qubits: int) -> str:
        return f"best_{n_qubits}qubit"

    def _meta_path(self, n_qubits: int) -> Path:
        return self._root / f"best_{n_qubits}qubit_meta.json"

    def get_meta(self, n_qubits: int) -> dict | None:
        p = self._meta_path(n_qubits)
        if p.exists():
            with open(p) as f:
                return json.load(f)
        return None

    def get_predictor(self, n_qubits: int) -> "IPredictor | None":
        try:
            return self._store.load(self._name(n_qubits))
        except Exception:
            return None

    def maybe_update(self, predictor: "IPredictor", metrics, n_qubits: int) -> bool:
        """Save predictor as new champion if its R² exceeds the stored best.

        Returns True if the registry was updated.
        """
        current = self.get_meta(n_qubits)
        new_r2 = metrics.r2
        n_samples = getattr(metrics, "n_samples", 0)
        if n_samples < 1 or not np.isfinite(new_r2):
            return False
        if current is None or new_r2 > current.get("r2", float("-inf")):
            self._store.save(predictor, self._name(n_qubits))
            meta = {
                "r2":       new_r2,
                "mae":      metrics.mae,
                "rmse":     metrics.rmse,
                "mape":     metrics.mape,
                "auroc":    metrics.risk_auroc,
                "n_samples": metrics.n_samples,
                "kind":     type(predictor).__name__,
            }
            with open(self._meta_path(n_qubits), "w") as f:
                json.dump(meta, f, indent=2)
            return True
        return False

    def summary(self) -> str:
        lines = []
        for nq in (1, 2):
            meta = self.get_meta(nq)
            if meta:
                lines.append(
                    f"{nq}-qubit: R²={meta['r2']:.4f}  MAE={meta['mae']:.4f}"
                    f"  AUROC={meta['auroc']:.4f}  n={meta['n_samples']}"
                )
            else:
                lines.append(f"{nq}-qubit: no model saved yet")
        return "\n".join(lines)


class HamiltonianModelRegistry:
    """Stores one best model per Hamiltonian type, compared by backtest R².

    Registry key: (n_qubits, interaction_type, dissipator_type)
    This is more granular than BestModelRegistry (which uses only n_qubits).

    Files on disk::

        <root>/best_XXZ.pt / best_XXZ_meta.json
        <root>/best_TFIM.pt / best_TFIM_meta.json
        <root>/best_1q_sigma_minus.pt / best_1q_sigma_minus_meta.json
        <root>/best_1q_sigma_z.pt / best_1q_sigma_z_meta.json

    Usage::

        registry = HamiltonianModelRegistry()
        updated = registry.maybe_update(predictor, metrics, config)
        pred = registry.get_predictor(config)  # None if not saved yet
    """

    def __init__(self, root: str | Path = "checkpoints") -> None:
        self._root = Path(root)
        self._root.mkdir(parents=True, exist_ok=True)
        self._store = PickleModelStore(root=root)

    @staticmethod
    def _name(
        n_qubits: int,
        interaction_type: "InteractionType",
        dissipator_type: "DissipatorType",
    ) -> str:
        """Return a filesystem-safe registry key string."""
        if n_qubits == 1:
            return f"best_1q_{dissipator_type.value}"
        return f"best_{interaction_type.value}"

    @staticmethod
    def _label(
        n_qubits: int,
        interaction_type: "InteractionType",
        dissipator_type: "DissipatorType",
    ) -> str:
        """Human-readable label for UI display."""
        if n_qubits == 1:
            return f"1q/{dissipator_type.value}"
        return f"2q/{interaction_type.value}"

    def _meta_path(self, name: str) -> Path:
        return self._root / f"{name}_meta.json"

    def get_meta(
        self,
        n_qubits: int,
        interaction_type: "InteractionType",
        dissipator_type: "DissipatorType",
    ) -> dict | None:
        p = self._meta_path(self._name(n_qubits, interaction_type, dissipator_type))
        if p.exists():
            with open(p) as f:
                return json.load(f)
        return None

    def get_predictor(
        self,
        n_qubits: int,
        interaction_type: "InteractionType",
        dissipator_type: "DissipatorType",
    ) -> "IPredictor | None":
        name = self._name(n_qubits, interaction_type, dissipator_type)
        try:
            return self._store.load(name)
        except Exception:
            return None

    def maybe_update(
        self,
        predictor: "IPredictor",
        metrics,
        n_qubits: int,
        interaction_type: "InteractionType",
        dissipator_type: "DissipatorType",
    ) -> bool:
        """Save predictor as new champion if its R² exceeds the stored best.

        Returns True if the registry was updated.
        """
        name = self._name(n_qubits, interaction_type, dissipator_type)
        label = self._label(n_qubits, interaction_type, dissipator_type)
        current = self.get_meta(n_qubits, interaction_type, dissipator_type)
        new_r2 = metrics.r2
        n_samples = getattr(metrics, "n_samples", 0)
        if n_samples < 1 or not np.isfinite(new_r2):
            return False
        if current is None or new_r2 > current.get("r2", float("-inf")):
            self._store.save(predictor, name)
            meta = {
                "label":      label,
                "n_qubits":   n_qubits,
                "interaction": interaction_type.value,
                "dissipator":  dissipator_type.value,
                "r2":          new_r2,
                "mae":         metrics.mae,
                "rmse":        metrics.rmse,
                "mape":        metrics.mape,
                "auroc":       metrics.risk_auroc,
                "n_samples":   metrics.n_samples,
                "kind":        type(predictor).__name__,
            }
            with open(self._meta_path(name), "w") as f:
                json.dump(meta, f, indent=2)
            return True
        return False

    def summary(self) -> str:
        """Return a human-readable summary of all Hamiltonian-specific slots."""
        slots = (
            (1, InteractionType.XXZ, DissipatorType.SIGMA_MINUS),
            (1, InteractionType.XXZ, DissipatorType.SIGMA_Z),
            (2, InteractionType.XXZ, DissipatorType.SIGMA_MINUS),
            (2, InteractionType.TFIM, DissipatorType.SIGMA_MINUS),
        )
        lines = []
        for nq, it, dt in slots:
            label = self._label(nq, it, dt)
            meta = self.get_meta(nq, it, dt)
            if meta:
                lines.append(
                    f"{label}: R²={meta['r2']:.4f}  MAE={meta['mae']:.4f}"
                    f"  AUROC={meta['auroc']:.4f}  n={meta['n_samples']}"
                )
            else:
                lines.append(f"{label}: no model saved yet")
        return "\n".join(lines)


def predictor_class_registry() -> Dict[str, Type[IPredictor]]:
    """Lazy map class-name → IPredictor. Avoids importing torch at module import."""
    from ...infrastructure.ml.lstm_predictor import LSTMPredictor
    from ...infrastructure.ml.physics_only_predictor import PhysicsOnlyPredictor
    from ...infrastructure.ml.stretched_exp_predictor import StretchedExpPredictor
    from ...infrastructure.ml.transformer_predictor import TransformerPredictor

    return {
        "LSTMPredictor": LSTMPredictor,
        "TransformerPredictor": TransformerPredictor,
        "PhysicsOnlyPredictor": PhysicsOnlyPredictor,
        "StretchedExpPredictor": StretchedExpPredictor,
    }


def infer_predictor_kind(data: bytes) -> str:
    """Guess predictor class from raw checkpoint bytes (old files without sidecar)."""
    if data[:1] in (b"{", b"[") or data[:1].lstrip()[:1] == b"{":
        try:
            payload = json.loads(data.decode())
            if isinstance(payload, dict) and "t_max" in payload and "dt" in payload:
                return "StretchedExpPredictor"
        except (UnicodeDecodeError, json.JSONDecodeError, ValueError):
            pass
    if len(data) == 24:
        return "PhysicsOnlyPredictor"
    try:
        import io
        import torch

        ckpt = torch.load(io.BytesIO(data), map_location="cpu", weights_only=False)
        if isinstance(ckpt, dict) and "hidden_size" in ckpt:
            return "LSTMPredictor"
        if isinstance(ckpt, dict) and "d_model" in ckpt:
            return "TransformerPredictor"
    except Exception:
        pass
    return "LSTMPredictor"


class PickleModelStore(IModelStore):
    """Persist any IPredictor via ``state_bytes`` plus a kind sidecar.

    Layout::

        <root>/<name>.pt         – predictor.state_bytes()
        <root>/<name>_kind.json  – {"kind": "TransformerPredictor"}
    """

    def __init__(self, root: str | Path = "checkpoints") -> None:
        self._root = Path(root)
        self._root.mkdir(parents=True, exist_ok=True)

    def _kind_path(self, name: str) -> Path:
        return self._root / f"{name}_kind.json"

    def save(self, predictor: IPredictor, name: str) -> None:
        data = predictor.state_bytes()  # type: ignore[attr-defined]
        out_path = self._root / f"{name}.pt"
        out_path.write_bytes(data)
        kind = type(predictor).__name__
        with open(self._kind_path(name), "w") as f:
            json.dump({"kind": kind}, f)

    def load(self, name: str) -> IPredictor:
        pt_path = self._root / f"{name}.pt"
        if not pt_path.exists():
            raise FileNotFoundError(f"No model checkpoint at {pt_path}")
        data = pt_path.read_bytes()
        kind_path = self._kind_path(name)
        if kind_path.exists():
            with open(kind_path) as f:
                kind = json.load(f)["kind"]
        else:
            kind = infer_predictor_kind(data)
        registry = predictor_class_registry()
        if kind not in registry:
            raise ValueError(f"Unknown predictor kind {kind!r} for {name}")
        return registry[kind].from_bytes(data)


# ---------------------------------------------------------------------------
# Serialisation helpers for SystemConfig ↔ dict
# ---------------------------------------------------------------------------

def _config_to_dict(config: SystemConfig) -> dict:
    d = config.dissipator
    return {
        "n_qubits": config.n_qubits.value,
        "omega": config.omega,
        "dissipator": {
            "operator_type": d.operator_type.value,
            "gamma_const": d.gamma_const,
            "gamma_coeffs": list(d.gamma_coeffs) if d.gamma_coeffs else None,
        },
        "interaction_type": config.interaction_type.value,
        "J": config.J,
        "t_max": config.t_max,
        "dt": config.dt,
        "decoherence_criterion": config.decoherence_criterion.value,
        "decoherence_threshold": config.decoherence_threshold,
    }


def _config_from_dict(d: dict) -> SystemConfig:
    dis_d = d["dissipator"]
    gamma_coeffs = tuple(dis_d["gamma_coeffs"]) if dis_d["gamma_coeffs"] else None
    dissipator = DissipatorConfig(
        operator_type=DissipatorType(dis_d["operator_type"]),
        gamma_const=dis_d["gamma_const"],
        gamma_coeffs=gamma_coeffs,
    )
    return SystemConfig(
        n_qubits=QubitCount(d["n_qubits"]),
        omega=d["omega"],
        dissipator=dissipator,
        interaction_type=InteractionType(d["interaction_type"]),
        J=d["J"],
        t_max=d["t_max"],
        dt=d["dt"],
        decoherence_criterion=DecoherenceCriterion(d["decoherence_criterion"]),
        decoherence_threshold=d["decoherence_threshold"],
    )
