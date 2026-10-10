"""File-based adapters for ITrajectoryStore and IModelStore.

NumpyTrajectoryStore: trajectory collections as .npz archives.
PickleModelStore: any IPredictor via `state_bytes` plus a kind sidecar.
BestModelRegistry, HamiltonianModelRegistry: champion checkpoints.
"""
from __future__ import annotations

import io
import json
import os
import pathlib
import zipfile

import numpy as np

from src.domain import ports
from src.domain import value_objects


class NumpyTrajectoryStore(ports.ITrajectoryStore):
    """Persists trajectory lists as compressed NumPy archives.

    Layout::

        <root>/<name>.npz          arrays
        <root>/<name>_meta.json    metadata (system configs, feature names)
    """

    def __init__(self, root: str | pathlib.Path = "data/trajectories") -> None:
        """Opens (and creates) the store directory.

        Args:
            root: Directory of the archives.
        """
        self._root = pathlib.Path(root)
        self._root.mkdir(parents=True, exist_ok=True)

    def save(
        self, trajectories: list[value_objects.TrajectoryResult], name: str
    ) -> None:
        """Writes a trajectory collection under `name`, replacing any.

        Args:
            trajectories: The trajectories to store.
            name: Store key.
        """
        arrays: dict[str, np.ndarray] = {}
        meta: dict = {
            "n": len(trajectories), "configs": [], "feature_names": []}

        for i, traj in enumerate(trajectories):
            arrays[f"times_{i}"] = traj.times
            arrays[f"t_decoh_{i}"] = np.array([traj.t_decoh])
            arrays[f"censored_{i}"] = np.array(
                [int(getattr(traj, "censored", False))])
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
        # It also carries the process id: concurrent experiments that save
        # under the same name must not rename each other's temporary file.
        npz_tmp = self._root / f"{name}.{os.getpid()}.tmp.npz"
        np.savez_compressed(npz_tmp, **arrays)
        npz_tmp.replace(npz_path)

        meta_path = self._root / f"{name}_meta.json"
        meta_tmp = meta_path.with_suffix(f".json.{os.getpid()}.tmp")
        with open(meta_tmp, "w") as f:
            json.dump(meta, f, indent=2)
        meta_tmp.replace(meta_path)

    def load(self, name: str) -> list[value_objects.TrajectoryResult]:
        """Reads the collection stored under `name`.

        Args:
            name: Store key.

        Returns:
            The trajectories.

        Raises:
            FileNotFoundError: If there is no entry, or it is damaged;
                callers treat both as "not cached" and re-simulate.
        """
        npz_path = self._root / f"{name}.npz"
        meta_path = self._root / f"{name}_meta.json"

        if not npz_path.exists():
            raise FileNotFoundError(
                f"No trajectory store found at {npz_path}")

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
            feature_names = meta.get("feature_names", [])
            return [
                _trajectory_from_archive(data, i, cfg_dict, feature_names)
                for i, cfg_dict in enumerate(meta["configs"])
            ]
        except (OSError, ValueError, KeyError, EOFError,
                json.JSONDecodeError, zipfile.BadZipFile) as exc:
            raise FileNotFoundError(
                f"Unreadable trajectory cache entry {name}: {exc}"
            ) from exc

    def list_available(self) -> list[str]:
        """Returns the keys of all stored collections."""
        return [p.stem for p in self._root.glob("*.npz")]


def _trajectory_from_archive(
    data, index: int, cfg_dict: dict, feature_names: list[str]
) -> value_objects.TrajectoryResult:
    censored_key = f"censored_{index}"
    censored = bool(int(data[censored_key][0])) if censored_key in data \
        else False
    observables = {}
    for key in feature_names:
        arr_key = f"obs_{index}_{key}"
        if arr_key in data:
            observables[key] = data[arr_key]
    return value_objects.TrajectoryResult(
        times=data[f"times_{index}"],
        observables=observables,
        t_decoh=float(data[f"t_decoh_{index}"][0]),
        system_config=_config_from_dict(cfg_dict),
        censored=censored,
    )


def _write_json(path: pathlib.Path, payload: dict) -> None:
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)


def _metrics_row(metrics) -> dict:
    return {
        "r2": metrics.r2,
        "mae": metrics.mae,
        "rmse": metrics.rmse,
        "mape": metrics.mape,
        "auroc": metrics.risk_auroc,
        "n_samples": metrics.n_samples,
    }


def _is_better(metrics, current: dict | None) -> bool:
    """True if `metrics` should replace the stored champion `current`."""
    n_samples = getattr(metrics, "n_samples", 0)
    if n_samples < 1 or not np.isfinite(metrics.r2):
        return False
    return current is None or metrics.r2 > current.get("r2", float("-inf"))


def _summary_line(label: str, meta: dict | None) -> str:
    if not meta:
        return f"{label}: no model saved yet"
    return (f"{label}: R²={meta['r2']:.4f}  MAE={meta['mae']:.4f}"
            f"  AUROC={meta['auroc']:.4f}  n={meta['n_samples']}")


class BestModelRegistry:
    """Stores one best model per qubit count, compared by backtest R².

    Files on disk::

        <root>/best_1qubit.pt         weights of the 1-qubit champion
        <root>/best_1qubit_meta.json  its metrics
        <root>/best_2qubit.pt / meta  the same for 2 qubits
    """

    def __init__(self, root: str | pathlib.Path = "checkpoints") -> None:
        """Opens (and creates) the checkpoint directory.

        Args:
            root: Directory of the checkpoints.
        """
        self._root = pathlib.Path(root)
        self._root.mkdir(parents=True, exist_ok=True)
        self._store = PickleModelStore(root=root)

    def _name(self, n_qubits: int) -> str:
        return f"best_{n_qubits}qubit"

    def _meta_path(self, n_qubits: int) -> pathlib.Path:
        return self._root / f"best_{n_qubits}qubit_meta.json"

    def get_meta(self, n_qubits: int) -> dict | None:
        """Returns the champion's metrics, or None if there is none."""
        path = self._meta_path(n_qubits)
        if path.exists():
            with open(path) as f:
                return json.load(f)
        return None

    def get_predictor(self, n_qubits: int) -> ports.IPredictor | None:
        """Loads the champion; None if missing or unreadable."""
        try:
            return self._store.load(self._name(n_qubits))
        except Exception:  # A bad checkpoint must not break the caller.
            return None

    def maybe_update(
        self, predictor: ports.IPredictor, metrics, n_qubits: int
    ) -> bool:
        """Saves `predictor` as the new champion if its R² is higher.

        Args:
            predictor: Trained predictor.
            metrics: Backtest metrics of `predictor`.
            n_qubits: Slot (1 or 2).

        Returns:
            True if the registry was updated.
        """
        if not _is_better(metrics, self.get_meta(n_qubits)):
            return False
        self._store.save(predictor, self._name(n_qubits))
        meta = {**_metrics_row(metrics), "kind": type(predictor).__name__}
        _write_json(self._meta_path(n_qubits), meta)
        return True

    def summary(self) -> str:
        """Returns one line per slot with the champion's metrics."""
        return "\n".join(
            _summary_line(f"{nq}-qubit", self.get_meta(nq)) for nq in (1, 2))


def slot_name(
    n_qubits: int,
    interaction_type: value_objects.InteractionType,
    dissipator_type: value_objects.DissipatorType,
) -> str:
    """Returns the filesystem-safe checkpoint name of a Hamiltonian slot."""
    if n_qubits == 1:
        return f"best_1q_{dissipator_type.value}"
    return f"best_{interaction_type.value}"


def slot_label(
    n_qubits: int,
    interaction_type: value_objects.InteractionType,
    dissipator_type: value_objects.DissipatorType,
) -> str:
    """Returns the human-readable label of a Hamiltonian slot."""
    if n_qubits == 1:
        return f"1q/{dissipator_type.value}"
    return f"2q/{interaction_type.value}"


class HamiltonianModelRegistry:
    """Stores one best model per Hamiltonian type, compared by backtest R².

    The key is (n_qubits, interaction_type, dissipator_type), which is
    finer than the qubit count alone of `BestModelRegistry`.

    Files on disk::

        <root>/best_XXZ.pt / best_XXZ_meta.json
        <root>/best_TFIM.pt / best_TFIM_meta.json
        <root>/best_1q_sigma_minus.pt / best_1q_sigma_minus_meta.json
        <root>/best_1q_sigma_z.pt / best_1q_sigma_z_meta.json
    """

    def __init__(self, root: str | pathlib.Path = "checkpoints") -> None:
        """Opens (and creates) the checkpoint directory.

        Args:
            root: Directory of the checkpoints.
        """
        self._root = pathlib.Path(root)
        self._root.mkdir(parents=True, exist_ok=True)
        self._store = PickleModelStore(root=root)

    def _meta_path(self, name: str) -> pathlib.Path:
        return self._root / f"{name}_meta.json"

    def get_meta(
        self,
        n_qubits: int,
        interaction_type: value_objects.InteractionType,
        dissipator_type: value_objects.DissipatorType,
    ) -> dict | None:
        """Returns the slot's champion metrics, or None if there is none."""
        name = slot_name(n_qubits, interaction_type, dissipator_type)
        path = self._meta_path(name)
        if path.exists():
            with open(path) as f:
                return json.load(f)
        return None

    def get_predictor(
        self,
        n_qubits: int,
        interaction_type: value_objects.InteractionType,
        dissipator_type: value_objects.DissipatorType,
    ) -> ports.IPredictor | None:
        """Loads the slot's champion; None if missing or unreadable."""
        name = slot_name(n_qubits, interaction_type, dissipator_type)
        try:
            return self._store.load(name)
        except Exception:  # A bad checkpoint must not break the caller.
            return None

    def maybe_update(
        self,
        predictor: ports.IPredictor,
        metrics,
        n_qubits: int,
        interaction_type: value_objects.InteractionType,
        dissipator_type: value_objects.DissipatorType,
    ) -> bool:
        """Saves `predictor` as the slot's champion if its R² is higher.

        Args:
            predictor: Trained predictor.
            metrics: Backtest metrics of `predictor`.
            n_qubits: Number of qubits of the slot.
            interaction_type: Interaction of the slot.
            dissipator_type: Dissipator of the slot.

        Returns:
            True if the registry was updated.
        """
        current = self.get_meta(n_qubits, interaction_type, dissipator_type)
        if not _is_better(metrics, current):
            return False
        name = slot_name(n_qubits, interaction_type, dissipator_type)
        self._store.save(predictor, name)
        meta = {
            "label": slot_label(n_qubits, interaction_type, dissipator_type),
            "n_qubits": n_qubits,
            "interaction": interaction_type.value,
            "dissipator": dissipator_type.value,
            **_metrics_row(metrics),
            "kind": type(predictor).__name__,
        }
        _write_json(self._meta_path(name), meta)
        return True

    def summary(self) -> str:
        """Returns one line per Hamiltonian slot with its metrics."""
        slots = (
            (1, value_objects.InteractionType.XXZ,
             value_objects.DissipatorType.SIGMA_MINUS),
            (1, value_objects.InteractionType.XXZ,
             value_objects.DissipatorType.SIGMA_Z),
            (2, value_objects.InteractionType.XXZ,
             value_objects.DissipatorType.SIGMA_MINUS),
            (2, value_objects.InteractionType.TFIM,
             value_objects.DissipatorType.SIGMA_MINUS),
        )
        return "\n".join(
            _summary_line(slot_label(nq, it, dt), self.get_meta(nq, it, dt))
            for nq, it, dt in slots)


def predictor_class_registry() -> dict[str, type[ports.IPredictor]]:
    """Maps class name to predictor class.

    The imports are inside the function so that importing the stores does
    not import torch.

    Returns:
        Class name -> class, for every predictor a checkpoint can hold.
    """
    from src.models import lstm
    from src.models import physics_only
    from src.models import stretched_exp
    from src.models import transformer

    return {
        "LSTMPredictor": lstm.LSTMPredictor,
        "TransformerPredictor": transformer.TransformerPredictor,
        "PhysicsOnlyPredictor": physics_only.PhysicsOnlyPredictor,
        "StretchedExpPredictor": stretched_exp.StretchedExpPredictor,
    }


def infer_predictor_kind(data: bytes) -> str:
    """Guesses the predictor class from raw checkpoint bytes.

    For old files that have no kind sidecar.

    Args:
        data: Contents of a checkpoint file.

    Returns:
        A key of `predictor_class_registry()`; "LSTMPredictor" when
        nothing else matches.
    """
    if data[:1] in (b"{", b"[") or data[:1].lstrip()[:1] == b"{":
        try:
            payload = json.loads(data.decode())
            if (isinstance(payload, dict)
                    and "t_max" in payload and "dt" in payload):
                return "StretchedExpPredictor"
        except (UnicodeDecodeError, json.JSONDecodeError, ValueError):
            pass
    if len(data) == 24:
        return "PhysicsOnlyPredictor"
    try:
        import torch

        ckpt = torch.load(
            io.BytesIO(data), map_location="cpu", weights_only=False)
        if isinstance(ckpt, dict) and "hidden_size" in ckpt:
            return "LSTMPredictor"
        if isinstance(ckpt, dict) and "d_model" in ckpt:
            return "TransformerPredictor"
    except Exception:  # Not a torch file: fall through to the default.
        pass
    return "LSTMPredictor"


class PickleModelStore(ports.IModelStore):
    """Persists any IPredictor via ``state_bytes`` plus a kind sidecar.

    Layout::

        <root>/<name>.pt         predictor.state_bytes()
        <root>/<name>_kind.json  {"kind": "TransformerPredictor"}
    """

    def __init__(self, root: str | pathlib.Path = "checkpoints") -> None:
        """Opens (and creates) the checkpoint directory.

        Args:
            root: Directory of the checkpoints.
        """
        self._root = pathlib.Path(root)
        self._root.mkdir(parents=True, exist_ok=True)

    def _kind_path(self, name: str) -> pathlib.Path:
        return self._root / f"{name}_kind.json"

    def save(self, predictor: ports.IPredictor, name: str) -> None:
        """Writes a predictor's checkpoint and kind sidecar.

        Args:
            predictor: A predictor with a `state_bytes` method.
            name: Checkpoint name.
        """
        data = predictor.state_bytes()
        (self._root / f"{name}.pt").write_bytes(data)
        with open(self._kind_path(name), "w") as f:
            json.dump({"kind": type(predictor).__name__}, f)

    def load(self, name: str) -> ports.IPredictor:
        """Restores the predictor saved under `name`.

        Args:
            name: Checkpoint name.

        Returns:
            The predictor.

        Raises:
            FileNotFoundError: If there is no checkpoint.
            ValueError: If the stored kind is not a known predictor.
        """
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
        classes = predictor_class_registry()
        if kind not in classes:
            raise ValueError(f"Unknown predictor kind {kind!r} for {name}")
        return classes[kind].from_bytes(data)


# ---------------------------------------------------------------------------
# SystemConfig <-> dict
# ---------------------------------------------------------------------------

def _config_to_dict(config: value_objects.SystemConfig) -> dict:
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


def _config_from_dict(d: dict) -> value_objects.SystemConfig:
    dis_d = d["dissipator"]
    gamma_coeffs = (tuple(dis_d["gamma_coeffs"])
                    if dis_d["gamma_coeffs"] else None)
    dissipator = value_objects.DissipatorConfig(
        operator_type=value_objects.DissipatorType(dis_d["operator_type"]),
        gamma_const=dis_d["gamma_const"],
        gamma_coeffs=gamma_coeffs,
    )
    return value_objects.SystemConfig(
        n_qubits=value_objects.QubitCount(d["n_qubits"]),
        omega=d["omega"],
        dissipator=dissipator,
        interaction_type=value_objects.InteractionType(d["interaction_type"]),
        J=d["J"],
        t_max=d["t_max"],
        dt=d["dt"],
        decoherence_criterion=value_objects.DecoherenceCriterion(
            d["decoherence_criterion"]),
        decoherence_threshold=d["decoherence_threshold"],
    )
