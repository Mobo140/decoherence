"""The labelled dataset passed to training and backtesting."""
from __future__ import annotations

import dataclasses
from collections.abc import Iterable
from typing import Any

import numpy as np


# Arrays with one entry per window; `censored` may be empty.
_REQUIRED_PER_WINDOW = (
    "t_obs", "t_decoh_abs", "remaining_time", "risk_labels")
_OPTIONAL_PER_WINDOW = ("censored",)


@dataclasses.dataclass
class Dataset:
    """Labelled windows ready for training and backtesting.

    Only pre-decoherence windows are included (t_obs < t_decoh_abs), so
    remaining_time is strictly positive everywhere.

    The `with_*`, `select_features`, `drop_features` and `as_test_only`
    methods return a new Dataset that shares every array it does not change
    with the original.

    Attributes:
        sequences: Input windows, shape (N, window_length, n_features).
        t_obs: Observation time at the end of each window, shape (N,).
        t_decoh_abs: Absolute decoherence time T2 of the window's
            trajectory, used for evaluation, shape (N,).
        remaining_time: Regression target t_decoh_abs - t_obs, shape (N,).
        risk_labels: 1 if remaining_time <= horizon, else 0, shape (N,).
        train_idx: Indices into the first axis of `sequences`.
        val_idx: Validation indices.
        test_idx: Test indices.
        feature_names: Names of the observables in the last axis of
            `sequences`.
        metadata: Provenance of the dataset.
        context: Per-window scalars describing the system, by name, each
            of length N. `windows.build_dataset` fills `interaction_type`
            (4 classes: 0.0 = 1-qubit sigma_-, 1.0 = 1-qubit sigma_z,
            2.0 = 2-qubit XXZ, 3.0 = 2-qubit TFIM), `dissipator_type`
            (0.0 = sigma_- amplitude damping, 1.0 = sigma_z dephasing) and
            `J` (coupling; 0.0 for one qubit), all float32. A missing key
            means the value is unknown.
        censored: True where t_decoh_abs is only a lower bound (the
            threshold was not crossed by t_max). Empty if unknown.

    Raises:
        ValueError: If `sequences` is not 3-D, its last axis does not match
            `feature_names`, a per-window array does not have one entry per
            window (`censored` may be empty), a `context` array does not
            have one entry per window, or an index is not a window.
    """

    sequences: np.ndarray
    t_obs: np.ndarray
    t_decoh_abs: np.ndarray
    remaining_time: np.ndarray
    risk_labels: np.ndarray
    train_idx: np.ndarray
    val_idx: np.ndarray
    test_idx: np.ndarray
    feature_names: list[str]
    metadata: dict[str, Any]
    context: dict[str, np.ndarray] = dataclasses.field(
        default_factory=dict)
    censored: np.ndarray = dataclasses.field(
        default_factory=lambda: np.zeros(0, dtype=bool))

    def __post_init__(self) -> None:
        n_windows = len(self.sequences)
        if self.sequences.ndim != 3:
            raise ValueError(
                "sequences must have shape (N, window_length, n_features), "
                f"got {self.sequences.shape}.")
        if self.sequences.shape[2] != len(self.feature_names):
            raise ValueError(
                f"sequences have {self.sequences.shape[2]} features but "
                f"feature_names has {len(self.feature_names)}.")
        for name in _REQUIRED_PER_WINDOW:
            if len(getattr(self, name)) != n_windows:
                raise ValueError(
                    f"{name} has {len(getattr(self, name))} entries for "
                    f"{n_windows} windows.")
        for name in _OPTIONAL_PER_WINDOW:
            if len(getattr(self, name)) not in (0, n_windows):
                raise ValueError(
                    f"{name} has {len(getattr(self, name))} entries for "
                    f"{n_windows} windows; it must be empty or complete.")
        for name, values in self.context.items():
            if len(values) != n_windows:
                raise ValueError(
                    f"context[{name!r}] has {len(values)} entries for "
                    f"{n_windows} windows.")
        for name in ("train_idx", "val_idx", "test_idx"):
            index = np.asarray(getattr(self, name))
            if index.size and (index.min() < 0 or index.max() >= n_windows):
                raise ValueError(
                    f"{name} contains an index outside 0..{n_windows - 1}.")

    def with_sequences(self, sequences: np.ndarray) -> Dataset:
        """Returns the dataset with other input windows.

        Args:
            sequences: New windows, shape (N, window_length, n_features).
                Converted to float32. The feature names are kept, so the
                last axis must still match them.

        Returns:
            A new Dataset; labels, indices and context are unchanged.

        Raises:
            ValueError: If `sequences` has a different number of windows,
                is not 3-D, or its last axis does not match the feature
                names.
        """
        if len(sequences) != len(self.sequences):
            raise ValueError(
                f"Expected {len(self.sequences)} windows, "
                f"got {len(sequences)}")
        return dataclasses.replace(
            self, sequences=np.asarray(sequences, dtype=np.float32))

    def select_features(self, names: Iterable[str]) -> Dataset:
        """Keeps only the named feature columns.

        Args:
            names: Feature names to keep. The columns stay in the order
                of `feature_names`, not of `names`.

        Returns:
            A new Dataset with the selected columns.

        Raises:
            TypeError: If `names` is a single string.
            ValueError: If a name is not in `feature_names`, or no column
                is left.
        """
        wanted = _name_set(names)
        missing = wanted - set(self.feature_names)
        if missing:
            raise ValueError(f"Unknown features: {sorted(missing)}")
        keep = [i for i, name in enumerate(self.feature_names)
                if name in wanted]
        return self._take_columns(keep)

    def drop_features(self, names: Iterable[str]) -> Dataset:
        """Removes the named feature columns.

        Args:
            names: Feature names to remove. Names that are not in
                `feature_names` are ignored.

        Returns:
            A new Dataset without the dropped columns.

        Raises:
            TypeError: If `names` is a single string.
            ValueError: If every column would be removed.
        """
        dropped = _name_set(names)
        keep = [i for i, name in enumerate(self.feature_names)
                if name not in dropped]
        return self._take_columns(keep)

    def as_test_only(self) -> Dataset:
        """Puts every window into the test split.

        Returns:
            A new Dataset whose `test_idx` covers all windows and whose
            `train_idx` and `val_idx` are empty.
        """
        return dataclasses.replace(
            self,
            test_idx=np.arange(len(self.sequences)),
            train_idx=np.array([], dtype=np.int64),
            val_idx=np.array([], dtype=np.int64),
        )

    def _take_columns(self, keep: list[int]) -> Dataset:
        if not keep:
            raise ValueError("A dataset needs at least one feature column.")
        return dataclasses.replace(
            self,
            sequences=np.ascontiguousarray(self.sequences[:, :, keep]),
            feature_names=[self.feature_names[i] for i in keep],
        )


def _name_set(names: Iterable[str]) -> set[str]:
    """Returns the names as a set; a bare string is a mistake, not a name."""
    if isinstance(names, str):
        raise TypeError(
            f"Expected a collection of feature names, got the string "
            f"{names!r}; pass [{names!r}].")
    return set(names)
