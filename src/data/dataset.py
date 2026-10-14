"""The labelled dataset passed to training and backtesting."""
from __future__ import annotations

import dataclasses
from collections.abc import Iterable
from typing import Any

import numpy as np


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
        interaction_type_codes: System type per window (4 classes):
            0.0 = 1-qubit sigma_-, 1.0 = 1-qubit sigma_z, 2.0 = 2-qubit
            XXZ, 3.0 = 2-qubit TFIM. Empty if unknown.
        dissipator_type_codes: 0.0 = sigma_- (amplitude damping), 1.0 =
            sigma_z (dephasing) per window. Empty if unknown.
        J_values: Coupling J per window; 0.0 for one qubit. Empty if
            unknown.
        censored: True where t_decoh_abs is only a lower bound (the
            threshold was not crossed by t_max). Empty if unknown.
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
    interaction_type_codes: np.ndarray = dataclasses.field(
        default_factory=lambda: np.zeros(0, dtype=np.float32))
    dissipator_type_codes: np.ndarray = dataclasses.field(
        default_factory=lambda: np.zeros(0, dtype=np.float32))
    J_values: np.ndarray = dataclasses.field(
        default_factory=lambda: np.zeros(0, dtype=np.float32))
    censored: np.ndarray = dataclasses.field(
        default_factory=lambda: np.zeros(0, dtype=bool))

    def with_sequences(self, sequences: np.ndarray) -> Dataset:
        """Returns the dataset with other input windows.

        Args:
            sequences: New windows, shape (N, window_length, n_features).
                Converted to float32. The feature names are kept, so the
                last axis must still match them.

        Returns:
            A new Dataset; labels, indices and codes are unchanged.

        Raises:
            ValueError: If `sequences` has a different number of windows.
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
            ValueError: If a name is not in `feature_names`.
        """
        wanted = set(names)
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
        """
        dropped = set(names)
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
        return dataclasses.replace(
            self,
            sequences=np.ascontiguousarray(self.sequences[:, :, keep]),
            feature_names=[self.feature_names[i] for i in keep],
        )
