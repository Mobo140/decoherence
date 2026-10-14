"""Cuts simulated trajectories into labelled windows.

`build_dataset` is the entry point. It splits the trajectories into
train/val/test first, then samples windows that end before decoherence, so
no trajectory contributes to more than one split.
"""
from __future__ import annotations

import dataclasses
from collections.abc import Sequence

import numpy as np

from src.data import dataset as dataset_lib
from src.domain import value_objects

# Windows are padded with zeros before the last PHYSICS_COLUMNS columns
# (coherence_l1 and purity) so that those stay at a fixed position.
PHYSICS_COLUMNS = 2


@dataclasses.dataclass
class _Windows:
    """Per-window columns collected from the trajectories, one list each."""

    sequences: list[np.ndarray] = dataclasses.field(default_factory=list)
    t_obs: list[float] = dataclasses.field(default_factory=list)
    t_decoh_abs: list[float] = dataclasses.field(default_factory=list)
    remaining: list[float] = dataclasses.field(default_factory=list)
    risk: list[int] = dataclasses.field(default_factory=list)
    split: list[str] = dataclasses.field(default_factory=list)
    interaction_codes: list[float] = dataclasses.field(default_factory=list)
    dissipator_codes: list[float] = dataclasses.field(default_factory=list)
    coupling: list[float] = dataclasses.field(default_factory=list)
    censored: list[bool] = dataclasses.field(default_factory=list)


def assign_splits(
    n_trajectories: int,
    train_ratio: float,
    val_ratio: float,
    rng: np.random.Generator,
) -> list[str]:
    """Assigns every trajectory to the train, val or test split.

    Draws one permutation from `rng`; the first `train_ratio` share of it
    is train, the next `val_ratio` share is val and the rest is test.

    Args:
        n_trajectories: Number of trajectories.
        train_ratio: Share of trajectories in the train split.
        val_ratio: Share of trajectories in the val split.
        rng: Source of randomness.

    Returns:
        One of "train", "val" or "test" per trajectory.
    """
    order = rng.permutation(n_trajectories)
    n_train = int(n_trajectories * train_ratio)
    n_val = int(n_trajectories * val_ratio)
    split = ["test"] * n_trajectories
    for i in order[:n_train].tolist():
        split[i] = "train"
    for i in order[n_train:n_train + n_val].tolist():
        split[i] = "val"
    return split


def context_codes(
    config: value_objects.SystemConfig,
) -> tuple[float, float, float]:
    """Encodes the physics regime of a system as model input scalars.

    The four system types get distinct interaction codes, so that XXZ and
    one-qubit sigma_- do not look identical to the model.

    Args:
        config: The system configuration of a trajectory.

    Returns:
        A tuple (interaction code, dissipator code, coupling J). The
        interaction code is 0 for one qubit with sigma_-, 1 for one qubit
        with sigma_z, 2 for XXZ and 3 for TFIM. The dissipator code is 1
        for sigma_z and 0 otherwise. J is 0 for one qubit.
    """
    is_sigma_z = (
        config.dissipator.operator_type
        == value_objects.DissipatorType.SIGMA_Z)
    if config.n_qubits == value_objects.QubitCount.TWO:
        is_tfim = (
            config.interaction_type == value_objects.InteractionType.TFIM)
        interaction = 3.0 if is_tfim else 2.0
        coupling = float(config.J)
    else:
        interaction = 1.0 if is_sigma_z else 0.0
        coupling = 0.0
    return interaction, 1.0 if is_sigma_z else 0.0, coupling


def valid_window_ends(
    times: np.ndarray,
    t_decoh: float,
    window_length: int,
) -> np.ndarray:
    """Lists the time indices at which a window may end.

    A window ends strictly before decoherence, so the remaining time is
    positive, and it needs `window_length` earlier steps.

    Args:
        times: Time grid of the trajectory.
        t_decoh: Decoherence time of the trajectory.
        window_length: Number of steps in a window.

    Returns:
        Increasing indices into `times`; empty if T2 is shorter than a
        window.
    """
    ends = np.where(times < t_decoh)[0]
    return ends[ends >= window_length]


def sample_window_ends(
    valid_ends: np.ndarray,
    n_samples: int,
    min_gap: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Draws window end indices without replacement.

    Args:
        valid_ends: Candidates from `valid_window_ends`.
        n_samples: Number of windows wanted; at most `len(valid_ends)`
            are returned.
        min_gap: Minimum distance in steps between chosen ends, to reduce
            overlap within a trajectory. 0 switches the constraint off.
        rng: Source of randomness.

    Returns:
        The chosen end indices. With a gap, fewer than `n_samples` may be
        found.
    """
    n_sample = min(n_samples, len(valid_ends))
    if min_gap > 0 and len(valid_ends) > 1:
        # Greedy: walk a random order and keep ends far from those kept.
        selected: list[int] = []
        for idx in rng.permutation(valid_ends):
            if all(abs(int(idx) - int(s)) >= min_gap for s in selected):
                selected.append(idx)
            if len(selected) == n_sample:
                break
        return np.array(selected)
    return rng.choice(valid_ends, size=n_sample, replace=False)


def pad_features(sequences: list[np.ndarray]) -> list[np.ndarray]:
    """Pads windows to a common number of features.

    This is needed when one-qubit and two-qubit systems are mixed. Zero
    columns go before the last `PHYSICS_COLUMNS` columns.

    Args:
        sequences: Windows of shape (window_length, n_features_i).

    Returns:
        Windows that all have the largest n_features.
    """
    width = max(s.shape[-1] for s in sequences)
    if width == min(s.shape[-1] for s in sequences):
        return sequences
    padded = []
    for window in sequences:
        n_pad = width - window.shape[-1]
        if n_pad > 0:
            window = np.concatenate(
                [window[:, :-PHYSICS_COLUMNS],
                 np.zeros((window.shape[0], n_pad), dtype=window.dtype),
                 window[:, -PHYSICS_COLUMNS:]],
                axis=-1,
            )
        padded.append(window)
    return padded


def _add_trajectory_windows(
    rows: _Windows,
    trajectory: value_objects.TrajectoryResult,
    ends: np.ndarray,
    split: str,
    window_length: int,
    horizon: float,
    censored: bool,
) -> None:
    """Appends one labelled window per end index to `rows`."""
    interaction, dissipator, coupling = context_codes(
        trajectory.system_config)
    features = trajectory.feature_matrix
    for end in ends:
        window = features[end - window_length:end]
        t_obs = float(trajectory.times[end])
        # Lower bound if the trajectory is censored.
        remaining = trajectory.t_decoh - t_obs
        # A censored window near the end may or may not be close to the
        # event: label 0 and let training mask it out of the risk loss.
        risk = 0 if censored else int(remaining <= horizon)
        rows.sequences.append(window)
        rows.t_obs.append(t_obs)
        rows.t_decoh_abs.append(trajectory.t_decoh)
        rows.remaining.append(remaining)
        rows.risk.append(risk)
        rows.split.append(split)
        rows.interaction_codes.append(interaction)
        rows.dissipator_codes.append(dissipator)
        rows.coupling.append(coupling)
        rows.censored.append(censored)


def build_dataset(
    trajectories: Sequence[value_objects.TrajectoryResult],
    *,
    window_length: int,
    horizon: float,
    samples_per_trajectory: int,
    train_ratio: float,
    val_ratio: float,
    min_window_gap: int,
    include_censored: bool,
    seed: int | None,
    rng: np.random.Generator,
) -> dataset_lib.Dataset:
    """Cuts trajectories into windows and labels them.

    The split is made per trajectory, so windows of one trajectory never
    appear in two splits. Randomness is drawn from `rng` in a fixed order:
    one permutation for the split, then per trajectory the window ends.

    Args:
        trajectories: Simulated trajectories.
        window_length: Number of time steps in a window.
        horizon: A window is at risk if decoherence follows within this
            time.
        samples_per_trajectory: Windows drawn from each trajectory.
        train_ratio: Share of trajectories in the train split.
        val_ratio: Share of trajectories in the val split.
        min_window_gap: Minimum gap in steps between windows of one
            trajectory; 0 switches it off.
        include_censored: Keep trajectories whose threshold was not crossed
            by t_max. Their remaining time is a lower bound.
        seed: Recorded in the metadata only.
        rng: Source of randomness.

    Returns:
        The dataset.

    Raises:
        RuntimeError: If no trajectory yields a pre-decoherence window.
    """
    split_of = assign_splits(len(trajectories), train_ratio, val_ratio, rng)
    rows = _Windows()
    n_skipped = 0
    n_censored = 0
    for i, traj in enumerate(trajectories):
        censored = bool(getattr(traj, "censored", False))
        n_censored += censored
        if censored and not include_censored:
            continue
        if len(traj.times) <= window_length:
            continue  # trajectory too short
        valid = valid_window_ends(traj.times, traj.t_decoh, window_length)
        if len(valid) == 0:
            n_skipped += 1
            continue  # T2 shorter than a window
        ends = sample_window_ends(
            valid, samples_per_trajectory, min_window_gap, rng)
        _add_trajectory_windows(
            rows, traj, ends, split_of[i], window_length, horizon, censored)
    if not rows.sequences:
        raise RuntimeError(
            "No valid pre-decoherence windows found. "
            "Reduce window_length or increase t_max / T2 (lower gamma).")
    return _assemble(
        trajectories, rows, n_skipped, n_censored, window_length, horizon,
        include_censored, seed)


def _assemble(
    trajectories: Sequence[value_objects.TrajectoryResult],
    rows: _Windows,
    n_skipped: int,
    n_censored: int,
    window_length: int,
    horizon: float,
    include_censored: bool,
    seed: int | None,
) -> dataset_lib.Dataset:
    """Converts the collected columns into the arrays of a Dataset."""
    sequences = np.array(pad_features(rows.sequences), dtype=np.float32)
    split = np.array(rows.split)
    censored = np.array(rows.censored, dtype=bool)
    train_idx = np.where(split == "train")[0]
    val_idx = np.where(split == "val")[0]
    test_idx = np.where(split == "test")[0]
    # Take names from the trajectory with the most features, so that lookups
    # of coherence_l1 and purity stay valid after zero-padding.
    feature_names = max(
        (t.feature_names for t in trajectories), key=len, default=[])
    return dataset_lib.Dataset(
        sequences=sequences,
        t_obs=np.array(rows.t_obs, dtype=np.float32),
        t_decoh_abs=np.array(rows.t_decoh_abs, dtype=np.float32),
        remaining_time=np.array(rows.remaining, dtype=np.float32),
        risk_labels=np.array(rows.risk, dtype=np.float32),
        train_idx=train_idx,
        val_idx=val_idx,
        test_idx=test_idx,
        feature_names=feature_names,
        interaction_type_codes=np.array(
            rows.interaction_codes, dtype=np.float32),
        dissipator_type_codes=np.array(
            rows.dissipator_codes, dtype=np.float32),
        J_values=np.array(rows.coupling, dtype=np.float32),
        censored=censored,
        metadata={
            "n_trajectories": len(trajectories),
            "n_skipped": n_skipped,
            "n_censored": n_censored,
            "include_censored": include_censored,
            "n_censored_windows": int(censored.sum()),
            "window_length": window_length,
            "horizon": horizon,
            "seed": seed,
            "n_train": len(train_idx),
            "n_val": len(val_idx),
            "n_test": len(test_idx),
            "dt": _time_step(trajectories),
        },
    )


def _time_step(
    trajectories: Sequence[value_objects.TrajectoryResult],
) -> float:
    """Time step of the first trajectory (robust to linspace vs arange)."""
    if trajectories and len(trajectories[0].times) > 1:
        times = trajectories[0].times
        return float(times[1] - times[0])
    return 0.1
