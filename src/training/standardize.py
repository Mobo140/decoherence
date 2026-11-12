"""Standardisation of model inputs and targets.

A `Standardizer` is fitted on the training rows only and applied unchanged
to validation rows and to every window seen at prediction time.
"""
from __future__ import annotations

import dataclasses

import numpy as np

# Added to the standard deviation so that a constant column does not divide
# by zero.
EPS = 1e-8


@dataclasses.dataclass
class Standardizer:
    """Affine map x -> (x - mean) / std.

    Works elementwise on NumPy arrays, torch tensors and Python floats. The
    defaults (mean 0, std 1) are the identity, which is also what a
    checkpoint without these statistics gets.

    Attributes:
        mean: Per-channel array, or a float for a scalar quantity.
        std: Same shape as `mean`; never zero.
    """

    mean: np.ndarray | float = 0.0
    std: np.ndarray | float = 1.0

    @classmethod
    def fit(
        cls,
        values: np.ndarray,
        axis: int | tuple[int, ...] | None = None,
        min_std: float | None = None,
    ) -> Standardizer:
        """Computes the mean and standard deviation of training values.

        Args:
            values: Training values.
            axis: Axes to reduce. None reduces over everything and stores
                Python floats; a tuple such as (0, 1) keeps the remaining
                channel axis (windows of shape (N, T, F) -> F channels).
            min_std: If given, the standard deviation is floored at this
                value instead of having `EPS` added. Use it for a quantity
                that is nearly constant in the training rows, where
                dividing by its tiny spread would amplify noise. Only for
                `axis=None`.

        Returns:
            The fitted standardizer.
        """
        mean = values.mean(axis=axis)
        std = values.std(axis=axis)
        if axis is None:
            mean, std = float(mean), float(std)
        if min_std is None:
            std = std + EPS
        else:
            std = max(std, min_std)
        return cls(mean=mean, std=std)

    def apply(self, values):
        """Returns (values - mean) / std."""
        return (values - self.mean) / self.std
