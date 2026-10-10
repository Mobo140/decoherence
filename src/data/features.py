"""Feature sets: which observables a model sees.

`FEATURE_SETS` maps an alias to the columns it removes from the full set
recorded by `EntanglementSimulator`. A column that a dataset does not have
is ignored, so "base" also applies to data without concurrence.

    base       the base observables only
    corr       base plus the nine two-body Pauli correlators
    corr_conc  base plus the correlators and the concurrence
"""
from __future__ import annotations

import types
from collections.abc import Mapping

from src.data import dataset as dataset_lib
from src.simulation import variants

FEATURE_SETS: Mapping[str, frozenset[str]] = types.MappingProxyType({
    "base": frozenset(variants.CORR) | {"concurrence"},
    "corr": frozenset({"concurrence"}),
    "corr_conc": frozenset(),
})


def select(dataset: dataset_lib.Dataset, name: str) -> dataset_lib.Dataset:
    """Restricts a dataset to the columns of a feature set.

    Args:
        dataset: Dataset with the full feature columns.
        name: Alias in `FEATURE_SETS`.

    Returns:
        A new dataset without the columns the set removes.

    Raises:
        KeyError: If `name` is not an alias in `FEATURE_SETS`.
    """
    return dataset.drop_features(FEATURE_SETS[name])
