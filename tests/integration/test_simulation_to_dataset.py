"""Simulator variants feeding the dataset builder."""
import numpy as np

from src.application.generate_dataset import (
    GenerateDatasetCommand,
    GenerateDatasetUseCase,
)
from src.data import features
from src.data.scenarios import build_configs
from src.simulation.simulator import QuTipSimulator
from src.simulation.simulator import QuTipSimulator


def test_correlator_base_features_equal_the_plain_simulator():
    """R23: the correlators leave the base features unchanged."""
    groups = build_configs(n_per_scenario=3, seed=0)
    cmd = dict(configs=groups["D"] + groups["E"], window_length=10,
               horizon=1.0, samples_per_trajectory=3, seed=4,
               dataset_name="r23_test")
    plain = GenerateDatasetUseCase(QuTipSimulator(), store=None).execute(
        GenerateDatasetCommand(**cmd))
    full = GenerateDatasetUseCase(
        QuTipSimulator("correlators"), store=None).execute(
        GenerateDatasetCommand(**cmd))
    base = features.select(full, "base")
    assert base.feature_names == plain.feature_names
    np.testing.assert_array_equal(base.sequences, plain.sequences)
    np.testing.assert_array_equal(base.remaining_time, plain.remaining_time)
