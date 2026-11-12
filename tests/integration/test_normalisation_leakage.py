"""Normalisation must be fitted on training rows only.

Fitting the scaler on all rows would leak test information into the model.
"""
from __future__ import annotations

import numpy as np

from src.application.generate_dataset import (
    GenerateDatasetCommand,
    GenerateDatasetUseCase,
)
from src.data.dataset import Dataset
from src.models.lstm import LSTMPredictor
from tests import support


def _dataset(n_traj: int = 12) -> Dataset:
    # Distinct T2 per trajectory, as in the use-case split tests.
    trajs = [support.scripted_trajectory(2.0 + 0.5 * i)
             for i in range(n_traj)]
    uc = GenerateDatasetUseCase(support.ScriptedSimulator(trajs), store=None)
    return uc.execute(GenerateDatasetCommand(
        configs=[support.make_system_config(0.4, 12.0, 0.1)
                 for _ in range(n_traj)],
        window_length=10,
        horizon=1.0,
        samples_per_trajectory=4,
        seed=7,
        train_ratio=0.6,
        val_ratio=0.2,
    ))


def test_train_fits_normalisation_statistics_on_training_rows_only():
    ds = _dataset()
    predictor = LSTMPredictor(hidden_size=8, num_layers=1, dropout=0.0)
    predictor.train(ds, n_epochs=1, batch_size=8, verbose=False, seed=0)
    fitted_mean = np.array(
        predictor.feature_scaler.mean, dtype=np.float64).copy()

    # Corrupt only the rows the model must not have looked at.
    poisoned = Dataset(**{**ds.__dict__})
    poisoned.sequences = ds.sequences.copy()
    poisoned.sequences[ds.test_idx] += 1000.0

    predictor2 = LSTMPredictor(hidden_size=8, num_layers=1, dropout=0.0)
    predictor2.train(poisoned, n_epochs=1, batch_size=8, verbose=False, seed=0)
    again = np.array(predictor2.feature_scaler.mean, dtype=np.float64)

    assert np.allclose(fitted_mean, again), (
        "feature normalisation changed when only test rows changed"
    )
