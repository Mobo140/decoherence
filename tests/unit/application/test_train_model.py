"""TrainModelCommand: the knobs experiment scripts rely on."""
from __future__ import annotations

import inspect

from src.application.train_model import TrainModelCommand


def test_train_model_command_accepts_a_seed():
    assert "seed" in inspect.signature(TrainModelCommand).parameters
