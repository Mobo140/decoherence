"""Domain ports are abstract: adapters must implement every method."""
from __future__ import annotations

import numpy as np
import pytest

from src.domain import ports, value_objects


class _Predictor(ports.IPredictor):
    """Smallest concrete predictor."""

    @property
    def is_trained(self) -> bool:
        return True

    def predict(self, window, t_obs, horizon):
        return value_objects.PredictionResult(1.0, 0.5, horizon)

    def train(self, dataset, **kwargs):
        return None


def test_a_complete_predictor_subclass_can_be_instantiated():
    predictor = _Predictor()
    assert predictor.is_trained
    result = predictor.predict(np.zeros((3, 2)), 1.0, 2.0)
    assert result.horizon == 2.0


# --- Negative cases ---


@pytest.mark.parametrize("port", [
    ports.ISimulator, ports.IPredictor, ports.ITrajectoryStore,
    ports.IModelStore, ports.ILossFunction])
def test_port_cannot_be_instantiated_directly(port):
    with pytest.raises(TypeError, match="abstract"):
        port()


def test_predictor_missing_a_method_cannot_be_instantiated():
    class Partial(ports.IPredictor):
        @property
        def is_trained(self) -> bool:
            return False

        def predict(self, window, t_obs, horizon):
            raise NotImplementedError

    with pytest.raises(TypeError, match="train"):
        Partial()


def test_loss_function_needs_call_and_name():
    class OnlyCall(ports.ILossFunction):
        def __call__(self, predictions, targets):
            return 0.0

    with pytest.raises(TypeError, match="name"):
        OnlyCall()
