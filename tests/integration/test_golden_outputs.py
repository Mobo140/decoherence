"""Behaviour regression: the recorded fingerprints must be reproduced exactly.

`scripts.golden_outputs` simulates under every criterion, builds datasets,
applies the noise models, trains every predictor for a few epochs and loads
stored checkpoints and cached trajectories. Its fingerprints are compared
with `scripts/golden_outputs.json` bit for bit, so any change of arithmetic
in the simulation, data or training layers shows up here.
"""
from __future__ import annotations

import json

import numpy as np
import pytest

from scripts import golden_outputs


@pytest.fixture
def restore_global_numpy_state():
    """`compute` reseeds numpy's global generator; put it back afterwards."""
    state = np.random.get_state()
    yield
    np.random.set_state(state)


def test_golden_outputs_match_the_recorded_fingerprints(
    restore_global_numpy_state,
):
    recorded = json.loads(golden_outputs.GOLDEN.read_text())
    computed = json.loads(json.dumps(golden_outputs.compute()))

    assert sorted(computed) == sorted(recorded)
    differing = [name for name in recorded if computed[name] != recorded[name]]
    assert not differing, f"fingerprints changed: {differing}"
