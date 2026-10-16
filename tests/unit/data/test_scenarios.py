"""Scenario generators: fixed-coupling groups."""
import numpy as np

from src.data import scenarios
from src.domain import value_objects


def test_fixed_coupling_configs_share_the_coupling():
    cfgs = scenarios.fixed_coupling_configs(
        7, 1.3, np.random.default_rng(0),
        interaction=value_objects.InteractionType.XXZ)
    assert len(cfgs) == 7 and {c.J for c in cfgs} == {1.3}
    assert {c.interaction_type for c in cfgs} == {
        value_objects.InteractionType.XXZ}
