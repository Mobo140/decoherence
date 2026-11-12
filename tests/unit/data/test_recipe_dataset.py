"""recipe_dataset.to_dataset: runs of a recipe become a Dataset."""
import dataclasses

import numpy as np
import pytest

from src.blocks import runner
from src.blocks.recipe import Recipe, block
from src.data import recipe_dataset

WINDOW = dict(
    window_length=5, horizon=1.0, samples_per_trajectory=2, train_ratio=0.6,
    val_ratio=0.2, min_window_gap=0, include_censored=True)


def _recipe(system_block, channel_blocks, **overrides):
    fields = dict(
        system=system_block, channels=channel_blocks,
        initial=block("initial.random_pure"),
        observe=(block("observe.pauli_means"), block("observe.purity"),
                 block("observe.coherence_l1")),
        target=block("target.coherence_crossing", threshold=0.5),
        t_max=4.0, dt=0.1)
    fields.update(overrides)
    return Recipe(**fields)


def _runs(recipe, n=6):
    return runner.run(recipe, n, seed=0)


ONE = _recipe(
    block("system.single_qubit", omega=1.0),
    (block("channels.jump", op_type="sigma_z", gamma=0.5),))
TWO = _recipe(
    block("system.two_qubit", J=0.7, interaction="TFIM"),
    (block("channels.jump", op_type="sigma_minus", gamma=0.5),))
KONDO_2Q = _recipe(
    block("system.two_qubit", J=0.7, interaction="XXZ"),
    (block("channels.kondo_bath", rho_J=0.1, kT=0.5, k_F=1.0, dim=2),))
KONDO_1Q = _recipe(
    block("system.single_qubit", omega=1.0),
    (block("channels.kondo_bath", rho_J=0.1, kT=0.5, k_F=1.0, dim=2),))


# --- Negative cases ---


def test_no_runs_raise_value_error():
    with pytest.raises(ValueError, match="no runs"):
        recipe_dataset.to_dataset(
            [], rng=np.random.default_rng(0), **WINDOW)


def test_run_of_an_unknown_system_raises_value_error():
    run_ = _runs(ONE, 1)[0]
    bad = dataclasses.replace(run_, system_fn="system.chain")
    with pytest.raises(ValueError, match="knows the systems"):
        recipe_dataset.to_dataset(
            [bad], rng=np.random.default_rng(0), **WINDOW)


def test_invalid_window_settings_raise_value_error():
    with pytest.raises(ValueError, match="window_length"):
        recipe_dataset.to_dataset(
            _runs(ONE), rng=np.random.default_rng(0),
            **{**WINDOW, "window_length": 0})


# --- Positive cases ---


@pytest.mark.parametrize("recipe, interaction, j", [
    (ONE, 1.0, 0.0), (TWO, 3.0, 0.7)])
def test_dataset_is_valid_and_carries_the_context_codes(
        recipe, interaction, j):
    ds = recipe_dataset.to_dataset(
        _runs(recipe), rng=np.random.default_rng(0), **WINDOW)
    n = len(ds.sequences)
    assert n > 0 and ds.sequences.shape[1] == 5
    assert ds.feature_names[-2:] == ["purity", "coherence_l1"]
    assert set(ds.context) == {"interaction_type", "dissipator_type", "J"}
    assert set(ds.context["interaction_type"]) == {interaction}
    assert set(ds.context["J"]) == {np.float32(j)}
    assert all(len(v) == n and v.dtype == np.float32
               for v in ds.context.values())
    assert len(ds.train_idx) + len(ds.val_idx) + len(ds.test_idx) == n


def test_dissipator_code_follows_the_jump_operator():
    ds = recipe_dataset.to_dataset(
        _runs(ONE), rng=np.random.default_rng(0), **WINDOW)
    assert set(ds.context["dissipator_type"]) == {1.0}


def test_codes_that_depend_on_a_missing_jump_channel_are_left_out():
    two = recipe_dataset.to_dataset(
        _runs(KONDO_2Q), rng=np.random.default_rng(0), **WINDOW)
    assert set(two.context) == {"interaction_type", "J"}
    assert set(two.context["interaction_type"]) == {2.0}
    one = recipe_dataset.to_dataset(
        _runs(KONDO_1Q), rng=np.random.default_rng(0), **WINDOW)
    assert set(one.context) == {"J"}


def test_same_rng_gives_the_same_dataset():
    runs = _runs(TWO)
    a = recipe_dataset.to_dataset(
        runs, rng=np.random.default_rng(3), **WINDOW)
    b = recipe_dataset.to_dataset(
        runs, rng=np.random.default_rng(3), **WINDOW)
    np.testing.assert_array_equal(a.sequences, b.sequences)
    np.testing.assert_array_equal(a.t_decoh_abs, b.t_decoh_abs)
