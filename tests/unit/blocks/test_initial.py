"""blocks.initial: random pure states and parity-sector preparation."""
import numpy as np
import pytest
import qutip as qt

from src.blocks import initial, system
from src.data.scenarios import build_configs
from src.physics import lindblad_systems
from src.simulation import variants

ONE = system.single_qubit(1.0)
TWO = system.two_qubit(0.5, "XXZ")


# --- Negative cases ---


@pytest.mark.parametrize("sector", [0, 2, -2, None, "1"])
def test_parity_sector_other_than_plus_or_minus_one_raises_value_error(
        sector):
    with pytest.raises(ValueError, match=r"sector must be \+1 or -1"):
        initial.parity_sector(TWO, sector)


def test_parity_sector_of_a_one_qubit_system_raises_value_error():
    with pytest.raises(ValueError, match="two-qubit system"):
        initial.parity_sector(ONE, +1)


def test_parity_sector_that_cannot_be_reached_raises_runtime_error():
    # No normalised state has a projection norm above one.
    with pytest.raises(RuntimeError, match="inside the requested sector"):
        initial.parity_sector(
            TWO, +1, np.random.default_rng(0), min_norm=2.0, max_draws=3)


# --- Positive cases ---


def test_random_pure_without_a_generator_follows_the_global_seed():
    np.random.seed(7)
    first = initial.random_pure(TWO)
    np.random.seed(7)
    legacy = lindblad_systems.sample_random_pure_state(n_qubits=2)
    np.testing.assert_array_equal(first.full(), legacy.full())


def test_random_pure_with_a_generator_ignores_the_global_state():
    np.random.seed(1)
    a = initial.random_pure(ONE, np.random.default_rng(3))
    np.random.seed(2)
    b = initial.random_pure(ONE, np.random.default_rng(3))
    np.testing.assert_array_equal(a.full(), b.full())


def test_random_pure_with_a_generator_matches_the_seeded_legacy_draw():
    drawn = initial.random_pure(TWO, np.random.default_rng(11))
    legacy = lindblad_systems.sample_random_pure_state(2, seed=11)
    np.testing.assert_array_equal(drawn.full(), legacy.full())


@pytest.mark.parametrize("n", [1, 2])
def test_random_pure_is_a_pure_density_matrix(n):
    sys = ONE if n == 1 else TWO
    rho = initial.random_pure(sys, np.random.default_rng(0))
    assert rho.shape == sys.H.shape
    assert rho.tr() == pytest.approx(1.0)
    assert (rho * rho).tr() == pytest.approx(1.0)


@pytest.mark.parametrize("sector", [+1, -1])
def test_parity_sector_equals_the_legacy_variant_under_the_global_seed(
        sector):
    config = build_configs(1, 3)["E"][0]
    np.random.seed(5)
    legacy = variants.ParitySectorSimulator(sector)._initial_state(config)
    np.random.seed(5)
    block = initial.parity_sector(TWO, sector)
    np.testing.assert_array_equal(block.full(), legacy.full())


@pytest.mark.parametrize("sector", [+1, -1])
def test_parity_sector_is_projected_from_the_left_and_unit_norm(sector):
    rng = np.random.default_rng(2)
    projector = (qt.qeye([2, 2]) + sector * initial.PARITY) / 2
    for _ in range(5):
        prepared = initial.parity_sector(TWO, sector, rng)
        assert prepared.norm() == pytest.approx(1.0)
        assert (projector * prepared - prepared).norm() < 1e-9


def test_parity_sector_redraws_a_state_with_a_vanishing_weight(monkeypatch):
    states = iter([
        qt.ket2dm(qt.basis([2, 2], [0, 1])),    # parity -1
        qt.ket2dm(qt.basis([2, 2], [0, 0])),    # parity +1
    ])
    monkeypatch.setattr(
        initial, "random_pure", lambda system, rng=None: next(states))
    prepared = initial.parity_sector(TWO, +1)
    assert prepared.full()[0, 0] == pytest.approx(1.0)
    assert next(states, None) is None
