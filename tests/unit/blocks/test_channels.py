"""blocks.channels: jump channels, rate modifiers, Kondo bath, mesolve."""
import numpy as np
import pytest
import qutip as qt

from src.blocks import channels, evolve, system
from src.physics import kondo_bath, lindblad_systems, simulate_qutip

NAN = float("nan")
INF = float("inf")
ONE = system.single_qubit(1.0)
TWO = system.two_qubit(0.8, "XXZ")
COEFFS = np.array([0.2, 0.5, 0.1])
T_MAX = 10.0


def _unit_channels(sys=TWO):
    return channels.jump(sys, "sigma_minus", 1.0)


# --- Negative cases ---


@pytest.mark.parametrize("rate", [-0.1, NAN, INF, -INF])
def test_channel_invalid_constant_rate_raises_value_error(rate):
    with pytest.raises(ValueError, match="constant rate must be"):
        channels.Channel(qt.sigmam(), rate)


@pytest.mark.parametrize("gamma", [-0.1, NAN, INF])
def test_jump_invalid_rate_raises_value_error(gamma):
    with pytest.raises(ValueError, match="constant rate must be"):
        channels.jump(ONE, "sigma_minus", gamma)


@pytest.mark.parametrize("op_type", ["sigma_plus", "", "sigma_x", None])
@pytest.mark.parametrize("sys", [ONE, TWO])
def test_jump_unknown_operator_raises_value_error(sys, op_type):
    with pytest.raises(ValueError, match="Unknown jump operator"):
        channels.jump(sys, op_type, 0.1)


def test_jump_of_three_qubits_raises_value_error():
    three = system.System(
        qt.tensor(*[qt.sigmaz()] * 3),
        tuple(kondo_bath.chain_spin_operators(3)), np.arange(3.0), 3)
    with pytest.raises(ValueError, match="one or two qubits"):
        channels.jump(three, "sigma_z", 0.1)


def test_bernstein_profile_empty_coefficients_raise_value_error():
    with pytest.raises(ValueError, match="must not be empty"):
        channels.bernstein_profile(_unit_channels(), [], T_MAX)


@pytest.mark.parametrize("t_max", [0.0, -1.0, NAN, INF])
def test_bernstein_profile_invalid_t_max_raises_value_error(t_max):
    with pytest.raises(ValueError, match="t_max must be"):
        channels.bernstein_profile(_unit_channels(), COEFFS, t_max)


@pytest.mark.parametrize("s", [-0.1, NAN, INF, -INF])
def test_ou_modulation_invalid_strength_raises_value_error(s):
    with pytest.raises(ValueError, match="s must be"):
        channels.ou_modulation(
            _unit_channels(), s, 1.0, T_MAX, np.random.default_rng(0))


@pytest.mark.parametrize("tau", [0.0, -1.0, NAN, INF, None])
def test_ou_modulation_invalid_correlation_time_raises_value_error(tau):
    with pytest.raises(ValueError, match="tau must be"):
        channels.ou_modulation(
            _unit_channels(), 0.3, tau, T_MAX, np.random.default_rng(0))


@pytest.mark.parametrize("t_max", [0.0, -1.0, NAN])
def test_ou_modulation_invalid_t_max_raises_value_error(t_max):
    with pytest.raises(ValueError, match="t_max must be"):
        channels.ou_modulation(
            _unit_channels(), 0.3, 1.0, t_max, np.random.default_rng(0))


@pytest.mark.parametrize("kwargs, match", [
    (dict(rho_J=1.0), "rho_J"), (dict(rho_J=-0.1), "rho_J"),
    (dict(kT=0.0), "kT"), (dict(k_F=0.0), "k_F"), (dict(dim=4), "dim")])
def test_kondo_bath_invalid_parameters_raise_value_error(kwargs, match):
    params = dict(rho_J=0.1, kT=0.5, k_F=1.0, dim=3)
    params.update(kwargs)
    with pytest.raises(ValueError, match=match):
        channels.kondo_bath(TWO, **params)


# --- Positive cases: jump ---


@pytest.mark.parametrize("op_type", ["sigma_minus", "sigma_z"])
def test_two_qubit_jump_equals_the_legacy_collapse_list(op_type):
    legacy = lindblad_systems.TwoQubitSystem(
        J=0.8, gamma_const=0.3, jump_operators=[op_type])
    chans = channels.jump(TWO, op_type, 0.3)
    assert len(chans) == len(legacy.L_list) == 2
    for channel, op in zip(chans, legacy.L_list):
        np.testing.assert_array_equal(channel.operator.full(), op.full())
        assert channel.rate == 0.3


@pytest.mark.parametrize("op_type", ["sigma_minus", "sigma_z"])
def test_single_qubit_jump_equals_the_legacy_collapse_list(op_type):
    legacy = lindblad_systems.SingleQubitSystem(
        gamma_const=0.3, jump_operators=[op_type])
    (channel,) = channels.jump(ONE, op_type, 0.3)
    np.testing.assert_array_equal(
        channel.operator.full(), legacy.L_list[0].full())


# --- Positive cases: modifiers ---


def test_bernstein_profile_rate_is_the_legacy_gamma():
    chans = channels.bernstein_profile(_unit_channels(), COEFFS, T_MAX)
    for t in (0.0, 1.3, 5.0, 10.0, 12.0):
        expected = lindblad_systems.bernstein_gamma(t, COEFFS, T_MAX)
        assert all(c.rate(t) == expected for c in chans)


def test_bernstein_profile_multiplies_a_constant_rate():
    chans = channels.bernstein_profile(
        channels.jump(TWO, "sigma_z", 2.0), COEFFS, T_MAX)
    assert chans[0].rate(3.0) == 2.0 * lindblad_systems.bernstein_gamma(
        3.0, COEFFS, T_MAX)


def test_modifiers_do_not_change_their_input():
    base = _unit_channels()
    channels.bernstein_profile(base, COEFFS, T_MAX)
    channels.ou_modulation(base, 0.3, 1.0, T_MAX, np.random.default_rng(0))
    assert [c.rate for c in base] == [1.0, 1.0]


def _reference_factor(s, tau, t_max, rng):
    """The OU factor, written out independently of the block."""
    step = channels.OU_GRID_DT
    n = int(round(t_max / step)) + 1
    eta = rng.standard_normal(n)
    a = np.exp(-step / tau)
    x = np.empty(n)
    x[0] = eta[0]
    for k in range(1, n):
        x[k] = a * x[k - 1] + np.sqrt(1.0 - a * a) * eta[k]
    grid = np.linspace(0.0, t_max, n)
    factor = np.exp(s * x - 0.5 * s * s)
    return lambda t: float(np.interp(t, grid, factor))


@pytest.mark.parametrize("s, tau", [(0.4, 1.0), (0.6, 0.5)])
def test_ou_modulation_follows_the_documented_arithmetic(s, tau):
    t_max = 6.0
    expected = _reference_factor(s, tau, t_max, np.random.default_rng(8))
    base = channels.bernstein_profile(
        _unit_channels(), (1.0,), t_max)
    block_rate = channels.ou_modulation(
        base, s, tau, t_max, np.random.default_rng(8))[0].rate
    for t in np.linspace(0.0, t_max, 37):
        assert block_rate(t) == 1.0 * expected(t)


def test_ou_modulation_without_a_fixed_strength_draws_it_after_the_path():
    rng = np.random.default_rng(8)
    out = channels.ou_modulation(
        channels.bernstein_profile(_unit_channels(), (1.0,), 3.0),
        None, None, 3.0, rng)
    ref = np.random.default_rng(8)
    n = int(round(3.0 / channels.OU_GRID_DT)) + 1
    ref.standard_normal(n)
    s = float(ref.uniform(0.0, 0.6))
    tau = float(ref.choice([0.5, 2.0]))
    expected = _reference_factor(s, tau, 3.0, np.random.default_rng(8))
    assert out[0].rate(1.7) == expected(1.7)


def test_ou_modulation_can_leave_constant_rates_alone():
    mixed = channels.jump(ONE, "sigma_z", 2.0) + channels.bernstein_profile(
        channels.jump(ONE, "sigma_minus", 1.0), (1.0,), 3.0)
    rng = np.random.default_rng(1)
    out = channels.ou_modulation(
        mixed, 0.5, 1.0, 3.0, rng, modulate_constant=False)
    assert out[0] is mixed[0] and out[0].rate == 2.0
    assert callable(out[1].rate)
    n = int(round(3.0 / channels.OU_GRID_DT)) + 1
    ref = np.random.default_rng(1)
    ref.standard_normal(n)
    assert rng.standard_normal() == ref.standard_normal()


def test_ou_modulation_consumes_the_generator_in_a_fixed_order():
    rng = np.random.default_rng(4)
    channels.ou_modulation(_unit_channels(), None, None, T_MAX, rng)
    n = int(round(T_MAX / channels.OU_GRID_DT)) + 1
    ref = np.random.default_rng(4)
    ref.standard_normal(n)
    ref.uniform(0.0, 0.6)
    ref.choice([0.5, 2.0])
    assert rng.standard_normal() == ref.standard_normal()


def test_ou_modulation_zero_strength_keeps_the_rates_but_draws():
    rng = np.random.default_rng(4)
    base = _unit_channels()
    out = channels.ou_modulation(base, 0.0, None, T_MAX, rng)
    assert [c.rate for c in out] == [1.0, 1.0]
    ref = np.random.default_rng(4)
    ref.standard_normal(int(round(T_MAX / channels.OU_GRID_DT)) + 1)
    assert rng.standard_normal() == ref.standard_normal()


def test_ou_modulation_factor_is_positive_with_mean_near_one():
    ts = np.linspace(0.0, T_MAX, 201)
    rng = np.random.default_rng(3)
    values = [
        [c.rate(t) for t in ts]
        for c in (channels.ou_modulation(
            _unit_channels(), 0.6, 0.5, T_MAX, rng)[0]
            for _ in range(100))]
    assert np.min(values) > 0
    assert abs(np.mean(values) - 1.0) < 0.05


def test_ou_modulation_modulates_a_constant_rate_too():
    out = channels.ou_modulation(
        channels.jump(ONE, "sigma_z", 2.0), 0.5, 1.0, T_MAX,
        np.random.default_rng(1))
    values = [out[0].rate(t) for t in np.linspace(0.0, T_MAX, 20)]
    assert np.std(values) > 0 and min(values) > 0


# --- Positive cases: Kondo ---


@pytest.mark.parametrize("dim", [1, 2, 3])
def test_kondo_channels_equal_the_collapse_operators(dim):
    chans = channels.kondo_bath(TWO, rho_J=0.1, kT=0.5, k_F=1.3, dim=dim)
    ops = kondo_bath.kondo_collapse_operators(
        TWO.H, TWO.spin_ops, TWO.positions, rho_J=0.1, kT=0.5, k_F=1.3,
        dim=dim, tol=1e-9)
    assert len(chans) == len(ops) > 0
    for channel, op in zip(chans, ops):
        np.testing.assert_array_equal(channel.operator.full(), op.full())
        assert channel.rate == 1.0


@pytest.mark.parametrize("spacing", [0.0, 1.7])
def test_kondo_spacing_scales_the_site_positions(spacing):
    chans = channels.kondo_bath(
        TWO, rho_J=0.1, kT=0.2, k_F=1.0, dim=2, spacing=spacing)
    ops = kondo_bath.kondo_collapse_operators(
        TWO.H, kondo_bath.chain_spin_operators(2), [0.0, spacing],
        rho_J=0.1, kT=0.2, k_F=1.0, dim=2, tol=1e-9)
    assert len(chans) == len(ops)
    for channel, op in zip(chans, ops):
        np.testing.assert_array_equal(channel.operator.full(), op.full())


@pytest.mark.parametrize("spacing", [-1.0, NAN, INF])
def test_kondo_invalid_spacing_raises_value_error(spacing):
    with pytest.raises(ValueError, match="spacing must be"):
        channels.kondo_bath(
            TWO, rho_J=0.1, kT=0.2, k_F=1.0, dim=2, spacing=spacing)


def test_single_qubit_kondo_bath_relaxes_to_the_thermal_state():
    sys_ = system.single_qubit(1.0)
    chans = channels.kondo_bath(sys_, rho_J=0.3, kT=0.5, k_F=1.0, dim=3)
    final = evolve.master_equation(
        sys_, chans, qt.fock_dm(2, 0), np.array([0.0, 2000.0])).rho[-1]
    gibbs = (-sys_.H / 0.5).expm().full()
    gibbs = gibbs / np.trace(gibbs)
    assert np.linalg.norm(final - gibbs) < 1e-3


# --- Positive cases: to_mesolve ---


def test_to_mesolve_constant_rates_are_scaled_operators():
    legacy = lindblad_systems.TwoQubitSystem(
        J=0.8, gamma_const=0.3, jump_operators=["sigma_minus", "sigma_z"])
    expected = simulate_qutip._collapse_operators(legacy)
    chans = (channels.jump(TWO, "sigma_minus", 0.3)
             + channels.jump(TWO, "sigma_z", 0.3))
    out = channels.to_mesolve(chans)
    assert len(out) == len(expected) == 4
    for ours, theirs in zip(out, expected):
        np.testing.assert_array_equal(ours.full(), theirs.full())


def test_to_mesolve_time_dependent_rates_match_the_legacy_convention():
    legacy = lindblad_systems.TwoQubitSystem(
        J=0.8, gamma_coeffs=COEFFS, gamma_t_max=T_MAX)
    expected = simulate_qutip._collapse_operators(legacy)
    out = channels.to_mesolve(
        channels.bernstein_profile(_unit_channels(), COEFFS, T_MAX))
    assert len(out) == len(expected)
    for (op, coeff), (legacy_op, legacy_coeff) in zip(out, expected):
        np.testing.assert_array_equal(op.full(), legacy_op.full())
        for t in (0.0, 2.5, 9.0):
            assert coeff(t, {}) == legacy_coeff(t, {})


def test_to_mesolve_zero_time_dependent_rate_gives_exactly_zero():
    chans = channels.bernstein_profile(_unit_channels(), [0.0], T_MAX)
    (_, coeff), _ = channels.to_mesolve(chans)
    assert coeff(1.0, {}) == 0.0 and isinstance(coeff(1.0, {}), float)


def test_to_mesolve_channels_sharing_a_rate_share_a_coefficient():
    out = channels.to_mesolve(
        channels.bernstein_profile(_unit_channels(), COEFFS, T_MAX))
    assert out[0][1] is out[1][1]
