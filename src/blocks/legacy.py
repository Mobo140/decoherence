"""Recipes of the published simulators.

Every experiment so far names its physics with a ``SystemConfig`` and a
simulator alias (``plain``, ``correlators``, ``entanglement``,
``stochastic_gamma``, ``parity_sector``, ``kondo``). ``recipe_for`` turns
such a pair into a ``Recipe`` with legacy seeds, so that the old and the new
path give the same numbers bit for bit: the initial state comes from the
global NumPy generator, and the random blocks share one generator that the
caller owns.

Known details of the published behaviour that the recipes keep:

* stochastic_gamma draws its Ornstein-Uhlenbeck path for every trajectory
  but leaves a constant rate unmodulated;
* parity_sector prepares sectors on two qubits only, one qubit is plain;
* kondo replaces the jump channels and the rate gamma(t) by the bath, with
  the two sites at distance kF_d / k_F.
"""
from __future__ import annotations

import math
from collections.abc import Mapping

from src.blocks.recipe import Block, Recipe, block
from src.domain import value_objects
from src.physics import kondo_bath

# Constructor arguments of every alias, in the order of the old classes.
ALIAS_PARAMS: Mapping[str, tuple[str, ...]] = {
    "plain": (),
    "correlators": (),
    "entanglement": (),
    "stochastic_gamma": ("s", "tau", "seed"),
    "parity_sector": ("sector",),
    "kondo": ("rho_J", "kT", "kF_d", "dim"),
}

# Bohr frequencies closer than this are one frequency in the Kondo bath.
_KONDO_TOLERANCE = 1e-9


def validate_simulator(alias: str, params: Mapping[str, object]) -> None:
    """Checks a simulator alias and its constructor arguments.

    Args:
        alias: A key of `ALIAS_PARAMS`.
        params: The constructor arguments.

    Raises:
        KeyError: If `alias` is unknown; the message lists the valid ones.
        TypeError: If an argument is missing or unexpected.
        ValueError: If a value is out of range.
    """
    if alias not in ALIAS_PARAMS:
        raise KeyError(
            f"Unknown simulator {alias!r}. "
            f"Valid aliases: {sorted(ALIAS_PARAMS)}.")
    expected = ALIAS_PARAMS[alias]
    missing = [name for name in expected if name not in params]
    if missing:
        raise TypeError(
            f"Simulator {alias!r} is missing the required arguments "
            f"{missing}.")
    unexpected = sorted(set(params) - set(expected))
    if unexpected:
        raise TypeError(
            f"Simulator {alias!r} got unexpected arguments {unexpected}; "
            f"it takes {list(expected)}.")
    if alias == "stochastic_gamma":
        s, tau, seed = params["s"], params["tau"], params["seed"]
        if s is not None and not (math.isfinite(s) and s >= 0):
            raise ValueError(f"s must be finite and non-negative, got {s!r}.")
        if s and not (tau is not None and math.isfinite(tau) and tau > 0):
            raise ValueError(f"tau must be finite and positive, got {tau!r}.")
        if int(seed) != seed or seed < 0:
            raise ValueError(
                f"seed must be a non-negative integer, got {seed!r}.")
    elif alias == "parity_sector":
        if params["sector"] not in (+1, -1):
            raise ValueError("sector must be +1 or -1")
    elif alias == "kondo":
        kF_d = params["kF_d"]
        if not (math.isfinite(kF_d) and kF_d >= 0):
            raise ValueError(
                f"kF_d must be finite and non-negative, got {kF_d!r}.")
        kondo_bath.validate_bath_parameters(
            params["rho_J"], params["kT"], 1.0, params["dim"],
            tol=_KONDO_TOLERANCE)


def _system_block(config: value_objects.SystemConfig) -> Block:
    if config.n_qubits == value_objects.QubitCount.ONE:
        return block("system.single_qubit", omega=config.omega)
    return block(
        "system.two_qubit", J=config.J,
        interaction=config.interaction_type.value)


def _jump_blocks(config: value_objects.SystemConfig) -> list[Block]:
    dissipator = config.dissipator
    op_type = dissipator.operator_type.value
    if dissipator.gamma_const is not None:
        return [block(
            "channels.jump", op_type=op_type, gamma=dissipator.gamma_const)]
    return [
        block("channels.jump", op_type=op_type, gamma=1.0),
        block("channels.bernstein_profile",
              coeffs=tuple(dissipator.gamma_coeffs))]


def _target_block(config: value_objects.SystemConfig) -> Block:
    criteria = value_objects.DecoherenceCriterion
    criterion = config.decoherence_criterion
    threshold = config.decoherence_threshold
    if criterion == criteria.COHERENCE_ENERGY:
        return block("target.energy_basis_crossing", threshold=threshold)
    if criterion == criteria.ENTANGLEMENT_DEATH:
        return block("target.sudden_death")
    return block(
        "target.coherence_crossing", threshold=threshold,
        criterion=criterion.value)


def recipe_for(
    config: value_objects.SystemConfig,
    alias: str = "plain",
    **params: object,
) -> Recipe:
    """Builds the recipe of a config under a simulator alias.

    Args:
        config: The system, with its dissipator, criterion and time grid.
        alias: A key of `ALIAS_PARAMS`.
        **params: The constructor arguments of the alias, except that the
            ``seed`` of stochastic_gamma is not part of the recipe (it
            seeds the shared generator that the caller passes to
            `runner.run_with_generator`).

    Returns:
        A recipe with legacy seeds and no sampler.

    Raises:
        KeyError: If `alias` is unknown.
        TypeError: If an argument is missing or unexpected.
        ValueError: If a value is out of range.
    """
    validate_simulator(alias, params)
    two_qubits = config.n_qubits == value_objects.QubitCount.TWO
    channels = _jump_blocks(config)
    initial = block("initial.random_pure")
    extras = []
    if two_qubits and alias in ("correlators", "entanglement"):
        extras.append(block("observe.correlators"))
    if two_qubits and alias == "entanglement":
        extras.append(block("observe.concurrence"))

    if alias == "stochastic_gamma":
        channels.append(block(
            "channels.ou_modulation", s=params["s"], tau=params["tau"],
            modulate_constant=config.dissipator.gamma_const is None))
    elif alias == "parity_sector" and two_qubits:
        initial = block("initial.parity_sector", sector=params["sector"])
    elif alias == "kondo":
        channels = [block(
            "channels.kondo_bath", rho_J=params["rho_J"], kT=params["kT"],
            k_F=1.0, dim=params["dim"], spacing=params["kF_d"])]

    return Recipe(
        system=_system_block(config),
        channels=tuple(channels),
        initial=initial,
        observe=(block("observe.pauli_means"), *extras,
                 block("observe.purity"), block("observe.coherence_l1")),
        target=_target_block(config),
        t_max=config.t_max, dt=config.dt, seeds="legacy")
