"""Recipes: a simulation described as data, not as a class hierarchy.

A ``Block`` names a function of ``src.blocks`` and its parameters; a
``Recipe`` lists the blocks of every stage (system, channels, initial
state, observation, target) and how per-trajectory parameters are sampled.
Both are frozen and hashable, so a recipe can be a dictionary key and its
resolved parts can be hashed into a cache key. ``runner.run`` executes a
recipe.

The runner supplies the data a block takes from the pipeline, by parameter
name: ``system``, ``channels``, ``states``, ``obs``, ``rng`` and, for the
channel blocks, ``t_max``. A block therefore lists only its own parameters.
"""
from __future__ import annotations

import importlib
import inspect
import math
import numbers
from dataclasses import dataclass
from typing import Any, Callable

# Names the runner supplies; a block cannot be given them as parameters.
RESERVED = frozenset({"system", "channels", "states", "obs", "rng"})
# Parameters the runner supplies unless the block sets them.
OPTIONAL_INJECTED = frozenset({"t_max"})
SEED_MODES = ("spawn", "legacy")

_PRODUCER, _MODIFIER = "producer", "modifier"


@dataclass(frozen=True)
class Param:
    """A placeholder, replaced per trajectory by a sampled value.

    Attributes:
        name: Key in the dict that the sampler returned for the trajectory.
    """

    name: str

    def __post_init__(self) -> None:
        """Raises ValueError unless `name` is a non-empty string."""
        if not isinstance(self.name, str) or not self.name:
            raise ValueError(
                f"A Param needs a non-empty name, got {self.name!r}.")


def freeze(value: Any) -> Any:
    """Converts a parameter value to a hashable one.

    Args:
        value: A number, string, bool, None, `Param`, or a list or tuple of
            these, nested.

    Returns:
        The value with every list turned into a tuple.

    Raises:
        ValueError: If the value, or something inside it, is of another
            type (dicts, arrays, objects).
    """
    if value is None or isinstance(value, (str, bool, numbers.Real, Param)):
        return value
    if isinstance(value, (list, tuple)):
        return tuple(freeze(v) for v in value)
    raise ValueError(
        f"A block parameter must be a number, string, None, Param or a "
        f"tuple of them, got {type(value).__name__}: {value!r}.")


def resolve_fn(fn: str) -> Callable:
    """Finds the function a dotted name refers to.

    Args:
        fn: Name relative to ``src.blocks``, e.g. ``"channels.jump"``.

    Returns:
        The function.

    Raises:
        ValueError: If `fn` is not ``module.function`` or does not name a
            callable in a module of ``src.blocks``.
    """
    if not isinstance(fn, str):
        raise ValueError(f"fn must be a string, got {fn!r}.")
    module_name, _, attribute = fn.rpartition(".")
    if not module_name or not attribute:
        raise ValueError(
            f"fn must be 'module.function' relative to src.blocks, "
            f"got {fn!r}.")
    try:
        module = importlib.import_module(f"src.blocks.{module_name}")
    except ImportError as error:
        raise ValueError(
            f"fn {fn!r}: there is no module src.blocks.{module_name}."
        ) from error
    function = getattr(module, attribute, None)
    if not callable(function) or attribute.startswith("_"):
        raise ValueError(f"fn {fn!r} is not a public function of the blocks.")
    return function


@dataclass(frozen=True)
class Block:
    """A function of ``src.blocks`` with its parameters.

    Build blocks with `block`, which validates them.

    Attributes:
        fn: Dotted name relative to ``src.blocks``.
        params: Sorted (name, value) pairs; values are hashable.
    """

    fn: str
    params: tuple[tuple[str, Any], ...] = ()

    def function(self) -> Callable:
        """Returns the function the block calls."""
        return resolve_fn(self.fn)

    def param_dict(self) -> dict[str, Any]:
        """Returns the parameters as a dict."""
        return dict(self.params)

    def param_names(self) -> set[str]:
        """Returns the names of the `Param` placeholders inside."""
        names: set[str] = set()

        def collect(value: Any) -> None:
            if isinstance(value, Param):
                names.add(value.name)
            elif isinstance(value, tuple):
                for item in value:
                    collect(item)

        for _, value in self.params:
            collect(value)
        return names


def block(fn: str, **params: Any) -> Block:
    """Builds a validated block.

    Args:
        fn: Dotted function name relative to ``src.blocks``.
        **params: Parameters of the function other than the ones the runner
            supplies. Lists become tuples.

    Returns:
        The block.

    Raises:
        ValueError: If `fn` does not resolve, a parameter is not accepted
            by the signature or is reserved for the runner, or a value is
            not hashable data.
    """
    function = resolve_fn(fn)
    signature = inspect.signature(function)
    accepts_any = any(
        p.kind is inspect.Parameter.VAR_KEYWORD
        for p in signature.parameters.values())
    for name in params:
        if name in RESERVED:
            raise ValueError(
                f"{fn}: {name!r} is supplied by the runner, not a "
                "parameter of the block.")
        if name not in signature.parameters and not accepts_any:
            raise ValueError(
                f"{fn} has no parameter {name!r}; it takes "
                f"{list(signature.parameters)}.")
    frozen = tuple(sorted((k, freeze(v)) for k, v in params.items()))
    return Block(fn, frozen)


def _first_parameter(b: Block) -> str:
    """Returns the name of the first parameter of the block's function."""
    parameters = list(inspect.signature(b.function()).parameters)
    return parameters[0] if parameters else ""


def check_block(b: Block, available: set[str], stage: str) -> None:
    """Checks that a block's required parameters are all supplied.

    Args:
        b: The block.
        available: Names the runner supplies in this stage.
        stage: Name of the stage, for the error message.

    Raises:
        ValueError: If a required parameter is neither given nor supplied.
    """
    function = b.function()
    given = dict(b.params)
    for name, p in inspect.signature(function).parameters.items():
        if p.kind in (inspect.Parameter.VAR_KEYWORD,
                      inspect.Parameter.VAR_POSITIONAL):
            continue
        if p.default is inspect.Parameter.empty and (
                name not in given and name not in available):
            raise ValueError(
                f"{stage} block {b.fn}: the required parameter {name!r} is "
                "neither given nor supplied by the runner.")


def channel_role(b: Block) -> str:
    """Classifies a channel block as 'producer' or 'modifier'.

    Args:
        b: A block of the channels stage.

    Returns:
        'producer' if the function's first parameter is ``system``,
        'modifier' if it is ``channels``.

    Raises:
        ValueError: If the first parameter is neither.
    """
    first = _first_parameter(b)
    if first == "system":
        return _PRODUCER
    if first == "channels":
        return _MODIFIER
    raise ValueError(
        f"Channel block {b.fn} must take 'system' (producer) or 'channels' "
        f"(modifier) as its first parameter, not {first!r}.")


def observe_role(b: Block) -> str:
    """Classifies an observation block as 'producer' or 'modifier'.

    Args:
        b: A block of the observe stage.

    Returns:
        'producer' if the first parameter is ``states``, 'modifier' if it
        is ``obs``.

    Raises:
        ValueError: If the first parameter is neither.
    """
    first = _first_parameter(b)
    if first == "states":
        return _PRODUCER
    if first == "obs":
        return _MODIFIER
    raise ValueError(
        f"Observe block {b.fn} must take 'states' (producer) or 'obs' "
        f"(transform) as its first parameter, not {first!r}.")


@dataclass(frozen=True)
class Recipe:
    """Everything that defines a simulated trajectory except its seed.

    Attributes:
        system: Block that returns a `System`.
        channels: Producers (first parameter ``system``) add channels;
            modifiers (first parameter ``channels``) change everything
            produced before them.
        initial: Block that returns the initial density matrix; it takes
            ``system`` and may take ``rng``.
        t_max: End time, or a `Param`.
        dt: Time step, or a `Param`; the grid is built as
            ``linspace(0, t_max, round(t_max / dt) + 1)``.
        observe: Producers (first parameter ``states``) and transforms
            (first parameter ``obs``), applied in order; the series are
            put in model layout at the end if both tail series exist.
        target: Block that takes ``states`` and ``system``.
        sampler: Block ``(rng, n, **params) -> list[dict]`` that gives the
            values of the `Param` placeholders, or None if there are none.
        seeds: 'spawn' (one independent stream per trajectory and stage) or
            'legacy' (the global NumPy generator for the initial state and
            one shared generator for the other random blocks).
        legacy_rng_offset: In 'legacy' mode the shared generator is
            ``default_rng(seed + legacy_rng_offset)``.

    Raises:
        ValueError: If a block does not validate, a role is wrong, a
            `Param` is used without a sampler, or a field is out of range.
    """

    system: Block
    channels: tuple[Block, ...]
    initial: Block
    observe: tuple[Block, ...]
    target: Block
    t_max: float | Param = 20.0
    dt: float | Param = 0.1
    sampler: Block | None = None
    seeds: str = "spawn"
    legacy_rng_offset: int = 0

    def __post_init__(self) -> None:
        if self.seeds not in SEED_MODES:
            raise ValueError(
                f"seeds must be one of {SEED_MODES}, got {self.seeds!r}.")
        if int(self.legacy_rng_offset) != self.legacy_rng_offset:
            raise ValueError("legacy_rng_offset must be an integer.")
        for name in ("t_max", "dt"):
            value = getattr(self, name)
            if not isinstance(value, Param) and not (
                    isinstance(value, numbers.Real)
                    and math.isfinite(value) and value > 0):
                raise ValueError(
                    f"{name} must be a positive number or a Param, "
                    f"got {value!r}.")
        if len(self.observe) == 0 or len(self.channels) == 0:
            raise ValueError(
                "A recipe needs at least one channel and one observe "
                "block.")
        check_block(self.system, set(), "system")
        produced = False
        for b in self.channels:
            if channel_role(b) == _PRODUCER:
                produced = True
                check_block(b, {"system", "rng", "t_max"}, "channels")
            else:
                if not produced:
                    raise ValueError(
                        f"The modifier {b.fn} comes before any channel "
                        "producer.")
                check_block(b, {"channels", "rng", "t_max"}, "channels")
        check_block(self.initial, {"system", "rng"}, "initial")
        for i, b in enumerate(self.observe):
            role = observe_role(b)
            if i == 0 and role != _PRODUCER:
                raise ValueError(
                    f"The first observe block must be a producer, "
                    f"got {b.fn}.")
            check_block(
                b, {"states" if role == _PRODUCER else "obs", "rng"},
                "observe")
        check_block(self.target, {"states", "system"}, "target")
        if self.sampler is not None:
            if self.sampler.param_names():
                raise ValueError("The sampler's parameters cannot be Params.")
            positional = [
                p for p in inspect.signature(
                    self.sampler.function()).parameters.values()
                if p.kind is inspect.Parameter.POSITIONAL_OR_KEYWORD]
            if len(positional) < 2:
                raise ValueError(
                    f"The sampler {self.sampler.fn} must take (rng, n, ...).")
        elif self.referenced_params():
            raise ValueError(
                f"The recipe uses Params {sorted(self.referenced_params())} "
                "but has no sampler.")

    def blocks(self) -> tuple[Block, ...]:
        """Returns the blocks of the pipeline stages, in order."""
        return ((self.system,) + self.channels + (self.initial,)
                + self.observe + (self.target,))

    def referenced_params(self) -> set[str]:
        """Returns the names of every `Param` the recipe uses."""
        names: set[str] = set()
        for b in self.blocks():
            names |= b.param_names()
        for value in (self.t_max, self.dt):
            if isinstance(value, Param):
                names.add(value.name)
        return names
