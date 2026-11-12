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
# Evolution blocks and the first parameter that the observe and target
# blocks of a recipe take for each: the integrated states, or the record of
# measurement outcomes.
EVOLVERS = {
    "evolve.master_equation": "states",
    "evolve.sequential_record": "record",
}

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


def first_parameter_of(function: Callable) -> str:
    """Returns the name of the first parameter of a function, or ''."""
    parameters = list(inspect.signature(function).parameters)
    return parameters[0] if parameters else ""


def _first_parameter(b: Block) -> str:
    """Returns the name of the first parameter of the block's function."""
    return first_parameter_of(b.function())


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
        'producer' if the first parameter is ``states`` or ``record``,
        'modifier' if it is ``obs``.

    Raises:
        ValueError: If the first parameter is none of these.
    """
    first = _first_parameter(b)
    if first in ("states", "record"):
        return _PRODUCER
    if first == "obs":
        return _MODIFIER
    raise ValueError(
        f"Observe block {b.fn} must take 'states' or 'record' (producer) "
        f"or 'obs' (transform) as its first parameter, not {first!r}.")


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
        target: Block that takes ``states`` (or ``record``) and ``system``,
            or None if the recipe has no target (for instance with
            ``sequential_record``, which has no target block yet).
        sampler: Block ``(rng, n, **params) -> list[dict]`` that gives the
            values of the `Param` placeholders, or None if there are none.
        seeds: 'spawn' (one independent stream per trajectory and stage) or
            'legacy' (the global NumPy generator for the initial state and
            one shared generator for the other random blocks).
        legacy_rng_offset: In 'legacy' mode the shared generator is
            ``default_rng(seed + legacy_rng_offset)``.
        evolve: ``evolve.master_equation`` (default; the observe blocks
            take ``states``) or ``evolve.sequential_record``, e.g.
            ``block("evolve.sequential_record", basis="z")`` (the observe
            blocks take ``record``, e.g. ``observe.record_series``).

    Raises:
        ValueError: If a block does not validate, a role is wrong, a
            `Param` is used without a sampler, or a field is out of range.
    """

    system: Block
    channels: tuple[Block, ...]
    initial: Block
    observe: tuple[Block, ...]
    target: Block | None
    t_max: float | Param = 20.0
    dt: float | Param = 0.1
    sampler: Block | None = None
    seeds: str = "spawn"
    legacy_rng_offset: int = 0
    evolve: Block = Block("evolve.master_equation")

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
        if self.evolve.fn not in EVOLVERS:
            raise ValueError(
                f"evolve must be one of {sorted(EVOLVERS)}, "
                f"got {self.evolve.fn!r}.")
        evolved = EVOLVERS[self.evolve.fn]
        check_block(
            self.evolve, {"system", "channels", "rho0", "times", "rng"},
            "evolve")
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
            if role == _PRODUCER and _first_parameter(b) != evolved:
                raise ValueError(
                    f"The observe block {b.fn} takes "
                    f"{_first_parameter(b)!r} but {self.evolve.fn} "
                    f"produces {evolved!r}.")
            check_block(
                b, {evolved if role == _PRODUCER else "obs", "rng"},
                "observe")
        if self.target is not None:
            check_block(self.target, {evolved, "system"}, "target")
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
        target = () if self.target is None else (self.target,)
        return ((self.system,) + self.channels + (self.initial, self.evolve)
                + self.observe + target)

    def referenced_params(self) -> set[str]:
        """Returns the names of every `Param` the recipe uses."""
        names: set[str] = set()
        for b in self.blocks():
            names |= b.param_names()
        for value in (self.t_max, self.dt):
            if isinstance(value, Param):
                names.add(value.name)
        return names


# --- JSON round trip ---

_REQUIRED_FIELDS = (
    "system", "channels", "initial", "observe", "target")
_RECIPE_FIELDS = _REQUIRED_FIELDS + (
    "evolve", "t_max", "dt", "sampler", "seeds",
    "legacy_rng_offset")


def _encode(value: Any) -> Any:
    """Turns a block parameter value into JSON data."""
    if isinstance(value, Param):
        return {"param": value.name}
    if isinstance(value, tuple):
        return [_encode(v) for v in value]
    return value


def _decode(value: Any, where: str) -> Any:
    """Turns JSON data into a block parameter value.

    Raises:
        ValueError: If the data holds a dict that is not ``{"param": name}``.
    """
    if isinstance(value, dict):
        if set(value) != {"param"} or not isinstance(value["param"], str):
            raise ValueError(
                f"{where}: an object must be {{\"param\": \"name\"}}, "
                f"got {value!r}.")
        return Param(value["param"])
    if isinstance(value, list):
        return tuple(_decode(v, where) for v in value)
    return value


def block_to_dict(b: Block) -> dict[str, Any]:
    """Encodes a block as ``{"fn": ..., "params": {...}}`` (JSON-safe)."""
    return {"fn": b.fn, "params": {k: _encode(v) for k, v in b.params}}


def block_from_dict(d: Any, where: str) -> Block:
    """Decodes and validates a block.

    Args:
        d: ``{"fn": "module.function", "params": {...}}``; `params` may be
            left out. A value ``{"param": name}`` is a `Param`; lists
            become tuples.
        where: Location in the recipe, e.g. ``"channels[1]"``, put in
            front of every error message.

    Returns:
        The block.

    Raises:
        ValueError: If `d` is not of that form or `block` rejects it.
    """
    if not isinstance(d, dict):
        raise ValueError(
            f"{where}: a block must be an object with 'fn' and 'params', "
            f"got {d!r}.")
    unknown = set(d) - {"fn", "params"}
    if unknown:
        raise ValueError(
            f"{where}: unknown fields {sorted(unknown)}; a block has "
            "'fn' and 'params'.")
    if "fn" not in d:
        raise ValueError(f"{where}: the field 'fn' is missing.")
    params = d.get("params", {})
    if not isinstance(params, dict):
        raise ValueError(
            f"{where}.params: must be an object, got {params!r}.")
    decoded = {
        k: _decode(v, f"{where}.params.{k}") for k, v in params.items()}
    try:
        return block(d["fn"], **decoded)
    except ValueError as error:
        raise ValueError(f"{where}: {error}") from error


def recipe_to_dict(r: Recipe) -> dict[str, Any]:
    """Encodes a recipe as JSON-safe data.

    Blocks become ``{"fn", "params"}``, a `Param` becomes
    ``{"param": name}`` and tuples become lists.

    Args:
        r: The recipe.

    Returns:
        A dict that `recipe_from_dict` turns back into an equal recipe.
    """
    return {
        "system": block_to_dict(r.system),
        "channels": [block_to_dict(b) for b in r.channels],
        "initial": block_to_dict(r.initial),
        "evolve": block_to_dict(r.evolve),
        "observe": [block_to_dict(b) for b in r.observe],
        "target": None if r.target is None else block_to_dict(r.target),
        "t_max": _encode(r.t_max),
        "dt": _encode(r.dt),
        "sampler": None if r.sampler is None else block_to_dict(r.sampler),
        "seeds": r.seeds,
        "legacy_rng_offset": r.legacy_rng_offset,
    }


def _block_list(d: dict, field: str) -> tuple[Block, ...]:
    """Decodes a field that holds a list of blocks."""
    items = d[field]
    if not isinstance(items, list):
        raise ValueError(
            f"{field}: must be a list of blocks, got {items!r}.")
    return tuple(
        block_from_dict(item, f"{field}[{i}]")
        for i, item in enumerate(items))


def recipe_from_dict(d: Any) -> Recipe:
    """Decodes and validates a recipe.

    Args:
        d: Data as `recipe_to_dict` writes it. `system`, `channels`,
            `initial`, `observe` and `target` (which may be null) are
            required; the other fields take the defaults of `Recipe` when
            left out.

    Returns:
        The recipe.

    Raises:
        ValueError: If `d` is not an object, a field is missing, unknown or
            of the wrong shape, or a block or the recipe does not validate.
            The message begins with the stage and field at fault.
    """
    if not isinstance(d, dict):
        raise ValueError(f"A recipe must be an object, got {d!r}.")
    for field in _REQUIRED_FIELDS:
        if field not in d:
            raise ValueError(f"{field}: the field is missing.")
    unknown = set(d) - set(_RECIPE_FIELDS)
    if unknown:
        raise ValueError(
            f"Unknown recipe fields {sorted(unknown)}; "
            f"known: {list(_RECIPE_FIELDS)}.")
    fields: dict[str, Any] = {
        "system": block_from_dict(d["system"], "system"),
        "channels": _block_list(d, "channels"),
        "initial": block_from_dict(d["initial"], "initial"),
        "observe": _block_list(d, "observe"),
    }
    if "evolve" in d:
        fields["evolve"] = block_from_dict(d["evolve"], "evolve")
    for field in ("target", "sampler"):
        if d.get(field) is not None:
            fields[field] = block_from_dict(d[field], field)
        elif field == "target":
            fields[field] = None
    for field in ("t_max", "dt"):
        if field in d:
            fields[field] = _decode(d[field], field)
    for field in ("seeds", "legacy_rng_offset"):
        if field in d:
            fields[field] = d[field]
    try:
        return Recipe(**fields)
    except (ValueError, TypeError) as error:
        raise ValueError(f"recipe: {error}") from error
