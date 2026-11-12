"""The block catalog: what a UI needs to offer every block of a recipe.

``catalog`` reads the signatures, annotations and docstrings of the public
functions of the stage modules, so a new block shows up without a second
list to keep. Value sets come from ``typing.Literal`` annotations.
"""
from __future__ import annotations

import importlib
import inspect
import types
import typing
from collections.abc import Iterable, Sequence
from typing import Any

from src.blocks.recipe import (
    EVOLVERS,
    OPTIONAL_INJECTED,
    RESERVED,
    first_parameter_of,
)

STAGES = (
    "system", "channels", "initial", "evolve", "observe", "target",
    "samplers")
# Stages of which a recipe holds several blocks, and which have roles.
_ROLE_STAGES = ("channels", "observe")
_ROLES = {
    "channels": {"system": "producer", "channels": "modifier"},
    "observe": {
        "states": "producer", "record": "producer", "obs": "modifier"},
}
# Public functions that are not blocks of a recipe.
_NOT_BLOCKS = {
    "channels.to_mesolve",    # QuTiP's collapse-operator format
    "observe.model_layout",   # the runner applies it by itself
}
# Everything the runner passes to a block by name, besides ``t_max``.
_INJECTED = RESERVED | {"record", "rho0", "times"}
# The first parameter of a block that tells which stage output it takes.
_OUTPUT_PARAMETERS = ("states", "record")
_KINDS = {"float": "float", "int": "int", "str": "str", "bool": "bool"}


def _unwrap(hint: Any) -> tuple[Any, bool]:
    """Splits an annotation into its type and whether None is allowed."""
    origin = typing.get_origin(hint)
    if origin in (typing.Union, types.UnionType):
        arguments = [
            a for a in typing.get_args(hint) if a is not type(None)]
        nullable = len(arguments) < len(typing.get_args(hint))
        if len(arguments) == 1:
            return arguments[0], nullable
        return typing.Union[tuple(arguments)], nullable
    return hint, False


def _kind(hint: Any, where: str) -> tuple[str, list | None]:
    """Maps an annotation to a UI kind and its finite set of values.

    Args:
        hint: The annotation of a parameter, without None.
        where: Name of the parameter, for the error message.

    Returns:
        ``(kind, choices)``; `choices` is None unless the hint is a
        ``Literal``.

    Raises:
        ValueError: If the annotation is of a type the UI has no form for.
    """
    if typing.get_origin(hint) is typing.Literal:
        values = list(typing.get_args(hint))
        kinds = {type(v).__name__ for v in values}
        if len(kinds) != 1 or not kinds <= set(_KINDS):
            raise ValueError(
                f"{where}: Literal values must all be of one of "
                f"{sorted(_KINDS)}, got {values!r}.")
        return _KINDS[kinds.pop()], values
    if hint in (float, int, str, bool):
        return hint.__name__, None
    origin = typing.get_origin(hint) or hint
    if origin in (tuple, list, Sequence, Iterable) or (
            inspect.isclass(origin) and issubclass(origin, (tuple, list))):
        return "tuple", None
    raise ValueError(f"{where}: no form for the annotation {hint!r}.")


def _json_default(value: Any) -> Any:
    """Turns a default into JSON data: tuples become lists."""
    if isinstance(value, (tuple, list)):
        return [_json_default(v) for v in value]
    return value


def _params(function: Any, stage: str = "") -> list[dict[str, Any]]:
    """Describes the parameters of a block, without the injected ones.

    Samplers get ``(rng, n)`` from the runner, so `n` is left out for them.
    """
    hints = typing.get_type_hints(function)
    out = []
    for name, p in inspect.signature(function).parameters.items():
        injected = name in _INJECTED or (stage == "samplers" and name == "n")
        if injected or p.kind in (
                inspect.Parameter.VAR_KEYWORD,
                inspect.Parameter.VAR_POSITIONAL):
            continue
        required = p.default is inspect.Parameter.empty
        entry: dict[str, Any] = {"name": name}
        if name in OPTIONAL_INJECTED:
            entry.update(
                kind="float", nullable=False, required=False,
                default=None, injected=name)
            out.append(entry)
            continue
        where = f"{function.__module__}.{function.__name__}({name})"
        if name not in hints:
            raise ValueError(f"{where}: the parameter has no annotation.")
        hint, nullable = _unwrap(hints[name])
        kind, choices = _kind(hint, where)
        entry.update(
            kind=kind, nullable=nullable, required=required,
            default=None if required else _json_default(p.default),
            injected=None)
        if choices is not None:
            entry["choices"] = choices
        out.append(entry)
    return out


def _block(stage: str, name: str, function: Any) -> dict[str, Any] | None:
    """Describes one function as a block, or None if it is not one."""
    fn = f"{stage}.{name}"
    if fn in _NOT_BLOCKS:
        return None
    first = first_parameter_of(function)
    role = None
    needs = None
    if stage in _ROLE_STAGES:
        role = _ROLES[stage].get(first)
        if role is None:
            return None
    if stage == "target" and first not in _OUTPUT_PARAMETERS:
        return None
    if (stage == "observe" and role == "producer") or stage == "target":
        needs = first
    doc = inspect.getdoc(function) or ""
    return {
        "fn": fn,
        "name": name,
        "stage": stage,
        "role": role,
        "needs": needs,
        "doc": doc.splitlines()[0] if doc else "",
        "params": _params(function, stage),
    }


def catalog() -> dict[str, Any]:
    """Lists the blocks of every stage with their parameters.

    Returns:
        A JSON-safe dict:

        - ``stages``: for each stage in `STAGES` the list of its blocks,
          each ``{fn, name, stage, role, needs, doc, params}``. `role` is
          'producer' or 'modifier' for the channels and observe stages,
          else None. `needs` is 'states' or 'record' for the observe
          producers and the target blocks (the output of the evolve block
          they can follow), else None. Each parameter is ``{name, kind,
          nullable, required, default, injected}`` plus ``choices`` when
          its values are a finite set. `kind` is float, int, str, bool or
          tuple. The only injected parameter listed is ``t_max``, which
          the runner supplies unless it is set (`injected` is 't_max');
          the other runner-supplied parameters are left out.
        - ``evolve_output``: for each evolve block, 'states' or 'record'.

    Raises:
        ValueError: If a block parameter has no annotation or one that has
            no form.
    """
    stages: dict[str, list[dict[str, Any]]] = {}
    for stage in STAGES:
        module = importlib.import_module(f"src.blocks.{stage}")
        blocks = []
        for name, function in vars(module).items():
            if (name.startswith("_") or not inspect.isfunction(function)
                    or function.__module__ != module.__name__):
                continue
            described = _block(stage, name, function)
            if described is not None:
                blocks.append(described)
        stages[stage] = blocks
    return {"stages": stages, "evolve_output": dict(EVOLVERS)}
