"""Turning an execution back into the command that produced it.

The provenance log already records what every parameter was set to and where
each value came from. That answers "what was done"; it does not answer "how do
I do it again", and the gap between those two is where reproducibility usually
fails -- you can read the record perfectly well and still have to reconstruct
the invocation by hand, guessing which values were defaults.

So every run records the command that reproduces it, written out in full:
every value that was not a default stated explicitly, nothing implicit, no
dependence on what happens to be in `harwood.yaml` today. Paste it into a
terminal a year later and it does the same thing.

This module is the renderer. It lives in `core` rather than `cli` because the
canonical form of a run is part of the record, not part of the interface.
"""

from __future__ import annotations

import shlex
from typing import Any


def option_name(field: str, kebab: bool = True) -> str:
    """`low_reject` -> `--low-reject`. The canonical spelling on the command line."""
    return f"--{field.replace('_', '-')}" if kebab else f"--{field}"


def _render_value(value: Any) -> list[str]:
    """One parameter's value, as the words a shell would pass."""
    from pathlib import Path

    if isinstance(value, bool):
        return []
    if isinstance(value, (list, tuple)):
        return [shlex.quote(str(item)) for item in value]
    if isinstance(value, Path):
        return [shlex.quote(str(value))]
    if isinstance(value, float):
        return [f"{value:g}"]
    return [shlex.quote(str(value))]


def command(task: str, model: type, values: dict[str, Any],
            program: str = "harwood") -> str:
    """The command line that reproduces one run.

    Required parameters become positional arguments in the order the model
    declares them; everything else becomes an option, and only when it differs
    from the model's own default. Booleans become `--flag` or `--no-flag`.
    """
    positional: list[str] = []
    options: list[str] = []
    variadic_taken = False

    for name, field in model.model_fields.items():
        if name not in values:
            continue
        value = values[name]

        # The same `positional` marker the command line honours, so the
        # command written into the log is the one a person would type.
        extra = field.json_schema_extra or {}
        if field.is_required() or extra.get("positional"):
            # Only the first list among the positionals takes a word per
            # item (cli/build.py); a later one takes them joined by commas,
            # or the first would swallow them.
            is_list = _is_list(field)
            if value is None and field.is_required():
                positional.append("INDEF")        # required, and given as no value
            elif value is not None:
                if isinstance(value, (list, tuple)) and variadic_taken:
                    positional.append(shlex.quote(",".join(str(v) for v in value)))
                else:
                    positional.extend(_render_value(value))
            variadic_taken = variadic_taken or is_list
            continue

        default = field.get_default(call_default_factory=True)
        if _same(value, default):
            continue

        if isinstance(value, bool):
            flag = name.replace("_", "-")
            options.append(f"--{flag}" if value else f"--no-{flag}")
            continue
        if value is None:
            if _numeric(field):
                options.extend([option_name(name), "INDEF"])   # unset over a default
            continue
        rendered = _render_value(value)
        if isinstance(value, (list, tuple)) and len(rendered) > 1:
            # Repeat the option, which is how click reads a multiple value.
            for item in rendered:
                options.extend([option_name(name), item])
        else:
            options.extend([option_name(name), *rendered])

    return " ".join([program, task, *positional, *options])


def _numeric(field) -> bool:
    """An int or float that may be None: the command line reads INDEF for it."""
    import types
    import typing
    from typing import get_args, get_origin

    annotation = field.annotation
    if get_origin(annotation) in (typing.Union, types.UnionType):
        return any(a in (int, float) for a in get_args(annotation))
    return False


def _is_list(field) -> bool:
    import types
    import typing
    from typing import get_args, get_origin

    annotation = field.annotation
    if get_origin(annotation) in (typing.Union, types.UnionType):
        annotation = next(a for a in get_args(annotation) if a is not type(None))
    return get_origin(annotation) is list


def _same(value: Any, default: Any) -> bool:
    """Would writing this value change anything?"""
    from pathlib import Path

    if isinstance(value, Path) or isinstance(default, Path):
        return str(value) == str(default)
    if isinstance(value, (list, tuple)) and isinstance(default, (list, tuple)):
        return list(value) == list(default)
    try:
        return bool(value == default)
    except Exception:
        return False
