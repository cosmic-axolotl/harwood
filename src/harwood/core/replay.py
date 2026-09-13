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

    for name, field in model.model_fields.items():
        if name not in values:
            continue
        value = values[name]

        # The same `positional` marker the command line honours, so the
        # command written into the log is the one a person would type.
        extra = field.json_schema_extra or {}
        if field.is_required() or extra.get("positional"):
            if value is not None:
                positional.extend(_render_value(value))
            continue

        default = field.get_default(call_default_factory=True)
        if _same(value, default):
            continue

        if isinstance(value, bool):
            flag = name.replace("_", "-")
            options.append(f"--{flag}" if value else f"--no-{flag}")
            continue
        if value is None:
            continue
        rendered = _render_value(value)
        if isinstance(value, (list, tuple)) and len(rendered) > 1:
            # Repeat the option, which is how click reads a multiple value.
            for item in rendered:
                options.extend([option_name(name), item])
        else:
            options.extend([option_name(name), *rendered])

    return " ".join([program, task, *positional, *options])


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
