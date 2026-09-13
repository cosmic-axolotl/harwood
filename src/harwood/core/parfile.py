"""The parameter file you edit in a text editor.

IRAF's `epar` puts you in a full-screen cursor form that owns your terminal
until you leave it, with one line of help at the bottom and no room to say why
a parameter exists. It is the part of IRAF people describe as "you just have to
know".

harwood renders the same information as plain text, opens it in whatever editor
you use, and reads it back when you save. Every parameter carries its
explanation, its type, its allowed values, its default, and -- the part IRAF
never showed -- the value in force right now and where that value came from.
You can open it as often as you like; it is generated fresh each time from the
project file, so it always shows the current state rather than a snapshot.

What you save goes into `harwood.yaml` in the reduction directory, which stays
the only place harwood keeps parameters (section 2). This file is a view of
that one, never a second copy of it.
"""

from __future__ import annotations

import types
import typing
from pathlib import Path
from typing import Any, Literal, get_args, get_origin

MARKER = "harwood parameters"


def _typename(annotation: Any) -> str:
    origin = get_origin(annotation)
    if origin is Literal:
        return " | ".join(str(a) for a in get_args(annotation))
    if origin in (typing.Union, types.UnionType):
        inner = [a for a in get_args(annotation) if a is not type(None)]
        return " or ".join(_typename(a) for a in inner) + ", may be empty"
    if origin is list:
        return f"list of {_typename(get_args(annotation)[0])}, comma separated"
    if annotation is bool:
        return "yes | no"
    return getattr(annotation, "__name__", str(annotation))


def _render_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, (list, tuple)):
        return ", ".join(_render_value(item) for item in value)
    if isinstance(value, float):
        return f"{value:g}"
    return str(value)


def render(task: str, model: type, resolved: dict, description: str = "",
           project: Path | None = None) -> str:
    """Write the editable parameter file for one task."""
    import textwrap

    lines = [
        f"# {MARKER} · {task}",
        f"# {description}" if description else "#",
        "#",
        "# Edit the value to the right of '=' and save. Everything starting with",
        "# '#' is explanation and is ignored. Empty a value to go back to the",
        "# task's default.",
        "#",
        f"# What you save is written to {project or 'harwood.yaml'}, in the",
        "# reduction directory -- the only place harwood keeps parameters.",
        "# Nothing is written to your home directory.",
        "",
    ]

    for name, field in model.model_fields.items():
        entry = resolved.get(name)
        default = (None if field.is_required()
                   else field.get_default(call_default_factory=True))

        for line in textwrap.wrap(field.description or "no description", width=74):
            lines.append(f"# {line}")

        facts = [_typename(field.annotation)]
        if field.is_required():
            facts.append("required")
        else:
            facts.append(f"default {_render_value(default) or '(empty)'}")
        if entry is not None and entry.origin != "default":
            facts.append(f"now from {entry.describe()}")
        lines.append(f"#   {'  ·  '.join(facts)}")

        value = _render_value(entry.value if entry is not None else default)
        lines.append(f"{name} = {value}")
        lines.append("")

    return "\n".join(lines) + "\n"


def parse(text: str, model: type) -> dict[str, Any]:
    """Read an edited parameter file back into values.

    Only what the user actually set comes back. An empty value means "use the
    default", and is returned as a removal rather than as an empty string --
    otherwise clearing a field would silently set it to nothing.
    """
    values: dict[str, Any] = {}
    cleared: list[str] = []

    for number, raw in enumerate(text.splitlines(), start=1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            raise ValueError(f"line {number}: expected 'name = value', got {line!r}")
        name, _, value = line.partition("=")
        name, value = name.strip(), value.strip()
        if name not in model.model_fields:
            raise ValueError(
                f"line {number}: '{name}' is not a parameter of this task. "
                "Delete the line, or check the spelling."
            )
        if not value:
            cleared.append(name)
            continue
        values[name] = _coerce(value, model.model_fields[name].annotation, name, number)

    return {"values": values, "cleared": cleared}


def _coerce(text: str, annotation: Any, name: str, number: int) -> Any:
    """Turn the text a person typed into the type the model wants."""
    origin = get_origin(annotation)
    if origin in (typing.Union, types.UnionType):
        inner = [a for a in get_args(annotation) if a is not type(None)]
        annotation = inner[0] if inner else str
        origin = get_origin(annotation)

    if origin is list:
        item = get_args(annotation)[0]
        return [_coerce(part.strip(), item, name, number)
                for part in text.split(",") if part.strip()]
    if origin is Literal:
        allowed = [str(a) for a in get_args(annotation)]
        if text not in allowed:
            raise ValueError(
                f"line {number}: {name} must be one of {', '.join(allowed)}; "
                f"got '{text}'"
            )
        return text
    if annotation is bool:
        lowered = text.lower()
        if lowered in ("yes", "true", "y", "1", "on"):
            return True
        if lowered in ("no", "false", "n", "0", "off"):
            return False
        raise ValueError(f"line {number}: {name} takes yes or no; got '{text}'")
    if annotation is int:
        try:
            return int(float(text))
        except ValueError:
            raise ValueError(f"line {number}: {name} takes a whole number; "
                             f"got '{text}'") from None
    if annotation is float:
        try:
            return float(text)
        except ValueError:
            raise ValueError(f"line {number}: {name} takes a number; "
                             f"got '{text}'") from None
    return text


def apply(task: str, edited: dict[str, Any], model: type,
          project: Path) -> dict[str, Any]:
    """Write the edited values into the project file. Returns what changed."""
    import yaml

    document = {}
    if project.exists():
        document = yaml.safe_load(project.read_text(encoding="utf-8")) or {}
    params = document.setdefault("params", {})
    before = dict(params.get(task) or {})

    after = dict(before)
    for name, value in edited["values"].items():
        field = model.model_fields[name]
        if not field.is_required():
            default = field.get_default(call_default_factory=True)
            if _equal(value, default):
                after.pop(name, None)
                continue
        after[name] = value
    for name in edited["cleared"]:
        after.pop(name, None)

    if after:
        params[task] = after
    else:
        params.pop(task, None)

    header = _keep_comments(project)
    body = yaml.safe_dump(document, sort_keys=False, allow_unicode=False)
    project.write_text(header + body, encoding="utf-8")

    changed = {name: (before.get(name), after.get(name))
               for name in set(before) | set(after)
               if before.get(name) != after.get(name)}
    return changed


def _equal(value: Any, default: Any) -> bool:
    if isinstance(value, (list, tuple)) and isinstance(default, (list, tuple)):
        return list(value) == list(default)
    try:
        return bool(value == default) and type(value) is type(default)
    except Exception:
        return False


def _keep_comments(project: Path) -> str:
    """Keep the explanatory header `harwood init` wrote at the top of the file."""
    if not project.exists():
        return ""
    kept = []
    for line in project.read_text(encoding="utf-8").splitlines():
        if line.startswith("#") or not line.strip():
            kept.append(line)
        else:
            break
    return ("\n".join(kept) + "\n") if kept else ""
