"""Turning a parameter model into a command line.

Section 3.1 fixes the shape: IRAF's automatic parameters become positional
arguments, its hidden parameters become options, and booleans become a
`--flag/--no-flag` pair::

    harwood imcombine @bias.lis Zero.fits --combine median --reject ccdclip

The names stay IRAF's; only the way you write them changes. `low_reject` is the
canonical name -- in Python, in `harwood.yaml`, in recipes -- and the command
line accepts both `--low-reject` and `--low_reject`, because a user who has
typed the underscore for twenty years should not have to stop and remember.

Nothing here knows about any particular task. Add a task, get a command.
"""

from __future__ import annotations

import types
import typing
from pathlib import Path
from typing import Any, Literal, get_args, get_origin

import click


def option_names(field: str) -> list[str]:
    """`low_reject` -> ['--low-reject', '--low_reject'] (section 3.1)."""
    kebab = f"--{field.replace('_', '-')}"
    snake = f"--{field}"
    return [kebab] if kebab == snake else [kebab, snake]


def _unwrap(annotation: Any) -> tuple[Any, bool]:
    """Strip Optional[...] and report whether None was allowed."""
    origin = get_origin(annotation)
    if origin in (typing.Union, types.UnionType):
        args = [a for a in get_args(annotation) if a is not type(None)]
        if len(args) == 1:
            return args[0], True
        return args[0], type(None) in get_args(annotation)
    return annotation, False


#: IRAF's "no value", as typed: `harwood autoidentify arc.fits INDEF INDEF`.
#: A marker rather than None, because None is what click passes for a value
#: that was not typed at all, and collect() leaves those out.
INDEF = type("Indef", (), {"__repr__": lambda self: "INDEF"})()


class _IndefNumber(click.ParamType):
    """A number, or INDEF -- for a parameter that may be left unset."""

    def __init__(self, base) -> None:
        self.base = base
        self.name = f"{base.name} or INDEF"

    def convert(self, value, param, ctx):
        if value is INDEF or (isinstance(value, str) and value.strip().upper() == "INDEF"):
            return INDEF
        return self.base.convert(value, param, ctx)


def _click_type(annotation: Any):
    """Map a Python annotation to a click type. Returns (type, is_multiple)."""
    inner, optional = _unwrap(annotation)
    origin = get_origin(inner)

    if origin is list:
        item, _ = _unwrap(get_args(inner)[0])
        return _scalar_type(item), True
    scalar = _scalar_type(inner)
    if optional and scalar in (click.INT, click.FLOAT):
        return _IndefNumber(scalar), False
    return scalar, False


def _scalar_type(annotation: Any):
    if get_origin(annotation) is Literal:
        return click.Choice([str(v) for v in get_args(annotation)])
    if annotation is bool:
        return bool
    if annotation is int:
        return click.INT
    if annotation is float:
        return click.FLOAT
    if annotation is Path:
        return click.Path(path_type=Path)
    return click.STRING


def parameters_for(model: type) -> list[click.Parameter]:
    """Build the click parameters for one task's model, in declaration order."""
    params: list[click.Parameter] = []
    variadic_taken = False
    for name, field in model.model_fields.items():
        annotation = field.annotation
        click_type, multiple = _click_type(annotation)
        inner, _ = _unwrap(annotation)
        help_text = field.description or ""

        # A field may ask to be positional even when it is optional: some
        # tasks read naturally as `setinstrument kpno/direct` and awkwardly
        # as `setinstrument --instrument kpno/direct`.
        extra = field.json_schema_extra or {}
        if field.is_required() or extra.get("positional"):
            # Automatic parameters: positional, in the order declared.
            # Only one of them can swallow a variable number of words, so a
            # second list-valued positional takes a comma-separated value --
            # which is how IRAF writes them too ("OBJECT,EXPTIME,FILTER").
            variadic = multiple and not variadic_taken
            if variadic_taken and not field.is_required():
                # A variadic positional swallows everything up to the fixed
                # ones behind it, so an *optional* positional after one can
                # never be omitted: give one argument and it goes to the
                # optional slot, leaving the list empty. There is no way to
                # write it that is not a trap, so it is refused here rather
                # than discovered by a user.
                raise ValueError(
                    f"{model.__name__}.{name} is an optional positional following "
                    "a variadic one, which cannot be omitted. Make it required, "
                    "or drop the 'positional' marker so it becomes an option."
                )
            variadic_taken = variadic_taken or variadic
            # Never required at the click level, even though the model requires
            # it. Two reasons: `--editpar` and `--help` have to work without
            # supplying the inputs, and when something really is missing the
            # model says which field and why, where click only says "Missing
            # argument 'OUTPUT'".
            params.append(click.Argument(
                [name],
                type=click_type if variadic or not multiple else click.STRING,
                nargs=-1 if variadic else 1,
                required=False,
                default=None if not variadic else (),
                metavar=name.upper() + (",..." if multiple and not variadic else ""),
            ))
            continue

        if inner is bool:
            flag = name.replace("_", "-")
            params.append(click.Option(
                [f"--{flag}/--no-{flag}"],
                default=None,
                help=help_text,
            ))
            continue

        params.append(click.Option(
            option_names(name),
            # A list option takes text, so that IRAF's '3,5,10' reaches
            # collect() to be split; the model checks each value's type.
            type=click.STRING if multiple else click_type,
            multiple=multiple,
            default=None,
            show_default=False,
            metavar=_metavar(click_type, multiple),
            help=help_text,
        ))
    return params


def _metavar(click_type, multiple: bool) -> str | None:
    if isinstance(click_type, click.Choice):
        return "|".join(click_type.choices)
    if multiple:
        return "VALUE..."
    return None


def collect(model: type, kwargs: dict[str, Any]) -> dict[str, Any]:
    """Keep only what the user actually typed.

    Everything else is left out on purpose, so that the resolution chain in
    `core.params` can put the project file or a preset underneath and report
    where each value came from. Passing click's defaults through would make
    every parameter look as if it came from the command line.
    """
    given: dict[str, Any] = {}
    for name, field in model.model_fields.items():
        if name not in kwargs:
            continue
        value = kwargs[name]
        if value is None:
            continue
        if value is INDEF:
            given[name] = None          # typed, and typed as "no value"
            continue
        if isinstance(value, tuple):
            if not value:
                continue
            # '--apertures 3,5,10' and '--apertures 3 --apertures 5' both,
            # the first being how IRAF writes a list. A comma inside brackets
            # is a section's, 'obj.fits[1:100,*]', and not a separator.
            value = [part for item in value for part in split_list(str(item))] \
                if _is_list_field(field) else list(value)
        elif isinstance(value, str) and _is_list_field(field):
            value = split_list(value)
        given[name] = value
    return given


def split_list(text: str) -> list[str]:
    from ..core.images import split_list as split

    return split(text)


def _is_list_field(field) -> bool:
    inner, _ = _unwrap(field.annotation)
    return get_origin(inner) is list
