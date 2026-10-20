"""Running a task the way the command line does, from anywhere.

`harwood ccdproc ...` does more than call the task. It puts the project file
and the preset underneath what was typed and remembers where each value came
from; it gives the run a directory of its own under output/; it writes the
log, the run.yaml and the timing. Python had only `Task.run`, which does none
of that -- so the same reduction typed and scripted ended up in different
places, with one of them missing from the log. Both now come through here.

    import harwood
    result = harwood.run("daofind", image="obj001.fits", fwhmpsf=3.0)

is the command `harwood daofind obj001.fits --fwhmpsf 3.0`, with the same
values, the same output directory and the same entry in the log.
"""

from __future__ import annotations

import contextlib
import time
import types
import typing
from dataclasses import dataclass
from pathlib import Path
from typing import Any, get_args, get_origin


@dataclass
class Prepared:
    """A run with its values worked out, not yet made."""

    name: str
    cls: Any
    task: Any
    resolved: dict
    validated: Any
    workdir: Path
    timed: bool
    size: Any
    expected: Any

    def plan(self) -> list:
        return self.task.plan(self.validated)


@dataclass
class Finished:
    result: Any
    folder: Path | None
    written: bool


def prepare(name: str, given: dict[str, Any], workdir: Path | str | None = None) -> Prepared:
    """Resolve and validate. Raises pydantic's ValidationError for a bad
    value, ValueError or KeyError for a bad project file."""
    from . import params as store
    from . import timing
    from .registry import get_class, get_task

    cls = get_class(name)
    resolved = store.resolve(name, cls.Params, overrides=given, preset=cls.defaults,
                             preset_name=name if cls.defaults else None)
    validated = cls.Params(**store.values(resolved))
    # A task that declares its own `workdir` keeps it: the value belongs to the
    # model, and the reduction directory is whatever it says.
    if "workdir" in cls.Params.model_fields:
        workdir = getattr(validated, "workdir", None) or "."
    values = validated.model_dump()
    # A person deciding at a panel is most of an interactive run's time, and
    # says nothing about how long the next one will take.
    timed = not (cls.needs_window or values.get("interactive"))
    size = timing.size_of(values) if timed else timing.Size()
    expected = timing.estimate(name, size) if timed else None
    return Prepared(name, cls, get_task(name), resolved, validated, Path(workdir or "."),
                    timed, size, expected)


def execute(prepared: Prepared, argv: list[str] | None = None,
            status: Any = None) -> Finished:
    """Make the run: its own output directory, the log, run.yaml, the timing.
    `status`, when given, is a context manager to show while it works."""
    from . import output, provenance, replay, timing
    from . import params as store

    name, validated = prepared.name, prepared.validated
    # try/finally: a task that raises must not leave this run's directory set
    # for whatever runs next in the same process -- the session, a script, a test.
    token = output.begin(name, prepared.workdir)
    timing.current = timing.Running(name, time.monotonic(), prepared.expected)
    try:
        with status if status is not None else contextlib.nullcontext():
            result = prepared.task.run(validated)
        values = validated.model_dump()
        command = replay.command(name, prepared.cls.Params, values)
        origins = store.origins(prepared.resolved)
        provenance.record(name, values, result, prepared.workdir, origins=origins,
                          command=command, argv=argv or [])
        written = output.record(name, values, result, command, origins)
        folder = output.current()
        if prepared.timed:
            timing.record(name, prepared.size, result.duration_s, result.backend)
    finally:
        output.end(token)
        timing.current = None
    return Finished(result, folder, bool(written))


def run(name: str, /, **params: Any) -> Any:
    """`harwood.run`: one task, as the command line runs it. Returns the
    TaskResult -- outputs, data, warnings, steps."""
    from .registry import exists, get_class

    if not exists(name):
        import difflib

        from .registry import list_tasks

        near = difflib.get_close_matches(name, list_tasks(), n=3)
        raise KeyError(f"no task called '{name}'"
                       + (f"; did you mean {', '.join(near)}?" if near else ""))
    cls = get_class(name)
    fields = cls.Params.model_fields
    workdir = params.pop("workdir", None) if "workdir" not in fields else None
    unknown = [key for key in params if key not in fields]
    if unknown:
        import difflib

        hints = [f"{key} (did you mean {', '.join(m)}?)" if (m := difflib.get_close_matches(
            key, fields, n=2)) else key for key in unknown]
        raise TypeError(f"{name} has no parameter {', '.join(hints)}; "
                        f"`harwood help {name}` lists them")
    given = {key: [value] if isinstance(value, str | Path) and _takes_list(fields[key])
             else value for key, value in params.items()}
    return execute(prepare(name, given, workdir)).result


def plan(name: str, /, **params: Any) -> list:
    """`harwood.plan`: the steps a run would take, with real values -- `--plan`."""
    from .registry import get_class

    fields = get_class(name).Params.model_fields
    workdir = params.pop("workdir", None) if "workdir" not in fields else None
    given = {key: [value] if isinstance(value, str | Path) and _takes_list(fields[key])
             else value for key, value in params.items()}
    return prepare(name, given, workdir).plan()


def _takes_list(field) -> bool:
    """A field typed list[...] (or Optional of one): a single name is a list of one,
    as it is on the command line."""
    annotation = field.annotation
    if get_origin(annotation) in (typing.Union, types.UnionType):
        annotation = next((a for a in get_args(annotation) if a is not type(None)), None)
    return get_origin(annotation) is list
