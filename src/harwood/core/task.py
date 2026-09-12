"""Task: the unit of work, and its parameters.

A harwood task is the modern equivalent of an IRAF task. The difference that
everything else follows from: its parameters are a Pydantic model rather than
a `.par` text file. Declare them once and you get, for free and permanently in
sync -- validation, the command line, the JSON Schema behind the parameter
editor, the documentation, and the provenance record.

Two project rules for anyone writing one:

1. **Heavy imports go inside `execute()`**, never at module scope. Section 3.2
   requires `list` and `--help` to answer instantly, and they cannot if
   importing the task pulls in astropy.
2. **Roughly sixty lines of logic, maximum.** Past that you are usually
   reimplementing something `photutils` or `ccdproc` already does, and section 1
   says harwood orchestrates rather than reimplements.
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, ClassVar, Generic, TypeVar

from pydantic import BaseModel

from . import provenance


class TaskParams(BaseModel):
    """Base class for a task's parameters.

    Always give every field a `description`: it becomes the CLI help, the
    explanation in the parameter editor, and the text the translation layer
    picks up. A test fails if any field is missing one.

    Field names are the canonical form -- `low_reject`, as in Python, YAML and
    the project file. The command line accepts `--low-reject` and `--low_reject`
    for the same field (section 3.1).
    """

    model_config = {
        "extra": "forbid",
        "validate_assignment": True,
        "populate_by_name": True,
    }


P = TypeVar("P", bound=TaskParams)


@dataclass
class Step:
    """One line of an execution plan (section 2.1).

    The point is that a user can read, before anything is written, exactly what
    is about to happen with real files and real values -- not a generic
    description of the task. `noproc` in ccdproc is the embryo of this idea.
    """

    action: str
    detail: str = ""
    skipped: bool = False
    reason: str = ""

    def render(self) -> str:
        if self.skipped:
            return f"{self.action} -- skipped: {self.reason}" if self.reason else \
                   f"{self.action} -- skipped"
        return f"{self.action}: {self.detail}" if self.detail else self.action


@dataclass
class TaskResult:
    """What a task gives back.

    `history` lines are written as HISTORY cards into the output headers, so the
    trail lives in the data itself and survives being e-mailed to a collaborator.
    """

    outputs: list[Path] = field(default_factory=list)
    data: dict[str, Any] = field(default_factory=dict)
    history: list[str] = field(default_factory=list)
    steps: list[Step] = field(default_factory=list)
    duration_s: float = 0.0
    backend: str | None = None


class Task(ABC, Generic[P]):
    """Base class of every harwood task."""

    name: ClassVar[str]
    description: ClassVar[str] = ""
    package: ClassVar[str] = "misc"
    iraf: ClassVar[str | None] = None       # IRAF name, when it differs
    Params: ClassVar[type[TaskParams]]

    #: Named parameter sets, the way IRAF used packages (section 3.2). The task
    #: appears once; `harwood list` shows its presets.
    presets: ClassVar[dict[str, dict[str, Any]]] = {}

    #: Defaults this task carries because it *is* a preset -- `zerocombine` is
    #: `imcombine` with these. They sit above the model's own defaults and
    #: below the project file, and `lpar` names them as their own origin, so a
    #: value that came from the preset never looks like something you chose.
    defaults: ClassVar[dict[str, Any]] = {}

    @abstractmethod
    def execute(self, params: P) -> TaskResult:
        """The work. Do not call directly -- use `run()`."""
        raise NotImplementedError

    def plan(self, params: P) -> list[Step]:
        """The steps `execute` will take, with real files and values.

        The default is honest but uninformative; any task that does more than
        one thing should override it. `harwood run --plan` prints this and
        stops.
        """
        return [Step(self.name, "no plan written for this task yet")]

    def run(self, params: P | dict[str, Any] | None = None, **kwargs: Any) -> TaskResult:
        """Validate, stamp, execute, time.

        Three equivalent call forms::

            task.run(MyParams(input="a.fits"))
            task.run({"input": "a.fits"})
            task.run(input="a.fits")
        """
        if params is None:
            params = {}
        if isinstance(params, dict):
            params = self.Params(**{**params, **kwargs})
        elif kwargs:
            params = params.model_copy(update=kwargs)

        # The stamp has to exist *before* execute(), because that is where the
        # task writes headers and saves files. See core/provenance.py.
        token = provenance.set_call_stamp(f"harwood.{self.name} {params.model_dump_json()}")
        stamp = provenance.current_stamp() or ""

        started = time.perf_counter()
        try:
            result = self.execute(params)  # type: ignore[arg-type]
        finally:
            provenance.reset_call_stamp(token)
        result.duration_s = time.perf_counter() - started
        result.history.insert(0, stamp)
        return result

    @classmethod
    def schema(cls) -> dict[str, Any]:
        """JSON Schema of the parameters. Feeds the CLI, the editor and the docs."""
        return cls.Params.model_json_schema()

    @classmethod
    def field_help(cls) -> dict[str, str]:
        """Parameter name -> description, for help and the parameter editor."""
        return {
            name: (f.description or "")
            for name, f in cls.Params.model_fields.items()
        }
