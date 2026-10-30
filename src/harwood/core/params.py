"""Parameter resolution: the `epar` of IRAF, without `uparm$`.

IRAF got one thing right that almost every modern pipeline got wrong: you set
the rejection sigma once and it is still there tomorrow. It also got one thing
badly wrong, and section 2 of the design contract names it -- values learned in
silence, in a directory (`uparm$`) that belongs to the user rather than to the
reduction. Two reductions in two directories shared one hidden state, and a
result could not be reproduced without it.

harwood keeps the persistence and drops the hiding:

* parameters live in ``harwood.yaml``, in the reduction directory, visible and
  commented -- created by ``harwood init``;
* nothing is ever written to the user's home;
* every execution records the effective value **and where it came from**.

Precedence, highest first::

    command line   --fwhm 3.8
    project file   harwood.yaml in this directory, or the nearest ancestor
                   below the home directory
    preset         what makes zerocombine differ from imcombine
    task default   the Pydantic model

There is deliberately no level above the command line and none below the
model: nothing is read from the user's home directory, ever.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

PROJECT_FILE = "harwood.yaml"

Origin = Literal["command line", "project", "preset", "default"]


@dataclass(frozen=True)
class Resolved:
    """One parameter's effective value and where it came from."""

    name: str
    value: Any
    origin: Origin
    source: str | None = None  # path of the project file, when origin == project

    def describe(self) -> str:
        """Where this value came from, short enough to sit in a table column.

        The project file is named relative to where you are standing, because
        an absolute path is both longer and less informative -- you already
        know which reduction you are in.
        """
        if not self.source:
            return self.origin
        source = self.source
        try:
            # ../harwood.yaml when it is above: the name alone would say it
            # was here, which is exactly what someone checking needs to know.
            source = os.path.relpath(Path(self.source).resolve(), Path.cwd())
        except (ValueError, OSError):
            source = Path(self.source).name
        return f"{self.origin}: {source}"


def find_project(start: Path | None = None) -> Path | None:
    """Find the project file for a reduction directory.

    Walks up from ``start`` so that a subdirectory of a reduction still sees
    the reduction's parameters -- the same reason ``git`` looks upwards. It
    stops below the home directory, and above it when it starts outside: a
    harwood.yaml left in the home by a command run there would otherwise hold
    for every night under it, the global place for parameters this module
    exists to prevent. Standing in the home itself, it is that directory's.
    """
    here = (start or Path.cwd()).resolve()
    home = home_directory()
    for directory in (here, *here.parents):
        if directory == home and directory != here:
            return None
        candidate = directory / PROJECT_FILE
        if candidate.is_file():
            return candidate
        if directory == home:
            return None
    return None


def home_directory() -> Path | None:
    try:
        return Path.home().resolve()
    except (RuntimeError, OSError):
        return None


#: What to say on writing a project file in the home: what happened, what to do.
HOME_NOTE = ("this harwood.yaml is in your home directory, so it holds only for "
             "commands run right there, and not in the directories under it",
             "cd to the night's directory and do this there")


def in_home(project: Path) -> bool:
    return Path(project).resolve().parent == home_directory()


def inherited_project(start: Path | None = None) -> Path | None:
    """The project file in use, when it is in a directory above this one."""
    path = find_project(start)
    if path is None or path.parent == (start or Path.cwd()).resolve():
        return None
    return path


def load_project(start: Path | None = None) -> tuple[dict[str, Any], Path | None]:
    """Read the project file. Returns (contents, path); ({}, None) if absent."""
    path = find_project(start)
    if path is None:
        return {}, None
    import yaml

    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        raise ValueError(f"{path}: expected a mapping at the top level")
    return data, path


def project_params(task: str, start: Path | None = None) -> tuple[dict[str, Any], Path | None]:
    """The project file's parameters for one task: its own, over those under
    `params.all`, which hold for every task that has the parameter -- how a
    reduction says once that it wants its uncertainties as ERR."""
    data, path = load_project(start)
    every = (data.get("params") or {}).get("all") or {}
    params = (data.get("params") or {}).get(task) or {}
    for name, value in (("all", every), (task, params)):
        if not isinstance(value, dict):
            raise ValueError(f"{path}: params.{name} should be a mapping of name to value")
    return {**every, **params}, path


def resolve(
    task: str,
    model: type,
    overrides: dict[str, Any] | None = None,
    start: Path | None = None,
    preset: dict[str, Any] | None = None,
    preset_name: str | None = None,
) -> dict[str, Resolved]:
    """Resolve every parameter of a task, recording the origin of each.

    ``model`` is the task's ``TaskParams`` subclass; its field defaults are the
    bottom of the stack. The result covers every field the model declares, so
    ``harwood lpar`` can show the complete picture rather than only what was
    overridden.
    """
    overrides = dict(overrides or {})
    preset = dict(preset or {})
    from_project, project_path = project_params(task, start)

    resolved: dict[str, Resolved] = {}
    for name, field in model.model_fields.items():
        if name in overrides:
            resolved[name] = Resolved(name, overrides.pop(name), "command line")
        elif name in from_project:
            resolved[name] = Resolved(
                name, from_project[name], "project", str(project_path)
            )
        elif name in preset:
            resolved[name] = Resolved(name, preset[name], "preset", preset_name)
        elif field.is_required():
            continue  # no value anywhere; validation will say so, by name
        else:
            resolved[name] = Resolved(name, field.get_default(call_default_factory=True),
                                      "default")

    # An override with no matching field is a typo, and saying so here beats a
    # pydantic error that arrives after the file has been read.
    for stray in overrides:
        raise KeyError(stray)

    return resolved


def values(resolved: dict[str, Resolved]) -> dict[str, Any]:
    """Just the values, for handing to the parameter model."""
    return {name: r.value for name, r in resolved.items()}


def origins(resolved: dict[str, Resolved]) -> dict[str, str]:
    """Just the origins, for the provenance record."""
    return {name: r.describe() for name, r in resolved.items()}


def write_project(
    path: Path,
    params: dict[str, dict[str, Any]] | None = None,
    instrument: str | None = None,
) -> Path:
    """Write a fresh, commented project file -- what ``harwood init`` creates."""
    import yaml

    lines = [
        "# harwood project file.",
        "#",
        "# Parameters set here apply to every task run in this directory and",
        "# below. They are visible on purpose: a reduction that cannot be read",
        "# off the disk cannot be reproduced.",
        "#",
        "# Nothing is stored in your home directory. If a value is not here and",
        "# not on the command line, it is a task default -- so this file plus the",
        "# log beside it are the whole record of how the reduction was made.",
        "#",
        "# See the effective value of every parameter, and where it came from:",
        "#     harwood lpar <task>",
        "",
    ]
    body: dict[str, Any] = {}
    if instrument:
        lines += [
            "# Header translation for this instrument: which keyword says what",
            "# the image type is, which says the filter, where the bias section",
            "# is. The equivalent of IRAF's setinstrument.",
            "",
        ]
        body["instrument"] = instrument
    body["params"] = params or {}
    text = "\n".join(lines) + yaml.safe_dump(body, sort_keys=False, allow_unicode=False)
    if not body["params"]:
        text += ("  # imcombine:\n  #   reject: ccdclip\n  #   rdnoise: 6.5\n"
                 "  # all:                 # for every task that has the parameter\n"
                 "  #   uncertainty: ERR    # VAR (default), ERR or UNCERT\n"
                 "  #   format: iraf        # write images as IRAF reads them\n")
    path.write_text(text, encoding="utf-8")
    return path
