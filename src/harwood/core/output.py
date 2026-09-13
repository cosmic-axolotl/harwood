"""Where results go when you do not say.

Two things go wrong when a task writes wherever it is told and no further
thought is given. Files pile into the working directory with names that say
nothing about where they came from, and six months later nobody can tell
`n92010.fits` from `n92010a.fits`. Or worse, a task overwrites its own input
because that was the shortest thing to type.

So every run that is not given somewhere to write gets its own directory:

    output/hrwd_ccdproc_2026_09_13/
    output/hrwd_ccdproc_2026_09_13_2/      a second run the same day

with the files inside keeping the names of the frames they came from, because
`m92010.fits` is the only thing that says which frame it is. Each directory
also holds a `run.yaml`: the command, every parameter with where its value came
from, and what each step did. The directory is the deliverable -- hand it to
someone and they have the results and how they were made.

**Where `output/` lives.** In the reduction directory, beside `harwood.yaml`
and the log, not in the installation. An installation is shared between users
on an observatory machine, replaced on upgrade, and often not writable; and a
global place for results is the same mistake as a global place for parameters,
which section 2 exists to prevent. A reduction carries its own results.
"""

from __future__ import annotations

import contextvars
from datetime import UTC, date
from pathlib import Path
from typing import Any

DIRECTORY = "output"
PREFIX = "hrwd"

#: The run directory for the task in progress, created on first use. A
#: ContextVar rather than a global so that concurrent runs cannot collide.
_current: contextvars.ContextVar[dict | None] = contextvars.ContextVar(
    "harwood_output_run", default=None
)


def begin(task: str, workdir: Path | str = ".") -> object:
    """Start a run. Returns a token for `end`."""
    return _current.set({"task": task, "workdir": Path(workdir), "path": None})


def end(token: object) -> None:
    _current.reset(token)


def active() -> bool:
    return _current.get() is not None


def name_for(task: str, when: date | None = None, suffix: int = 0) -> str:
    """`hrwd_ccdproc_2026_09_13`, and `..._2` for the next run the same day."""
    stamp = (when or date.today()).strftime("%Y_%m_%d")
    base = f"{PREFIX}_{task}_{stamp}"
    return base if suffix < 2 else f"{base}_{suffix}"


def directory(task: str | None = None, workdir: Path | str = ".") -> Path:
    """The directory this run writes to, created the first time it is asked for.

    Created lazily: a task that is given somewhere to write, or that writes
    nothing at all, should not leave an empty directory behind.
    """
    run = _current.get()
    if run is None:
        run = {"task": task or "harwood", "workdir": Path(workdir), "path": None}
    if run["path"] is not None:
        return run["path"]

    root = Path(run["workdir"]) / DIRECTORY
    suffix = 1
    while True:
        candidate = root / name_for(run["task"], suffix=suffix)
        if not candidate.exists():
            break
        suffix += 1
    candidate.mkdir(parents=True)
    run["path"] = candidate
    return candidate


def current() -> Path | None:
    """The run directory, if one has been created. Does not create it."""
    run = _current.get()
    return None if run is None else run["path"]


def record(task: str, params: dict[str, Any], result: Any,
           command: str = "", origins: dict[str, str] | None = None) -> Path | None:
    """Write `run.yaml` beside the outputs: what ran, with what, and what it did.

    Only when a run directory was actually used. A run that wrote to a place
    the user chose leaves its record in the reduction log, which is where
    someone would look for it.
    """
    import yaml

    path = current()
    if path is None:
        return None

    steps = []
    for step in getattr(result, "steps", []):
        entry: dict[str, Any] = {"step": step.action}
        if step.skipped:
            entry["skipped"] = step.reason or True
        else:
            if step.detail:
                entry["detail"] = step.detail
            if step.why:
                entry["why"] = step.why
            if step.measured:
                entry["measured"] = {k: str(v) for k, v in step.measured.items()}
            if step.before and step.after:
                entry["before"] = _rounded(step.before)
                entry["after"] = _rounded(step.after)
        steps.append(entry)

    import json
    from datetime import datetime

    document = {
        "task": task,
        "when": datetime.now(UTC).isoformat(timespec="seconds"),
        "command": command,
        "parameters": json.loads(json.dumps(params, default=str)),
        "origins": origins or {},
        "outputs": [p.name for p in getattr(result, "outputs", [])],
        "steps": steps,
    }
    if getattr(result, "warnings", None):
        document["notes"] = list(result.warnings)
    if getattr(result, "backend", None):
        document["backend"] = result.backend

    header = (
        "# What made the files in this directory.\n"
        "#\n"
        "# `command` reproduces the run: every value that was not a default is\n"
        "# written out, so it does not depend on what harwood.yaml says today.\n"
        "# `origins` says where each value came from when it ran.\n"
        "\n"
    )
    destination = path / "run.yaml"
    destination.write_text(
        header + yaml.safe_dump(document, sort_keys=False, allow_unicode=False,
                                width=88),
        encoding="utf-8")
    return destination


def _rounded(statistics: dict) -> dict:
    """Round the measurements, keeping counts as counts."""
    return {key: int(value) if key == "npix" else round(float(value), 6)
            for key, value in statistics.items()}


def named_for(source: str | Path, suffix: str, task: str | None = None,
              workdir: Path | str = ".") -> Path:
    """A name in this run's directory, derived from the frame it came from.

    `daofind obj.fits` with nowhere named writes `obj.coo` into the run
    directory, which is IRAF's convention and the only name that says which
    frame the catalogue belongs to.
    """
    # Brackets first: 'obj.fits[SCI,1]' does not end in '.fits', so stripping
    # the extension before the section leaves 'obj.fits.coo'.
    stem = Path(str(source).split("[")[0]).name
    for known in (".fits", ".fit", ".imh", ".fits.gz", ".fz"):
        if stem.endswith(known):
            stem = stem[: -len(known)]
            break
    return directory(task, workdir) / f"{stem}{suffix}"


def listing(workdir: Path | str = ".") -> list[Path]:
    """Every run directory in this reduction, newest last."""
    root = Path(workdir) / DIRECTORY
    if not root.is_dir():
        return []
    return sorted(p for p in root.iterdir() if p.is_dir())
