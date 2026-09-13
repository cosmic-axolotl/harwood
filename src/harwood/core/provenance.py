"""Provenance: what was done, to what, with which values, by which code.

Two records, deliberately redundant:

* ``harwood.log.jsonl`` in the reduction directory -- one JSON object per
  execution, holding the parameters, **where each value came from**, the
  compiled backend and its version, the inputs, the outputs and the timing.
* ``HISTORY`` cards in the output FITS headers, plus ``HARWOOD``/``HRWDDATE``
  keywords and ``IMCMB`` cards naming combined inputs.

The redundancy is the point. The log can reconstruct a runnable recipe
(``harwood recipe from-log``); the header travels with the data when someone
e-mails a single file to a collaborator.

Header text is ASCII only -- section 6.2: FITS cards do not take "ção" or "ñ",
so anything written into the data stays in English, whatever the interface
language is.

Design note on the call stamp
-----------------------------
A task writes its own output inside ``execute()``, so the stamp describing the
call has to be available *during* execution, not after it. It lives in a
``ContextVar`` set by ``Task.run`` and read by ``stamp_header``; that also makes
it correct under concurrency, where a module-level global would not be.
"""

from __future__ import annotations

import contextvars
import json
import os
import platform
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

LOG_NAME = "harwood.log.jsonl"

#: FITS cards take 80 columns; HISTORY leaves 72 for text.
_CARD_TEXT = 72

_call_stamp: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "harwood_call_stamp", default=None
)


def set_call_stamp(text: str):
    """Record the call about to run. Returns a token for `reset_call_stamp`."""
    return _call_stamp.set(_ascii(_truncate(text)))


def reset_call_stamp(token) -> None:
    _call_stamp.reset(token)


def current_stamp() -> str | None:
    """The call stamp of the execution in progress, if any."""
    return _call_stamp.get()


def _truncate(text: str, limit: int = 240) -> str:
    return text if len(text) <= limit else text[: limit - 3] + "..."


def _ascii(text: str) -> str:
    """FITS headers are ASCII; drop what will not fit rather than crash."""
    return text.encode("ascii", "replace").decode("ascii")


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


# ---------------------------------------------------------------- the log


def log_path(workdir: Path | str = ".") -> Path:
    return Path(workdir) / LOG_NAME


def record(
    task: str,
    params: dict[str, Any],
    result: Any,
    workdir: Path | str = ".",
    origins: dict[str, str] | None = None,
    command: str | None = None,
    argv: list[str] | None = None,
) -> Path:
    """Append one execution to the reduction's log.

    `command` is the canonical invocation that reproduces this run, with every
    non-default value written out; `argv` is what was actually typed, which is
    usually shorter because it leaned on the project file. Keeping both means
    the log answers "what did I run" and "how do I run it again" separately,
    and those are different questions.
    """
    entry = {
        "timestamp": _now(),
        "task": task,
        "params": json.loads(json.dumps(params, default=str)),
        "origins": origins or {},
        "command": command or "",
        "argv": argv or [],
        "outputs": [str(p) for p in getattr(result, "outputs", [])],
        "data": json.loads(json.dumps(getattr(result, "data", {}), default=str)),
        "duration_s": round(getattr(result, "duration_s", 0.0), 4),
        "backend": getattr(result, "backend", None),
        "harwood": _version(),
        "python": platform.python_version(),
    }
    path = log_path(workdir)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(entry, ensure_ascii=False) + "\n")
    return path


def read_log(workdir: Path | str = ".") -> list[dict[str, Any]]:
    path = log_path(workdir)
    if not path.exists():
        return []
    entries = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            entries.append(json.loads(line))
    return entries


def to_recipe(workdir: Path | str = ".") -> dict[str, Any]:
    """Freeze the log into a re-runnable recipe.

    You explore interactively, then keep exactly what worked -- ready to
    publish alongside the paper.
    """
    steps = [
        {"task": entry["task"], "params": entry["params"]}
        for entry in read_log(workdir)
    ]
    return {"name": f"from-log-{Path(workdir).resolve().name}", "steps": steps}


def history_of(task: str, workdir: Path | str = ".") -> list[dict[str, Any]]:
    """Every run of one task in this reduction, oldest first."""
    return [entry for entry in read_log(workdir) if entry["task"] == task]


def _version() -> str:
    try:
        from importlib.metadata import version

        return version("harwood")
    except Exception:
        return "unknown"


# ------------------------------------------------------------- the header


def stamp_header(header: Any, extra: list[str] | None = None) -> Any:
    """Write the harwood trail into a FITS header, in place."""
    stamp = current_stamp()
    header["HARWOOD"] = (_version(), "harwood version that wrote this file")
    header["HRWDDATE"] = (_now(), "UTC of the harwood call")
    if stamp:
        _history(header, stamp)
    for line in extra or []:
        _history(header, line)
    return header


def _history(header: Any, text: str) -> None:
    """Write one logical line as one or more HISTORY cards.

    Wrapping at 72 characters regardless of what sits there splits numbers in
    half, and "mean level 505" reads as a complete -- and wrong -- statement
    when the rest of "505.53" went to the next card. So the break happens
    between words, and every continuation is indented, which makes it obvious
    to a reader and detectable to anything parsing the header back.
    """
    import textwrap

    text = _ascii(text)
    if not text:
        return
    lines = textwrap.wrap(text, width=_CARD_TEXT, subsequent_indent="    ",
                          break_long_words=True, break_on_hyphens=False) or [text]
    for line in lines:
        header.add_history(line)


def record_inputs(header: Any, inputs: list[Path | str], keyword: str = "IMCMB") -> Any:
    """Name every input that went into a combined image.

    IRAF writes ``IMCMB001``..``IMCMBnnn``; above 999 the keyword no longer
    fits in eight characters, so the count is written instead and the names go
    to HISTORY. Same convention, same limit, stated out loud.
    """
    names = [Path(i).name for i in inputs]
    header["NCOMBINE"] = (len(names), "number of images combined")
    if len(names) > 999:
        _history(header, f"{len(names)} images combined; names in the reduction log")
        return header
    for index, name in enumerate(names, start=1):
        header[f"{keyword}{index:03d}"] = _ascii(name)
    return header


def backend_note(name: str, version: str | None = None) -> str:
    """The HISTORY line naming which implementation actually ran (section 7.1)."""
    return f"kernel backend: {name}" + (f" {version}" if version else "")


def env_backend() -> str | None:
    """The backend the user asked for, if any."""
    return os.environ.get("HARWOOD_BACKEND")
