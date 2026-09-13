"""The IRAF migration map.

Reads `data/iraf_migration.yaml`: all 610 task names of Community IRAF 2.18.2,
each with its package, its domain, its status and its milestone.

Section 3.2 explains why it exists. Someone coming from IRAF types the name
they know. If harwood answers "command not found", they cannot tell whether the
task never existed, changed name, is coming later, or is deliberately gone. The
map makes every answer informative -- `harwood help rfits` says the FITS reader
is unnecessary because FITS is read directly, rather than pretending not to
recognise the word.

No heavy imports: this module is loaded by `list` and `help`, which section 3.2
requires to be instant.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any, Literal

Status = Literal["available", "partial", "planned", "replaced", "wontport"]

_DATA = Path(__file__).resolve().parent.parent / "data" / "iraf_migration.yaml"


@lru_cache(maxsize=1)
def _load() -> dict[str, Any]:
    import yaml

    return yaml.safe_load(_DATA.read_text(encoding="utf-8"))


def tasks() -> dict[str, dict[str, Any]]:
    """Every IRAF task, by name."""
    return _load()["tasks"]


def domains() -> dict[str, dict[str, Any]]:
    """The 22 domains of the inventory, with the scope decided for each."""
    return _load()["domains"]


def lookup(name: str) -> dict[str, Any] | None:
    return tasks().get(name)


def by_milestone(milestone: str) -> list[str]:
    """Task names in one milestone -- 'mvp' or 'later'."""
    return sorted(n for n, e in tasks().items() if e.get("milestone") == milestone)


def explain(name: str) -> str | None:
    """What to say about an IRAF name harwood does not run.

    Returns None when the task is available here: then the registry answers,
    not the map.
    """
    entry = lookup(name)
    if entry is None:
        return None
    status, note = entry["status"], entry.get("note")
    if status in ("available", "partial"):
        # An IRAF name that harwood runs under a different one still needs
        # answering: `apedit` is part of `apall`, and "did you mean apall?"
        # makes that look like a guess rather than the answer.
        harwood = entry.get("harwood")
        if harwood and harwood != name:
            return (f"'{name}' ({entry['pkg']}) is harwood's '{harwood}'"
                    + (f": {note}." if note else ". Run "
                       f"`harwood help {harwood}`."))
        return None
    if status == "planned":
        when = "in the MVP" if entry.get("milestone") == "mvp" else "after the MVP"
        base = f"'{name}' ({entry['pkg']}) is planned {when}"
    elif status == "replaced":
        base = f"'{name}' ({entry['pkg']}) is replaced"
    else:
        base = f"'{name}' ({entry['pkg']}) will not be ported"
    return f"{base}: {note}." if note else f"{base}."
