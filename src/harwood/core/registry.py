"""The task registry: IRAF's package tree, flattened and explicit.

IRAF organised tasks in a tree you had to `load` before you could use --
`noao`, then `imred`, then `ccdred`. The tree carried two different things at
once: a namespace, and a set of parameter defaults per instrument. Section 3.2
splits them. The namespace becomes a flat registry, because a task called
`imcombine` should be reachable by typing `imcombine`; the defaults become
**presets**, so a task appears once and its instrument variants are listed
beside it rather than duplicated across eight packages.

Task modules must stay cheap to import -- heavy imports belong inside
`execute()` -- because `list` and `--help` import all of them and section 3.2
requires those to answer instantly.
"""

from __future__ import annotations

import difflib
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .task import Task

_REGISTRY: dict[str, type[Task]] = {}
_LOADED = False

#: Task modules, in the order `list` should present their packages.
_MODULES = (
    "harwood.tasks.imutil",
    "harwood.tasks.artdata",
    "harwood.tasks.utilities",
    "harwood.tasks.imcombine",
    "harwood.tasks.immatch",
    "harwood.tasks.ccdred",
    "harwood.tasks.photometry",
    "harwood.tasks.photcal",
    "harwood.tasks.onedspec",
    "harwood.tasks.onedtools",
    "harwood.tasks.apextract",
    "harwood.tasks.astutil",
    "harwood.tasks.fluxcal",
    "harwood.tasks.crutil",
    "harwood.tasks.tv",
    "harwood.tasks.proto",
    "harwood.tasks.imfilter",
    "harwood.tasks.modern",
    "harwood.tasks.timeseries",
)


def register(cls: type[Task]) -> type[Task]:
    """Register a task by its `name` attribute."""
    name = getattr(cls, "name", None)
    if not name:
        raise ValueError(f"{cls.__name__} must set the class attribute `name`.")
    if name in _REGISTRY and _REGISTRY[name] is not cls:
        raise ValueError(f"task '{name}' is already registered by {_REGISTRY[name].__name__}")
    _REGISTRY[name] = cls
    return cls


def _load() -> None:
    global _LOADED
    if _LOADED:
        return
    _LOADED = True
    import importlib

    for module in _MODULES:
        importlib.import_module(module)


def get_task(name: str) -> Task:
    """Instantiate a task by name.

    On a miss, raise `KeyError` with near matches. The command line turns that
    into a better answer still, by consulting the IRAF migration map: a name
    that exists in IRAF gets told whether it is planned, replaced, or never
    coming.
    """
    _load()
    if name not in _REGISTRY:
        near = difflib.get_close_matches(name, _REGISTRY, n=3)
        hint = f" Did you mean: {', '.join(near)}?" if near else ""
        raise KeyError(f"task '{name}' does not exist.{hint}")
    return _REGISTRY[name]()


def get_class(name: str) -> type[Task]:
    _load()
    if name not in _REGISTRY:
        raise KeyError(f"task '{name}' does not exist.")
    return _REGISTRY[name]


def list_tasks() -> dict[str, type[Task]]:
    """Every available task, ordered by package then name."""
    _load()
    return dict(sorted(_REGISTRY.items(), key=lambda item: (item[1].package, item[0])))


def packages() -> dict[str, list[str]]:
    """Package -> task names, for `harwood list <package>`."""
    grouped: dict[str, list[str]] = {}
    for name, cls in list_tasks().items():
        grouped.setdefault(cls.package, []).append(name)
    return grouped


def exists(name: str) -> bool:
    _load()
    return name in _REGISTRY
