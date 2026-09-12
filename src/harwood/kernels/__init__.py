"""Numeric kernels and the backend that runs them.

Section 7 puts a single line through harwood: above it, tasks, parameter
models, FITS and provenance, all in Python; below it, functions that take
NumPy arrays and numbers and return arrays. The compiled backends live
entirely below that line, which is why they can be swapped at runtime::

    reference   pure NumPy -- the specification, readable
    numba       the same algorithm as loops, compiled on first call
    rust        a compiled module, imported like any other

The dispatcher picks the fastest one installed, or obeys ``HARWOOD_BACKEND``.
Being able to force ``reference`` matters: when a result looks strange, you can
re-run the whole reduction through the readable implementation and compare.

The backend that actually ran, and its version, go into the provenance log and
the FITS history. A reduction that cannot say which code produced its numbers
is not reproducible.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

#: Compilation cache, kept out of the package and out of the reduction (7.2).
CACHE_DIR = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache")) / "harwood"

#: Fastest first. The dispatcher walks this in order.
ORDER = ("rust", "numba", "reference")

__all__ = ["Kernel", "get", "available", "requested", "CACHE_DIR", "ORDER"]


@dataclass(frozen=True)
class Kernel:
    """A kernel bound to the backend that will run it."""

    name: str
    backend: str
    version: str
    function: Callable[..., Any]

    @property
    def label(self) -> str:
        """What goes in the log: ``numba 0.67.0``."""
        return f"{self.backend} {self.version}".strip()

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return self.function(*args, **kwargs)


def requested() -> str | None:
    """The backend the user asked for, if any."""
    value = os.environ.get("HARWOOD_BACKEND")
    return value.strip().lower() if value else None


def _load(backend: str):
    """Import one backend module, or return None if it is not installed."""
    if backend == "reference":
        from . import reference

        import numpy

        return reference, numpy.__version__
    if backend == "numba":
        try:
            CACHE_DIR.mkdir(parents=True, exist_ok=True)
            os.environ.setdefault("NUMBA_CACHE_DIR", str(CACHE_DIR))
            from . import numba_impl

            return numba_impl, numba_impl.version()
        except ImportError:
            return None
    if backend == "rust":
        try:
            import harwood_kernels  # type: ignore

            return harwood_kernels, getattr(harwood_kernels, "__version__", "")
        except ImportError:
            return None
    raise ValueError(f"unknown backend '{backend}'")


def available() -> dict[str, str]:
    """Installed backends, mapped to their versions. Always includes reference."""
    found = {}
    for backend in ORDER:
        loaded = _load(backend)
        if loaded is not None:
            found[backend] = loaded[1]
    return found


def get(name: str, backend: str | None = None) -> Kernel:
    """Bind a kernel by name to the best available backend.

    Raises ``RuntimeError`` when a backend was asked for explicitly and is not
    installed -- silently falling back would make ``HARWOOD_BACKEND=numba``
    a lie, and the log would record a different backend than the user believes
    they used.
    """
    wanted = (backend or requested())
    order = (wanted,) if wanted else ORDER

    if wanted and wanted not in ORDER:
        raise RuntimeError(
            f"unknown backend '{wanted}'; choose one of {', '.join(ORDER)}"
        )

    for candidate in order:
        loaded = _load(candidate)
        if loaded is None:
            continue
        module, version = loaded
        function = getattr(module, name, None)
        if function is None:
            if wanted:
                raise RuntimeError(
                    f"backend '{candidate}' does not implement kernel '{name}'"
                )
            continue
        return Kernel(name, candidate, version, function)

    if wanted:
        raise RuntimeError(
            f"backend '{wanted}' is not installed. "
            f"Available: {', '.join(available())}"
        )
    raise RuntimeError(f"no backend implements kernel '{name}'")
