"""The display interface -- IRAF's `stdimage`, made explicit.

Section 8.2: **tasks never import Qt.** They talk to this interface, and one of
three backends answers::

    native      the harwood window; the default
    headless    writes a PNG; ssh, batch, automated tests
    ds9         through XPA or SAMP; a last resort

Same principle as the numeric kernels: a stable interface, swappable
implementations. It is what lets `display obj.fits` mean something over ssh,
and what keeps a task testable without a screen.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

BACKENDS = ("native", "headless", "ds9")


@dataclass
class Marker:
    """Something drawn over the image, in 1-based (x, y)."""

    x: float
    y: float
    shape: str = "circle"
    radius: float = 5.0
    label: str = ""          #: drawn beside the marker; empty draws nothing
    colour: str = "lime"
    tag: str = ""            #: internal grouping, never drawn


@runtime_checkable
class Display(Protocol):
    """What every backend provides."""

    def show(self, data: Any, header: Any = None, frame: int = 1,
             name: str = "", scale: str = "zscale") -> None: ...

    def mark(self, markers: list[Marker], frame: int = 1) -> None: ...

    def erase(self, frame: int | None = None) -> None: ...

    def close(self) -> None: ...


_current: Display | None = None


def get(backend: str | None = None, **options: Any) -> Display:
    """Get the display, starting it if necessary.

    With no backend named, pick the one that can work here: the native window
    when there is a session to draw on, headless otherwise. Falling back is
    right for the *default*; an explicitly requested backend is never
    downgraded silently.
    """
    global _current
    if _current is not None and backend is None:
        return _current

    chosen = backend or _autodetect()
    if chosen == "native":
        from ..viewer.backend import NativeDisplay

        _current = NativeDisplay(**options)
    elif chosen == "headless":
        from ..viewer.headless import HeadlessDisplay

        _current = HeadlessDisplay(**options)
    elif chosen == "ds9":
        from ..viewer.ds9 import DS9Display

        _current = DS9Display(**options)
    else:
        raise ValueError(f"unknown display backend '{chosen}'; "
                         f"choose one of {', '.join(BACKENDS)}")
    return _current


def reset() -> None:
    """Forget the current display. Mostly for tests."""
    global _current
    if _current is not None:
        _current.close()
    _current = None


def _autodetect() -> str:
    import os
    import shutil

    if os.environ.get("HARWOOD_DISPLAY"):
        return os.environ["HARWOOD_DISPLAY"]
    if not (os.environ.get("WAYLAND_DISPLAY") or os.environ.get("DISPLAY")):
        return "headless"
    try:
        import PySide6  # noqa: F401

        return "native"
    except ImportError:
        pass
    return "ds9" if shutil.which("ds9") else "headless"
