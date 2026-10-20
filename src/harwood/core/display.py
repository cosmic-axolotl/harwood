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
from typing import Any, Protocol, runtime_checkable

BACKENDS = ("native", "headless", "ds9")


@dataclass
class Marker:
    """Something drawn over the image, in 1-based (x, y). Shapes: circle,
    rectangle (radius is half the width), point, plus, cross, line (to x2,
    y2), box and fill (outlined and filled rectangles), text (the label)."""

    x: float
    y: float
    shape: str = "circle"
    radius: float = 5.0
    label: str = ""          #: drawn beside the marker; empty draws nothing
    colour: str = "lime"
    tag: str = ""            #: internal grouping, never drawn
    width: float = 1.0       #: a rectangle's height over its width
    x2: float | None = None  #: the far end of a line
    y2: float | None = None


@runtime_checkable
class Display(Protocol):
    """What every backend provides."""

    def show(self, data: Any, header: Any = None, frame: int = 1,
             name: str = "", scale: str = "zscale",
             window: tuple[float, float] | None = None, stretch: str | None = None,
             table: Any = None) -> None:
        """`window` fixes z1 and z2 (overriding `scale`); `stretch` is the
        transfer from them to grey, and `table` the lookup table of the
        "user" one."""

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
        # Section 8.2 names it and nothing has needed it yet. Saying so beats
        # the ModuleNotFoundError that importing a file that is not there gives.
        from . import advice

        advice.fail("the ds9 backend is not written yet",
                    "use the harwood window (leave --backend empty), or "
                    "--backend headless for a PNG")
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
    """Pick a backend that can actually work here.

    Never ds9: it is a last resort you ask for, not one to fall into. Falling
    into it would also mean the answer to "PySide6 is not installed" is a
    different program's window appearing, which is not an answer.
    """
    import os

    if os.environ.get("HARWOOD_DISPLAY"):
        return os.environ["HARWOOD_DISPLAY"]
    if not (os.environ.get("WAYLAND_DISPLAY") or os.environ.get("DISPLAY")):
        return "headless"
    try:
        import PySide6  # noqa: F401
    except ImportError:
        return "headless"
    return "native"
