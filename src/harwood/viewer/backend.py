"""The native display backend: the Qt window, behind the display interface."""

from __future__ import annotations

from typing import Any

from ..core.display import Marker


class NativeDisplay:
    """Drives `viewer.window.HarwoodWindow` through the `Display` protocol.

    One window, reused. Section 8.2's second mode -- a terminal session where
    `display` opens or reuses a single window while the prompt keeps answering
    -- works because IPython and the session loop both know how to run Qt's
    event loop alongside their own.
    """

    def __init__(self, **_: Any) -> None:
        from ._qt import application
        from .window import HarwoodWindow

        self.app = application()
        self.window = HarwoodWindow()
        self.window.show()

    def show(self, data: Any, header: Any = None, frame: int = 1,
             name: str = "", scale: str = "zscale") -> None:
        self.window.load(data, header, frame=frame, name=name, scale=scale)
        self._pump()

    def mark(self, markers: list[Marker], frame: int = 1) -> None:
        self.window.add_markers(markers, frame)
        self._pump()

    def erase(self, frame: int | None = None) -> None:
        if frame is None:
            self.window.frames.clear()
        else:
            self.window.frames.pop(frame, None)
        self.window._redraw()
        self._pump()

    def run(self) -> int:
        """Hand control to Qt. Used by `harwood gui`."""
        return self.app.exec()

    def close(self) -> None:
        self.window.close()

    def _pump(self) -> None:
        self.app.processEvents()
