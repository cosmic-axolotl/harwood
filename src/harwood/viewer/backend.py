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
        from ._qt import application, window
        from .window import HarwoodWindow

        self.app = application()
        # Reuse the window that is already open. `display` typed at the prompt
        # inside `harwood gui` has to load into the frame you are looking at;
        # opening a second window would be two of everything and a second event
        # loop that never returns.
        existing = window()
        self.owns = existing is None
        self.window = existing if existing is not None else HarwoodWindow()
        if self.owns:
            self.window.show()

    def show(self, data: Any, header: Any = None, frame: int = 1,
             name: str = "", scale: str = "zscale",
             window: tuple[float, float] | None = None, stretch: str | None = None,
             table: Any = None) -> None:
        self.window.load(data, header, frame=frame, name=name, scale=scale,
                         window=window, stretch=stretch, table=table)
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
        """Hand control to Qt. Used by `harwood gui`.

        Does nothing when the window was already open and running: the event
        loop is somebody else's, and entering a second one from inside it is
        how a prompt stops answering.
        """
        if not self.owns:
            return 0
        return self.app.exec()

    def close(self) -> None:
        self.window.close()

    def _pump(self) -> None:
        self.app.processEvents()
