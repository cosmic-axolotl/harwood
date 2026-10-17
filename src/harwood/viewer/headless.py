"""The headless display: writes a PNG and says where it put it.

For ssh, for batch, and for tests. Section 8.4 is blunt about the alternative:
X11 forwarding works but is slow, and Wayland has no network transparency at
all. On an observatory server this is the backend that makes sense.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ..core.display import Marker


class HeadlessDisplay:
    def __init__(self, output: Path | str = "display.png", **_: Any) -> None:
        self.output = Path(output)
        self._frames: dict[int, Any] = {}
        self._markers: dict[int, list[Marker]] = {}

    def show(self, data: Any, header: Any = None, frame: int = 1,
             name: str = "", scale: str = "zscale") -> None:
        self._frames[frame] = (data, name, scale)
        self._render(frame)

    def mark(self, markers: list[Marker], frame: int = 1) -> None:
        self._markers.setdefault(frame, []).extend(markers)
        if frame in self._frames:
            self._render(frame)

    def erase(self, frame: int | None = None) -> None:
        if frame is None:
            self._frames.clear()
            self._markers.clear()
        else:
            self._frames.pop(frame, None)
            self._markers.pop(frame, None)

    def close(self) -> None:
        pass

    def _render(self, frame: int) -> None:
        import numpy as np

        from ..core import render
        from .png import write_grey

        data, _, scale = self._frames[frame]
        # The window's limits, so the picture shows what the window would.
        low, high = render.limits(data, scale if scale in render.SCALES else "minmax")
        scaled = np.nan_to_num(render.stretch(data, low, high), nan=0.0)
        # FITS counts rows from the bottom and PNG from the top.
        write_grey(self.output, scaled[::-1] * 255)
