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
        from astropy.visualization import ZScaleInterval
        from PIL import Image  # pillow ships with matplotlib-free stacks too

        data, _, scale = self._frames[frame]
        if scale == "zscale":
            low, high = ZScaleInterval().get_limits(data)
        else:
            low, high = float(np.nanmin(data)), float(np.nanmax(data))
        scaled = np.clip((np.asarray(data) - low) / max(high - low, 1e-12), 0, 1)
        image = Image.fromarray((scaled[::-1] * 255).astype("uint8"))
        image.save(self.output)
