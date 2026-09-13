"""implot: reading numbers off an image by plotting cuts through it.

This is how the exercises decide things. You do not guess the overscan
section; you plot a line across the frame, expand the right-hand end, and read
where the illuminated pixels stop and the overscan begins. `implot` is the
instrument for that, and the keys matter because they are the ones the
tutorials tell you to press.

    l          plot the line under the cursor
    c          plot the column under the cursor
    :l n1 n2   plot the average of lines n1 to n2
    :c n1 n2   plot the average of columns n1 to n2
    e          expand: press once at one corner, again at the other
    r  or  a   redraw the whole plot
    C          read off the coordinate and value under the cursor
    o          keep the current plot and overplot the next one
    ?          the keys
    q          leave

Positions are 1-based, as everywhere a user can see.
"""

from __future__ import annotations

import numpy as np
import pyqtgraph as pg
from PySide6 import QtCore, QtWidgets

from ._qt import application

INK = "#d7dee5"
MUTED = "#8b9aa8"
GROUND = "#141a21"
PANEL = "#1b222b"
LINE = "#2a3541"
ACCENT = "#e0a33e"
SKY = "#4fb59b"
OVER = ("#4fb59b", "#e0a33e", "#7db2d8", "#d2686a", "#a98fd0")

HELP = [
    ("l", "plot the line under the cursor"),
    ("c", "plot the column under the cursor"),
    (":l n1 n2", "average of lines n1 to n2"),
    (":c n1 n2", "average of columns n1 to n2"),
    ("e", "expand: press at one corner, then the other"),
    ("r", "redraw the whole plot"),
    ("C", "read the coordinate and value under the cursor"),
    ("o", "overplot the next cut instead of replacing"),
    ("q", "leave"),
]


class ImplotPanel(QtWidgets.QWidget):
    """One plot, and the keys that decide what it shows."""

    finished = QtCore.Signal()

    def __init__(self, data, name: str = "image", parent=None) -> None:
        super().__init__(parent)
        self.data = np.asarray(data, dtype=float)
        self.name = name
        self.ny, self.nx = self.data.shape
        self.overplot = False
        self.curves = 0
        self._cursor = (self.nx / 2, self.ny / 2)
        self._corner: tuple[float, float] | None = None

        self.setStyleSheet(f"""
            QWidget {{ background: {GROUND}; color: {INK};
                font-family: "IBM Plex Sans", "DejaVu Sans", sans-serif;
                font-size: 12px; }}
            QLabel#status {{ background: {PANEL}; border-top: 1px solid {LINE};
                padding: 5px 9px; font-family: "IBM Plex Mono", monospace;
                font-size: 11px; }}
            QLabel#keys {{ color: {MUTED}; font-family: "IBM Plex Mono", monospace;
                font-size: 11px; padding: 4px 9px; }}
            QLineEdit {{ background: {PANEL}; color: {INK}; border: 1px solid {LINE};
                border-radius: 3px; padding: 4px 8px;
                font-family: "IBM Plex Mono", monospace; }}
            QLineEdit:focus {{ border-color: {ACCENT}; }}
        """)

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        self.plot = pg.PlotWidget()
        self.plot.setBackground(GROUND)
        for axis in ("left", "bottom"):
            item = self.plot.getAxis(axis)
            item.setPen(pg.mkPen(LINE))
            item.setTextPen(pg.mkPen(MUTED))
            item.enableAutoSIPrefix(False)
        self.plot.setLabel("left", "counts", color=MUTED)
        self.plot.scene().sigMouseMoved.connect(self._mouse_moved)
        self.plot.setFocusPolicy(QtCore.Qt.StrongFocus)
        self.plot.installEventFilter(self)
        layout.addWidget(self.plot, stretch=1)

        self.status = QtWidgets.QLabel("")
        self.status.setObjectName("status")
        layout.addWidget(self.status)

        self.keys = QtWidgets.QLabel(
            "   ".join(f"[{key}] {what}" for key, what in HELP[:5]))
        self.keys.setObjectName("keys")
        layout.addWidget(self.keys)

        self.command = QtWidgets.QLineEdit()
        self.command.setPlaceholderText(":l 250 260   ·   :c 335 350   ·   ? for the keys")
        self.command.returnPressed.connect(self._run_command)
        layout.addWidget(self.command)

        self.setFocusProxy(self.plot)
        self.plot_lines(self.ny // 2, self.ny // 2)

    # --------------------------------------------------------------- plots

    def plot_lines(self, first: int, last: int) -> None:
        """Average of image lines `first` to `last`, both 1-based and inclusive."""
        first, last = sorted((int(first), int(last)))
        first = max(1, min(first, self.ny))
        last = max(1, min(last, self.ny))
        cut = self.data[first - 1:last, :].mean(axis=0)
        label = (f"line {first}" if first == last
                 else f"lines {first} to {last}, averaged")
        self._draw(np.arange(1.0, self.nx + 1.0), cut, "column (x)", label)

    def plot_columns(self, first: int, last: int) -> None:
        """Average of image columns `first` to `last`, both 1-based and inclusive."""
        first, last = sorted((int(first), int(last)))
        first = max(1, min(first, self.nx))
        last = max(1, min(last, self.nx))
        cut = self.data[:, first - 1:last].mean(axis=1)
        label = (f"column {first}" if first == last
                 else f"columns {first} to {last}, averaged")
        self._draw(np.arange(1.0, self.ny + 1.0), cut, "line (y)", label)

    def _draw(self, x, y, xlabel: str, label: str) -> None:
        if not self.overplot:
            self.plot.clear()
            self.curves = 0
        colour = OVER[self.curves % len(OVER)]
        self.plot.plot(x, y, pen=pg.mkPen(colour, width=1.2))
        self.curves += 1
        self.plot.setLabel("bottom", xlabel, color=MUTED)
        self.plot.setTitle(f"implot · {self.name} · {label}", color=INK, size="10pt")
        if not self.overplot:
            self.plot.autoRange(padding=0.02)
        self.last = (x, y, label)
        self.status.setText(
            f"{label}   min {y.min():.1f}   max {y.max():.1f}   "
            f"mean {y.mean():.1f}   median {np.median(y):.1f}\n"
            + ("overplotting; press o again to stop" if self.overplot
               else "press ? for the keys"))

    # -------------------------------------------------------------- input

    def eventFilter(self, obj, event) -> bool:
        if event.type() == QtCore.QEvent.KeyPress and self.key(event.text()):
            return True
        return super().eventFilter(obj, event)

    def _mouse_moved(self, position) -> None:
        box = self.plot.getPlotItem().vb
        if box.sceneBoundingRect().contains(position):
            point = box.mapSceneToView(position)
            self._cursor = (point.x(), point.y())

    def key(self, key: str) -> bool:
        x, y = self._cursor
        if key == "l":
            self.plot_lines(round(y), round(y)) if self.last[2].startswith("column") \
                else self.plot_lines(round(self._row_hint()), round(self._row_hint()))
        elif key == "c":
            self.plot_columns(round(x), round(x))
        elif key == "e":
            self._expand()
        elif key in ("r", "a"):
            self.plot.autoRange(padding=0.02)
            self._corner = None
            self.status.setText(f"{self.last[2]}\nredrawn whole")
        elif key == "C":
            self._read_off()
        elif key == "o":
            self.overplot = not self.overplot
            self.status.setText(
                f"{self.last[2]}\n"
                + ("overplotting: the next cut is added, not replaced"
                   if self.overplot else "replacing again"))
        elif key == "?":
            self.keys.setText("   ".join(f"[{k}] {w}" for k, w in HELP))
        elif key == "q":
            self.finished.emit()
        elif key == ":":
            self.command.setFocus()
            self.command.setText(":")
        else:
            return False
        return True

    def _row_hint(self) -> float:
        """Which image line the cursor is over, when a line is being plotted."""
        return self._cursor[0] if self.last[2].startswith("column") else self.ny / 2

    def _expand(self) -> None:
        """Two presses mark opposite corners of the region to magnify."""
        if self._corner is None:
            self._corner = self._cursor
            self.status.setText(f"{self.last[2]}\n"
                                f"corner at x={self._corner[0]:.1f} "
                                f"y={self._corner[1]:.1f} -- press e again at the "
                                f"opposite corner")
            return
        (x0, y0), (x1, y1) = self._corner, self._cursor
        self._corner = None
        self.plot.setRange(xRange=sorted((x0, x1)), yRange=sorted((y0, y1)),
                           padding=0)
        self.status.setText(f"{self.last[2]}\n"
                            f"expanded to x {min(x0, x1):.1f}..{max(x0, x1):.1f}, "
                            f"y {min(y0, y1):.1f}..{max(y0, y1):.1f}")

    def _read_off(self) -> None:
        """What the plotted curve is at the cursor -- IRAF's C."""
        x, y = self.last[0], self.last[1]
        index = int(np.abs(x - self._cursor[0]).argmin())
        self.status.setText(
            f"{self.last[2]}\n"
            f"at {x[index]:.0f}  value {y[index]:.2f}   "
            f"(cursor y = {self._cursor[1]:.2f})")

    def _run_command(self) -> None:
        text = self.command.text().strip().lstrip(":")
        self.command.clear()
        self.plot.setFocus()
        if not text:
            return
        parts = text.split()
        what = parts[0].lower()
        try:
            numbers = [int(float(p)) for p in parts[1:]]
        except ValueError:
            self.status.setText(f"{self.last[2]}\n'{text}' -- expected numbers, "
                                f"as in ':l 250 260'")
            return
        if what not in ("l", "c"):
            self.status.setText(f"{self.last[2]}\nno command ':{what}'. "
                                f"Use :l for lines or :c for columns.")
            return
        if not numbers:
            self.status.setText(f"{self.last[2]}\n':{what}' needs one or two "
                                f"numbers, as in ':{what} 250 260'")
            return
        first = numbers[0]
        last = numbers[1] if len(numbers) > 1 else first
        (self.plot_lines if what == "l" else self.plot_columns)(first, last)


def run(data, name: str = "image") -> None:
    """Open implot and wait for `q`."""
    application()
    dialog = QtWidgets.QDialog()
    dialog.setWindowTitle(f"implot · {name}")
    dialog.resize(1080, 640)
    panel = ImplotPanel(data, name)
    panel.finished.connect(dialog.accept)
    layout = QtWidgets.QVBoxLayout(dialog)
    layout.setContentsMargins(0, 0, 0, 0)
    layout.addWidget(panel)
    dialog.exec()
