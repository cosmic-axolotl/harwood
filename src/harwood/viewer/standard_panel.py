"""standard's bands, on the spectrum: add one by marking its ends, delete one.

The spectrum is drawn with each band as a box at its mean counts per pixel,
as IRAF's std_graph draws it. The keys are IRAF's and core.stdstar.Editor
answers them -- the same engine a cursor file drives -- so the bands chosen
here are the ones a cursor file with the same keys would choose.
"""

from __future__ import annotations

import numpy as np
import pyqtgraph as pg
from PySide6 import QtCore, QtWidgets

from . import keyguide, theme
from ._qt import on_main
from .theme import palette as _c

HELP = [("a", "add a band: press at both ends"), ("d", "delete the nearest band"),
        (":show", "list the bands"), ("r", "redraw"), ("q", "done with this spectrum")]


class StandardPanel(QtWidgets.QWidget):
    finished = QtCore.Signal()

    def __init__(self, editor, parent=None) -> None:
        super().__init__(parent)
        self.editor = editor
        self._x = float(editor.spec.wave[0])
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        self.plot = pg.PlotWidget()
        self.plot.setBackground(_c().ground)
        theme.follow(self, [self.plot])
        for axis in ("left", "bottom"):
            item = self.plot.getAxis(axis)
            item.setPen(pg.mkPen(_c().line))
            item.setTextPen(pg.mkPen(_c().muted))
            item.enableAutoSIPrefix(False)
        spec = editor.spec
        self.plot.setTitle(f"{spec.image}{spec.section}({spec.ap}): {spec.title}",
                           color=_c().muted, size="10pt")
        self.plot.setLabel("bottom", "wavelength (Angstroms)", color=_c().muted)
        self.plot.setLabel("left", "instrumental flux", color=_c().muted)
        self.plot.scene().sigMouseMoved.connect(self._moved)
        self.plot.setFocusPolicy(QtCore.Qt.FocusPolicy.StrongFocus)
        self.plot.installEventFilter(self)
        layout.addWidget(self.plot, stretch=1)
        self.status = QtWidgets.QLabel("")
        self.status.setObjectName("status")
        layout.addWidget(self.status)
        keys = keyguide.strip(HELP)
        keys.setObjectName("keys")
        layout.addWidget(keys)
        self.command = QtWidgets.QLineEdit()
        self.command.setObjectName("command")
        self.command.setPlaceholderText(":show")
        self.command.returnPressed.connect(self._run_command)
        layout.addWidget(self.command)
        self.render()

    def boxes(self) -> list[tuple[float, float, float]]:
        """(centre, width, counts per pixel) of each band measured: what
        std_graph marks."""
        spec, bands = self.editor.spec, self.editor.bands
        out = []
        for wave, dwave, total in zip(bands.waves, bands.dwaves, bands.fluxes, strict=True):
            if total == 0:
                continue
            pixels = abs(spec.wl(float(wave - dwave / 2)) - spec.wl(float(wave + dwave / 2)))
            out.append((float(wave), float(dwave), float(total) / pixels))
        return out

    def render(self) -> None:
        self.plot.clear()
        spec = self.editor.spec
        self.plot.plot(spec.wave, np.asarray(spec.y, float),
                       pen=pg.mkPen(_c().accent, width=1.2))
        for wave, dwave, level in self.boxes():
            x = [wave - dwave / 2, wave + dwave / 2, wave + dwave / 2, wave - dwave / 2,
                 wave - dwave / 2]
            height = 0.03 * max(abs(level), 1e-30)
            y = [level - height, level - height, level + height, level + height,
                 level - height]
            self.plot.plot(x, y, pen=pg.mkPen(_c().warning, width=1.4))
        pending = self.editor.pending
        if pending is not None:
            self.plot.addItem(pg.InfiniteLine(angle=90, pos=float(pending),
                                              pen=pg.mkPen(_c().warning, width=1)))
        self.status.setText(f"{len(self.editor.bands)} bands" + (
            " · a again at the other end" if pending is not None else ""))

    def _moved(self, position) -> None:
        self._x = float(self.plot.getViewBox().mapSceneToView(position).x())

    def key(self, key: str, command: str = "") -> None:
        self.editor.key(self._x, key, command)
        for message in self.editor.messages:
            if message.strip("\a"):
                self.status.setText(message.strip("\a").splitlines()[0])
        if command == "show" and self.editor.messages:
            QtWidgets.QMessageBox.information(self, "standard star data",
                                              self.editor.messages[-1])
        self.editor.messages.clear()
        if self.editor.done:
            self.finished.emit()
            return
        self.render()

    def eventFilter(self, obj, event) -> bool:  # noqa: N802 (Qt's name)
        if event.type() == QtCore.QEvent.Type.KeyPress and event.text():
            text = event.text()
            if text == ":":
                self.command.setFocus()
                self.command.setText(":")
            elif text != "I":
                self.key(text)
            return True
        return super().eventFilter(obj, event)

    def _run_command(self) -> None:
        text = self.command.text().strip().lstrip(":").strip()
        self.command.clear()
        self.plot.setFocus()
        if text:
            self.key(":", text)


@on_main
def edit(editor) -> None:
    """Show the spectrum and its bands until q."""
    from .dock import show

    spec = editor.spec
    show(StandardPanel(editor), f"standard · {spec.image}({spec.ap})", size=(1000, 600))


class WindowCursor:
    """The window in place of a cursor file, for stdstar.run."""

    def edit(self, editor) -> None:
        edit(editor)
