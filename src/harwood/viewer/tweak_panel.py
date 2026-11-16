"""telluric and skytweak by eye -- IRAF's twk_fit -- and its keys.

Nothing is decided here: every key goes to `core.telluric.FitSession`, which is held
to IRAF's by files of cursor reads. Above, the three candidates: the spectrum corrected
with the current shift and scale in the middle, and a step less and a step more of
one of them below and above, the offset apart. Below, the calibration where the
spectrum is, or the spectrum (c and d). The keys act where the mouse is: x and y
choose the candidate under it, w and :/ change the window of the graph it is over.

Beside IRAF's graph each candidate says what it is, at its right end, and the sample
is shaded as well as marked: what is chosen with x and y is then read, not counted.
"""

from __future__ import annotations

import numpy as np
import pyqtgraph as pg
from PySide6 import QtCore, QtWidgets

from ..core import telluric
from ..core.i18n import N_, tr
from . import gtplot, keyguide, theme
from ._qt import on_main
from .theme import palette as _c

PLACEHOLDER = N_(":shift 0.5   ·   :scale 1.1   ·   :sample 6850:6900   ·   press a key "
                 "over the graphs, or type a colon command here")


class TweakPanel(QtWidgets.QWidget):
    """The candidates, the graph below, and the keys that choose."""

    finished = QtCore.Signal()

    def __init__(self, session: telluric.FitSession, parent=None) -> None:
        super().__init__(parent)
        self.session = session
        self._cursor = (0.0, 0.0)
        self._wcs = 1
        self._seen = len(session.out)

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        self.plots = [pg.PlotWidget(), pg.PlotWidget()]
        theme.follow(self, self.plots, lambda: self.redraw())
        for at, plot in enumerate(self.plots):
            plot.setBackground(_c().ground)
            for name in ("left", "bottom"):
                axis = plot.getAxis(name)
                axis.setPen(pg.mkPen(_c().line))
                axis.setTextPen(pg.mkPen(_c().muted))
                axis.enableAutoSIPrefix(False)
            plot.getAxis("left").setWidth(70)       # the two graphs' x axes line up
            plot.scene().sigMouseMoved.connect(
                lambda position, at=at: self._mouse_moved(at, position))
            plot.setFocusPolicy(QtCore.Qt.FocusPolicy.StrongFocus)
            plot.installEventFilter(self)
            layout.addWidget(plot, stretch=5 if at == 0 else 3)
        self.status = QtWidgets.QLabel("")
        self.status.setObjectName("status")
        self.status.setWordWrap(True)
        self.status.setTextInteractionFlags(
            QtCore.Qt.TextInteractionFlag.TextSelectableByMouse)
        layout.addWidget(self.status)
        layout.addWidget(keyguide.strip(telluric.KEYS))
        self.command = QtWidgets.QLineEdit()
        self.command.setObjectName("command")
        self.command.setPlaceholderText(tr(PLACEHOLDER))
        self.command.returnPressed.connect(self._run_command)
        layout.addWidget(self.command)
        self.setFocusProxy(self.plots[0])
        self.redraw(rescale=True)
        self._say()

    # ------------------------------------------------------------ drawing

    def labels(self) -> list[str]:
        """What each candidate is, lowest first."""
        session = self.session
        if session.graph1 == "x":
            return [f"shift {float(v):.2f}" for v in session.shifts]
        return [f"scale {float(v):.4g}" for v in session.scales]

    def redraw(self, rescale: bool = False) -> None:
        frame = self.session.frames[-1]
        upper, lower = self.plots
        upper.clear()
        upper.setTitle("<br>".join(frame.title), color=_c().ink, size="10pt")
        turned, logs = frame.transposed[0], frame.logs[0]
        x = frame.x.astype(float)
        colours = (_c().band, _c().sky, _c().band)
        for curve, colour, text in zip(frame.curves, colours, self.labels(), strict=True):
            xs, ys = (curve, x) if turned else (x, curve)
            px, py = gtplot.on_axes(logs, xs, ys)
            upper.plot(px, py, pen=pg.mkPen(colour, width=1.2))
            if not turned and np.isfinite(py).any():
                label = pg.TextItem(text, color=_c().ink, anchor=(1.0, 1.0))
                label.setPos(float(px[-1]), float(py[np.isfinite(py)][-1]))
                upper.addItem(label)
        # the sample: IRAF's ticks and bar, and the region shaded
        y1, y2, y3 = frame.heights
        pen = pg.mkPen(_c().accent, width=1.5)
        for xa, xb, tick_a, tick_b in frame.regions:
            if turned:
                continue
            upper.plot(*gtplot.on_axes(logs, [xa, xb], [y3, y3]), pen=pen)
            for at, tick in ((xa, tick_a), (xb, tick_b)):
                if tick:
                    upper.plot(*gtplot.on_axes(logs, [at, at], [y1, y2]), pen=pen)
            if not logs[0]:
                region = pg.LinearRegionItem(sorted((xa, xb)), movable=False,
                                             brush=pg.mkBrush(_c().accent + "22"),
                                             pen=pg.mkPen(None))
                region.setZValue(-10)
                upper.addItem(region)
        upper.setLabel("bottom", frame.xlabel if frame.below is None else "",
                       color=_c().muted)
        lower.setVisible(frame.below is not None)
        if frame.below is not None:
            lower.clear()
            turned2, logs2 = frame.transposed[1], frame.logs[1]
            xs, ys = (frame.below, x) if turned2 else (x, frame.below)
            lower.plot(*gtplot.on_axes(logs2, xs, ys), pen=pg.mkPen(_c().muted, width=1.2))
            lower.setLabel("bottom", frame.xlabel, color=_c().muted)
            what = tr(N_("calibration")) if self.session.graph2 == "cal" \
                else tr(N_("spectrum"))
            lower.setLabel("left", what, color=_c().muted)
        if rescale:
            gtplot.set_window(upper, frame.windows[0], logs)
            if frame.below is not None:
                gtplot.set_window(lower, frame.windows[1], frame.logs[1])

    def _say(self) -> None:
        """What IRAF would print: the last lines of it, and the prompt waiting."""
        session = self.session
        text = session.out[self._seen:].replace("\a", "")
        if session.err:
            text += session.err
            session.err = ""
        lines = [line for line in text.splitlines() if line.strip()]
        twk = session.twk
        which = tr(N_("candidates by shift")) if session.graph1 == "x" \
            else tr(N_("candidates by scale"))
        summary = (f"shift {float(twk.shift):.2f} (± {float(twk.dshift):.2f}), "
                   f"scale {float(twk.scale):.4g} (± {float(twk.dscale):.3g}), "
                   f"sample {twk.sample}: {which}.")
        self.status.setText("\n".join([*lines[-8:], summary]))

    # ------------------------------------------------------------ the keys

    def eventFilter(self, obj, event) -> bool:
        if event.type() == QtCore.QEvent.Type.KeyPress and self.key(event.text()):
            return True
        return super().eventFilter(obj, event)

    def _mouse_moved(self, at: int, position) -> None:
        box = self.plots[at].getPlotItem().vb
        if box.sceneBoundingRect().contains(position):
            point = box.mapSceneToView(position)
            frame = self.session.frames[-1]
            logs = frame.logs[at] if at < len(frame.logs) else (False, False)
            self._cursor = gtplot.world(logs, point.x(), point.y())
            self._wcs = at + 1
            self.plots[at].setFocus()

    def key(self, key: str, rest: str = "") -> bool:
        """Handle one keystroke (or, with ':', a colon command). True if it was ours."""
        session = self.session
        if not key:
            return False
        if key == ":" and not rest and not session.state:
            self.command.setFocus()
            self.command.setText(":")
            return True
        before = len(session.frames)
        if not session.state:
            self._seen = len(session.out)
        try:
            session.key(key, *self._cursor, wcs=self._wcs, rest=rest)
        except ValueError as error:
            # what ends IRAF's task (a calibration at or below zero) is said here
            session.out += f"{error}\n"
        if session.done:
            self.finished.emit()
            return True
        self.redraw(rescale=len(session.frames) != before)
        self._say()
        return True

    def _run_command(self) -> None:
        text = self.command.text().strip().lstrip(":")
        self.command.clear()
        self.plots[self._wcs - 1].setFocus()
        if text.startswith("."):
            self.session.out += gtplot.cursor_mode(self.plots[self._wcs - 1], text[1:],
                                                   "tweak.png")
            self._say()
        elif text:
            self.key(":", rest=text)


@on_main
def run(session: telluric.FitSession, title: str = "telluric") -> None:
    """Open the panel and wait for q."""
    from .dock import application, show

    application()   # before the panel: a widget made with no QApplication aborts Qt

    session.done = False
    show(TweakPanel(session), title, size=(1000, 720))
