"""The interactive half of icfit: the keys, the plot, the colon commands.

This is the interaction that the IRAF inventory says is "reused by dozens of tasks",
and it is the single strongest reason a long-time IRAF user can sit down in front of
harwood and work. The keys are IRAF's, all but w, for which the mouse is here.

Nothing is computed here. Every number and every line of text comes from
`core.icfitsession`, held to IRAF's by files of cursor commands and to what IRAF draws
by its graphics metacode. As in IRAF a deletion or a new parameter changes nothing
until f, and q takes the last fit: the line under the plot says when it is out of date.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pyqtgraph as pg
from PySide6 import QtCore, QtWidgets

from ..core import icfit, icfitsession
from . import keyguide, theme
from ._qt import on_main
from .theme import palette as _c

HELP = icfitsession.HELP


class IcfitPanel(QtWidgets.QWidget):
    """The fit, the points, and the keys that change them."""

    finished = QtCore.Signal()

    @classmethod
    def on(cls, session: icfitsession.IcfitSession, parent=None) -> IcfitPanel:
        """The panel of a session another has started, as identify's f."""
        return cls(session.x, session.y, parent=parent, session=session)

    def __init__(self, x, y, spec: icfit.FitSpec | None = None, title: str = "icfit",
                 xlabel: str = "x", ylabel: str = "y", parent=None, kind: str = "line",
                 xrange=None, weights=None, session=None) -> None:
        super().__init__(parent)
        self.session = session or icfitsession.IcfitSession(x, y, spec, weights, title, xlabel,
                                                            ylabel, kind, xrange)
        self._cursor = (float(np.mean(x)), float(np.mean(y))) if len(x) else (0.0, 0.0)

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        self.plot = pg.PlotWidget()
        self.plot.setBackground(_c().ground)
        theme.follow(self, [self.plot], lambda: self.redraw())
        for name in ("left", "bottom"):
            axis = self.plot.getAxis(name)
            axis.setPen(pg.mkPen(_c().line))
            axis.setTextPen(pg.mkPen(_c().muted))
            # Without this pyqtgraph factors a power of ten out of the tick
            # labels and appends it to the axis title, which then no longer
            # fits and is cut -- leaving a residual axis labelled "(x0.0".
            axis.enableAutoSIPrefix(False)
        self.plot.scene().sigMouseMoved.connect(self._mouse_moved)
        layout.addWidget(self.plot, stretch=1)

        self.status = QtWidgets.QLabel("")
        self.status.setObjectName("status")
        self.status.setWordWrap(True)
        layout.addWidget(self.status)

        self.keys = keyguide.strip(HELP)
        layout.addWidget(self.keys)

        self.command = QtWidgets.QLineEdit()
        self.command.setObjectName("command")
        self.command.returnPressed.connect(self._run_command)
        layout.addWidget(self.command)

        self.plot.setFocusPolicy(QtCore.Qt.FocusPolicy.StrongFocus)
        self.plot.installEventFilter(self)
        self.setFocusProxy(self.plot)

        self._curves: list = []
        self.redraw(rescale=True)
        self._offer()

    # ---------------------------------------------------------------- fit

    @property
    def fit(self) -> icfit.Fit:
        return self.session.result()

    @property
    def said(self) -> str:
        return self.session.stdout + self.session.stderr

    def redraw(self, rescale: bool = False) -> None:
        graph = self.session.graph()
        if graph.overplot:
            self._draw_fit(graph)
            self._draw_sample(graph)
            self.status.setText(self._status_text(graph))
            return
        self.plot.clear()
        self._curves = []
        self.plot.setLabel("bottom", graph.xlabel, color=_c().muted)
        self.plot.setLabel("left", graph.ylabel, color=_c().muted)
        self.plot.setTitle(graph.title.strip().replace("\n", "<br>"), color=_c().ink,
                           size="10pt")
        # Marks that differ in shape, so that colour is never the only channel.
        x, y = graph.kept
        if graph.kind == "line":
            self.plot.plot(x, y, pen=pg.mkPen(_c().sky, width=1.2))
        else:
            self._mark(x, y, _c().sky, "+")
        self._mark(*graph.deleted, _c().muted, "x")
        self._mark(*graph.averaged, _c().sky, "+")
        self._mark(*graph.rejected, _c().drop, "d")
        self._draw_fit(graph)
        self._draw_sample(graph)
        if rescale:
            x0, x1, y0, y1 = graph.window
            self.plot.setRange(xRange=(x0, x1), yRange=(y0, y1), padding=0)
        self.status.setText(self._status_text(graph))

    def _draw_fit(self, graph) -> None:
        x, y = graph.fit
        if len(x):
            self.plot.plot(x, y, pen=pg.mkPen(_c().accent, width=2,
                                              style=QtCore.Qt.PenStyle.DashLine))

    def _draw_sample(self, graph) -> None:
        for x1, y1, x2, y2 in graph.sample:
            self.plot.plot([x1, x2], [y1, y2], pen=pg.mkPen(_c().line, width=1.4))

    def _mark(self, x, y, colour, symbol) -> None:
        if len(x):
            self.plot.plot(x, y, pen=None, symbol=symbol, symbolSize=9, symbolBrush=None,
                           symbolPen=pg.mkPen(colour, width=1.4))

    def _status_text(self, graph) -> str:
        lines = list(graph.params)
        if self.session.stale:
            lines.append("Changed since the last fit: f fits again, q keeps the last fit.")
        return "\n".join(lines)

    # ------------------------------------------------------------- input

    def eventFilter(self, obj, event) -> bool:
        if event.type() == QtCore.QEvent.Type.KeyPress and self.key(event.text()):
            return True
        return super().eventFilter(obj, event)

    def _mouse_moved(self, position) -> None:
        box = self.plot.getPlotItem().vb
        if box.sceneBoundingRect().contains(position):
            point = box.mapSceneToView(position)
            self._cursor = (point.x(), point.y())

    def _scale(self) -> tuple[float, float]:
        """Pixels per unit on each axis, so that nearest means nearest on the
        screen, however the plot has been zoomed."""
        box = self.plot.getPlotItem().vb
        (x0, x1), (y0, y1) = box.viewRange()
        return (box.width() or 1.0) / max(x1 - x0, 1e-12), \
            (box.height() or 1.0) / max(y1 - y0, 1e-12)

    def key(self, key: str) -> bool:
        """Handle one keystroke. Returns True if it was ours."""
        if not key:
            return False
        session = self.session
        if key == ":":
            self.command.setFocus()
            self.command.setText(":")
            return True
        if key not in dict(HELP) and key != "w" and session.state != "second":
            return False
        session.key(key, *self._cursor, scale=self._scale())
        self._after()
        if session.done:
            self.finished.emit()
        return True

    def _after(self) -> None:
        """What the session said, and the graph, if it has changed."""
        said = self.session.take()
        if self.session.newgraph:
            self.redraw(rescale=True)
        else:
            self.redraw()
        if said.strip():
            self.status.setText(self.status.text() + "\n" + said.strip().splitlines()[-1])
        self._offer()

    def _offer(self) -> None:
        session = self.session
        asking = session.state not in ("fit", "second")
        if session.state == "second":
            text = "the other end of the range: s again"
        elif asking:
            text = session.prompt
        else:
            text = ":order 3   ·   press a key over the plot, or type a colon command here"
        self.command.setPlaceholderText(text)
        (self.command if asking else self.plot).setFocus()

    def _run_command(self) -> None:
        text = self.command.text().strip()
        self.command.clear()
        if self.session.state not in ("fit", "second"):
            self.session.answer(text)
        elif text.lstrip(":"):
            self.session.key(":", rest=text.lstrip(":"))
        self.plot.setFocus()
        self._after()

    def result(self) -> icfit.Fit:
        return self.fit


def apply_command(spec: icfit.FitSpec, text: str) -> icfit.FitSpec:
    """Apply one colon command to a fit specification.

    Kept out of the widget so it can be tested, and so the same strings work
    from a script, from a recipe, and from the keyboard.
    """
    parts = text.split(None, 1)
    name = parts[0].lower()
    value = parts[1].strip() if len(parts) > 1 else ""

    aliases = {"low": "low_reject", "high": "high_reject", "lowreject": "low_reject",
               "highreject": "high_reject", "func": "function", "ord": "order"}
    name = aliases.get(name, name)

    if name == "function":
        if value not in icfit.FUNCTIONS:
            raise ValueError(f"function must be one of {', '.join(icfit.FUNCTIONS)}")
        return replace(spec, function=value)
    if name == "sample":
        return replace(spec, sample=value or "*")
    if name in ("order", "niterate", "naverage"):
        try:
            number = int(float(value))
        except ValueError:
            raise ValueError(f"{name} takes a whole number") from None
        if name == "order" and number < 1:
            raise ValueError("order must be 1 or more")
        return replace(spec, **{name: number})
    if name in ("low_reject", "high_reject", "grow"):
        try:
            number = float(value)
        except ValueError:
            raise ValueError(f"{name} takes a number") from None
        return replace(spec, **{name: number})
    raise ValueError(f"no parameter called '{name}'")


@on_main
def run(x, y, spec: icfit.FitSpec | None = None, title: str = "icfit",
        xlabel: str = "x", ylabel: str = "y", kind: str = "line", xrange=None) -> icfit.Fit:
    """Open the panel, wait for `q`, and hand back the fit that was accepted."""
    from .dock import show

    panel = IcfitPanel(x, y, spec, title=title, xlabel=xlabel, ylabel=ylabel, kind=kind,
                       xrange=xrange)
    show(panel, title, size=(980, 620))
    return panel.result()
