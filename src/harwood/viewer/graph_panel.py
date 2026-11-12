"""A core.graph.Graph on the screen: the same curves, axes and title as the
SVG a task writes without a screen, in a panel, until q."""

from __future__ import annotations

import numpy as np
import pyqtgraph as pg
from PySide6 import QtCore, QtWidgets

from . import theme
from .theme import palette as _c


class GraphPanel(QtWidgets.QWidget):
    finished = QtCore.Signal()
    #: A graph is worth going back to; see the window's finish_panel.
    keep_after_quit = True

    def __init__(self, graph, parent=None) -> None:
        super().__init__(parent)
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self.plot = pg.PlotWidget()
        self.plot.setBackground(_c().ground)
        theme.follow(self, [self.plot])
        for axis in ("left", "bottom"):
            item = self.plot.getAxis(axis)
            item.setPen(pg.mkPen(_c().line))
            item.setTextPen(pg.mkPen(_c().muted))
            item.enableAutoSIPrefix(False)
        self.graph = graph
        self.plot.setTitle("<br>".join(graph.title), color=_c().muted, size="10pt")
        self.plot.setLabel("bottom", graph.xlabel, color=_c().muted)
        self.plot.setLabel("left", graph.ylabel, color=_c().muted)
        self.plot.setLogMode(x=graph.logx and not graph.xelog,
                             y=graph.logy and not graph.elog)
        for (x, y), (ltype, color) in zip(graph.curves, graph.styles, strict=True):
            if ltype == 0:
                continue
            if graph.logy and not graph.elog:
                y = np.where(y > 0, y, np.nan)
            if graph.logx and not graph.xelog:
                x = np.where(x > 0, x, np.nan)
            self.plot.plot(graph.on_xaxis(x), graph.on_axis(y),
                           pen=_pen(color, ltype, 1.4), connect="finite")
        if graph.marks is not None:
            self.plot.plot(graph.marks[0], graph.on_axis(graph.marks[1]), pen=None, symbol="+",
                           symbolSize=6, symbolPen=pg.mkPen(_c().accent))
        dots = [m for m in graph.marklist if m[2] == "point"]
        if dots:
            self.plot.plot(graph.on_xaxis([m[0] for m in dots]),
                           graph.on_axis([m[1] for m in dots]), pen=None, symbol="o",
                           symbolSize=3, symbolBrush=pg.mkBrush(_c().accent),
                           symbolPen=None)
        self._marks: list = []
        if any(m[2] != "point" for m in graph.marklist):
            box = self.plot.getViewBox()
            box.sigRangeChanged.connect(self._draw_marks)
            box.sigResized.connect(self._draw_marks)
        if getattr(graph, "polygons", None):
            from PySide6 import QtGui

            brush = pg.mkBrush(_c().ground)
            pen = _pen(1, 1, 0.6)
            for k, (x, y) in enumerate(graph.polygons):
                poly = QtGui.QPolygonF([QtCore.QPointF(float(a), float(b))
                                        for a, b in zip(x, y, strict=True)])
                item = QtWidgets.QGraphicsPolygonItem(poly)
                item.setBrush(brush)
                item.setPen(pen)
                item.setZValue(k)
                self.plot.addItem(item)
        for x, y, text, color in getattr(graph, "texts", []):
            item = pg.TextItem(text, color=_pen(color, 1, 1.0).color(), anchor=(0.5, 0.5))
            item.setPos(float(graph.on_xaxis([x])[0]), float(graph.on_axis([y])[0]))
            self.plot.addItem(item)
        if getattr(graph, "labels", None):
            from PySide6 import QtGui

            for x, y, text, angle, ax, ay, _size, color in graph.labels:
                if not text.strip():
                    continue
                item = pg.TextItem(text, color=_pen(color, 1, 1.0).color(), anchor=(ax, ay),
                                   angle=angle)
                font = QtGui.QFontDatabase.systemFont(QtGui.QFontDatabase.SystemFont.FixedFont)
                item.setFont(font)
                item.setPos(float(x), float(y))
                self.plot.addItem(item)
        if getattr(graph, "lock_aspect", False):
            self.plot.getViewBox().setAspectLocked(True)
        if not getattr(graph, "frame", True):
            for name in ("left", "bottom"):
                self.plot.hideAxis(name)
        if graph.window is not None:
            self._set_window(graph)
        self._draw_marks()
        self.plot.installEventFilter(self)
        self.plot.setFocusPolicy(QtCore.Qt.FocusPolicy.StrongFocus)
        self.setFocusProxy(self.plot)
        layout.addWidget(self.plot, stretch=1)
        hint = QtWidgets.QLabel(graph.hint or "q when you are done")
        hint.setObjectName("readout")
        layout.addWidget(hint)

    def _draw_marks(self, *_) -> None:
        """GIO's marks, as gumark scales them: a size in NDC is a fraction of the
        plot's height (or width), a negative one is in world units."""
        graph = self.graph
        if getattr(self, "_drawing", False) or \
                not any(m[2] != "point" for m in graph.marklist):
            return
        self._drawing = True
        box = self.plot.getViewBox()
        box.disableAutoRange()
        (x1, x2), (y1, y2) = box.viewRange()
        width, height = max(box.width(), 1.0), max(box.height(), 1.0)
        # axis units per NDC: the whole page is about the plot over 0.78 of it
        per_x = (x2 - x1) / width * height / 0.78
        per_y = (y2 - y1) / 0.78
        for item in self._marks:
            self.plot.removeItem(item)
        self._marks = []
        for x, y, color in graph.mark_lines(per_x, per_y, 1.0, 1.0):
            if graph.logx and not graph.xelog:
                x = 10.0 ** x
            if graph.logy and not graph.elog:
                y = 10.0 ** y
            item = self.plot.plot(x, y, pen=_pen(color, 1, 1.0), connect="finite")
            self._marks.append(item)
        self._drawing = False

    def _set_window(self, graph) -> None:
        """The window IRAF gives the plot, and on its elog scale, where the view is
        not in what the axis says, ticks that say what they are."""
        x1, x2, y1, y2 = graph.window
        if graph.xelog:
            x1, x2 = float(graph.on_xaxis(x1)), float(graph.on_xaxis(x2))
        elif graph.logx:
            x1, x2 = float(np.log10(x1)), float(np.log10(x2))
        if graph.elog:
            top = max(abs(y1), abs(y2))
            powers = [10.0 ** k for k in range(1, int(np.log10(top)) + 1)]
            values = sorted({0.0, *powers, *(-p for p in powers if y1 < 0)})
            self.plot.getAxis("left").setTicks(
                [[(float(graph.on_axis(v)), f"{v:g}") for v in values]])
            y1, y2 = float(graph.on_axis(y1)), float(graph.on_axis(y2))
        elif graph.logy:
            y1, y2 = float(np.log10(y1)), float(np.log10(y2))
        self.plot.setRange(xRange=(x1, x2), yRange=(y1, y2), padding=0)

    def eventFilter(self, obj, event) -> bool:  # noqa: N802 (Qt's name)
        if event.type() == QtCore.QEvent.Type.KeyPress and event.text() == "q":
            self.finished.emit()
            return True
        return super().eventFilter(obj, event)


#: IRAF's colours 2-9 on the screen; 1 is the theme's accent.
_COLORS = ("#d62728", "#2ca02c", "#3f6fe0", "#17becf", "#c9a400", "#c51bb5", "#8a4fd0",
           "#5f7f7f")
_STYLES = {2: QtCore.Qt.PenStyle.DashLine, 3: QtCore.Qt.PenStyle.DotLine,
           4: QtCore.Qt.PenStyle.DashDotLine}


def _pen(color: int, ltype: int, width: float):
    """IRAF's colour and line type (1 solid, 2 dashed, 3 dotted, 4 dot-dashed)."""
    index = (max(1, int(color)) - 1) % 9
    pen = pg.mkPen(_c().accent if index == 0 else _COLORS[index - 1], width=width)
    pen.setStyle(_STYLES.get(int(ltype), QtCore.Qt.PenStyle.SolidLine))
    return pen


def run(graph, title: str) -> None:
    from .dock import application, show

    application()   # before the panel: a widget made with no QApplication aborts Qt

    show(GraphPanel(graph), title, size=(900, 560))


class MarkPanel(GraphPanel):
    """A graph to mark a value on: space (or a click) at the cursor marks,
    q ends -- histplot's and radplot's graphics cursor."""
    keep_after_quit = False     # a step in a task, not a result

    def __init__(self, graph, start: float, parent=None) -> None:
        super().__init__(graph, parent)
        self.events: list[tuple[float, float, str]] = []
        self.line = pg.InfiniteLine(angle=90 if "Histogram" in " ".join(graph.title) else 0,
                                    pos=start, pen=pg.mkPen(_c().warning, width=1.2))
        self.plot.addItem(self.line)
        self.plot.scene().sigMouseClicked.connect(self._clicked)
        self._cursor = (start, start)
        self.plot.scene().sigMouseMoved.connect(self._moved)

    def _moved(self, position) -> None:
        point = self.plot.getViewBox().mapSceneToView(position)
        self._cursor = (point.x(), point.y())

    def _mark(self, key: str) -> None:
        x, y = self._cursor
        self.events.append((x, y, key))
        self.line.setValue(x if self.line.angle == 90 else y)

    def _clicked(self, event) -> None:
        self._moved(event.scenePos())
        self._mark(" ")

    def eventFilter(self, obj, event) -> bool:  # noqa: N802 (Qt's name)
        if event.type() == QtCore.QEvent.Type.KeyPress and event.text() and \
                event.text() != "q":
            self._mark(event.text())
            return True
        return super().eventFilter(obj, event)


def mark(graph, start: float):
    """Show the graph and give back the marks made on it, q last."""
    from .dock import application, show

    application()   # before the panel: a widget made with no QApplication aborts Qt

    panel = show(MarkPanel(graph, start), graph.title[0], size=(900, 560))
    return [*panel.events, (float("nan"), float("nan"), "q")]
