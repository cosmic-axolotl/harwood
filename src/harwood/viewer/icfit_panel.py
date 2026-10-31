"""The interactive half of icfit: the keys, the plot, the colon commands.

This is the interaction that the IRAF inventory says is "reused by dozens of
tasks", and it is the single strongest reason a long-time IRAF user can sit
down in front of harwood and work. The keys are IRAF's:

    f           refit with the current parameters
    d           delete the point nearest the cursor
    u           undelete the point nearest the cursor
    y           cycle the vertical axis: data, residuals, ratio
    r           redraw
    ?           list the keys
    q           accept the fit and leave

    :function legendre|chebyshev|spline1|spline3
    :order 3        :sample 20:500       :naverage -4
    :niterate 3     :low 3    :high 3    :grow 1

Nothing is computed here. Every number comes from `core.icfit`, so a fit made
by hand and a fit made in a script are the same fit -- which is what makes an
interactive session reproducible afterwards.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pyqtgraph as pg
from PySide6 import QtCore, QtWidgets

from ..core import gio, icfit
from . import keyguide, theme
from ._qt import on_main
from .theme import palette as _c

AXES = ("data", "residuals", "ratio")

HELP = [
    ("f", "refit with the current parameters"),
    ("d", "delete the point nearest the cursor"),
    ("u", "undelete the point nearest the cursor"),
    ("y", "cycle the vertical axis: data, residuals, ratio"),
    ("r", "redraw"),
    ("q", "accept this fit and leave"),
    (":order 3", "any icfit parameter: function, order, sample, naverage, "
                 "niterate, low, high, grow"),
]


class IcfitPanel(QtWidgets.QWidget):
    """The fit, the points, and the keys that change them."""

    finished = QtCore.Signal()

    def __init__(self, x, y, spec: icfit.FitSpec | None = None,
                 title: str = "icfit", xlabel: str = "x", ylabel: str = "y",
                 parent=None) -> None:
        super().__init__(parent)
        self.xdata = np.asarray(x, dtype=float)
        self.ydata = np.asarray(y, dtype=float)
        self.spec = spec or icfit.FitSpec()
        self.deleted = np.zeros(self.xdata.shape, dtype=bool)
        self.axis = "data"
        self.title = title
        self._cursor = (float(self.xdata.mean()), float(self.ydata.mean()))


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
        self.ylabel = ylabel
        self.plot.setLabel("bottom", xlabel, color=_c().muted)
        self.plot.setLabel("left", ylabel, color=_c().muted)
        self.plot.scene().sigMouseMoved.connect(self._mouse_moved)
        layout.addWidget(self.plot, stretch=1)

        self.status = QtWidgets.QLabel("")
        self.status.setObjectName("status")
        layout.addWidget(self.status)

        self.keys = keyguide.strip(HELP)
        layout.addWidget(self.keys)

        self.command = QtWidgets.QLineEdit()
        self.command.setObjectName("command")
        self.command.setPlaceholderText(":order 3   ·   press a key over the plot, "
                                        "or type a colon command here")
        self.command.returnPressed.connect(self._run_command)
        layout.addWidget(self.command)

        self.plot.setFocusPolicy(QtCore.Qt.FocusPolicy.StrongFocus)
        self.plot.installEventFilter(self)
        self.setFocusProxy(self.plot)

        self.refit()

    # ---------------------------------------------------------------- fit

    def refit(self) -> None:
        self.fit = icfit.fit(self.xdata, self.ydata, self.spec, deleted=self.deleted)
        self.redraw()

    def redraw(self) -> None:
        self.plot.clear()
        used = self.fit.mask & ~self.deleted
        rejected = ~self.fit.mask & ~self.deleted

        if self.axis == "residuals":
            values = self.fit.residuals
            curve_y = np.zeros_like(self.xdata)
            label = f"residual ({self.ylabel})" if self.ylabel else "residual"
        elif self.axis == "ratio":
            with np.errstate(divide="ignore", invalid="ignore"):
                values = np.where(self.fit.values != 0, self.ydata / self.fit.values, np.nan)
            curve_y = np.ones_like(self.xdata)
            label = "data / fit"
        else:
            values = self.ydata
            curve_y = None
            label = self.ylabel or "data"
        self.plot.setLabel("left", label, color=_c().muted)

        self._scatter(self.xdata[used], values[used], _c().sky, 5.0)
        self._scatter(self.xdata[rejected], values[rejected], _c().drop, 6.5, symbol="x")
        self._scatter(self.xdata[self.deleted], values[self.deleted], _c().muted, 7.0,
                      symbol="s")

        if self.fit.model is not None:
            grid = np.linspace(float(self.xdata.min()), float(self.xdata.max()), 400)
            curve = (np.interp(grid, self.xdata, curve_y) if curve_y is not None
                     else self.fit(grid))
            self.plot.plot(grid, curve, pen=pg.mkPen(_c().accent, width=2,
                                                     style=QtCore.Qt.PenStyle.DashLine))

        (x0, x1), (y0, y1) = gio.with_room(self.xdata), gio.with_room(values)
        self.plot.setRange(xRange=(x0, x1), yRange=(y0, y1), padding=0)
        self.plot.setTitle(self.title, color=_c().ink, size="10pt")
        self.status.setText(self._status_text())

    def _scatter(self, x, y, colour, size, symbol="o") -> None:
        if len(x) == 0:
            return
        self.plot.plot(x, y, pen=None, symbol=symbol, symbolSize=size,
                       symbolBrush=pg.mkBrush(colour) if symbol != "x" else None,
                       symbolPen=pg.mkPen(colour, width=1.4))

    def _status_text(self) -> str:
        spec = self.spec
        left = (f"{spec.function} order {spec.order}   sample {spec.sample}   "
                f"naverage {spec.naverage}   reject {spec.low_reject}/"
                f"{spec.high_reject} x{spec.niterate}   grow {spec.grow:g}")
        used = int((self.fit.mask & ~self.deleted).sum())
        right = (f"rms {self.fit.rms:.4g}   {used} used   "
                 f"{self.fit.nrejected} rejected   {int(self.deleted.sum())} deleted")
        if self.fit.message:
            right += f"   [{self.fit.message}]"
        return f"{left}\n{right}"

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

    def key(self, key: str) -> bool:
        """Handle one keystroke. Returns True if it was ours."""
        if not key:
            return False
        if key == "f":
            self.refit()
        elif key == "r":
            self.redraw()
        elif key == "d":
            self._toggle_nearest(True)
        elif key == "u":
            self._toggle_nearest(False)
        elif key == "y":
            self.axis = AXES[(AXES.index(self.axis) + 1) % len(AXES)]
            self.redraw()
        elif key == "?":
            self.command.setText("")
            self.keys.show()                      # every key is listed already
        elif key == "q":
            self.finished.emit()
        elif key == ":":
            self.command.setFocus()
            self.command.setText(":")
        else:
            return False
        return True

    def _toggle_nearest(self, delete: bool) -> None:
        """Delete or restore the point nearest the cursor.

        Distance is measured in plot units scaled by the visible range, so
        "nearest" means nearest *on screen* -- otherwise whichever axis happens
        to carry bigger numbers decides, and the wrong point is picked.
        """
        (x0, x1), (y0, y1) = self.plot.getPlotItem().vb.viewRange()
        span_x = max(x1 - x0, 1e-12)
        span_y = max(y1 - y0, 1e-12)
        values = self.ydata if self.axis == "data" else self.fit.residuals
        distance = np.hypot((self.xdata - self._cursor[0]) / span_x,
                            (values - self._cursor[1]) / span_y)
        candidates = (np.flatnonzero(~self.deleted) if delete
                      else np.flatnonzero(self.deleted))
        if candidates.size == 0:
            return
        index = candidates[np.argmin(distance[candidates])]
        self.deleted[index] = delete
        self.refit()

    def _run_command(self) -> None:
        text = self.command.text().strip().lstrip(":")
        self.command.clear()
        self.plot.setFocus()
        if not text:
            return
        try:
            self.spec = apply_command(self.spec, text)
        except ValueError as error:
            self.status.setText(f"{self._status_text()}\n[{error}]")
            return
        self.refit()

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
        xlabel: str = "x", ylabel: str = "y") -> icfit.Fit:
    """Open the panel, wait for `q`, and hand back the fit that was accepted."""
    from .dock import show

    panel = IcfitPanel(x, y, spec, title=title, xlabel=xlabel, ylabel=ylabel)
    show(panel, title, size=(980, 620))
    return panel.result()
