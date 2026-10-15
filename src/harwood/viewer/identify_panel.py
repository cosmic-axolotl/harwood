"""Marking arc lines by hand: IRAF's identify, with its keys.

The arc is on top, the dispersion fit below. Mark a few lines, fit, then let
the line list place the rest -- the sequence the IRAF tutorial walks through,
and the only reliable way through a thorium lamp whose dispersion you do not
already know.

    m     mark the line nearest the cursor, then type its wavelength
    d     delete the identification nearest the cursor
    l     add every catalogue line the current fit explains
    f     refit
    .     list the identifications
    q     accept and leave

The fit half is the same `IcfitPanel` the overscan uses, so `:function
chebyshev`, `:order 3` and `:niterate 2` behave identically here -- which is
the point of icfit being infrastructure rather than a feature of one task.
"""

from __future__ import annotations

import numpy as np
import pyqtgraph as pg
from PySide6 import QtCore, QtWidgets

from ..core import arc, icfit
from .icfit_panel import ACCENT, GROUND, INK, LINE, MUTED, PANEL, SKY, IcfitPanel

HELP = [
    ("m", "mark the line nearest the cursor, then type its wavelength"),
    ("d", "delete the identification nearest the cursor"),
    ("l", "add every catalogue line the current fit explains"),
    ("f", "refit"),
    (".", "list the identifications"),
    ("q", "accept this solution and leave"),
]


class IdentifyPanel(QtWidgets.QWidget):
    """The arc, the identifications, and the dispersion fit."""

    finished = QtCore.Signal()

    def __init__(self, flux, peaks, catalogue, spec: icfit.FitSpec | None = None,
                 match: float = 1.5, units: str = "Angstroms", parent=None) -> None:
        super().__init__(parent)
        self.flux = np.asarray(flux, dtype=float)
        self.pixels = np.arange(1.0, self.flux.size + 1.0)
        self.peaks = list(peaks)
        self.catalogue = np.asarray(catalogue, dtype=float)
        self.spec = spec or icfit.FitSpec(function="chebyshev", order=3)
        self.match = match
        self.units = units

        self.marked_pixel: list[float] = []
        self.marked_wavelength: list[float] = []
        self._cursor = float(self.flux.size / 2)
        self._pending: float | None = None

        self.setStyleSheet(f"""
            QWidget {{ background: {GROUND}; color: {INK};
                font-family: "IBM Plex Sans", "DejaVu Sans", sans-serif; font-size: 12px; }}
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

        self.arc_plot = pg.PlotWidget()
        self.arc_plot.setBackground(GROUND)
        for name in ("left", "bottom"):
            self.arc_plot.getAxis(name).setPen(pg.mkPen(LINE))
            self.arc_plot.getAxis(name).setTextPen(pg.mkPen(MUTED))
        self.arc_plot.setLabel("bottom", "pixel", color=MUTED)
        self.arc_plot.setLabel("left", "counts", color=MUTED)
        self.arc_plot.scene().sigMouseMoved.connect(self._mouse_moved)
        self.arc_plot.setFocusPolicy(QtCore.Qt.FocusPolicy.StrongFocus)
        self.arc_plot.installEventFilter(self)
        layout.addWidget(self.arc_plot, stretch=3)

        self.status = QtWidgets.QLabel("")
        self.status.setObjectName("status")
        layout.addWidget(self.status)

        self.fit_panel = IcfitPanel([1.0, 2.0], [1.0, 2.0], self.spec,
                                    title="dispersion", xlabel="pixel",
                                    ylabel=units)
        self.fit_panel.keys.hide()
        self.fit_panel.command.hide()
        layout.addWidget(self.fit_panel, stretch=2)

        self.keys = QtWidgets.QLabel(
            "   ".join(f"[{key}] {what}" for key, what in HELP[:4]))
        self.keys.setObjectName("keys")
        layout.addWidget(self.keys)

        self.command = QtWidgets.QLineEdit()
        self.command.setPlaceholderText(
            "press m over a line, or type :order 3 / :function chebyshev")
        self.command.returnPressed.connect(self._run_command)
        layout.addWidget(self.command)

        self.setFocusProxy(self.arc_plot)
        self.redraw_arc()
        self.refit()

    # --------------------------------------------------------------- draw

    def redraw_arc(self) -> None:
        self.arc_plot.clear()
        self.arc_plot.plot(self.pixels, self.flux, pen=pg.mkPen(SKY, width=1))

        span = float(self.flux.max() - self.flux.min()) or 1.0
        for peak in self.peaks:
            self.arc_plot.addItem(pg.InfiniteLine(
                peak.pixel, angle=90, pen=pg.mkPen(LINE, width=1)))
        # Every identification gets its marker; only those with room get their
        # wavelength written. An arc is crowded, and labels printed on top of
        # one another are worse than no labels -- they read as one wrong number.
        spacing = max(12.0, self.flux.size / 45)
        last_labelled = -1e9
        for pixel, wavelength in zip(self.marked_pixel, self.marked_wavelength, strict=False):
            self.arc_plot.addItem(pg.InfiniteLine(
                pixel, angle=90, pen=pg.mkPen(ACCENT, width=1.4)))
            if pixel - last_labelled < spacing:
                continue
            last_labelled = pixel
            label = pg.TextItem(f"{wavelength:.2f}", color=ACCENT, anchor=(0.5, 1.1),
                                angle=90)
            label.setPos(pixel, self.flux.min() + 0.82 * span)
            self.arc_plot.addItem(label)
        self.arc_plot.setTitle(
            f"identify · {len(self.marked_pixel)} lines marked of "
            f"{len(self.peaks)} found", color=INK, size="10pt")

    def refit(self) -> None:
        if len(self.marked_pixel) >= 2:
            self.fit_panel.xdata = np.array(self.marked_pixel)
            self.fit_panel.ydata = np.array(self.marked_wavelength)
            self.fit_panel.deleted = np.zeros(len(self.marked_pixel), dtype=bool)
            self.fit_panel.spec = self.spec
            self.fit_panel.refit()
        self._update_status()

    def _update_status(self) -> None:
        count = len(self.marked_pixel)
        if count < 2:
            self.status.setText(
                f"{count} line(s) marked. Mark at least four spread across the "
                f"spectrum, then press f.\n{len(self.peaks)} peaks found.")
            return
        fit = self.fit_panel.fit
        scale = abs(float(np.median(np.gradient(fit(self.pixels))))) if fit.model \
            is not None else float("nan")
        rms_px = fit.rms / scale if scale else float("nan")
        self.status.setText(
            f"{self.spec.function} order {self.spec.order}   {count} lines   "
            f"rms {fit.rms:.4f} {self.units} ({rms_px:.2f} px)\n"
            + (f"range {fit(np.array([1.0]))[0]:.2f} to "
               f"{fit(np.array([float(self.flux.size)]))[0]:.2f} {self.units}"
               if fit.model is not None else fit.message))

    # -------------------------------------------------------------- input

    def eventFilter(self, obj, event) -> bool:
        if event.type() == QtCore.QEvent.Type.KeyPress and self.key(event.text()):
            return True
        return super().eventFilter(obj, event)

    def _mouse_moved(self, position) -> None:
        box = self.arc_plot.getPlotItem().vb
        if box.sceneBoundingRect().contains(position):
            self._cursor = float(box.mapSceneToView(position).x())

    def key(self, key: str) -> bool:
        if key == "m":
            self._mark()
        elif key == "d":
            self._delete()
        elif key == "l":
            self._from_list()
        elif key == "f":
            self.refit()
        elif key == ".":
            self._list()
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

    def _nearest_peak(self, pixel: float):
        if not self.peaks:
            return None
        return min(self.peaks, key=lambda p: abs(p.pixel - pixel))

    def _mark(self) -> None:
        """Centre the nearest peak and ask what wavelength it is."""
        peak = self._nearest_peak(self._cursor)
        if peak is None or abs(peak.pixel - self._cursor) > 10:
            self.status.setText("no arc line near the cursor.\n"
                                "Point closer, or lower --threshold and run again.")
            return
        self._pending = peak.pixel
        suggestion = self._suggest(peak.pixel)
        self.command.setText(f"{suggestion:.4f}" if suggestion else "")
        self.command.setPlaceholderText(
            f"wavelength at pixel {peak.pixel:.2f}, in {self.units}")
        self.command.setFocus()
        self.command.selectAll()

    def _suggest(self, pixel: float) -> float | None:
        """What the current solution says this line is, if there is one."""
        if len(self.marked_pixel) < 2 or self.fit_panel.fit.model is None:
            return None
        predicted = float(self.fit_panel.fit(np.array([pixel]))[0])
        if self.catalogue.size == 0:
            return predicted
        return float(self.catalogue[np.abs(self.catalogue - predicted).argmin()])

    def _delete(self) -> None:
        if not self.marked_pixel:
            return
        index = int(np.argmin(np.abs(np.array(self.marked_pixel) - self._cursor)))
        gone = self.marked_wavelength.pop(index)
        self.marked_pixel.pop(index)
        self.redraw_arc()
        self.refit()
        self.status.setText(f"deleted {gone:.4f} {self.units}\n"
                            f"{len(self.marked_pixel)} lines left")

    def _from_list(self) -> None:
        """IRAF's `l`: the current solution places the rest of the catalogue."""
        if len(self.marked_pixel) < 2 or self.fit_panel.fit.model is None:
            self.status.setText("mark at least two lines and fit before pressing l.\n")
            return
        pixel, wavelength = arc.add_from_list(
            self.peaks, self.fit_panel.fit, self.catalogue, tolerance=self.match)
        if pixel.size <= len(self.marked_pixel):
            self.status.setText("the line list adds nothing the fit does not "
                                "already explain.\n")
            return
        self.marked_pixel = list(pixel)
        self.marked_wavelength = list(wavelength)
        self.redraw_arc()
        self.refit()

    def _list(self) -> None:
        lines = [f"{p:9.3f}  {w:12.4f}" for p, w in
                 zip(self.marked_pixel, self.marked_wavelength, strict=False)]
        self.status.setText("pixel     wavelength\n" + "   ".join(lines[:6])
                            + (" ..." if len(lines) > 6 else ""))

    def _run_command(self) -> None:
        from .icfit_panel import apply_command

        text = self.command.text().strip()
        self.command.clear()
        self.command.setPlaceholderText(
            "press m over a line, or type :order 3 / :function chebyshev")
        self.arc_plot.setFocus()
        if not text:
            self._pending = None
            return

        if self._pending is not None and not text.startswith(":"):
            try:
                wavelength = float(text)
            except ValueError:
                self.status.setText(f"'{text}' is not a wavelength.\n")
                self._pending = None
                return
            self.marked_pixel.append(self._pending)
            self.marked_wavelength.append(wavelength)
            order = np.argsort(self.marked_pixel)
            self.marked_pixel = [self.marked_pixel[i] for i in order]
            self.marked_wavelength = [self.marked_wavelength[i] for i in order]
            self._pending = None
            self.redraw_arc()
            self.refit()
            return

        try:
            self.spec = apply_command(self.spec, text.lstrip(":"))
        except ValueError as error:
            self.status.setText(f"{error}\n")
            return
        self.refit()

    def result(self):
        return (self.fit_panel.fit, np.array(self.marked_pixel),
                np.array(self.marked_wavelength))


def run(flux, peaks, catalogue, spec=None, title="identify", match: float = 1.5,
        units: str = "Angstroms"):
    """Open the panel and hand back the solution that was accepted."""
    from .dock import show

    panel = IdentifyPanel(flux, peaks, catalogue, spec, match=match, units=units)
    show(panel, title, size=(1150, 780))
    return panel.result()
