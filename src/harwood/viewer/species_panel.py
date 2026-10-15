"""The spectrum with its lines named on it.

A wavelength-calibrated spectrum drawn with every catalogued line marked where
it falls, coloured by what produces it. The colour is the part that carries the
information: a stellar line and a telluric band look nothing alike on the
screen, because confusing the two is the mistake this panel exists to prevent.
An absorption at 7594 in an otherwise clean spectrum is oxygen above the
telescope, and a panel that draws it the same colour as the star's own iron is
a panel that helped you get it wrong.

    a/s/i/t   show or hide the atmospheric, stellar, interstellar or nebular marks
    f         only the features actually detected, rather than every line in range
    v         type a radial velocity and move the marks to match
    +/-       raise or lower the prominence cut
    q         leave
"""

from __future__ import annotations

import numpy as np
import pyqtgraph as pg
from PySide6 import QtCore, QtWidgets

from ..core import arc, species
from .icfit_panel import ACCENT, GROUND, INK, LINE, MUTED, PANEL, SKY

#: One colour per origin. Telluric is deliberately the odd one out -- it is
#: the only category that is not in the object at all.
COLOURS = {
    "stellar": ACCENT,
    "nebular": SKY,
    "interstellar": "#b48ead",
    "telluric": "#d16a6a",
    "lamp": MUTED,
}

HELP = [
    ("s / n / i / t", "show or hide stellar, nebular, interstellar, telluric"),
    ("f", "only the detected features, not every line in range"),
    ("v", "type a radial velocity, in km/s"),
    ("+ / -", "raise or lower the prominence cut"),
    ("q", "leave"),
]


class SpeciesPanel(QtWidgets.QWidget):
    """A calibrated spectrum with its lines identified."""

    finished = QtCore.Signal()

    def __init__(self, aperture, velocity: float = 0.0, tolerance: float = 1.0,
                 name: str = "spectrum", parent=None) -> None:
        super().__init__(parent)
        self.wave = np.asarray(aperture.wavelength, dtype=float)
        self.flux = np.asarray(aperture.flux, dtype=float)
        self.pixel = np.asarray(aperture.pixel, dtype=float)
        self.title = aperture.title or name
        self.velocity = float(velocity)
        self.tolerance = float(tolerance)
        self.prominence = 2
        self.detected_only = False
        self.showing = {name: True for name in COLOURS}
        self._marks: list = []

        self.detected = self._detect()

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(6)

        self.plot = pg.PlotWidget(background=GROUND)
        self.plot.showGrid(x=True, y=True, alpha=0.15)
        self.plot.setLabel("bottom", "wavelength", units="A")
        self.plot.setLabel("left", "flux")
        self.plot.plot(self.wave, self.flux, pen=pg.mkPen(INK, width=1))
        layout.addWidget(self.plot, 1)

        self.status = QtWidgets.QLabel()
        self.status.setStyleSheet(
            f"background:{PANEL}; color:{MUTED}; padding:6px; "
            f"border:1px solid {LINE}; font-family:monospace;")
        self.status.setWordWrap(True)
        layout.addWidget(self.status)

        self.entry = QtWidgets.QLineEdit()
        self.entry.setStyleSheet(
            f"background:{PANEL}; color:{INK}; border:1px solid {LINE}; padding:4px;")
        self.entry.hide()
        self.entry.returnPressed.connect(self._took_velocity)
        layout.addWidget(self.entry)

        self.plot.scene().sigMouseMoved.connect(self._moved)
        self.setFocusPolicy(QtCore.Qt.FocusPolicy.StrongFocus)
        self._redraw()

    # ------------------------------------------------------------ drawing

    def _detect(self) -> list[tuple[float, str]]:
        """Where the real features are, in wavelength."""
        peaks = ([(peak, "absorption") for peak in
                  arc.find_peaks(self.flux, threshold=5.0, absorption=True)]
                 + [(peak, "emission") for peak in
                    arc.find_peaks(self.flux, threshold=5.0)])
        return [(float(np.interp(peak.pixel, self.pixel, self.wave)), kind)
                for peak, kind in peaks]

    def _wanted(self) -> list:
        """The catalogue lines to mark, given the current switches.

        The choosing is `species.marks`, which needs no screen and is tested
        without one; what is left here is the drawing.
        """
        found = species.marks(
            float(self.wave.min()), float(self.wave.max()), self.velocity,
            origins={name for name, on in self.showing.items() if on},
            minimum=self.prominence,
            measured=[value for value, _ in self.detected],
            tolerance=self.tolerance)
        return [mark for mark in found if mark.detected or not self.detected_only]

    def _redraw(self) -> None:
        for item in self._marks:
            self.plot.removeItem(item)
        self._marks = []

        top = float(np.nanmax(self.flux))
        for mark in self._wanted():
            colour = COLOURS.get(mark.colour_key, MUTED)
            rule = pg.InfiniteLine(
                pos=mark.at, angle=90,
                pen=pg.mkPen(colour, width=2 if mark.detected else 1,
                             style=QtCore.Qt.PenStyle.DashLine
                             if "telluric" in mark.line.origin
                             else QtCore.Qt.PenStyle.SolidLine))
            self.plot.addItem(rule)
            label = pg.TextItem(mark.line.name, color=colour, anchor=(0, 1),
                                angle=90)
            label.setPos(mark.at, top)
            self.plot.addItem(label)
            self._marks += [rule, label]

        shown = ", ".join(name for name, on in self.showing.items() if on)
        self.status.setText(
            f"{self.title}   {len(self._marks) // 2} lines marked   "
            f"prominence >= {self.prominence}   velocity {self.velocity:+.0f} km/s\n"
            f"showing: {shown}"
            + ("   [detected features only]" if self.detected_only else "")
            + "\n" + "   ".join(f"{key} {what}" for key, what in HELP))

    # ---------------------------------------------------------- the mouse

    def _moved(self, position) -> None:
        if not self.plot.sceneBoundingRect().contains(position):
            return
        point = self.plot.getPlotItem().vb.mapSceneToView(position)
        match = species.at(float(point.x()), self.tolerance, self.velocity)
        if match.best is None:
            self.status.setText(f"{point.x():.2f} A -- nothing catalogued within "
                                f"{self.tolerance} A")
            return
        line = match.best.line
        text = (f"{point.x():.2f} A -- {line.name}: {line.describe()}"
                f"  [{match.best.offset:+.2f} A]")
        if not match.confident:
            text += "   ambiguous: " + ", ".join(
                item.line.name for item in match.candidates[1:3])
        if line.note:
            text += "\n" + line.note
        self.status.setText(text)

    # ----------------------------------------------------------- the keys

    def key(self, key: str) -> bool:
        """Handle one keystroke. Returns False when it means 'leave'."""
        toggles = {"s": "stellar", "n": "nebular", "i": "interstellar",
                   "t": "telluric"}
        if key in toggles:
            name = toggles[key]
            self.showing[name] = not self.showing[name]
        elif key == "f":
            self.detected_only = not self.detected_only
        elif key == "+":
            self.prominence = min(3, self.prominence + 1)
        elif key == "-":
            self.prominence = max(1, self.prominence - 1)
        elif key == "v":
            self.entry.setPlaceholderText("radial velocity in km/s, then Enter")
            self.entry.setText("")
            self.entry.show()
            self.entry.setFocus()
            return True
        elif key == "q":
            self.finished.emit()
            return False
        else:
            return True
        self._redraw()
        return True

    def keyPressEvent(self, event) -> None:
        # q emits `finished`, and whoever showed the panel closes it: the
        # dialog when it is on its own, the dock when it sits in the main
        # window. Closing self.window() here used to close *that* -- the
        # whole of harwood gui -- from inside lineid.
        self.key(event.text().lower())

    def _took_velocity(self) -> None:
        text = self.entry.text().strip()
        self.entry.hide()
        self.setFocus()
        try:
            self.velocity = float(text)
        except ValueError:
            self.status.setText(f"'{text}' -- expected a number, in km/s.")
            return
        self._redraw()


def run(aperture, velocity: float = 0.0, tolerance: float = 1.0,
        name: str = "spectrum") -> None:
    """Open the panel and wait for `q`."""
    from .dock import show

    show(SpeciesPanel(aperture, velocity, tolerance, name),
         f"lineid · {name}", size=(1180, 660))
