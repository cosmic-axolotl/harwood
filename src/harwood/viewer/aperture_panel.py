"""apedit: setting the apertures by hand, on the profile they come from.

`apfind` puts an aperture where the profile peaks and sizes it where the
profile falls away, and most of the time that is right. The times it is not are
the ones that matter: two stars on the slit that the finder merged, a galaxy
whose light reaches further than a threshold says, a sky band that lands on a
neighbour. None of those is a parameter you can set in advance -- you have to
look at the profile.

So the profile is what this shows, with every aperture drawn on it, and the
keys are IRAF's:

    a          add an aperture at the cursor
    d          delete the one nearest the cursor
    c          re-centre the nearest aperture on the cursor
    l  u       set its lower or upper edge at the cursor
    b          set the background band edges, inner then outer
    g          re-centre every aperture on the profile
    z          fit every aperture's edges to the profile again
    n  p       select the next or previous aperture
    r          redraw
    ?          the keys
    q          accept and leave

Positions across the slit are 1-based, as everywhere a user can see.
"""

from __future__ import annotations

import numpy as np
import pyqtgraph as pg
from PySide6 import QtCore, QtWidgets

from ..core import apextract
from . import theme
from ._qt import application, on_main
from .theme import palette as _c

HELP = [
    ("a", "add an aperture at the cursor"),
    ("d", "delete the nearest one"),
    ("c", "re-centre the nearest on the cursor"),
    ("l / u", "set its lower or upper edge here"),
    ("b", "set the background band, inner then outer"),
    ("g", "re-centre every aperture"),
    ("z", "fit every aperture's edges again"),
    ("n / p", "select the next or previous"),
    ("q", "accept and leave"),
]


class AperturePanel(QtWidgets.QWidget):
    """The spatial profile, the apertures on it, and the keys that move them."""

    finished = QtCore.Signal()

    def __init__(self, profile, apertures: list[dict], background=None,
                 ylevel: float = 0.1, parent=None) -> None:
        super().__init__(parent)
        self.profile = np.asarray(profile, dtype=float)
        self.across = np.arange(1.0, self.profile.size + 1.0)
        self.apertures = [dict(aperture) for aperture in apertures]
        self.background = list(background) if background else None
        self.ylevel = ylevel
        self.selected = 0
        self._cursor = float(self.profile.size / 2)
        self._pending_band: float | None = None


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
            axis.enableAutoSIPrefix(False)
        self.plot.setLabel("bottom", "across the slit (pixels)", color=_c().muted)
        self.plot.setLabel("left", "summed counts", color=_c().muted)
        self.plot.scene().sigMouseMoved.connect(self._mouse_moved)
        self.plot.setFocusPolicy(QtCore.Qt.FocusPolicy.StrongFocus)
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
        self.command.setObjectName("command")
        self.command.setPlaceholderText(
            ":lower -4   ·   :upper 4   ·   :centre 53.9   ·   ? for the keys")
        self.command.returnPressed.connect(self._run_command)
        layout.addWidget(self.command)

        self.setFocusProxy(self.plot)
        self.redraw()

    # --------------------------------------------------------------- draw

    def redraw(self) -> None:
        self.plot.clear()
        self.plot.plot(self.across, self.profile, pen=pg.mkPen(_c().sky, width=1.2))

        span = float(np.ptp(self.profile)) or 1.0
        floor = float(self.profile.min())
        for index, aperture in enumerate(self.apertures):
            chosen = index == self.selected
            width = 2.0 if chosen else 1.2
            centre = aperture["centre"]
            for offset, colour in ((aperture["low"], _c().accent),
                                   (aperture["high"], _c().accent)):
                self.plot.addItem(pg.InfiniteLine(
                    centre + offset, angle=90,
                    pen=pg.mkPen(colour, width=width,
                                 style=QtCore.Qt.PenStyle.SolidLine if chosen
                                 else QtCore.Qt.PenStyle.DashLine)))
            self.plot.addItem(pg.InfiniteLine(
                centre, angle=90,
                pen=pg.mkPen(_c().accent if chosen else _c().muted, width=width,
                             style=QtCore.Qt.PenStyle.DotLine)))

            label = pg.TextItem(f"{index + 1}", color=_c().accent if chosen else _c().muted,
                                anchor=(0.5, 1.0))
            label.setPos(centre, floor + 0.95 * span)
            self.plot.addItem(label)

            if self.background:
                inner, outer = self.background
                for low, high in ((centre + aperture["low"] - outer,
                                   centre + aperture["low"] - inner),
                                  (centre + aperture["high"] + inner,
                                   centre + aperture["high"] + outer)):
                    fill = pg.mkBrush(theme.translucent(_c().band, 40))
                    region = pg.LinearRegionItem([low, high], movable=False, brush=fill)
                    region.setZValue(-10)
                    self.plot.addItem(region)

        self.plot.setTitle(f"apedit · {len(self.apertures)} aperture(s)",
                           color=_c().ink, size="10pt")
        self._update_status()

    def _update_status(self) -> None:
        if not self.apertures:
            self.status.setText("no apertures. Press a over a peak to add one.\n"
                                "q accepts what is here; with none, apall stops.")
            return
        aperture = self.apertures[self.selected]
        band = (f"   background {self.background[0]:g} to {self.background[1]:g} "
                f"px outside" if self.background else "   no background")
        self.status.setText(
            f"aperture {self.selected + 1} of {len(self.apertures)}   "
            f"centre {aperture['centre']:.2f}   "
            f"{aperture['low']:+.2f} to {aperture['high']:+.2f}   "
            f"({aperture['high'] - aperture['low']:.2f} px wide){band}\n"
            "press ? for the keys")

    # -------------------------------------------------------------- input

    def eventFilter(self, obj, event) -> bool:
        if event.type() == QtCore.QEvent.Type.KeyPress and self.key(event.text()):
            return True
        return super().eventFilter(obj, event)

    def _mouse_moved(self, position) -> None:
        box = self.plot.getPlotItem().vb
        if box.sceneBoundingRect().contains(position):
            self._cursor = float(box.mapSceneToView(position).x())

    def _nearest(self) -> int | None:
        if not self.apertures:
            return None
        distances = [abs(aperture["centre"] - self._cursor)
                     for aperture in self.apertures]
        return int(np.argmin(distances))

    def key(self, key: str) -> bool:
        if key == "a":
            self._add()
        elif key == "d":
            self._delete()
        elif key == "c":
            self._recentre_one()
        elif key in ("l", "u"):
            self._set_edge(key)
        elif key == "b":
            self._set_band()
        elif key == "g":
            self._recentre_all()
        elif key == "z":
            self._resize_all()
        elif key in ("n", "p"):
            if self.apertures:
                step = 1 if key == "n" else -1
                self.selected = (self.selected + step) % len(self.apertures)
                self.redraw()
        elif key == "r":
            self.plot.autoRange(padding=0.02)
            self.redraw()
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

    def _add(self) -> None:
        """A new aperture, centred on whatever the cursor is nearest."""
        centre = self._centre_on(self._cursor)
        low, high = apextract.ylevel(self.profile, centre, self.ylevel,
                                     low=-self.profile.size, high=self.profile.size)
        self.apertures.append({"centre": centre, "low": low, "high": high})
        self.apertures.sort(key=lambda aperture: aperture["centre"])
        self.selected = next(index for index, aperture in enumerate(self.apertures)
                             if aperture["centre"] == centre)
        self.redraw()

    def _delete(self) -> None:
        index = self._nearest()
        if index is None:
            return
        gone = self.apertures.pop(index)
        self.selected = max(0, min(self.selected, len(self.apertures) - 1))
        self.redraw()
        self.status.setText(f"deleted the aperture at {gone['centre']:.2f}\n"
                            f"{len(self.apertures)} left")

    def _recentre_one(self) -> None:
        index = self._nearest()
        if index is None:
            return
        self.selected = index
        self.apertures[index]["centre"] = self._centre_on(self._cursor)
        self.redraw()

    def _recentre_all(self) -> None:
        for aperture in self.apertures:
            aperture["centre"] = self._centre_on(aperture["centre"])
        self.redraw()

    def _resize_all(self) -> None:
        for aperture in self.apertures:
            low, high = apextract.ylevel(self.profile, aperture["centre"], self.ylevel,
                                         low=-self.profile.size, high=self.profile.size)
            aperture["low"], aperture["high"] = low, high
        self.redraw()

    def _set_edge(self, which: str) -> None:
        """An edge goes where the cursor is, relative to the centre.

        Refuses to put the lower edge above the centre or the upper edge below
        it: an aperture that does not contain its own centre is not a mistake
        anyone means to make, and it extracts nothing.
        """
        if not self.apertures:
            return
        aperture = self.apertures[self.selected]
        offset = self._cursor - aperture["centre"]
        if which == "l" and offset >= 0:
            self.status.setText("the lower edge has to be below the centre.\n"
                                "Point to the left of the dotted line.")
            return
        if which == "u" and offset <= 0:
            self.status.setText("the upper edge has to be above the centre.\n"
                                "Point to the right of the dotted line.")
            return
        aperture["low" if which == "l" else "high"] = float(offset)
        self.redraw()

    def _set_band(self) -> None:
        """Two presses: the inner edge of the background band, then the outer."""
        if not self.apertures:
            return
        aperture = self.apertures[self.selected]
        distance = abs(self._cursor - aperture["centre"])
        edge = max(abs(aperture["low"]), abs(aperture["high"]))
        offset = max(0.0, distance - edge)

        if self._pending_band is None:
            self._pending_band = offset
            self.status.setText(
                f"inner edge of the background band at {offset:.1f} px "
                f"outside the aperture.\nPress b again at the outer edge.")
            return
        inner, outer = sorted((self._pending_band, offset))
        self._pending_band = None
        if outer - inner < 1.0:
            self.status.setText("that background band is less than a pixel wide.\n"
                                "Press b twice, further apart.")
            return
        self.background = [inner, outer]
        self.redraw()

    def _centre_on(self, position: float, half: int = 4,
                   passes: int = 4) -> float:
        """Centroid the profile near a position, falling back to the position.

        Iterated, because a centroid measured in a box that is not centred on
        the peak is pulled towards the middle of the box: point three pixels
        off a star and one pass lands a pixel short. Each pass re-centres the
        box on the last answer, and it settles in two or three.
        """
        current = float(position)
        floor = float(np.median(self.profile))

        # Centroiding is a refinement, not a search, and on featureless profile
        # it is meaningless: the iteration walks off down whatever slope the
        # noise offers. Whether there is anything to centre on is a question
        # about signal, not about distance -- snapping two pixels onto a faint
        # star is wanted, drifting two pixels across bare sky is not.
        noise = 1.4826 * float(np.median(np.abs(self.profile - floor))) or 1.0
        index = int(round(current)) - 1
        low = max(0, index - half)
        high = min(self.profile.size, index + half + 1)
        if high - low < 3 or (self.profile[low:high].max() - floor) < 3 * noise:
            return current

        for _ in range(max(1, passes)):
            index = int(round(current)) - 1
            low = max(0, index - half)
            high = min(self.profile.size, index + half + 1)
            if high - low < 3:
                return current
            weights = np.clip(self.profile[low:high] - floor, 0, None)
            if weights.sum() <= 0:
                return current
            positions = np.arange(low, high)
            moved = float((weights * positions).sum() / weights.sum()) + 1.0
            if abs(moved - current) < 0.01:
                return moved
            current = moved
        return current

    def _run_command(self) -> None:
        text = self.command.text().strip().lstrip(":")
        self.command.clear()
        self.plot.setFocus()
        if not text or not self.apertures:
            return
        parts = text.split()
        name = parts[0].lower()
        try:
            value = float(parts[1]) if len(parts) > 1 else None
        except ValueError:
            self.status.setText(f"'{parts[1]}' is not a number.\n")
            return
        if value is None:
            self.status.setText(f":{name} needs a number.\n")
            return

        aperture = self.apertures[self.selected]
        if name in ("lower", "low", "l"):
            aperture["low"] = value
        elif name in ("upper", "high", "u"):
            aperture["high"] = value
        elif name in ("centre", "center", "c"):
            aperture["centre"] = value
        elif name in ("ylevel", "y"):
            self.ylevel = value
            self._resize_all()
        else:
            self.status.setText(f"no parameter called '{name}'.\n"
                                "Try lower, upper, centre or ylevel.")
            return
        self.redraw()

    def result(self) -> tuple[list[dict], list | None]:
        return self.apertures, self.background


@on_main
def run(profile, apertures: list[dict], background=None, ylevel: float = 0.1,
        title: str = "apedit") -> tuple[list[dict], list | None]:
    """Open the panel and hand back the apertures that were accepted."""
    from .dock import show

    application()
    panel = AperturePanel(profile, apertures, background, ylevel)
    show(panel, title, size=(1000, 560))
    return panel.result()
