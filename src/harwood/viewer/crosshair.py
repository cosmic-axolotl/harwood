"""The cursor's helpers, chosen in the strip under the tabs, each on its own:

  cross     a horizontal and a vertical line across the whole plot, crossing
            where the mouse is -- IRAF's graphics cursor, for reading a value off
            an axis or lining a feature up with another;
  axes      with the cross, its values written where the lines meet the plot's
            edges (on the image, the 1-based pixel x and y);
  aperture  on the image, imexamine's circles around the mouse: the radius it
            measures within, and the sky annulus (radius + buffer, + width) --
            what a neighbour or a cosmic ray would fall into before r or a;
  pixel     on the image, the square of the pixel under the mouse, the one the
            value at the bottom of the window is;
  centroid  on the image, where r would centre the star near the mouse
            (ie_center's moments of the marginals, analysis.centroid).

One Crosshair per pyqtgraph view (the image, each graph tab). What it draws goes
into whichever ViewBox the mouse is over, outside the data's bounds (it never
changes the range), letting every click and key through, in the palette's own
cursor colour, which no curve, mark or fit uses. A panel that clears its plot
takes the items with it: they go back in at the next move of the mouse.
"""

from __future__ import annotations

import weakref
from collections.abc import Callable
from typing import Any

import numpy as np
import pyqtgraph as pg
from PySide6 import QtCore, QtWidgets

from . import theme

OPTIONS = ("cross", "axes", "aperture", "pixel", "centroid")
#: The ones that only mean something on an image.
IMAGE_ONLY = ("aperture", "pixel", "centroid")

_all: weakref.WeakSet = weakref.WeakSet()


def on(name: str) -> bool:
    prefs = theme.current().preferences
    if name == "cross":
        return prefs.cursor == "cross"
    return bool(getattr(prefs, f"cursor_{name}"))


def enabled() -> bool:
    """The crosshair itself."""
    return on("cross")


def set_option(name: str, value: bool, save: bool = True) -> None:
    """Turn one helper on or off, for every view, and keep the choice."""
    if name == "cross":
        theme.update(save=save, cursor="cross" if value else "arrow")
    else:
        theme.update(save=save, **{f"cursor_{name}": bool(value)})
    for cross in list(_all):
        cross.refresh()


def set_enabled(value: bool, save: bool = True) -> None:
    set_option("cross", value, save)


def attach(view: Any, image: Callable[[], Any] | None = None,
           image_box: Any = None) -> Crosshair | None:
    """The helpers on a pyqtgraph view (GraphicsView: PlotWidget,
    GraphicsLayoutWidget), once. `image` gives the image's pixels and imexamine's
    radius, buffer and width -- (data, radius, buffer, width), or None when there
    is no image -- for the helpers drawn over `image_box`."""
    if not isinstance(view, pg.GraphicsView):
        return None
    for cross in _all:
        if cross.view is view:
            if image is not None:
                cross.image, cross.image_box = image, image_box
            return cross
    return Crosshair(view, image, image_box)


def attach_all(widget: Any) -> list[Crosshair]:
    """The helpers on every pyqtgraph view inside a widget (a graph tab)."""
    views = [widget] if isinstance(widget, pg.GraphicsView) else []
    views += widget.findChildren(pg.GraphicsView)
    return [c for c in (attach(v) for v in views) if c is not None]


def _fmt(value: float) -> str:
    return f"{value:.6g}"


class Crosshair(QtCore.QObject):
    def __init__(self, view: Any, image: Callable[[], Any] | None = None,
                 image_box: Any = None) -> None:
        super().__init__(view)
        self.view = view
        self.image = image
        self.image_box = image_box
        self.lines: tuple[Any, Any] | None = None
        self.tags: tuple[Any, Any] | None = None
        self.circles: list[Any] = []
        self.square: Any = None
        self.mark: Any = None
        self._centre_for: tuple[int, int] | None = None
        self.centre: tuple[float, float] | None = None
        self.box: Any = None
        view.scene().sigMouseMoved.connect(self._moved)
        view.viewport().installEventFilter(self)
        theme.on_change(self._recolour, owner=view)
        _all.add(self)

    # ---- drawing
    def _pen(self):
        return pg.mkPen(theme.palette().cursor, width=1)

    def _quiet(self, item) -> Any:
        """An item that is only looked at: above all, no button, no hover."""
        item.setZValue(1e6)
        item.setAcceptedMouseButtons(QtCore.Qt.MouseButton.NoButton)
        item.setAcceptHoverEvents(False)
        return item

    def _items(self) -> list[Any]:
        out = [*(self.lines or ()), *(self.tags or ()), *self.circles]
        return out + [i for i in (self.square, self.mark) if i is not None]

    def _recolour(self) -> None:
        colour = theme.palette().cursor
        for item in self._items():
            if isinstance(item, pg.TextItem):
                item.setColor(colour)
                item.fill = pg.mkBrush(theme.palette().panel)
            elif isinstance(item, QtWidgets.QGraphicsEllipseItem):
                pen = item.pen()
                pen.setColor(pg.mkColor(colour))
                item.setPen(pen)
            else:
                item.setPen(self._pen())

    def _into(self, item, box) -> None:
        if item.parentItem() is not box.childGroup:
            box.addItem(item, ignoreBounds=True)
        item.show()

    def _box_at(self, position) -> Any:
        for item in self.view.scene().items(position):
            while item is not None and not isinstance(item, pg.ViewBox):
                item = item.parentItem()
            if item is not None:
                return item
        return None

    # ---- the mouse
    def _moved(self, position) -> None:
        box = self._box_at(position) if any(on(n) for n in OPTIONS) else None
        if box is None:
            self.hide()
            return
        self.box = box
        point = box.mapSceneToView(position)
        x, y = point.x(), point.y()
        over_image = self.image is not None and box is self.image_box
        self._cross(box, x, y, over_image)
        self._image_helpers(box, x, y if over_image else None)

    def _cross(self, box, x: float, y: float, over_image: bool) -> None:
        if not on("cross"):
            for item in (*(self.lines or ()), *(self.tags or ())):
                item.hide()
            self.view.viewport().unsetCursor()
            return
        if self.lines is None:
            self.lines = tuple(self._quiet(pg.InfiniteLine(angle=a, movable=False,
                                                           pen=self._pen()))
                               for a in (90, 0))
        for line in self.lines:
            self._into(line, box)
        self.lines[0].setPos(x)
        self.lines[1].setPos(y)
        # the lines are the pointer, while they are there
        self.view.viewport().setCursor(QtCore.Qt.CursorShape.BlankCursor)
        if not on("axes"):
            for item in self.tags or ():
                item.hide()
            return
        if self.tags is None:
            self.tags = (self._quiet(pg.TextItem(anchor=(0.5, 1.0))),
                         self._quiet(pg.TextItem(anchor=(0.0, 0.5))))
            self._recolour()
        (x0, _), (y0, _) = box.viewRange()
        if over_image:
            from ..core import pixels

            px, py = pixels.from_scene(x, y)
            xs, ys = f"x {float(px):.1f}", f"y {float(py):.1f}"
        else:
            logx, logy = box.state.get("logMode", [False, False])
            xs = _fmt(10 ** x if logx else x)
            ys = _fmt(10 ** y if logy else y)
        self.tags[0].setText(xs)
        self.tags[1].setText(ys)
        self.tags[0].setPos(x, y0)
        self.tags[1].setPos(x0, y)
        for tag in self.tags:
            self._into(tag, box)

    def _image_helpers(self, box, sx: float, sy: float | None) -> None:
        got = self.image() if (sy is not None and self.image is not None) else None
        if got is None:
            for item in [*self.circles, self.square, self.mark]:
                if item is not None:
                    item.hide()
            return
        from ..core import pixels

        data, radius, buffer, width = got
        x, y = (float(v) for v in pixels.from_scene(sx, sy))
        pen = self._pen()
        # imexamine's circles: the radius, and the sky annulus beyond the buffer
        if on("aperture"):
            if not self.circles:
                self.circles = [self._quiet(QtWidgets.QGraphicsEllipseItem())
                                for _ in range(3)]
                dashed = pg.mkPen(theme.palette().cursor, width=1,
                                  style=QtCore.Qt.PenStyle.DashLine)
                for k, circle in enumerate(self.circles):
                    circle.setPen(pen if k == 0 else dashed)
            for circle, r in zip(self.circles, (radius, radius + buffer,
                                                radius + buffer + width), strict=True):
                circle.setRect(sx - r, sy - r, 2 * r, 2 * r)
                self._into(circle, box)
        else:
            for circle in self.circles:
                circle.hide()
        ix, iy = int(np.floor(x + 0.5)), int(np.floor(y + 0.5))
        if on("pixel"):
            if self.square is None:
                self.square = self._quiet(QtWidgets.QGraphicsRectItem())
                self.square.setPen(pen)
            cx, cy = (float(v) for v in pixels.to_scene(ix, iy))
            self.square.setRect(cx - 0.5, cy - 0.5, 1.0, 1.0)
            self._into(self.square, box)
        elif self.square is not None:
            self.square.hide()
        if on("centroid"):
            ny, nx = data.shape[-2:]
            if not (1 <= ix <= nx and 1 <= iy <= ny):
                if self.mark is not None:
                    self.mark.hide()
                return
            if self._centre_for != (ix, iy):
                from .analysis import centroid

                self.centre = centroid(np.asarray(data, float), x, y, radius=radius)
                self._centre_for = (ix, iy)
            if self.mark is None:
                self.mark = self._quiet(pg.ScatterPlotItem(size=14, symbol="x",
                                                           pen=pen, brush=None))
            mx, my = (float(v) for v in pixels.to_scene(*self.centre))
            self.mark.setData([mx], [my])
            self._into(self.mark, box)
        elif self.mark is not None:
            self.mark.hide()

    def hide(self) -> None:
        for item in self._items():
            item.hide()
        self.view.viewport().unsetCursor()

    def refresh(self) -> None:
        self._centre_for = None
        if not any(on(n) for n in OPTIONS):
            self.hide()

    def eventFilter(self, watched, event) -> bool:
        if event.type() == QtCore.QEvent.Type.Leave:
            self.hide()
        return False
