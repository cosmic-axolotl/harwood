"""harwood's icon: a lowercase h whose arch is the Sun's spectrum, drawn in the colours of
the theme the person chose.

The stem is a straight line, a little thinner than the arch; the arch is the Planck
curve of a black body at 5777 K, computed point by point from 300 to 1200 nm against
linear wavelength -- its peak, near 502 nm, at the x-height, its tail shifted to reach
the baseline. Over the peak, a four-pointed star. The square under it is rounded.

The colours are the palette's: the ground, the ink (the stem), the colour of data in
the graphs (the arch) and the accent (the star). `theme.apply` sets the application's
icon from here every time the look changes, so the icon follows the theme -- dark,
light, the desktop's, and the colour-blind palettes.
"""

from __future__ import annotations

import contextlib
import math
import os
from functools import lru_cache
from pathlib import Path

H, C, K = 6.62607015e-34, 2.99792458e8, 1.380649e-23
T = 5777.0                                  #: K, the Sun
PEAK = 2.897771955e6 / T                    #: nm, Wien's law: ~502 nm
START, END = 300.0, 1200.0                  #: nm, the part of the curve that is the arch

#: In Comfortaa's units, y up from the baseline, the stem at x = 0.
ASCENT, XHEIGHT, WIDTH = 781.0, 547.0, 600.0
STROKE = 88.0                               #: the arch's; the stem is STEM of it
STEM = 0.78
STAR_R, STAR_GAP = 60.0, 45.0
#: The middle of the letter and its star, which goes in the middle of the square.
MID_X, MID_Y = 300.0, 390.0
SIZE, SCALE, RADIUS = 1000.0, 0.92, 200.0
#: The sizes the icon is drawn at for the window system to choose from.
SIZES = (16, 24, 32, 48, 64, 128, 256)


def planck(nm: float) -> float:
    lam = nm * 1e-9
    return (2 * H * C * C / lam**5) / math.expm1(H * C / (lam * K * T))


@lru_cache(maxsize=1)
def arch(n: int = 180) -> tuple[tuple[float, float], ...]:
    """The arch, (x, y) from the stem and the baseline."""
    bmax, bend = planck(PEAK), planck(END)
    return tuple(((nm - START) / (END - START) * WIDTH,
                  (planck(nm) - bend) / (bmax - bend) * XHEIGHT)
                 for nm in (START + (END - START) * i / n for i in range(n + 1)))


def _star(x: float, y: float, r: float, fill: str, pinch: float = 0.18) -> str:
    """Four points, the sides drawn in: upright."""
    tips = [(x + r * math.cos(math.radians(90 * k)), y + r * math.sin(math.radians(90 * k)))
            for k in range(4)]
    waists = [(x + pinch * r * math.cos(math.radians(90 * k + 45)),
               y + pinch * r * math.sin(math.radians(90 * k + 45))) for k in range(4)]
    d = f"M{tips[0][0]:.1f},{tips[0][1]:.1f}"
    for k in range(4):
        (cx, cy), (tx, ty) = waists[k], tips[(k + 1) % 4]
        d += f" Q{cx:.1f},{cy:.1f} {tx:.1f},{ty:.1f}"
    return f'<path d="{d} Z" fill="{fill}"/>'


def svg(palette) -> str:
    """The icon as SVG, in a palette's colours (viewer.theme.Palette)."""
    s = SCALE
    ox, oy = SIZE / 2 - MID_X * s, SIZE / 2 + MID_Y * s
    curve = "M" + " L".join(f"{ox + x * s:.2f},{oy - y * s:.2f}" for x, y in arch())
    peak_x = (PEAK - START) / (END - START) * WIDTH
    star_y = XHEIGHT + STROKE / 2 + STAR_GAP + STAR_R
    return (f'<svg xmlns="http://www.w3.org/2000/svg" width="{SIZE:g}" height="{SIZE:g}" '
            f'viewBox="0 0 {SIZE:g} {SIZE:g}">'
            f'<rect width="{SIZE:g}" height="{SIZE:g}" rx="{RADIUS:g}" '
            f'fill="{palette.ground}"/>'
            f'<g fill="none" stroke-linecap="round" stroke-linejoin="round">'
            f'<path d="M{ox:.2f},{oy:.2f} L{ox:.2f},{oy - ASCENT * s:.2f}" '
            f'stroke="{palette.ink}" stroke-width="{STROKE * STEM * s:.2f}"/>'
            f'<path d="{curve}" stroke="{palette.sky}" stroke-width="{STROKE * s:.2f}"/></g>'
            + _star(ox + peak_x * s, oy - star_y * s, STAR_R * s, palette.accent)
            + "</svg>")


def qicon(palette):
    """The icon at every size in SIZES, for setWindowIcon."""
    from PySide6 import QtCore, QtGui, QtSvg

    renderer = QtSvg.QSvgRenderer(QtCore.QByteArray(svg(palette).encode("utf-8")))
    icon = QtGui.QIcon()
    for size in SIZES:
        image = QtGui.QImage(size, size, QtGui.QImage.Format.Format_ARGB32_Premultiplied)
        image.fill(QtCore.Qt.GlobalColor.transparent)
        painter = QtGui.QPainter(image)
        painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
        renderer.render(painter)
        painter.end()
        icon.addPixmap(QtGui.QPixmap.fromImage(image))
    return icon


_shown: tuple | None = None


def apply(app, palette) -> None:
    """Set the application's icon to the palette's, if it is not that already; name
    the desktop file, which is what Wayland's docks find the icon by; and, when
    `harwood shortcut` installed one, write the launcher's icon in these colours too
    (GNOME shows it the next time harwood opens)."""
    global _shown
    key = (palette.ground, palette.ink, palette.sky, palette.accent)
    if _shown == key and not app.windowIcon().isNull():
        return
    app.setWindowIcon(qicon(palette))
    if not app.desktopFileName():
        app.setDesktopFileName("harwood")
    _shown = key
    path = launcher_icon()
    if path.is_file():
        # a launcher's icon is not worth a failed window
        with contextlib.suppress(OSError):
            path.write_text(svg(palette), encoding="utf-8")


# ------------------------------------------------------------- the launcher

def _data_home() -> Path:
    return Path(os.environ.get("XDG_DATA_HOME") or Path.home() / ".local" / "share")


def launcher_icon() -> Path:
    return _data_home() / "icons" / "hicolor" / "scalable" / "apps" / "harwood.svg"


def launcher_file() -> Path:
    return _data_home() / "applications" / "harwood.desktop"


DESKTOP = """[Desktop Entry]
Type=Application
Name=harwood
GenericName=Astronomical data reduction
Comment=Reduce a night of observing, with the tasks you know from IRAF
Exec={exec} gui %f
Icon=harwood
Terminal=false
Categories=Science;Astronomy;
StartupWMClass=harwood
"""


def install_launcher(palette, command: str) -> list[Path]:
    """A freedesktop launcher (Linux): the .desktop file, which opens the window with
    `command gui`, and the icon in the palette's colours. The files written."""
    icon_path, desktop = launcher_icon(), launcher_file()
    icon_path.parent.mkdir(parents=True, exist_ok=True)
    desktop.parent.mkdir(parents=True, exist_ok=True)
    icon_path.write_text(svg(palette), encoding="utf-8")
    quoted = f'"{command}"' if " " in command else command
    desktop.write_text(DESKTOP.format(exec=quoted), encoding="utf-8")
    return [desktop, icon_path]


def remove_launcher() -> list[Path]:
    removed = []
    for path in (launcher_file(), launcher_icon()):
        if path.is_file():
            path.unlink()
            removed.append(path)
    return removed
