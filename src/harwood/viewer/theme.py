"""How the window looks: light or dark, colour-blind safe, dyslexia friendly.

Every colour the viewer uses is a token here, never a literal in a panel, so
one switch moves the whole interface: the window, every panel and every
plot. Three choices, independent of each other:

``theme``       dark (the default: quiet at night beside a terminal), light,
                or system (follow the desktop).
``colorblind``  data colours from Okabe and Ito's palette, which stays
                distinguishable in the common colour-vision deficiencies --
                the red-green pair most plots lean on does not. Colour is
                never the only channel anyway: rejected points also change
                symbol, and warnings carry a word and a sign.
``dyslexia``    Atkinson Hyperlegible (Braille Institute; or OpenDyslexic,
                for those who prefer it), wider letter and line spacing, no
                justified text, and a cream ground in the light theme.

Text size is one number, `scale`, moved with Ctrl+/Ctrl- and, when
`autoscale` is on, with the size of the window.

The choices are kept in ~/.config/harwood/interface.json -- a file a person
can read and edit, shown and changed from the View menu. They are not
reduction parameters, so they live with the user rather than the project.

Every text/ground pair is held to WCAG's contrast ratio in the tests: 4.5:1
for text, 3:1 for the accent and the data colours on their ground.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field, fields, replace
from pathlib import Path
from typing import Any

FONTS = Path(__file__).resolve().parent.parent / "data" / "fonts"


@dataclass(frozen=True)
class Palette:
    """The colour tokens."""

    ground: str          #: the background
    panel: str           #: toolbars, dock titles
    line: str            #: borders, grid
    ink: str             #: text
    muted: str           #: secondary text
    accent: str          #: focus, fits, the current thing
    sky: str             #: data points
    band: str            #: a second data colour (bands, apertures)
    drop: str            #: rejected, deleted
    comment: str         #: comments in the editor
    note: str            #: advice levels: note, warning, problem
    warning: str
    problem: str
    series: tuple[str, ...] = ()        #: overplotted curves, in order
    interstellar: str = ""
    telluric: str = ""


DARK = Palette(
    ground="#141a21", panel="#1b222b", line="#2a3541", ink="#d7dee5",
    muted="#8b9aa8", accent="#e0a33e", sky="#4fb59b", band="#7db2d8",
    drop="#d2686a", comment="#7f93a6", note="#56b4e9", warning="#e69f00",
    problem="#e86a33",
    series=("#4fb59b", "#e0a33e", "#7db2d8", "#d2686a", "#a98fd0"),
    interstellar="#b48ead", telluric="#d16a6a")

LIGHT = Palette(
    ground="#f7f5f0", panel="#ebe7df", line="#c9c2b6", ink="#1d252c",
    muted="#525e69", accent="#9a5b00", sky="#0f7660", band="#2c6a9a",
    drop="#b0383b", comment="#55697b", note="#1f6f9f", warning="#9a5b00",
    problem="#b0383b",
    series=("#0f7660", "#9a5b00", "#2c6a9a", "#b0383b", "#6b4f9a"),
    interstellar="#7d4f8a", telluric="#a83434")

#: Okabe and Ito (2008): blue, orange, bluish green, vermillion, reddish
#: purple, sky blue, yellow -- as bright on the dark ground, and darkened to
#: keep their hue on the light one.
OKABE_ITO_DARK = ("#56b4e9", "#e69f00", "#009e73", "#d55e00", "#cc79a7",
                  "#f0e442", "#0072b2")
OKABE_ITO_LIGHT = ("#0072b2", "#a76f00", "#007a5a", "#b84f00", "#a8487f",
                   "#7a6f00", "#005a8c")


def _colorblind(base: Palette, okabe: tuple[str, ...]) -> Palette:
    blue, orange, green, vermillion, purple = okabe[:5]
    return replace(base, accent=orange, sky=blue, band=green, drop=vermillion,
                   note=blue, warning=orange, problem=vermillion,
                   series=(blue, orange, green, vermillion, purple),
                   interstellar=purple, telluric=vermillion)


DARK_CB = _colorblind(DARK, OKABE_ITO_DARK)
LIGHT_CB = _colorblind(LIGHT, OKABE_ITO_LIGHT)
#: The light ground for the dyslexia mode: cream, not white -- glare from a
#: pure white page is what many dyslexic readers ask to avoid.
CREAM = "#fbf5e6"

#: Text sizes in points at scale 1.0. Points rather than pixels: a pixel size
#: ignores the display's own scaling, which is how text ends up unreadable on
#: a high-density screen and enormous on a projector.
BASE = {"body": 9.0, "mono": 8.5, "small": 8.0}
SCALES = (0.7, 0.8, 0.9, 1.0, 1.1, 1.25, 1.4, 1.6, 1.8, 2.0, 2.3)

FAMILIES = {
    "default": ("IBM Plex Sans", "DejaVu Sans", "sans-serif"),
    "atkinson": ("Atkinson Hyperlegible", "DejaVu Sans", "sans-serif"),
    "opendyslexic": ("OpenDyslexic", "DejaVu Sans", "sans-serif"),
}
MONO = ("IBM Plex Mono", "DejaVu Sans Mono", "monospace")


@dataclass
class Preferences:
    """What the person chose, as saved."""

    theme: str = "dark"                 #: dark, light or system
    colorblind: bool = False
    dyslexia: bool = False
    dyslexia_font: str = "atkinson"     #: atkinson or opendyslexic
    scale: float = 1.0                  #: the person's own text size
    autoscale: bool = True              #: and the window's on top of it
    reduce_motion: bool = False

    @classmethod
    def path(cls) -> Path:
        base = os.environ.get("XDG_CONFIG_HOME") or str(Path.home() / ".config")
        return Path(base) / "harwood" / "interface.json"

    @classmethod
    def load(cls) -> Preferences:
        try:
            stored = json.loads(cls.path().read_text())
        except (OSError, ValueError):
            return cls()
        known = {f.name for f in fields(cls)}
        prefs = cls(**{k: v for k, v in stored.items() if k in known})
        return prefs.checked()

    def checked(self) -> Preferences:
        """Anything out of range back to its default, rather than an error
        the window would have to show before it could show anything."""
        default = Preferences()
        if self.theme not in ("dark", "light", "system"):
            self.theme = default.theme
        if self.dyslexia_font not in ("atkinson", "opendyslexic"):
            self.dyslexia_font = default.dyslexia_font
        try:
            self.scale = min(max(float(self.scale), SCALES[0]), SCALES[-1])
        except (TypeError, ValueError):
            self.scale = default.scale
        return self

    def save(self) -> Path:
        path = self.path()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(asdict(self), indent=2) + "\n")
        return path


@dataclass
class Theme:
    """Preferences resolved into what to draw with."""

    preferences: Preferences = field(default_factory=Preferences)
    system_dark: bool = True            #: what the desktop says, for "system"
    window_factor: float = 1.0          #: from the window's size, when autoscale

    @property
    def dark(self) -> bool:
        if self.preferences.theme == "system":
            return self.system_dark
        return self.preferences.theme == "dark"

    @property
    def palette(self) -> Palette:
        p = self.preferences
        palette = ((DARK_CB if p.colorblind else DARK) if self.dark
                   else (LIGHT_CB if p.colorblind else LIGHT))
        if p.dyslexia and not self.dark:
            palette = replace(palette, ground=CREAM)
        return palette

    @property
    def scale(self) -> float:
        factor = self.window_factor if self.preferences.autoscale else 1.0
        # The dyslexia-friendly faces run small for their point size: a
        # tenth more keeps the text as large as the person set it.
        legible = 1.1 if self.preferences.dyslexia else 1.0
        return self.preferences.scale * factor * legible

    @property
    def family(self) -> tuple[str, ...]:
        p = self.preferences
        return FAMILIES[p.dyslexia_font if p.dyslexia else "default"]

    @property
    def letter_spacing(self) -> float:
        """Percent of the normal spacing, as QFont takes it."""
        return 110.0 if self.preferences.dyslexia else 100.0

    @property
    def line_height(self) -> float:
        """Percent of the font's height between baselines, for text panes."""
        return 150.0 if self.preferences.dyslexia else 115.0

    def stylesheet(self) -> str:
        """The window's appearance. Everything that sets a size does it
        through BASE and the scale, so one number moves the whole interface;
        anything that hard-coded a width in pixels would let the text
        outgrow its box, which is the usual way a zoom feature half works."""
        c, s = self.palette, self.scale
        body, mono, small = (BASE["body"] * s, BASE["mono"] * s, BASE["small"] * s)
        family = ", ".join(f'"{f}"' if " " in f else f for f in self.family)
        monofam = ", ".join(f'"{f}"' if " " in f else f for f in MONO)
        spacing = "letter-spacing: 0.06em; word-spacing: 0.12em;" \
            if self.preferences.dyslexia else ""
        return f"""
QMainWindow, QWidget {{ background: {c.ground}; color: {c.ink};
    font-family: {family}; font-size: {body:.1f}pt; {spacing} }}
QDockWidget {{ color: {c.muted}; font-size: {body:.1f}pt; }}
QDockWidget::title {{ background: {c.panel}; padding: {5 * s:.0f}px 9px;
    border-bottom: 1px solid {c.line}; font-weight: 600; }}
QToolBar {{ background: {c.panel}; border-bottom: 1px solid {c.line};
    spacing: {7 * s:.0f}px; padding: 4px 8px; }}
QMenuBar {{ background: {c.panel}; color: {c.ink}; }}
QMenuBar::item:selected, QMenu::item:selected {{ background: {c.line}; }}
QMenu {{ background: {c.panel}; color: {c.ink}; border: 1px solid {c.line}; }}
QMenu::indicator:checked {{ background: {c.accent}; border-radius: 2px; }}
QComboBox, QLineEdit, QSpinBox, QDoubleSpinBox {{ background: {c.ground};
    color: {c.ink}; border: 1px solid {c.line}; border-radius: 4px;
    padding: 3px 7px; font-size: {body:.1f}pt;
    selection-background-color: {c.accent}; selection-color: {c.ground}; }}
QComboBox::drop-down {{ border: none; width: {16 * s:.0f}px; }}
QLineEdit:focus, QComboBox:focus {{ border: 2px solid {c.accent}; }}
QToolButton, QPushButton {{ color: {c.ink}; padding: 4px 10px;
    border-radius: 6px; font-size: {body:.1f}pt; }}
QPushButton {{ background: {c.panel}; border: 1px solid {c.line}; }}
QToolButton:hover, QPushButton:hover {{ background: {c.line}; }}
QToolButton:focus, QPushButton:focus {{ outline: none;
    border: 2px solid {c.accent}; }}
QToolButton:checked {{ background: {c.line}; border-bottom: 2px solid {c.accent}; }}
QPlainTextEdit, QTextEdit {{ background: {c.ground}; color: {c.ink};
    border: none; font-family: {monofam}; font-size: {mono:.1f}pt; }}
QTabBar::tab {{ background: {c.panel}; color: {c.muted}; padding: 5px 12px;
    border: 1px solid {c.line}; border-bottom: none; }}
QTabBar::tab:selected {{ color: {c.ink}; border-top: 2px solid {c.accent}; }}
QStatusBar {{ background: {c.panel}; border-top: 1px solid {c.line};
    color: {c.ink}; font-family: {monofam}; font-size: {mono:.1f}pt; }}
QLabel#hint {{ color: {c.muted}; font-size: {small:.1f}pt; }}
QLabel#readout {{ font-family: {monofam}; font-size: {mono:.1f}pt; color: {c.ink}; }}
QLabel#param {{ color: {c.muted}; font-weight: 600; font-size: {body:.1f}pt; }}
QToolTip {{ background: {c.panel}; color: {c.ink}; border: 1px solid {c.line};
    padding: 5px; font-size: {small:.1f}pt; }}
QLabel#status {{ background: {c.panel}; border-top: 1px solid {c.line};
    padding: 5px 9px; font-family: {monofam}; font-size: {mono:.1f}pt; }}
QLabel#keys {{ color: {c.muted}; font-family: {monofam}; font-size: {small:.1f}pt;
    padding: 4px 9px; }}
QLineEdit#command {{ font-family: {monofam}; background: {c.panel}; }}
QTreeView#completions {{ background: {c.panel}; color: {c.ink};
    border: 2px solid {c.accent}; font-family: {monofam}; font-size: {mono:.1f}pt;
    selection-background-color: {c.accent}; selection-color: {c.ground}; }}
QTreeView#completions::item {{ padding: {2 * s:.0f}px 8px; }}
QScrollBar {{ background: {c.ground}; }}
QScrollBar::handle {{ background: {c.line}; border-radius: 4px; }}
"""


# ----------------------------------------------------------------- live state

_theme = Theme(Preferences.load())
_listeners: list[Any] = []
_fonts_loaded = False


def current() -> Theme:
    return _theme


def palette() -> Palette:
    """The colours to draw with right now. Call it when drawing, not at
    import: the answer changes when the person changes the theme."""
    return _theme.palette


def on_change(callback) -> None:
    """Call `callback()` whenever the look changes (panels redraw)."""
    _listeners.append(callback)


def forget(callback) -> None:
    if callback in _listeners:
        _listeners.remove(callback)


def update(save: bool = True, **changes) -> Theme:
    """Change preferences (or system_dark / window_factor) and tell everyone."""
    global _theme
    prefs = {k: v for k, v in changes.items() if k in Preferences.__dataclass_fields__}
    state = {k: v for k, v in changes.items() if k in ("system_dark", "window_factor")}
    if prefs:
        _theme = replace(_theme, preferences=replace(_theme.preferences, **prefs).checked())
        if save:
            _theme.preferences.save()
    if state:
        _theme = replace(_theme, **state)
    apply()
    for callback in list(_listeners):
        try:
            callback()
        except RuntimeError:           # a panel whose widgets are gone
            forget(callback)
    return _theme


def load_fonts() -> None:
    """Register the bundled fonts with Qt (once)."""
    global _fonts_loaded
    if _fonts_loaded:
        return
    from PySide6 import QtGui

    for path in sorted(FONTS.glob("*.[ot]tf")):
        QtGui.QFontDatabase.addApplicationFont(str(path))
    _fonts_loaded = True


def apply() -> None:
    """Apply the current look to the running application and to pyqtgraph."""
    from PySide6 import QtGui, QtWidgets

    app = QtWidgets.QApplication.instance()
    if not isinstance(app, QtWidgets.QApplication):
        return
    load_fonts()
    t, c = _theme, _theme.palette
    font = QtGui.QFont(t.family[0])
    font.setPointSizeF(BASE["body"] * t.scale)
    font.setLetterSpacing(QtGui.QFont.SpacingType.PercentageSpacing, t.letter_spacing)
    app.setFont(font)
    app.setStyleSheet(t.stylesheet())
    try:
        import pyqtgraph as pg

        pg.setConfigOption("background", c.ground)
        pg.setConfigOption("foreground", c.muted)
    except ImportError:
        pass


def translucent(colour: str, alpha: int):
    """A token colour with an alpha (0-255), for fills."""
    from PySide6 import QtGui

    value = QtGui.QColor(colour)
    value.setAlpha(alpha)
    return value


def follow(widget, plots, redraw=None) -> None:
    """Keep a panel's plots in the current look: restyle them, then redraw,
    whenever the theme changes, until the widget is gone."""
    def again():
        for plot in plots:
            style_plot(plot)
        if redraw is not None:
            redraw()

    on_change(again)
    widget.destroyed.connect(lambda *_: forget(again))


def style_plot(plot) -> None:
    """Give a pyqtgraph PlotWidget the current colours (call on change too)."""
    import pyqtgraph as pg

    c = palette()
    plot.setBackground(c.ground)
    item = plot.getPlotItem() if hasattr(plot, "getPlotItem") else plot
    for name in ("left", "bottom", "right", "top"):
        axis = item.getAxis(name)
        axis.setPen(pg.mkPen(c.line))
        axis.setTextPen(pg.mkPen(c.muted))
    title = item.titleLabel
    if title is not None and title.text:
        title.setText(title.text, color=c.ink)


# -------------------------------------------------------------------- contrast


def luminance(colour: str) -> float:
    """WCAG relative luminance of '#rrggbb'."""
    rgb = [int(colour[i:i + 2], 16) / 255 for i in (1, 3, 5)]
    lin = [v / 12.92 if v <= 0.03928 else ((v + 0.055) / 1.055) ** 2.4 for v in rgb]
    return 0.2126 * lin[0] + 0.7152 * lin[1] + 0.0722 * lin[2]


def contrast(a: str, b: str) -> float:
    """WCAG contrast ratio, 1 to 21."""
    la, lb = sorted((luminance(a), luminance(b)), reverse=True)
    return (la + 0.05) / (lb + 0.05)
