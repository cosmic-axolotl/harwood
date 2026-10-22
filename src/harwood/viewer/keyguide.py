"""The keys that work here, always in view.

IRAF's cursor mode has a key for everything and a list of them nowhere on
the screen: `?` prints them and they scroll away. Here every place an image
or a plot takes keys shows them all, beside it, grouped the way section 8.4
splits them -- lower case is the task's, upper case and `:` the window's --
and the list changes with what is running: imexamine's keys while an image
is merely shown, tvmark's while tvmark holds the cursor, apedit's in the
aperture panel.

The pairs come from the same lists the keys are answered from
(core.examine.KEYS, core.tvmark.HELP, each panel's HELP), so a key cannot be
listed that does nothing, or work without being listed.
"""

from __future__ import annotations

from collections.abc import Iterable

from PySide6 import QtCore, QtWidgets

from . import theme

#: The window's own keys, whatever task holds the lower case ones.
WINDOW = [("Z", "back to the whole image"), ("D", "show or hide the DQ layer"),
          ("n  p", "next, previous frame"), ("?", "these keys, in the session log"),
          (":", "colon commands, at the prompt")]

#: What the mouse does on the image (pyqtgraph's view box).
MOUSE = [("wheel", "zoom about the cursor"), ("drag", "pan"),
         ("right drag", "stretch one axis"), ("hover", "position, value, DQ")]

Pairs = Iterable[tuple[str, str]]


def pairs(keys) -> list[tuple[str, str]]:
    """A key list as (key, what) pairs, from a dict or a list of pairs;
    entries without a key (a panel's spacer) are left out."""
    items = keys.items() if isinstance(keys, dict) else keys
    return [(str(key), str(what)) for key, what in items if key]


def brief(what: str) -> str:
    """The description as the strip shows it: up to its first semicolon or
    aside. The whole sentence stays in the tooltip and in `?`."""
    return what.split(";")[0].split(" (")[0].strip()


def _escape(text: str) -> str:
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def _row(key: str, what: str) -> str:
    colour = theme.palette().accent
    return (f"<tr><td style='padding:1px 8px 1px 0; white-space:nowrap; "
            f"font-family:monospace; font-weight:bold; color:{colour}'>"
            f"{_escape(key)}</td><td style='padding:1px 0'>{_escape(what)}</td></tr>")


class KeyGuide(QtWidgets.QScrollArea):
    """The keys of the place in view, grouped, in as many columns as the
    width allows -- a strip under the image rather than a list to scroll --
    with an optional legend: the DQ bits of the frame shown, with their
    colours and counts."""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setObjectName("keyguide")
        self.setWidgetResizable(True)
        self.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)
        self.label = QtWidgets.QLabel()
        self.label.setObjectName("keyguide-text")
        self.label.setTextFormat(QtCore.Qt.TextFormat.RichText)
        self.label.setWordWrap(True)
        self.label.setAlignment(QtCore.Qt.AlignmentFlag.AlignTop)
        self.label.setContentsMargins(8, 4, 8, 4)
        self.setWidget(self.label)
        self.title = ""
        self.groups: list[tuple[str, list[tuple[str, str]]]] = []
        self.legend: list[tuple[str, str, str]] = []

    def show_keys(self, title: str, groups: list[tuple[str, Pairs]],
                  legend: list[tuple[str, str, str]] | None = None) -> None:
        """`groups` are (heading, pairs); `legend` is (colour, name, detail)."""
        self.title = title
        self.groups = [(heading, pairs(keys)) for heading, keys in groups]
        self.legend = list(legend or [])
        self.redraw()

    def entries(self) -> list[tuple[str, str]]:
        """Every (key, what) on show, for tests and for `?`."""
        return [pair for _, keys in self.groups for pair in keys]

    def columns(self) -> int:
        """As many as fit: the widest key and description, measured in the
        font they are drawn in, set the column's width."""
        width = self.viewport().width() or self.width()
        metrics = self.label.fontMetrics()
        cells = [(key, brief(what)) for key, what in self.entries()]
        cells += [("#", f"{name} {detail}") for _, name, detail in self.legend]
        cells += [("", heading) for heading, _ in self.groups]
        if not cells:
            return 1
        column = max(int(metrics.horizontalAdvance(key) * 1.3)
                     + metrics.horizontalAdvance(text) for key, text in cells) + 40
        return max(1, int((width - 16) // column))

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        if self.groups and self.columns() != getattr(self, "_drawn_columns", 0):
            self.redraw()

    def redraw(self) -> None:
        palette = theme.palette()
        n = self.columns()
        self._drawn_columns = n
        cells = []                       # (kind, a, b): heading, key or legend
        for heading, keys in self.groups:
            if keys:
                cells.append(("heading", heading, ""))
                cells += [("key", key, what) for key, what in keys]
        if self.legend:
            cells.append(("heading", "DQ in this frame", ""))
            cells += [("legend", f"{colour}|{name}", detail)
                      for colour, name, detail in self.legend]
        # Column-major: read down a column, then the next, as a list is read.
        rows = -(-len(cells) // n)
        html = [f"<p style='margin:0 0 3px 0'><b>{_escape(self.title)}</b></p>"
                "<table cellspacing='0' cellpadding='0'>"]
        for r in range(rows):
            html.append("<tr>")
            for c in range(n):
                index = c * rows + r
                if index >= len(cells):
                    html.append("<td></td><td></td>")
                    continue
                kind, a, b = cells[index]
                gap = "padding:1px 18px 1px 0"
                if kind == "heading":
                    html.append(f"<td colspan='2' style='{gap}; color:{palette.muted}'>"
                                f"<i>{_escape(a)}</i></td>")
                elif kind == "key":
                    html.append(f"<td style='padding:1px 8px 1px 0; white-space:nowrap; "
                                f"font-family:monospace; font-weight:bold; "
                                f"color:{palette.accent}'>{_escape(a)}</td>"
                                f"<td style='{gap}; white-space:nowrap'>"
                                f"{_escape(brief(b))}</td>")
                else:
                    colour, name = a.split("|", 1)
                    html.append(f"<td style='padding:1px 8px 1px 0; color:{colour}'>"
                                f"&#9632;</td><td style='{gap}'>{_escape(name)} "
                                f"<span style='color:{palette.muted}'>{_escape(b)}"
                                "</span></td>")
            html.append("</tr>")
        html.append("</table>")
        self.label.setText("".join(html))
        self.label.setToolTip("\n".join(f"{key}   {what}" for key, what in self.entries()))
        self.rows = rows

    def wanted_height(self) -> int:
        """What the strip needs to show everything without scrolling."""
        line = self.label.fontMetrics().lineSpacing() + 3      # cell padding
        return (getattr(self, "rows", 6) + 2) * line + 24      # title, margins


def strip(keys, columns: int = 0) -> QtWidgets.QLabel:
    """A panel's keys, every one of them, wrapping to as many lines as they
    take -- what used to be the first five on one line."""
    label = QtWidgets.QLabel()
    label.setObjectName("keys")
    label.setWordWrap(True)
    label.setTextFormat(QtCore.Qt.TextFormat.RichText)
    colour = theme.palette().accent
    label.setText("&nbsp;&nbsp; ".join(
        f"<span style='white-space:nowrap'><b style='color:{colour}; "
        f"font-family:monospace'>{_escape(key)}</b>&nbsp;{_escape(what)}</span>"
        for key, what in pairs(keys)))
    label.keys = pairs(keys)
    return label
