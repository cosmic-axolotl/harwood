"""The documentation, in the window.

The pages are the built site (docsite/), the same files a browser opens with
`harwood docs --browser` -- so the two cannot say different things. The
window passes its own look along (theme, colour-blind palette, dyslexia
font, text size), and again whenever it changes, so the pages look like the
window around them.

Showing web pages needs QtWebEngine, which comes with PySide6's Addons and
not with PySide6-Essentials (`pip install harwood[docs]`). Without it the
panel says so and opens the same page in the browser instead.
"""

from __future__ import annotations

from pathlib import Path

from PySide6 import QtCore, QtWidgets

from . import theme


def available() -> bool:
    try:
        from PySide6 import QtWebEngineWidgets  # noqa: F401
    except ImportError:
        return False
    return True


def look_query() -> str:
    """The window's look, as the pages read it (docsite/assets/look.js)."""
    current = theme.current()
    words = ["dark" if current.dark else "light"]
    if current.preferences.colorblind:
        words.append("colorblind")
    if current.preferences.dyslexia:
        words.append("dyslexia")
    return (f"look={','.join(words)}&font={current.preferences.dyslexia_font}"
            f"&scale={current.preferences.scale:g}")


class DocsPanel(QtWidgets.QWidget):
    """Back, forward, home, the page, and the way out to a browser."""

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self.root: Path | None = None
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        bar = QtWidgets.QHBoxLayout()
        bar.setContentsMargins(6, 4, 6, 4)
        self.view = None
        if available():
            from PySide6 import QtWebEngineWidgets

            self.view = QtWebEngineWidgets.QWebEngineView()
            self.view.urlChanged.connect(self._where)
        for label, tip, slot in (
            ("◀", "Back", lambda: self.view and self.view.back()),
            ("▶", "Forward", lambda: self.view and self.view.forward()),
            ("⌂ Home", "The first page", lambda: self.show_page("index.html")),
            ("Open in browser", "The same page in your web browser",
             self.open_in_browser),
        ):
            button = QtWidgets.QPushButton(label)
            button.setToolTip(tip)
            button.setAccessibleName(tip)
            button.clicked.connect(slot)
            bar.addWidget(button)
        self.address = QtWidgets.QLabel("")
        self.address.setObjectName("hint")
        bar.addWidget(self.address, stretch=1)
        layout.addLayout(bar)

        if self.view is not None:
            layout.addWidget(self.view, stretch=1)
        else:
            missing = QtWidgets.QLabel(
                "Showing the documentation here needs QtWebEngine: "
                "pip install 'harwood[docs]'. Until then each page opens in your "
                "browser -- the same pages.")
            missing.setWordWrap(True)
            missing.setContentsMargins(12, 12, 12, 12)
            layout.addWidget(missing, stretch=1)
        self._page = "index.html"
        theme.on_change(self._restyle)

    def show_page(self, url: str = "index.html") -> None:
        """Open a page of the site, by its path from the site's root."""
        from ..docsite import ensure

        self.root = ensure()
        self._page = url or "index.html"
        if self.view is None:
            self.open_in_browser()
            return
        page, _, anchor = self._page.partition("#")
        address = QtCore.QUrl.fromLocalFile(str(self.root / page))
        address.setQuery(look_query())
        if anchor:
            address.setFragment(anchor)
        self.view.load(address)

    def current_file(self) -> Path | None:
        if self.view is not None and self.view.url().isLocalFile():
            return Path(self.view.url().toLocalFile())
        return (self.root / self._page.partition("#")[0]) if self.root else None

    def open_in_browser(self) -> None:
        import webbrowser

        from ..docsite import ensure

        self.root = self.root or ensure()
        path = self.current_file() or self.root / "index.html"
        webbrowser.open(path.as_uri())

    def _where(self, url: QtCore.QUrl) -> None:
        if self.root and url.isLocalFile():
            try:
                self.address.setText(str(Path(url.toLocalFile()).relative_to(self.root)))
            except ValueError:
                self.address.setText(url.toString())

    def _restyle(self) -> None:
        """The window's look changed: tell the page, without reloading it."""
        if self.view is None:
            return
        current = theme.current()
        look = {"theme": "dark" if current.dark else "light",
                "colorblind": current.preferences.colorblind,
                "dyslexia": current.preferences.dyslexia,
                "font": current.preferences.dyslexia_font,
                "scale": current.preferences.scale}
        import json

        self.view.page().runJavaScript(
            f"window.harwoodLook && window.harwoodLook({json.dumps(look)});")


def run(url: str = "index.html") -> int:
    """`harwood docs` with no window open: the documentation in one of its own."""
    from ._qt import application

    app = application()
    theme.apply()
    panel = DocsPanel()
    panel.setWindowTitle("harwood · documentation")
    panel.resize(1200, 860)
    panel.show()
    panel.show_page(url)
    return app.exec()
