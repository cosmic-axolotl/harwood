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

from . import lang, theme
from .lang import N_, tr


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
            ("◀", N_("Back"), lambda: self.view and self.view.back()),
            ("▶", N_("Forward"), lambda: self.view and self.view.forward()),
            (N_("⌂ Home"), N_("The first page"), lambda: self.show_page("index.html")),
            (N_("Open in browser"), N_("The same page in your web browser"),
             self.open_in_browser),
        ):
            button = lang.bind(QtWidgets.QPushButton(), "setText", label)
            lang.bind(button, "setToolTip", tip)
            lang.bind(button, "setAccessibleName", tip)
            button.clicked.connect(slot)
            bar.addWidget(button)
        self.address = QtWidgets.QLabel("")
        self.address.setObjectName("hint")
        bar.addWidget(self.address, stretch=1)
        layout.addLayout(bar)

        if self.view is not None:
            layout.addWidget(self.view, stretch=1)
        else:
            missing = lang.bind(QtWidgets.QLabel(), "setText",
                                "Showing the documentation here needs QtWebEngine: "
                                "pip install 'harwood[docs]'. Until then each page opens "
                                "in your browser -- the same pages.")
            missing.setWordWrap(True)
            missing.setContentsMargins(12, 12, 12, 12)
            layout.addWidget(missing, stretch=1)
        self._page = "index.html"
        theme.on_change(self._restyle)

    def home(self) -> Path:
        """Where the pages in the window's language are."""
        from ..core.i18n import language
        from ..docsite.builder import language_root

        return self.root / language_root(language()) if self.root else Path(".")

    def show_page(self, url: str = "index.html") -> None:
        """Open a page of the site, by its path from the site's root, in the
        window's language."""
        from ..docsite import ensure

        self.root = ensure()
        self._page = url or "index.html"
        if self.view is None:
            self.open_in_browser()
            return
        page, _, anchor = self._page.partition("#")
        address = QtCore.QUrl.fromLocalFile(str(self.home() / page))
        address.setQuery(look_query())
        if anchor:
            address.setFragment(anchor)
        self.view.load(address)

    def current_file(self) -> Path | None:
        if self.view is not None and self.view.url().isLocalFile():
            return Path(self.view.url().toLocalFile())
        return (self.home() / self._page.partition("#")[0]) if self.root else None

    def open_in_browser(self) -> None:
        import webbrowser

        from ..docsite import ensure

        self.root = self.root or ensure()
        path = self.current_file() or self.home() / "index.html"
        webbrowser.open(path.as_uri())

    def retranslate(self) -> None:
        """The window's language changed: the page shown, in it."""
        if self.root is None:
            return
        current = self.current_file()
        page = self._page
        if current is not None:
            from ..core.i18n import LANGUAGES

            try:
                relative = current.relative_to(self.root).as_posix()
            except ValueError:
                relative = page
            first, _, rest = relative.partition("/")
            page = rest if first in LANGUAGES and rest else relative
            anchor = self.view.url().fragment() if self.view is not None else ""
            page += f"#{anchor}" if anchor else ""
        if self.view is not None and self.parent() is not None and self.isVisible():
            self.show_page(page)
        else:
            self._page = page

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
    panel.setWindowTitle(tr("harwood · documentation"))
    panel.resize(1200, 860)
    panel.show()
    panel.show_page(url)
    return app.exec()
