"""harwood's own text editor, for when there is no $EDITOR to hand.

Small on purpose. It exists so that `--editpar` works the same way on a
machine where nobody has set EDITOR as on one where somebody has, without
falling back to `vi` and stranding a person who has never used it.

It is a plain text editor with one job: open, change, save, close. The
comments in the parameter file do the explaining, so the editor does not have
to. Close it and open it again as often as you like -- the file is regenerated
from the project each time, so it always shows where things actually stand.
"""

from __future__ import annotations

from pathlib import Path

from PySide6 import QtGui, QtWidgets

from ._qt import application

INK = "#d7dee5"
MUTED = "#8b9aa8"
GROUND = "#141a21"
PANEL = "#1b222b"
LINE = "#2a3541"
ACCENT = "#e0a33e"
COMMENT = "#7f93a6"

STYLE = f"""
QWidget {{ background: {GROUND}; color: {INK};
    font-family: "IBM Plex Sans", "DejaVu Sans", sans-serif; font-size: 12px; }}
QPlainTextEdit {{ background: {GROUND}; color: {INK}; border: none;
    font-family: "IBM Plex Mono", "DejaVu Sans Mono", monospace; font-size: 12px;
    selection-background-color: {LINE}; }}
QLabel#hint {{ color: {MUTED}; padding: 6px 10px; background: {PANEL};
    border-top: 1px solid {LINE}; }}
QPushButton {{ background: {PANEL}; color: {INK}; border: 1px solid {LINE};
    border-radius: 3px; padding: 5px 14px; }}
QPushButton:hover {{ border-color: {ACCENT}; }}
QPushButton#save {{ color: {ACCENT}; font-weight: 600; }}
"""


class CommentHighlighter(QtGui.QSyntaxHighlighter):
    """Dim the explanation, so the values stand out."""

    def highlightBlock(self, text: str) -> None:
        stripped = text.lstrip()
        if stripped.startswith("#"):
            comment = QtGui.QTextCharFormat()
            comment.setForeground(QtGui.QColor(COMMENT))
            self.setFormat(0, len(text), comment)
        elif "=" in text:
            value = QtGui.QTextCharFormat()
            value.setForeground(QtGui.QColor(ACCENT))
            start = text.index("=") + 1
            self.setFormat(start, len(text) - start, value)


class Editor(QtWidgets.QDialog):
    """Open a file, let it be changed, hand the text back."""

    def __init__(self, path: Path, title: str = "") -> None:
        super().__init__()
        self.path = Path(path)
        self.saved = False
        self.setWindowTitle(title or f"harwood · {self.path.name}")
        self.resize(860, 720)
        self.setStyleSheet(STYLE)

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        self.text = QtWidgets.QPlainTextEdit()
        self.text.setPlainText(self.path.read_text(encoding="utf-8"))
        self.text.setTabStopDistance(32)
        self.highlighter = CommentHighlighter(self.text.document())
        layout.addWidget(self.text, stretch=1)

        bar = QtWidgets.QWidget()
        row = QtWidgets.QHBoxLayout(bar)
        row.setContentsMargins(10, 8, 10, 8)
        hint = QtWidgets.QLabel("Ctrl+S saves and closes · Esc closes without saving")
        hint.setObjectName("hint")
        row.addWidget(hint, stretch=1)

        cancel = QtWidgets.QPushButton("Close")
        cancel.clicked.connect(self.reject)
        save = QtWidgets.QPushButton("Save and close")
        save.setObjectName("save")
        save.setDefault(True)
        save.clicked.connect(self._save)
        row.addWidget(cancel)
        row.addWidget(save)
        layout.addWidget(bar)

        QtGui.QShortcut(QtGui.QKeySequence.Save, self, self._save)

    def _save(self) -> None:
        self.path.write_text(self.text.toPlainText(), encoding="utf-8")
        self.saved = True
        self.accept()


def edit(path: Path, title: str = "") -> bool:
    """Open the file in harwood's editor. True if it was saved."""
    application()
    dialog = Editor(path, title)
    dialog.exec()
    return dialog.saved
