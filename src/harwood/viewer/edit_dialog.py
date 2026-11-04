"""A file to change, in a dialog that waits for Save or Cancel: IRAF's edit, in the window."""

from __future__ import annotations

from pathlib import Path

from PySide6 import QtGui, QtWidgets

from ._qt import on_main


@on_main
def edit(path: str) -> None:
    """Open `path`; Save writes what is in the box back, Cancel leaves the file as it was."""
    dialog = QtWidgets.QDialog()
    dialog.setWindowTitle(f"edit · {Path(path).name}")
    layout = QtWidgets.QVBoxLayout(dialog)
    box = QtWidgets.QPlainTextEdit()
    box.setFont(QtGui.QFontDatabase.systemFont(QtGui.QFontDatabase.SystemFont.FixedFont))
    box.setPlainText(Path(path).read_text(encoding="utf-8", errors="replace"))
    layout.addWidget(box)
    buttons = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.StandardButton.Save
                                         | QtWidgets.QDialogButtonBox.StandardButton.Cancel)
    buttons.accepted.connect(dialog.accept)
    buttons.rejected.connect(dialog.reject)
    layout.addWidget(buttons)
    dialog.resize(900, 640)
    if dialog.exec() == QtWidgets.QDialog.DialogCode.Accepted:
        Path(path).write_text(box.toPlainText(), encoding="utf-8")
    dialog.deleteLater()
