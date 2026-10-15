"""Putting a panel somewhere: inside the main window, or in one of its own.

Every interactive panel -- icfit, identify, implot, the parameter editor --
has to work two ways. Run from a shell it is a window of its own and the
command waits for it. Run inside `harwood gui` it belongs in the main window,
beside the image it refers to, because looking at a radial profile in a
separate window from the star it came from is most of what is wrong with
working across three windows in IRAF.

The panels do not know which of those is happening. They emit `finished` and
this decides where they lived.
"""

from __future__ import annotations

from typing import Any

from ._qt import application, window


def show(panel: Any, title: str, size: tuple[int, int] = (1000, 640)) -> Any:
    """Show a panel and wait for it to finish. Returns the panel.

    Docked into the main window when there is one, otherwise a dialog. Either
    way this returns only once the panel says it is done, so a task that opens
    one can go on to use what it produced.
    """
    from PySide6 import QtCore, QtWidgets

    application()
    main = window()
    if main is not None and hasattr(main, "open_panel"):
        main.open_panel(panel, title)
        loop = QtCore.QEventLoop()
        panel.finished.connect(loop.quit)
        loop.exec()
        main.close_panel(panel)
        return panel

    dialog = QtWidgets.QDialog()
    dialog.setWindowTitle(title)
    dialog.resize(*size)
    # Small enough to be arranged on a laptop, and resizable: a dialog with a
    # size grip and no maximise button is the one thing worse than a fixed one.
    dialog.setMinimumSize(520, 360)
    dialog.setSizeGripEnabled(True)
    dialog.setWindowFlag(QtCore.Qt.WindowType.WindowMaximizeButtonHint, True)

    panel.finished.connect(dialog.accept)
    layout = QtWidgets.QVBoxLayout(dialog)
    layout.setContentsMargins(0, 0, 0, 0)
    layout.addWidget(panel)
    _text_shortcuts(dialog)
    dialog.exec()
    return panel


def _text_shortcuts(dialog: Any) -> None:
    """Ctrl+ and Ctrl- in a panel that opened on its own.

    The main window scales its text through its stylesheet; a dialog has none
    of its own, so here the application font is moved instead. Same keys, so
    it does not matter to the person pressing them which of the two they are
    looking at.
    """
    from PySide6 import QtGui, QtWidgets

    app = application()
    start = app.font().pointSizeF()
    if start <= 0:
        start = 10.0

    def step(direction: int) -> None:
        font = app.font()
        size = max(6.0, min(28.0, font.pointSizeF() + direction))
        font.setPointSizeF(size)
        app.setFont(font)
        for widget in QtWidgets.QApplication.allWidgets():
            widget.setFont(font)
            widget.updateGeometry()

    def reset() -> None:
        font = app.font()
        font.setPointSizeF(start)
        app.setFont(font)

    for sequences, slot in (
        (("Ctrl++", "Ctrl+=", "Ctrl+Shift+=", "Ctrl+Shift++"), lambda: step(+1)),
        (("Ctrl+-", "Ctrl+_"), lambda: step(-1)),
        (("Ctrl+0",), reset),
        (("F11",), lambda: (dialog.showNormal() if dialog.isFullScreen()
                            else dialog.showFullScreen())),
        (("Ctrl+M",), lambda: (dialog.showNormal() if dialog.isMaximized()
                               else dialog.showMaximized())),
    ):
        for sequence in sequences:
            QtGui.QShortcut(QtGui.QKeySequence(sequence), dialog, slot)
