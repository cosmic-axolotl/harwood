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
    panel.finished.connect(dialog.accept)
    layout = QtWidgets.QVBoxLayout(dialog)
    layout.setContentsMargins(0, 0, 0, 0)
    layout.addWidget(panel)
    dialog.exec()
    return panel
