"""One QApplication, kept alive.

Qt needs exactly one application object, and it must outlive every window it
owns. Creating it in a local variable inside a function works by accident --
the reference happens to survive because the function has not returned yet --
and reads like dead code. Here it is deliberate, and module level is what
keeps it alive.
"""

from __future__ import annotations

from typing import Any

_application: Any = None
_window: Any = None


def set_window(window: Any) -> None:
    """Remember the main window, so panels can dock into it rather than
    opening as separate dialogs.

    Section 8.2 wants the whole environment in one window. A panel still has
    to work on its own -- `harwood implot` from a shell opens a window of its
    own -- so the panels do not know about the main window; they ask here, and
    get None when there is not one.
    """
    global _window
    _window = window


def window() -> Any:
    """The main window, if one is open."""
    return _window


def application() -> Any:
    """The QApplication, created on first use."""
    global _application
    from PySide6 import QtWidgets

    existing = QtWidgets.QApplication.instance()
    if existing is not None:
        _application = existing
    elif _application is None:
        _application = QtWidgets.QApplication([])
    return _application
