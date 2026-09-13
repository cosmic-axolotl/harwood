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
