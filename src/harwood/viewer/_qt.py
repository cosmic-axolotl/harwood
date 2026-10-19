"""One QApplication, kept alive.

Qt needs exactly one application object, and it must outlive every window it
owns. Creating it in a local variable inside a function works by accident --
the reference happens to survive because the function has not returned yet --
and reads like dead code. Here it is deliberate, and module level is what
keeps it alive.
"""

from __future__ import annotations

import functools
import threading
from typing import Any

_application: Any = None
_window: Any = None
_courier: Any = None


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

    if threading.current_thread() is not threading.main_thread():
        # Qt widgets live on one thread. Touching them from another does not
        # fail cleanly; it crashes later, somewhere else. Say so here instead.
        raise RuntimeError("the window was reached from a task's own thread; "
                           "the function that did it needs @on_main (viewer/_qt.py)")

    existing = QtWidgets.QApplication.instance()
    if existing is not None:
        _application = existing
    elif _application is None:
        _application = QtWidgets.QApplication([])
    return _application


def on_main(function):
    """Run `function` on the window's thread, whichever thread calls it.

    The window runs a task beside itself, on a thread of its own, so that it
    keeps drawing -- and can show a clock -- while the task works. A task that
    stops to open a panel (an icfit fit, apedit, identify) calls a function
    marked with this; the call is handed to the window's thread, and the
    task's thread waits for the panel's answer, exactly as it would have
    waited had everything been on one thread.
    """
    @functools.wraps(function)
    def wrapper(*args, **kwargs):
        if threading.current_thread() is threading.main_thread():
            return function(*args, **kwargs)
        done = threading.Event()
        outcome: dict[str, Any] = {}

        def job() -> None:
            try:
                outcome["value"] = function(*args, **kwargs)
            except BaseException as error:      # handed back to the task's thread
                outcome["error"] = error
            finally:
                done.set()

        if _courier is None:
            raise RuntimeError(f"{function.__name__} needs the window, and the window "
                               "is not taking calls")
        _courier.deliver.emit(job)
        done.wait()
        if "error" in outcome:
            raise outcome["error"]
        return outcome.get("value")

    return wrapper


def accept_calls() -> None:
    """Called once, on the window's thread, by the window that will run tasks
    beside itself: from here on, on_main has somewhere to send its calls."""
    global _courier
    if _courier is not None:
        return
    from PySide6 import QtCore

    class Courier(QtCore.QObject):
        deliver = QtCore.Signal(object)

        def __init__(self) -> None:
            super().__init__()
            self.deliver.connect(self.run, QtCore.Qt.ConnectionType.QueuedConnection)

        def run(self, job) -> None:
            job()

    _courier = Courier()
