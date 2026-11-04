"""A line asked for in the window, as a task would ask for it at the terminal."""

from __future__ import annotations

from PySide6 import QtWidgets

from ._qt import on_main


@on_main
def line(prompt: str, title: str = "harwood") -> str | None:
    """The question -- what the task printed since the last answer -- and a line to
    type; Cancel is the end of the input, as Ctrl-D at the terminal."""
    text, ok = QtWidgets.QInputDialog.getText(None, title, prompt.strip("\n") or "?")
    return text if ok else None
