"""The window's language: its texts, and changing it while the window is open.

`tr` gives a text in the language now (core/i18n.py). A text set on a widget
that stays -- a dock's title, a button, a placeholder -- is set with `bind`,
which remembers the widget, the setter and the English, so that `switch` can set
it again in the new language without the window being made again. Menus are
rebuilt instead, and what a task prints from then on comes in the new language;
what was printed before stays as it was.
"""

from __future__ import annotations

import weakref
from collections.abc import Callable

from ..core import i18n
from ..core.i18n import LANGUAGES, N_, NAMES, tr, trf  # noqa: F401 -- the window's names

_bound: list[tuple[weakref.ref, str, str, tuple]] = []
_listeners: list[Callable[[], None]] = []


def bind(widget, setter: str, text: str, *before):
    """widget.setter(*before, tr(text)), now and after every switch. Gives the widget."""
    getattr(widget, setter)(*before, tr(text))
    _bound.append((weakref.ref(widget), setter, text, before))
    return widget


def action(menu, text: str, slot=None, shortcut=None):
    """A menu's action, its text translated (the menu is rebuilt on a switch)."""
    from PySide6 import QtGui

    made = menu.addAction(tr(text))
    if slot is not None:
        made.triggered.connect(lambda *_: slot())
    if shortcut is not None:
        made.setShortcut(QtGui.QKeySequence(shortcut))
    return made


def on_switch(listener: Callable[[], None]) -> None:
    """Call `listener` after each switch (to rebuild menus, reload a page...)."""
    _listeners.append(listener)


def retranslate() -> None:
    """Every bound text again, in the language now; widgets gone are forgotten."""
    alive = []
    for ref, setter, text, before in _bound:
        widget = ref()
        if widget is None:
            continue
        try:
            getattr(widget, setter)(*before, tr(text))
        except RuntimeError:        # its C++ side was deleted
            continue
        alive.append((ref, setter, text, before))
    _bound[:] = alive


def switch(choice: str) -> None:
    """Speak `choice` (a language, or "system") from now on, save it with the other
    preferences, and say it again everywhere."""
    from . import theme

    i18n.set_language(choice if choice in LANGUAGES else None)
    theme.update(language=choice)
    retranslate()
    for listener in list(_listeners):
        try:
            listener()
        except RuntimeError:
            _listeners.remove(listener)
