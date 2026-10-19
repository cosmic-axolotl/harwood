"""The window's command line: the hw> prompt, with what a prompt should have.

    Tab        completes the word: a task, an option, a value it allows, a
               file. One candidate is written in; several are listed with
               what each one is, and the part they share is written in.
    Up, Down   walk the history. With something typed, only the lines that
               start with it -- type `imc` and Up finds the last imcombine.
    Ctrl+R     searches the whole history for what is typed, anywhere in a
               line, newest first.

The completions and the history are the terminal session's own (cli/line.py,
cli/history.py): what Tab offers here is what it offers at `hw>` in a
terminal, and a line typed in one is found by Up in the other.

Tab on an empty prompt still moves the focus, as everywhere else, so that a
keyboard can always get out of here; Shift+Tab always does.
"""

from __future__ import annotations

from PySide6 import QtCore, QtGui, QtWidgets

from ..cli import history
from ..cli import line as command_line

#: Most a list shows; past this it is a hint to type more, not a menu.
SHOWN = 200

_VALUE = QtCore.Qt.ItemDataRole.UserRole


class CommandLine(QtWidgets.QLineEdit):
    """A line edit that completes and remembers. Enter is the window's: it
    reads `take()`, which returns the line and keeps it in the history."""

    #: A word to say about a Tab that found nothing, or a history search
    #: that found nothing -- for the status bar, so silence is never the answer.
    said = QtCore.Signal(str)

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("command")
        self.lines = history.load()
        self._at = len(self.lines)       # where Up and Down are in the history
        self._typed = ""                 # what was there before walking it
        self._start = 0                  # where the word being completed begins
        self._searching = False          # the list is the history, not words

        self._model = QtGui.QStandardItemModel(self)
        self.completer_ = QtWidgets.QCompleter(self._model, self)
        self.completer_.setWidget(self)
        self.completer_.setCompletionMode(
            QtWidgets.QCompleter.CompletionMode.UnfilteredPopupCompletion)
        self.completer_.setMaxVisibleItems(12)
        popup = QtWidgets.QTreeView()
        popup.setHeaderHidden(True)
        popup.setRootIsDecorated(False)
        popup.setUniformRowHeights(True)
        popup.setObjectName("completions")
        self.completer_.setPopup(popup)
        self.completer_.activated[QtCore.QModelIndex].connect(self._chosen)
        self.textEdited.connect(self._edited)

    # ------------------------------------------------------------ history

    def take(self) -> str:
        """The line as typed, kept in the history; the prompt is left empty."""
        line = self.text().strip()
        self.clear()
        if line:
            history.add(line)
            if not self.lines or self.lines[-1] != line:
                self.lines.append(line)
        self._at, self._typed = len(self.lines), ""
        return line

    def _edited(self, _text: str) -> None:
        """Typing starts a new walk: the next Up searches for what is there now."""
        self._at, self._typed = len(self.lines), ""
        if self.completer_.popup().isVisible():
            if self._searching:
                self.search_history()
            else:
                self.complete(refresh=True)

    def walk(self, step: int) -> None:
        """Up (-1) or Down (+1) through the lines starting with what was typed."""
        if self._at == len(self.lines):
            self._typed = self.text()
        at = self._at + step
        while 0 <= at < len(self.lines) and not self.lines[at].startswith(self._typed):
            at += step
        if at < 0:
            self.said.emit("the start of the history" if not self._typed else
                           f"nothing earlier starts with '{self._typed}'")
            return
        if at >= len(self.lines):
            self._at = len(self.lines)
            self.setText(self._typed)
            return
        self._at = at
        self.setText(self.lines[at])

    def search_history(self) -> None:
        """Ctrl+R: every line containing what is typed, newest first, once each."""
        text = self.text().strip()
        found, seen = [], set()
        for line in reversed(self.lines):
            if text in line and line not in seen:
                seen.add(line)
                found.append(line)
        self._searching = True
        if not found:
            self.completer_.popup().hide()
            self.said.emit(f"nothing typed before contains '{text}'" if text
                           else "nothing typed yet")
            return
        self._show([(line, "") for line in found[:SHOWN]])

    # ------------------------------------------------------------ completion

    def complete(self, refresh: bool = False) -> None:
        """Tab: write in what is certain, list what is not."""
        self._searching = False
        cursor = self.cursorPosition()
        found = command_line.complete(self.text()[:cursor])
        self._start = found.start
        if not found.items:
            self.completer_.popup().hide()
            if not refresh:
                self.said.emit("nothing to complete here")
            return
        if len(found.items) == 1 and not refresh:
            item = found.items[0]
            self._replace(item.value + ("" if item.kind == "dir" else " "))
            return
        typed = cursor - found.start
        common = found.common()
        if len(common) > typed and not refresh:
            self._replace(common)
        self._show([(item.value, item.help) for item in found.items[:SHOWN]])

    def _show(self, rows: list[tuple[str, str]]) -> None:
        from .theme import palette

        self._model.clear()
        quiet = QtGui.QColor(palette().muted)
        for value, help_text in rows:
            shown = QtGui.QStandardItem(value)
            shown.setData(value, _VALUE)
            shown.setToolTip(help_text or value)
            about = QtGui.QStandardItem(help_text)
            about.setData(value, _VALUE)
            about.setForeground(quiet)
            self._model.appendRow([shown, about])
        popup = self.completer_.popup()
        popup.resizeColumnToContents(0)
        rectangle = self.cursorRect()
        rectangle.setWidth(max(self.width() - rectangle.x(), 320))
        self.completer_.complete(rectangle)
        popup.setCurrentIndex(self._model.index(0, 0))

    def _chosen(self, index: QtCore.QModelIndex) -> None:
        value = index.data(_VALUE)
        if value is None:
            return
        if self._searching:
            self.setText(value)
        else:
            self._replace(value + ("" if value.endswith("/") else " "))
        self._searching = False

    def _replace(self, value: str) -> None:
        """Put `value` in place of the word being completed."""
        text, cursor = self.text(), self.cursorPosition()
        self.setText(text[:self._start] + value + text[cursor:])
        self.setCursorPosition(self._start + len(value))

    # ------------------------------------------------------------ keys

    def event(self, event: QtCore.QEvent) -> bool:
        # Tab has to be caught before QWidget turns it into a focus change.
        if event.type() == QtCore.QEvent.Type.KeyPress and \
                event.key() == QtCore.Qt.Key.Key_Tab and \
                not event.modifiers() & ~QtCore.Qt.KeyboardModifier.KeypadModifier and \
                self.text().strip():
            popup = self.completer_.popup()
            if popup.isVisible():
                index = popup.currentIndex()
                popup.hide()
                if index.isValid():
                    self._chosen(index)
            else:
                self.complete()
            return True
        return super().event(event)

    def keyPressEvent(self, event: QtGui.QKeyEvent) -> None:
        key, modifiers = event.key(), event.modifiers()
        popup = self.completer_.popup()
        if popup.isVisible() and key in (QtCore.Qt.Key.Key_Return, QtCore.Qt.Key.Key_Enter,
                                         QtCore.Qt.Key.Key_Escape):
            event.ignore()               # the completer's: choose, or close the list
            return
        if key == QtCore.Qt.Key.Key_Space and \
                modifiers & QtCore.Qt.KeyboardModifier.ControlModifier:
            self.complete()
            return
        if key == QtCore.Qt.Key.Key_R and \
                modifiers & QtCore.Qt.KeyboardModifier.ControlModifier:
            self.search_history()
            return
        if key == QtCore.Qt.Key.Key_Up:
            self.walk(-1)
            return
        if key == QtCore.Qt.Key.Key_Down:
            self.walk(+1)
            return
        super().keyPressEvent(event)
