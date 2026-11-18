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

A long line wraps and the box grows with it (up to six lines, then it scrolls),
rather than running off to the side: the whole command stays in view, and a typo
at its start is fixed without scrolling back. Enter still runs it -- the command is
one line, however many it is shown on -- and Up and Down move through the wrapped
lines, going to the history only from the first line or the last.
"""

from __future__ import annotations

from PySide6 import QtCore, QtGui, QtWidgets

from ..cli import history
from ..cli import line as command_line

#: Most a list shows; past this it is a hint to type more, not a menu.
SHOWN = 200

_VALUE = QtCore.Qt.ItemDataRole.UserRole


#: The most lines the box grows to before it scrolls.
MAX_LINES = 6


class CommandLine(QtWidgets.QPlainTextEdit):
    """A command box that completes, remembers and wraps. Enter is the window's: it
    reads `take()`, which returns the line and keeps it in the history. It answers
    to what the window used of a QLineEdit (text, setText, cursorPosition,
    returnPressed, textEdited), the command always one line."""

    #: A word to say about a Tab that found nothing, or a history search
    #: that found nothing -- for the status bar, so silence is never the answer.
    said = QtCore.Signal(str)
    returnPressed = QtCore.Signal()
    textEdited = QtCore.Signal(str)

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("command")
        self.setLineWrapMode(QtWidgets.QPlainTextEdit.LineWrapMode.WidgetWidth)
        self.setWordWrapMode(QtGui.QTextOption.WrapMode.WrapAtWordBoundaryOrAnywhere)
        self.setTabChangesFocus(False)
        self.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setSizePolicy(QtWidgets.QSizePolicy.Policy.Expanding,
                           QtWidgets.QSizePolicy.Policy.Fixed)
        self._quiet = False              # a change made here, not typed
        self.document().contentsChanged.connect(self._grow)
        self.textChanged.connect(self._typed_in)
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
        self._grow()

    # ------------------------------------------------------- a line, wrapped

    def text(self) -> str:
        return self.toPlainText()

    def setText(self, text: str) -> None:
        self._quiet = True
        self.setPlainText(text.replace("\n", " "))
        self._quiet = False
        self.setCursorPosition(len(self.toPlainText()))

    def clear(self) -> None:
        self._quiet = True
        super().clear()
        self._quiet = False

    def cursorPosition(self) -> int:
        return self.textCursor().position()

    def setCursorPosition(self, position: int) -> None:
        cursor = self.textCursor()
        cursor.setPosition(max(0, min(position, len(self.toPlainText()))))
        self.setTextCursor(cursor)

    def _typed_in(self) -> None:
        if not self._quiet:
            self.textEdited.emit(self.toPlainText())

    def insertFromMimeData(self, source) -> None:
        """A pasted block is one command: its line breaks become spaces."""
        if source.hasText():
            self.insertPlainText(" ".join(source.text().split("\n")))
        else:
            super().insertFromMimeData(source)

    def _grow(self) -> None:
        """As tall as its wrapped lines, one to MAX_LINES; past that it scrolls."""
        metrics = QtGui.QFontMetricsF(self.font())
        lines = 0
        block = self.document().begin()
        while block.isValid():
            layout = block.layout()
            lines += max(1, layout.lineCount() if layout is not None else 1)
            block = block.next()
        lines = max(1, min(MAX_LINES, lines))
        margins = self.contentsMargins()
        frame = 2 * self.frameWidth() + margins.top() + margins.bottom() \
            + 2 * int(self.document().documentMargin())
        self.setFixedHeight(int(lines * metrics.lineSpacing() + frame + 2))

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        self._grow()

    def _on_edge(self, up: bool) -> bool:
        """The cursor on the first visual line (up) or the last (down)."""
        cursor = QtGui.QTextCursor(self.textCursor())
        op = QtGui.QTextCursor.MoveOperation.Up if up else QtGui.QTextCursor.MoveOperation.Down
        return not cursor.movePosition(op)

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
        # On an empty prompt Tab and Shift+Tab move the focus, as in any field
        if event.type() == QtCore.QEvent.Type.KeyPress and \
                event.key() in (QtCore.Qt.Key.Key_Tab, QtCore.Qt.Key.Key_Backtab) and \
                (event.key() == QtCore.Qt.Key.Key_Backtab
                 or event.modifiers() & QtCore.Qt.KeyboardModifier.ShiftModifier
                 or not self.text().strip()):
            back = event.key() == QtCore.Qt.Key.Key_Backtab or \
                bool(event.modifiers() & QtCore.Qt.KeyboardModifier.ShiftModifier)
            self.focusPreviousChild() if back else self.focusNextChild()
            return True
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
        if key in (QtCore.Qt.Key.Key_Return, QtCore.Qt.Key.Key_Enter):
            self.returnPressed.emit()    # the command is one line: Enter runs it
            return
        if key == QtCore.Qt.Key.Key_Up and self._on_edge(True):
            self.walk(-1)
            return
        if key == QtCore.Qt.Key.Key_Down and self._on_edge(False):
            self.walk(+1)
            return
        super().keyPressEvent(event)
