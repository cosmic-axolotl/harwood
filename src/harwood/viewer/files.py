"""Text files, several at once: lists, coordinate files, tables.

Each file has a tab. The line the cursor is on is read for its columns, so
the bar underneath says not just "line 12" but "column 3 of 7: MAG" --
which column a number is in is exactly what is hard to see in a table that
has lost its alignment.

"Edit together" makes one edit to a batch of the open files (core/columns.py
has the edits). Before it does, each file is read for its shape, and every
difference that matters to that edit -- a different number of columns,
columns in another order, rows that are short, a file that is not a table
at all -- is listed with what to do about it. With nothing to say, the edit
is made. With something to say, it asks first; going ahead anyway is a
choice, never a default. Either way the edit goes into the tabs and not to
disk: it can be read, undone with Ctrl+Z, and is only written by Save.
"""

from __future__ import annotations

import re
from pathlib import Path

from PySide6 import QtCore, QtGui, QtWidgets

from ..core import columns
from ._qt import application
from .theme import palette as _c

#: Kept short: a table that is too wide to read is not helped by being seen.
OPEN_FILTER = ("Text tables (*.coo *.mag *.lis *.txt *.dat *.csv *.ecsv *.ctr *.sky *.obs "
               "*.cat *.out *.tab);;All files (*)")


class Highlighter(QtGui.QSyntaxHighlighter):
    """Comments quieter than the data; the header row (#N, or a first row of
    names) stands out, because it is what says what each column is."""

    def highlightBlock(self, text: str) -> None:
        stripped = text.lstrip()
        if stripped.startswith("#N"):
            names = QtGui.QTextCharFormat()
            names.setForeground(QtGui.QColor(_c().accent))
            names.setFontWeight(QtGui.QFont.Weight.Bold)
            self.setFormat(0, len(text), names)
        elif stripped.startswith("#"):
            comment = QtGui.QTextCharFormat()
            comment.setForeground(QtGui.QColor(_c().comment))
            self.setFormat(0, len(text), comment)


class FileTab(QtWidgets.QPlainTextEdit):
    """One open file."""

    def __init__(self, path: Path, text: str) -> None:
        super().__init__()
        self.path = path
        self.setPlainText(text)
        self.setLineWrapMode(QtWidgets.QPlainTextEdit.LineWrapMode.NoWrap)
        self.setTabStopDistance(32)
        self.highlighter = Highlighter(self.document())
        self.document().setModified(False)

    @property
    def modified(self) -> bool:
        return self.document().isModified()

    def replace_all(self, text: str) -> None:
        """Put new text in as one step of the undo history, so Ctrl+Z takes the
        whole batch edit back in one go."""
        cursor = QtGui.QTextCursor(self.document())
        cursor.beginEditBlock()
        cursor.select(QtGui.QTextCursor.SelectionType.Document)
        cursor.insertText(text)
        cursor.endEditBlock()

    def save(self) -> None:
        self.path.write_text(self.toPlainText(), encoding="utf-8")
        self.document().setModified(False)


class FileEditor(QtWidgets.QWidget):
    """The tabs, the bar under them, and the buttons to open, save and edit
    together."""

    #: Something to put in the session: what an edit together did.
    said = QtCore.Signal(str)

    def __init__(self, parent: QtWidgets.QWidget | None = None,
                 open_shortcut: bool = True) -> None:
        super().__init__(parent)
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        bar = QtWidgets.QHBoxLayout()
        bar.setContentsMargins(6, 4, 6, 4)
        for label, tip, slot in (
            ("📂 Open…", "Open one or more text files (Ctrl+O)", self.choose),
            ("💾 Save", "Save this file (Ctrl+S)", self.save_current),
            ("Save all", "Save every file with changes (Ctrl+Shift+S)", self.save_all),
            ("🧰 Edit together…", "Make one edit to several of the open files, after "
                                 "checking they are alike (Ctrl+Shift+T)", self.together),
        ):
            button = QtWidgets.QPushButton(label)
            button.setToolTip(tip)
            button.clicked.connect(slot)
            bar.addWidget(button)
        bar.addStretch(1)
        layout.addLayout(bar)

        self.tabs = QtWidgets.QTabWidget()
        self.tabs.setTabsClosable(True)
        self.tabs.setMovable(True)
        self.tabs.setDocumentMode(True)
        self.tabs.tabCloseRequested.connect(self.close_tab)
        self.tabs.currentChanged.connect(self._where)
        layout.addWidget(self.tabs, stretch=1)

        self.empty = QtWidgets.QLabel("No files open. 📂 Open… picks one or several; "
                                      "`edit a.coo b.coo` at the prompt does the same.")
        self.empty.setObjectName("hint")
        self.empty.setWordWrap(True)
        self.empty.setContentsMargins(10, 10, 10, 10)
        layout.addWidget(self.empty)

        self.where = QtWidgets.QLabel("")
        self.where.setObjectName("readout")
        self.where.setContentsMargins(8, 3, 8, 3)
        layout.addWidget(self.where)

        keys = [("Ctrl+S", self.save_current), ("Ctrl+Shift+S", self.save_all),
                ("Ctrl+Shift+T", self.together),
                ("Ctrl+W", lambda: self.close_tab(self.tabs.currentIndex()))]
        if open_shortcut:
            keys.append(("Ctrl+O", self.choose))
        for sequence, slot in keys:
            shortcut = QtGui.QShortcut(QtGui.QKeySequence(sequence), self)
            shortcut.setContext(QtCore.Qt.ShortcutContext.WidgetWithChildrenShortcut)
            shortcut.activated.connect(slot)
        self._sync()

    # ------------------------------------------------------------ files

    def files(self) -> list[FileTab]:
        return [self.tabs.widget(i) for i in range(self.tabs.count())]

    def choose(self) -> None:
        names, _ = QtWidgets.QFileDialog.getOpenFileNames(self, "Open text files", "",
                                                          OPEN_FILTER)
        self.open(names)

    def open(self, paths) -> list[str]:
        """Open each file in a tab (or go to its tab). Returns what could not be
        opened, each with why."""
        refused = []
        for name in paths:
            # Absolute: a tab saves to where it was opened from, whatever
            # `cd` has done since.
            path = Path(name).expanduser().absolute()
            existing = next((tab for tab in self.files()
                             if tab.path.resolve() == path.resolve()), None)
            if existing is not None:
                self.tabs.setCurrentWidget(existing)
                continue
            try:
                raw = path.read_bytes()
            except OSError as error:
                refused.append(f"{name}: {error.strerror or error}")
                continue
            if raw[:6] == b"SIMPLE":
                refused.append(f"{name} is a FITS image; `display {name}` shows it, "
                               f"`imheader {name}` lists its header")
                continue
            if b"\x00" in raw[:4096]:
                refused.append(f"{name} is not a text file")
                continue
            tab = FileTab(path, raw.decode("utf-8", errors="replace"))
            tab.document().modificationChanged.connect(lambda _m, t=tab: self._title(t))
            tab.cursorPositionChanged.connect(self._where)
            index = self.tabs.addTab(tab, path.name)
            self.tabs.setTabToolTip(index, str(path))
            self.tabs.setCurrentIndex(index)
        self._sync()
        for reason in refused:
            self.said.emit("  " + reason)
        return refused

    def _title(self, tab: FileTab) -> None:
        index = self.tabs.indexOf(tab)
        if index < 0:
            return
        # A mark and a word, not a colour: "● name" and "unsaved" in the tip.
        self.tabs.setTabText(index, ("● " if tab.modified else "") + tab.path.name)
        self.tabs.setTabToolTip(index, f"{tab.path}" + ("  (unsaved changes)"
                                                        if tab.modified else ""))
        self._where()

    def save_current(self) -> None:
        tab = self.tabs.currentWidget()
        if tab is not None:
            tab.save()
            self.said.emit(f"  saved {tab.path}")

    def save_all(self) -> None:
        saved = [tab for tab in self.files() if tab.modified]
        for tab in saved:
            tab.save()
        if saved:
            self.said.emit("  saved " + ", ".join(str(tab.path) for tab in saved))

    def close_tab(self, index: int) -> bool:
        tab = self.tabs.widget(index)
        if tab is None:
            return True
        if tab.modified:
            answer = QtWidgets.QMessageBox.question(
                self, "Unsaved changes", f"{tab.path.name} has changes that are not "
                "saved. Save them before closing?",
                QtWidgets.QMessageBox.StandardButton.Save
                | QtWidgets.QMessageBox.StandardButton.Discard
                | QtWidgets.QMessageBox.StandardButton.Cancel,
                QtWidgets.QMessageBox.StandardButton.Cancel)
            if answer == QtWidgets.QMessageBox.StandardButton.Cancel:
                return False
            if answer == QtWidgets.QMessageBox.StandardButton.Save:
                tab.save()
        self.tabs.removeTab(index)
        tab.deleteLater()
        self._sync()
        return True

    def close_all(self) -> bool:
        """Every tab, asking about each with changes. False if one was kept."""
        while self.tabs.count():
            if not self.close_tab(0):
                return False
        return True

    def _sync(self) -> None:
        has = self.tabs.count() > 0
        self.tabs.setVisible(has)
        self.empty.setVisible(not has)
        self._where()

    def _where(self, *_args) -> None:
        """line 12 · column 3 of 7: MAG · iraf, 7 columns, 3 rows · unsaved"""
        tab = self.tabs.currentWidget()
        if tab is None:
            self.where.setText("")
            return
        cursor = tab.textCursor()
        line = cursor.block().text()
        number = cursor.blockNumber() + 1
        shape = columns.layout(tab.toPlainText(), tab.path.name)
        text = f"line {number}"
        if not line.lstrip().startswith("#") and line.strip():
            spans = [m.span() for m in re.finditer(r"\S+", line)]
            here = cursor.positionInBlock()
            column = next((i for i, (a, b) in enumerate(spans, start=1) if here <= b),
                          len(spans))
            text += f" · column {column} of {len(spans)}"
            if 0 < column <= len(shape.names):
                text += f": {shape.names[column - 1]}"
        text += f" · {shape.summary()}"
        if tab.modified:
            text += " · unsaved"
        self.where.setText(text)

    # ------------------------------------------------------------ together

    def together(self) -> None:
        if not self.tabs.count():
            self.said.emit("  open the files to edit together first (📂 Open…)")
            return
        dialog = TogetherDialog(self)
        dialog.exec()


class TogetherDialog(QtWidgets.QDialog):
    """One edit, to several files, after looking at how alike they are."""

    def __init__(self, editor: FileEditor) -> None:
        super().__init__(editor)
        self.editor = editor
        self.setWindowTitle("Edit together")
        self.resize(900, 640)
        self.setMinimumSize(560, 420)
        self.setSizeGripEnabled(True)
        layout = QtWidgets.QVBoxLayout(self)

        layout.addWidget(_heading("Files"))
        self.table = QtWidgets.QTableWidget(0, 5)
        self.table.setHorizontalHeaderLabels(["", "file", "format", "columns, rows",
                                              "names"])
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.verticalHeader().hide()
        self.table.setEditTriggers(QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.setSelectionMode(QtWidgets.QAbstractItemView.SelectionMode.NoSelection)
        for tab in editor.files():
            row = self.table.rowCount()
            self.table.insertRow(row)
            include = QtWidgets.QTableWidgetItem()
            include.setFlags(QtCore.Qt.ItemFlag.ItemIsUserCheckable
                             | QtCore.Qt.ItemFlag.ItemIsEnabled)
            include.setCheckState(QtCore.Qt.CheckState.Checked)
            include.setToolTip("Include this file in the edit")
            self.table.setItem(row, 0, include)
            shape = columns.layout(tab.toPlainText(), tab.path.name)
            cells = (tab.path.name, shape.format if not shape.unreadable else "—",
                     shape.unreadable or f"{shape.width} columns, {shape.rows} rows",
                     " ".join(shape.names) or "(no names)")
            for column, text in enumerate(cells, start=1):
                item = QtWidgets.QTableWidgetItem(text)
                item.setToolTip(str(tab.path) if column == 1 else text)
                self.table.setItem(row, column, item)
        self.table.resizeColumnsToContents()
        self.table.itemChanged.connect(self._recheck)
        layout.addWidget(self.table, stretch=2)

        layout.addWidget(_heading("Edit"))
        form = QtWidgets.QGridLayout()
        self.kind = QtWidgets.QComboBox()
        for kind, title in columns.TITLES.items():
            self.kind.addItem(title, kind)
        self.kind.currentIndexChanged.connect(self._fields)
        form.addWidget(QtWidgets.QLabel("Edit"), 0, 0)
        form.addWidget(self.kind, 0, 1, 1, 3)

        self.inputs: dict[str, tuple[QtWidgets.QLabel, QtWidgets.QWidget]] = {}

        def field(key: str, label: str, widget: QtWidgets.QWidget, row: int, col: int):
            caption = QtWidgets.QLabel(label)
            form.addWidget(caption, row, col)
            form.addWidget(widget, row, col + 1)
            self.inputs[key] = (caption, widget)
            if isinstance(widget, QtWidgets.QLineEdit):
                widget.textChanged.connect(self._recheck)
            elif isinstance(widget, QtWidgets.QComboBox):
                widget.currentIndexChanged.connect(self._recheck)
            elif isinstance(widget, QtWidgets.QCheckBox):
                widget.toggled.connect(self._recheck)

        self.columns_edit = QtWidgets.QLineEdit()
        self.columns_edit.setPlaceholderText("3, or 2,4, or 2-5")
        field("columns", "Column", self.columns_edit, 1, 0)
        self.find_edit = QtWidgets.QLineEdit()
        field("find", "Find", self.find_edit, 1, 2)
        self.value_edit = QtWidgets.QLineEdit()
        field("value", "Value", self.value_edit, 2, 0)
        self.operator = QtWidgets.QComboBox()
        self.operator.addItems(["+", "-", "*", "/"])
        field("operator", "Operation", self.operator, 2, 2)
        self.name_edit = QtWidgets.QLineEdit()
        self.name_edit.setPlaceholderText("its name in a header, if the file has one")
        field("name", "Name", self.name_edit, 3, 0)
        self.regex = QtWidgets.QCheckBox("a regular expression")
        field("regex", "", self.regex, 3, 2)
        self.reverse = QtWidgets.QCheckBox("largest first")
        field("reverse", "", self.reverse, 4, 0)
        layout.addLayout(form)

        layout.addWidget(_heading("What to know first"))
        self.notes = QtWidgets.QTextBrowser()
        self.notes.setOpenLinks(False)
        layout.addWidget(self.notes, stretch=2)

        buttons = QtWidgets.QHBoxLayout()
        buttons.addStretch(1)
        close = QtWidgets.QPushButton("Close")
        close.clicked.connect(self.reject)
        self.apply_button = QtWidgets.QPushButton("Apply to the checked files")
        self.apply_button.setDefault(True)
        self.apply_button.clicked.connect(self.apply)
        buttons.addWidget(close)
        buttons.addWidget(self.apply_button)
        layout.addLayout(buttons)

        #: Answers the "are you sure?" question; replaced by tests.
        self.confirm = lambda flags, operation: ask_to_force(self, flags, operation)
        self._fields()

    # ------------------------------------------------------------ the edit

    def chosen(self) -> list[FileTab]:
        tabs = self.editor.files()
        return [tabs[row] for row in range(self.table.rowCount())
                if self.table.item(row, 0).checkState() == QtCore.Qt.CheckState.Checked]

    def operation(self) -> tuple[columns.Operation | None, str]:
        """The edit as filled in, or None and what is missing."""
        kind = self.kind.currentData()
        wanted, problem = (), ""
        if kind not in ("replace", "delete_rows"):
            wanted, problem = parse_columns(self.columns_edit.text())
            if problem:
                return None, problem
            if kind in ("insert_column", "set_column", "arithmetic", "sort") and \
                    len(wanted) != 1:
                return None, "this edit takes one column"
        if kind in ("replace", "delete_rows") and not self.find_edit.text():
            return None, "say what to find"
        if kind in ("insert_column", "set_column", "arithmetic") and \
                not self.value_edit.text().strip():
            return None, "give the value"
        if kind == "arithmetic":
            try:
                float(self.value_edit.text())
            except ValueError:
                return None, "the value has to be a number"
            if self.operator.currentText() == "/" and float(self.value_edit.text()) == 0:
                return None, "dividing by zero"
        if kind == "replace" and self.regex.isChecked():
            try:
                re.compile(self.find_edit.text())
            except re.error as error:
                return None, f"not a regular expression: {error}"
        return columns.Operation(
            kind, wanted, self.value_edit.text().strip(), self.operator.currentText(),
            self.find_edit.text(), self.regex.isChecked(), self.reverse.isChecked(),
            self.name_edit.text().strip()), ""

    def _fields(self) -> None:
        kind = self.kind.currentData()
        shown = {"replace": {"find", "value", "regex"}, "delete_rows": {"find"},
                 "delete_columns": {"columns"}, "keep_columns": {"columns"},
                 "insert_column": {"columns", "value", "name"},
                 "set_column": {"columns", "value"},
                 "arithmetic": {"columns", "operator", "value"},
                 "sort": {"columns", "reverse"}}[kind]
        for key, (caption, widget) in self.inputs.items():
            caption.setVisible(key in shown)
            widget.setVisible(key in shown)
        self.inputs["value"][0].setText("Replace with" if kind == "replace" else "Value")
        self._recheck()

    def flags(self) -> list:
        operation, _ = self.operation()
        if operation is None:
            return []
        shapes = [columns.layout(tab.toPlainText(), tab.path.name) for tab in self.chosen()]
        return columns.check(shapes, operation)

    def _recheck(self, *_args) -> None:
        operation, problem = self.operation()
        chosen = self.chosen()
        self.apply_button.setEnabled(operation is not None and bool(chosen))
        if not chosen:
            self.notes.setHtml("<p>Check at least one file.</p>")
            return
        if operation is None:
            self.notes.setHtml(f"<p>{_escape(problem.capitalize())}.</p>")
            return
        flags = self.flags()
        if not flags:
            self.notes.setHtml(
                f"<p>✔ <b>ready</b> — the {len(chosen)} file"
                f"{'s' if len(chosen) != 1 else ''} agree as far as "
                f"<i>{_escape(operation.describe())}</i> needs them to.</p>")
            return
        self.notes.setHtml(_flags_html(flags))

    def apply(self) -> bool:
        """Make the edit, asking first if anything was found. True if made."""
        operation, problem = self.operation()
        if operation is None:
            return False
        flags = self.flags()
        if flags and not self.confirm(flags, operation):
            return False
        report = []
        for tab in self.chosen():
            change = columns.apply(tab.toPlainText(), operation, tab.path.name)
            if change.text != tab.toPlainText():
                tab.replace_all(change.text)
            line = f"{tab.path.name}: {change.lines} line{'s' if change.lines != 1 else ''}"
            if change.skipped:
                line += f" ({change.skipped})"
            report.append(line)
        forced = " anyway" if flags else ""
        self.editor.said.emit(f"  edited together{forced}: {operation.describe()} -- "
                              + "; ".join(report)
                              + ". Not saved yet: Save all (Ctrl+Shift+S) writes them, "
                                "Ctrl+Z in a tab takes it back.")
        self.accept()
        return True


def ask_to_force(parent, flags, operation) -> bool:
    """"Are you sure?", with the reasons. Going back is the default."""
    box, force = force_box(parent, flags, operation)
    box.exec()
    return box.clickedButton() is force


def force_box(parent, flags, operation):
    """The question itself, and the button that means yes."""
    box = QtWidgets.QMessageBox(parent)
    box.setIcon(QtWidgets.QMessageBox.Icon.Warning)
    box.setWindowTitle("Are you sure?")
    box.setText(f"These files differ in ways that matter to “{operation.describe()}”.")
    box.setInformativeText(
        "\n\n".join(f"{flag.symbol} {flag.level}: {flag.message}\n    → {flag.fix}"
                    for flag in flags))
    back = box.addButton("Go back", QtWidgets.QMessageBox.ButtonRole.RejectRole)
    force = box.addButton("Apply anyway", QtWidgets.QMessageBox.ButtonRole.DestructiveRole)
    box.setDefaultButton(back)
    box.setEscapeButton(back)
    return box, force


def parse_columns(text: str) -> tuple[tuple[int, ...], str]:
    """'3', '2,4', '2-5' or '2 4' -> column numbers, or what is wrong with it."""
    text = text.strip()
    if not text:
        return (), "say which column (counting from 1)"
    found: list[int] = []
    for piece in re.split(r"[,\s]+", text):
        match = re.fullmatch(r"(\d+)(?:-(\d+))?", piece)
        if not match:
            return (), f"'{piece}' is not a column number"
        first, last = int(match.group(1)), int(match.group(2) or match.group(1))
        if first < 1 or last < first:
            return (), f"'{piece}': columns count from 1, low to high"
        found.extend(range(first, last + 1))
    return tuple(sorted(set(found))), ""


def _flags_html(flags) -> str:
    rows = []
    for flag in flags:
        rows.append(f"<p><b>{flag.symbol} {flag.level}</b>: {_escape(flag.message)}<br>"
                    f"&nbsp;&nbsp;&nbsp;→ {_escape(flag.fix)}</p>")
    return "".join(rows)


def _escape(text: str) -> str:
    import html

    return html.escape(text)


def _heading(text: str) -> QtWidgets.QLabel:
    label = QtWidgets.QLabel(text.upper())
    label.setObjectName("param")
    return label


def run(paths) -> int:
    """`harwood edit` from a shell: the editor in a window of its own."""
    from . import theme

    app = application()
    theme.apply()
    window = QtWidgets.QWidget()
    window.setWindowTitle("harwood · files")
    window.resize(1000, 700)
    layout = QtWidgets.QVBoxLayout(window)
    layout.setContentsMargins(0, 0, 0, 0)
    editor = FileEditor()
    layout.addWidget(editor)
    editor.said.connect(lambda text: print(text.strip()))
    editor.open(paths)

    def closing(event) -> None:
        if editor.close_all():
            event.accept()
        else:
            event.ignore()

    window.closeEvent = closing
    window.show()
    return app.exec()
