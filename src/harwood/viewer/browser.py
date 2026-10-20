"""The click-only way in: IRAF's package tree, and a form for each task.

    all > noao > digiphot > apphot > daofind

The same walk as typing `noao`, `digiphot`, `apphot` at the cl prompt, with
the two things IRAF never had: a way back that is always on screen (the
"Back to ..." button and the path above it, every part of which is a link),
and a line or two at each level saying what is in there. The long
explanations live in the documentation; here they stay short on purpose.

A task opens as a form built from its parameter model -- nothing in here
knows about any particular task -- with the command it amounts to written
underneath as it is filled in. Plan and Run send exactly that command
through the window's command line, so clicking and typing are not two ways
of doing a thing that might drift apart: they are one way, and the form
shows you the typing.
"""

from __future__ import annotations

import html
import types
import typing
from collections.abc import Callable
from pathlib import Path
from typing import Any, Literal, get_args, get_origin

from PySide6 import QtCore, QtGui, QtWidgets

from ..core import packages as tree_of
from ..core.packages import Node, TaskEntry
from .theme import palette as _c

#: Fields that name files to read: the button opens files (several, when the
#: field takes a list).
OPEN = {"image", "images", "input", "reference", "references", "coords", "coordlist",
        "extinction", "imagecur", "refpoints", "bpm", "dark", "flat", "zero",
        "standards", "instrument", "shifts", "shifts_file", "sensitivity",
        "operand1", "operand2", "masks", "oldnames"}
#: Fields that name files to write.
SAVE = {"output", "result", "logfile", "nrejmasks", "sigmas", "crmask", "newnames"}
#: Fields that name a directory.
FOLDER = {"database", "workdir", "caldir"}

FITS = "FITS images (*.fits *.fit *.fts *.imh);;Lists (*.lis *.list *.txt);;All files (*)"

#: What runs a command line and says how it went: (ok, what it printed).
Runner = Callable[[str], tuple[bool, str]]


def _style() -> str:
    """The browser's additions to the window's look: cards to press, a path
    of links, and the command underneath."""
    from . import theme

    c, scale = _c(), theme.current().scale
    icon, title = theme.BASE["body"] * 2.0 * scale, theme.BASE["body"] * 1.65 * scale
    return f"""
QPushButton#card {{ background: {c.panel}; border: 1px solid {c.line};
    border-radius: 10px; padding: 0; text-align: left; }}
QPushButton#card:hover {{ border: 2px solid {c.accent}; }}
QPushButton#card:focus {{ border: 2px solid {c.accent}; }}
QPushButton#card:pressed {{ background: {c.line}; }}
QPushButton#card[muted="true"] {{ background: {c.ground}; border-style: dashed; }}
QLabel#cardicon {{ color: {c.accent}; font-size: {icon:.1f}pt; background: transparent; }}
QLabel#cardname {{ font-weight: 700; background: transparent; }}
QLabel#cardbrief, QLabel#cardcount {{ color: {c.muted}; background: transparent; }}
QLabel#cardname[muted="true"] {{ color: {c.muted}; font-weight: 600; }}
QPushButton#back {{ color: {c.accent}; font-weight: 700; border: 1px solid {c.accent};
    border-radius: 14px; padding: 4px 14px; }}
QLabel#title {{ font-size: {title:.1f}pt; font-weight: 700; }}
QLabel#brief {{ color: {c.ink}; }}
QLabel#problem {{ color: {c.problem}; }}
QPushButton#run {{ background: {c.accent}; color: {c.ground}; font-weight: 700;
    border: none; border-radius: 14px; padding: 6px 18px; }}
QPushButton#run:disabled {{ background: {c.line}; color: {c.muted}; }}
QPushButton#plan {{ border-radius: 14px; padding: 6px 16px; font-weight: 600; }}
QPushButton#more {{ border: none; color: {c.accent}; text-align: left;
    font-weight: 600; padding: 4px 0; }}
QLineEdit[invalid="true"] {{ border: 2px dashed {c.problem}; }}
QPlainTextEdit#outcome {{ background: {c.panel}; border-radius: 6px; }}
"""


# ------------------------------------------------------------------ cards


class Card(QtWidgets.QPushButton):
    """A large, friendly button: a symbol, a name, a line about it.

    A real button underneath -- focus, Enter, Space, a name for a screen
    reader -- with labels laid over it, so it can wrap its text and still
    behave like the button it looks like. The symbol always comes with a
    word; it decorates, it never carries the meaning alone.
    """

    def __init__(self, icon: str, name: str, brief: str, count: str = "",
                 muted: bool = False) -> None:
        super().__init__()
        self.setObjectName("card")
        self.setProperty("muted", "true" if muted else "false")
        self.setCursor(QtCore.Qt.CursorShape.PointingHandCursor)
        self.setAccessibleName(name)
        self.setAccessibleDescription(brief)
        self.setToolTip(brief)

        grid = QtWidgets.QGridLayout(self)
        grid.setContentsMargins(12, 9, 12, 10)
        grid.setHorizontalSpacing(12)
        grid.setVerticalSpacing(2)
        symbol = QtWidgets.QLabel(icon)
        symbol.setObjectName("cardicon")
        symbol.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        symbol.setMinimumWidth(34)
        title = QtWidgets.QLabel(name)
        title.setObjectName("cardname")
        title.setProperty("muted", "true" if muted else "false")
        count_label = QtWidgets.QLabel(count)
        count_label.setObjectName("cardcount")
        text = QtWidgets.QLabel(brief)
        text.setObjectName("cardbrief")
        text.setWordWrap(True)
        grid.addWidget(symbol, 0, 0, 2, 1, QtCore.Qt.AlignmentFlag.AlignTop)
        grid.addWidget(title, 0, 1)
        grid.addWidget(count_label, 0, 2, QtCore.Qt.AlignmentFlag.AlignRight)
        grid.addWidget(text, 1, 1, 1, 2)
        grid.setColumnStretch(1, 1)
        for label in (symbol, title, count_label, text):
            label.setAttribute(QtCore.Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        self.name, self.brief = name, brief

    def hasHeightForWidth(self) -> bool:
        return True

    def heightForWidth(self, width: int) -> int:
        return self.layout().totalHeightForWidth(width)

    def sizeHint(self) -> QtCore.QSize:
        hint = self.layout().totalSizeHint()
        return QtCore.QSize(hint.width(), self.heightForWidth(max(hint.width(), 260)))

    def minimumSizeHint(self) -> QtCore.QSize:
        return QtCore.QSize(160, self.layout().totalMinimumSize().height())


# ------------------------------------------------------------ the fields


def _unwrap(annotation: Any) -> tuple[Any, bool]:
    """Optional[X] -> (X, True)."""
    if get_origin(annotation) in (typing.Union, types.UnionType):
        args = [a for a in get_args(annotation) if a is not type(None)]
        return args[0], type(None) in get_args(annotation)
    return annotation, False


def _first_sentence(text: str) -> str:
    """The hint under a field: its first sentence. The rest is in the
    tooltip, and all of it in the manual."""
    text = " ".join(text.split())
    end = text.find(". ")
    return text if end < 0 else text[: end + 1]


class Field:
    """One parameter's widget, and the value it holds.

    Every value goes through the model's own type for that field, so the form
    accepts exactly what the command line accepts and says so as it is typed,
    not after Run.
    """

    def __init__(self, name: str, info, value: Any, origin: str,
                 changed: Callable[[], None]) -> None:
        from pydantic import TypeAdapter

        self.name, self.info = name, info
        self.annotation = info.annotation
        self.inner, self.optional = _unwrap(self.annotation)
        self.is_list = get_origin(self.inner) is list
        self.adapter = TypeAdapter(self.annotation)
        self.required = info.is_required()
        self.error = ""
        self.browse: QtWidgets.QPushButton | None = None

        if self.inner is bool:
            self.widget: QtWidgets.QWidget = QtWidgets.QCheckBox(
                name.replace("_", " "))
            self.widget.setChecked(bool(value))
            self.widget.toggled.connect(changed)
            self.editor: QtWidgets.QWidget = self.widget
        elif get_origin(self.inner) is Literal:
            box = QtWidgets.QComboBox()
            box.addItems([str(v) for v in get_args(self.inner)])
            if self.optional:
                box.insertItem(0, "")
            box.setCurrentText("" if value is None else str(value))
            box.currentTextChanged.connect(changed)
            self.widget = self.editor = box
        else:
            line = QtWidgets.QLineEdit(self._text(value))
            if self.optional and value is None:
                line.setPlaceholderText("INDEF (not set)" if self.inner in (int, float)
                                        else "not set")
            if self.is_list:
                line.setPlaceholderText("one, or several separated by commas, or @list")
            line.textChanged.connect(changed)
            self.editor = line
            kind = self._kind()
            if kind:
                row = QtWidgets.QWidget()
                layout = QtWidgets.QHBoxLayout(row)
                layout.setContentsMargins(0, 0, 0, 0)
                layout.addWidget(line, stretch=1)
                self.browse = QtWidgets.QPushButton(
                    {"open": "Choose...", "save": "Save as...",
                     "folder": "Folder..."}[kind])
                self.browse.setToolTip(
                    "Pick several with Ctrl or Shift." if self.is_list and kind == "open"
                    else "")
                self.browse.clicked.connect(lambda: self._pick(kind))
                layout.addWidget(self.browse)
                self.widget = row
            else:
                self.widget = line
        description = info.description or ""
        self.editor.setToolTip(description)
        self.editor.setAccessibleName(name)
        self.editor.setAccessibleDescription(description)
        self.hint = _first_sentence(description)
        self.origin = origin

    def _kind(self) -> str:
        if self.name in FOLDER:
            return "folder"
        if self.name in SAVE:
            return "save"
        if self.name in OPEN or self.inner is Path:
            return "open"
        return ""

    def _text(self, value: Any) -> str:
        if value is None:
            return ""
        if isinstance(value, (list, tuple)):
            return ", ".join(str(v) for v in value)
        if isinstance(value, float):
            return f"{value:g}"
        return str(value)

    def _pick(self, kind: str) -> None:
        line = self.editor
        start = str(Path.cwd())
        if kind == "folder":
            chosen = QtWidgets.QFileDialog.getExistingDirectory(
                line, self.name, start)
            names = [chosen] if chosen else []
        elif kind == "save":
            chosen, _ = QtWidgets.QFileDialog.getSaveFileName(
                line, self.name, start, FITS)
            names = [chosen] if chosen else []
            if self.is_list and names and line.text().strip():
                # Output names for a list of inputs are picked one at a time.
                names = [*self._split(line.text()), *names]
        elif self.is_list:
            names, _ = QtWidgets.QFileDialog.getOpenFileNames(line, self.name, start, FITS)
        else:
            chosen, _ = QtWidgets.QFileDialog.getOpenFileName(line, self.name, start, FITS)
            names = [chosen] if chosen else []
        if names:
            self.set_files(names)

    def set_files(self, names: list[str]) -> None:
        """Fill the field with chosen files, written relative to where
        harwood runs when they are under it, so the command stays short."""
        def short(name: str) -> str:
            try:
                return str(Path(name).resolve().relative_to(Path.cwd().resolve()))
            except ValueError:
                return name
        shown = [short(n) for n in names]
        self.editor.setText(", ".join(shown) if self.is_list else shown[0])

    @staticmethod
    def _split(text: str) -> list[str]:
        # A comma inside brackets belongs to a section, 'obj.fits[1:100,*]'.
        from ..core.images import split_list

        return split_list(text.replace("\n", ","))

    def value(self) -> tuple[bool, Any]:
        """(given, value): given is False for an empty optional field, which
        then keeps its default. Sets self.error when the text will not do."""
        self.error = ""
        if isinstance(self.editor, QtWidgets.QCheckBox):
            return True, self.editor.isChecked()
        if isinstance(self.editor, QtWidgets.QComboBox):
            text = self.editor.currentText()
        else:
            text = self.editor.text().strip()
        if text == "":
            if self.required:
                self.error = "needed"
                return False, None
            if self.optional:
                return True, None
            return False, None
        raw: Any = self._split(text) if self.is_list else text
        if not self.is_list and text.upper() == "INDEF" and self.optional:
            return True, None
        from pydantic import ValidationError

        try:
            return True, self.adapter.validate_python(raw)
        except ValidationError as error:
            message = error.errors()[0].get("msg", str(error))
            self.error = message.replace("Input should be", "should be")
            return False, None

    def mark(self) -> None:
        """Show a bad value with a dashed border *and* words -- never colour
        alone."""
        if isinstance(self.editor, QtWidgets.QLineEdit):
            invalid = bool(self.error) and self.error != "needed"
            self.editor.setProperty("invalid", "true" if invalid else "false")
            self.editor.style().unpolish(self.editor)
            self.editor.style().polish(self.editor)


# ------------------------------------------------------------- the browser


class TaskBrowser(QtWidgets.QWidget):
    """The package tree, one level at a time, and a form for each task."""

    #: A command was put at the prompt for editing.
    to_prompt = QtCore.Signal(str)
    #: The task whose page of the documentation was asked for.
    manual = QtCore.Signal(str)

    def __init__(self, runner: Runner, parent: QtWidgets.QWidget | None = None,
                 current_image: Callable[[], str] | None = None) -> None:
        super().__init__(parent)
        self.runner = runner
        #: The image the window is showing, if it is a file: a form that
        #: needs an image starts with that one.
        self.current_image = current_image
        self.root = tree_of.tree()
        self.node: Node = self.root
        self.task: str | None = None
        self.fields: list[Field] = []
        self.show_missing = False

        outer = QtWidgets.QVBoxLayout(self)
        outer.setContentsMargins(10, 8, 10, 8)
        outer.setSpacing(6)

        # The way back, always in the same place, always visible.
        top = QtWidgets.QHBoxLayout()
        self.back_button = QtWidgets.QPushButton("◀  Back")
        self.back_button.setObjectName("back")
        self.back_button.setCursor(QtCore.Qt.CursorShape.PointingHandCursor)
        self.back_button.clicked.connect(self.back)
        top.addWidget(self.back_button)
        top.addStretch(1)
        outer.addLayout(top)
        self.crumbs = QtWidgets.QLabel()
        self.crumbs.setObjectName("crumbs")
        self.crumbs.setWordWrap(True)
        self.crumbs.setTextFormat(QtCore.Qt.TextFormat.RichText)
        self.crumbs.setTextInteractionFlags(
            QtCore.Qt.TextInteractionFlag.LinksAccessibleByMouse
            | QtCore.Qt.TextInteractionFlag.LinksAccessibleByKeyboard)
        self.crumbs.linkActivated.connect(self._crumb)
        outer.addWidget(self.crumbs)

        self.pages = QtWidgets.QStackedWidget()
        outer.addWidget(self.pages, stretch=1)
        self.pages.addWidget(self._build_navigator())
        self.pages.addWidget(self._build_form_page())

        for sequence in ("Alt+Left", "Backspace"):
            shortcut = QtGui.QShortcut(QtGui.QKeySequence(sequence), self)
            shortcut.setContext(QtCore.Qt.ShortcutContext.WidgetWithChildrenShortcut)
            shortcut.activated.connect(self._back_key)
        escape = QtGui.QShortcut(QtGui.QKeySequence("Escape"), self)
        escape.setContext(QtCore.Qt.ShortcutContext.WidgetWithChildrenShortcut)
        escape.activated.connect(self.back)

        from . import theme

        theme.follow(self, [], self._restyle)
        self._restyle()
        self.go("")

    # ------------------------------------------------------------- pages

    def _build_navigator(self) -> QtWidgets.QWidget:
        page = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(page)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)
        self.search = QtWidgets.QLineEdit()
        self.search.setPlaceholderText("Find a task: daofind, combine, airmass...")
        self.search.setClearButtonEnabled(True)
        self.search.textChanged.connect(lambda _: self._fill())
        self.search.returnPressed.connect(self._open_first)
        layout.addWidget(self.search)
        self.title = QtWidgets.QLabel()
        self.title.setObjectName("title")
        layout.addWidget(self.title)
        self.brief = QtWidgets.QLabel()
        self.brief.setObjectName("brief")
        self.brief.setWordWrap(True)
        layout.addWidget(self.brief)

        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)
        scroll.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.cards_body = QtWidgets.QWidget()
        self.cards = QtWidgets.QVBoxLayout(self.cards_body)
        self.cards.setContentsMargins(0, 4, 4, 4)
        self.cards.setSpacing(8)
        scroll.setWidget(self.cards_body)
        layout.addWidget(scroll, stretch=1)

        self.missing_box = QtWidgets.QCheckBox(
            "Also show IRAF tasks harwood does not have yet")
        self.missing_box.toggled.connect(self._toggle_missing)
        layout.addWidget(self.missing_box)
        return page

    def _build_form_page(self) -> QtWidgets.QWidget:
        page = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(page)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)
        self.form_title = QtWidgets.QLabel()
        self.form_title.setObjectName("title")
        layout.addWidget(self.form_title)
        self.form_brief = QtWidgets.QLabel()
        self.form_brief.setObjectName("brief")
        self.form_brief.setWordWrap(True)
        layout.addWidget(self.form_brief)

        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)
        scroll.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.form_body = QtWidgets.QWidget()
        self.form_layout = QtWidgets.QVBoxLayout(self.form_body)
        self.form_layout.setContentsMargins(0, 0, 6, 0)
        scroll.setWidget(self.form_body)
        layout.addWidget(scroll, stretch=1)

        # The command it all amounts to, kept up to date as the form changes.
        label = QtWidgets.QLabel("The same, typed:")
        label.setObjectName("hint")
        layout.addWidget(label)
        row = QtWidgets.QHBoxLayout()
        self.command = QtWidgets.QLabel()
        self.command.setObjectName("readout")
        self.command.setWordWrap(True)
        self.command.setTextFormat(QtCore.Qt.TextFormat.PlainText)
        self.command.setTextInteractionFlags(
            QtCore.Qt.TextInteractionFlag.TextSelectableByMouse
            | QtCore.Qt.TextInteractionFlag.TextSelectableByKeyboard)
        self.command.setAccessibleName("equivalent command")
        layout.addWidget(self.command)
        row.addStretch(1)
        copy = QtWidgets.QPushButton("Copy")
        copy.setToolTip("Copy the command, to paste in a terminal or a script.")
        copy.clicked.connect(lambda: QtWidgets.QApplication.clipboard().setText(
            self.command.text()))
        row.addWidget(copy)
        edit = QtWidgets.QPushButton("Edit as text")
        edit.setToolTip("Put this command at the command line, to change it there.")
        edit.clicked.connect(lambda: self.to_prompt.emit(self.command.text()))
        row.addWidget(edit)
        layout.addLayout(row)
        self.problems = QtWidgets.QLabel()
        self.problems.setObjectName("problem")
        self.problems.setWordWrap(True)
        layout.addWidget(self.problems)

        buttons = QtWidgets.QHBoxLayout()
        manual = QtWidgets.QPushButton("📖 Manual")
        manual.setToolTip("This task's page of the documentation: what it does, "
                          "every parameter, examples three ways. `help <task>` at "
                          "the prompt prints the same manual.")
        manual.clicked.connect(lambda: self.manual.emit(self.task))
        buttons.addWidget(manual)
        buttons.addStretch(1)
        self.plan_button = QtWidgets.QPushButton("☰  Plan")
        self.plan_button.setObjectName("plan")
        self.plan_button.setToolTip("Show the steps it would take, with the real "
                                    "values, and change nothing.")
        self.plan_button.clicked.connect(lambda: self.run(plan=True))
        buttons.addWidget(self.plan_button)
        self.run_button = QtWidgets.QPushButton("▶  Run")
        self.run_button.setObjectName("run")
        self.run_button.setCursor(QtCore.Qt.CursorShape.PointingHandCursor)
        self.run_button.clicked.connect(lambda: self.run(plan=False))
        buttons.addWidget(self.run_button)
        layout.addLayout(buttons)

        self.outcome = QtWidgets.QPlainTextEdit()
        self.outcome.setObjectName("outcome")
        self.outcome.setReadOnly(True)
        self.outcome.setMaximumHeight(160)
        self.outcome.hide()
        layout.addWidget(self.outcome)
        return page

    def _restyle(self) -> None:
        from . import theme

        self.setStyleSheet(_style())
        accent = theme.palette().accent
        self._link = f"color: {accent}; font-weight: 600; text-decoration: underline;"
        self._show_crumbs()

    # ------------------------------------------------------------ moving

    def go(self, path: str) -> None:
        """Into a package ("" is the top)."""
        self.node = tree_of.find(path, self.root)
        self.task = None
        self.search.blockSignals(True)
        self.search.clear()
        self.search.blockSignals(False)
        self.pages.setCurrentIndex(0)
        self._fill()
        self._show_crumbs()

    def back(self) -> None:
        """One level up: from a task to its package, from a package to the
        one that holds it."""
        if self.task is not None:
            self.go(self.node.path)
        elif self.search.text():
            self.search.clear()
        elif self.node.parent_path is not None:
            self.go(self.node.parent_path)

    def _back_key(self) -> None:
        # A text field takes Backspace for itself (Qt asks it first), so this
        # only fires elsewhere -- "back", as in a file manager.
        self.back()

    def _crumb(self, path: str) -> None:
        self.go(path)

    def _show_crumbs(self) -> None:
        if not hasattr(self, "_link"):
            return
        parts = [("", "all")]
        if self.node.path:
            pieces = self.node.path.split(".")
            parts += [(".".join(pieces[: i + 1]), piece) for i, piece in enumerate(pieces)]
        links = [f'<a href="{html.escape(p)}" style="{self._link}">{html.escape(n)}</a>'
                 for p, n in parts]
        if self.task is None:
            links[-1] = f"<b>{html.escape(parts[-1][1])}</b>"
        else:
            links.append(f"<b>{html.escape(self.task)}</b>")
        self.crumbs.setText(" › ".join(links))
        if self.task is not None:
            self.back_button.setText(f"◀  Back to {self.node.name}")
            self.back_button.setEnabled(True)
        elif self.node.parent_path is not None:
            parent = self.node.parent_path or "all"
            self.back_button.setText(f"◀  Back to {parent.rsplit('.', 1)[-1]}")
            self.back_button.setEnabled(True)
        else:
            self.back_button.setText("◀  Back")
            self.back_button.setEnabled(False)
        self.back_button.setToolTip("Alt+Left, Esc, or Backspace outside a text field.")

    def _toggle_missing(self, on: bool) -> None:
        self.show_missing = on
        self._fill()

    # ------------------------------------------------------------- cards

    def _clear_cards(self) -> None:
        _clear(self.cards)

    def _visible(self, node: Node) -> bool:
        return self.show_missing or node.available_count() > 0

    def _fill(self) -> None:
        """The cards for where we are, or for what the search finds."""
        self._clear_cards()
        query = self.search.text().strip().lower()
        if query:
            self._fill_search(query)
            return
        node = self.node
        self.title.setText(f"{node.icon}  {node.name}")
        self.brief.setText(node.brief)
        packages = [child for child in node.children if self._visible(child)]
        if packages:
            self.cards.addWidget(_heading("Packages"))
        for child in packages:
            have, total = child.available_count(), child.total_count()
            card = Card(child.icon, child.name, child.brief,
                        f"{have} of {total}" if self.show_missing else _plural(have, "task"),
                        muted=have == 0)
            card.setAccessibleDescription(f"package. {child.brief}")
            card.clicked.connect(lambda _=False, p=child.path: self.go(p))
            self.cards.addWidget(card)
        tasks = [t for t in node.tasks if t.available or self.show_missing]
        if tasks:
            self.cards.addWidget(_heading("Tasks"))
        for entry in tasks:
            self.cards.addWidget(self._task_card(entry))
        if not packages and not tasks:
            empty = QtWidgets.QLabel("harwood has nothing here yet. Tick the box "
                                     "below to see what IRAF has.")
            empty.setWordWrap(True)
            empty.setObjectName("hint")
            self.cards.addWidget(empty)
        self.cards.addStretch(1)

    def _task_card(self, entry: TaskEntry, where: str = "") -> Card:
        brief = entry.brief if entry.available else _missing_note(entry)
        if where:
            brief = f"{brief}  ({where.replace('.', ' › ')})"
        name = entry.name if entry.name == entry.iraf or not entry.iraf \
            else f"{entry.name}  (IRAF {entry.iraf})"
        card = Card("▸" if entry.available else "·", name, brief,
                    "" if entry.available else entry.status, muted=not entry.available)
        if entry.available:
            card.clicked.connect(lambda _=False, n=entry.name: self.open_task(n))
        else:
            card.clicked.connect(lambda _=False, e=entry: self._explain_missing(e))
        return card

    def _fill_search(self, query: str) -> None:
        self.title.setText("Search")
        found: list[tuple[TaskEntry, str]] = []
        seen = set()

        def walk(node: Node) -> None:
            for entry in node.tasks:
                if not (entry.available or self.show_missing):
                    continue
                text = f"{entry.name} {entry.iraf} {entry.brief}".lower()
                if query in text and (entry.name, node.path) not in seen:
                    seen.add((entry.name, node.path))
                    found.append((entry, node.path))
            for child in node.children:
                walk(child)

        walk(self.root)
        # Names that start with the query first, then names that contain it,
        # then the ones that only mention it.
        found.sort(key=lambda item: (not item[0].name.startswith(query),
                                     query not in item[0].name, not item[0].available,
                                     item[0].name))
        self.brief.setText(f"{_plural(len(found), 'task')} found" + ("" if found else
                           " -- try a shorter word, or tick the box below."))
        for entry, where in found[:60]:
            self.cards.addWidget(self._task_card(entry, where))
        self.cards.addStretch(1)
        self._found = found

    def _open_first(self) -> None:
        found = getattr(self, "_found", [])
        available = [entry for entry, _ in found if entry.available]
        if available:
            where = next(w for e, w in found if e is available[0])
            self.node = tree_of.find(where, self.root)
            self.open_task(available[0].name)

    def _explain_missing(self, entry: TaskEntry) -> None:
        QtWidgets.QToolTip.showText(QtGui.QCursor.pos(), _missing_note(entry), self)

    # -------------------------------------------------------------- form

    def open_task(self, name: str) -> None:
        """The form for one task, with what the project has set already."""
        from ..core import params as store
        from ..core.registry import get_class

        cls = get_class(name)
        if name not in {t.name for t in self.node.tasks}:
            places = tree_of.locate(name, self.root)
            if places:
                self.node = tree_of.find(places[0], self.root)
        self.task = name
        self.model = cls.Params
        try:
            resolved = store.resolve(name, cls.Params, preset=cls.defaults,
                                     preset_name=name if cls.defaults else None)
        except Exception:
            resolved = {}

        _clear(self.form_layout)
        self.fields = []
        self.form_title.setText(f"▸  {name}")
        self.form_brief.setText(cls.description or "")
        needed, rest = [], []
        for field_name, info in cls.Params.model_fields.items():
            entry = resolved.get(field_name)
            value = entry.value if entry is not None else None
            origin = entry.describe() \
                if entry is not None and entry.origin != "default" else ""
            if value is None and field_name in ("image", "images", "input") \
                    and info.is_required():
                shown = self.current_image() if self.current_image else ""
                if shown:
                    is_list = get_origin(_unwrap(info.annotation)[0]) is list
                    value = [shown] if is_list else shown
                    origin = "the image on display"
            field = Field(field_name, info, value, origin, self._changed)
            extra = info.json_schema_extra or {}
            (needed if info.is_required() or extra.get("positional") else rest).append(field)
            self.fields.append(field)

        if needed:
            self.form_layout.addWidget(_heading("What it works on"))
            for field in needed:
                self._add_field(self.form_layout, field)
        if rest:
            more = QtWidgets.QPushButton(f"▸  More options ({len(rest)})")
            more.setObjectName("more")
            more.setCheckable(True)
            box = QtWidgets.QWidget()
            inner = QtWidgets.QVBoxLayout(box)
            inner.setContentsMargins(0, 0, 0, 0)
            for field in rest:
                self._add_field(inner, field)
            box.setVisible(not needed)
            more.setChecked(not needed)

            count = len(rest)

            def toggle(on: bool, button=more, count=count, body=box) -> None:
                body.setVisible(on)
                button.setText(f"{'▾' if on else '▸'}  More options ({count})")
            more.toggled.connect(toggle)
            toggle(more.isChecked())
            self.form_layout.addWidget(more)
            self.form_layout.addWidget(box)
        self.form_layout.addStretch(1)
        self.outcome.hide()
        self.pages.setCurrentIndex(1)
        self._show_crumbs()
        self._changed()

    def _add_field(self, layout: QtWidgets.QVBoxLayout, field: Field) -> None:
        if not isinstance(field.editor, QtWidgets.QCheckBox):
            label = QtWidgets.QLabel(field.name + ("  (needed)" if field.required else ""))
            label.setObjectName("param")
            label.setBuddy(field.editor)
            layout.addWidget(label)
        layout.addWidget(field.widget)
        hint = field.hint + (f"  [{field.origin}]" if field.origin else "")
        if hint:
            text = QtWidgets.QLabel(hint)
            text.setObjectName("hint")
            text.setWordWrap(True)
            layout.addWidget(text)
        field.message = QtWidgets.QLabel()
        field.message.setObjectName("problem")
        field.message.setWordWrap(True)
        field.message.hide()
        layout.addWidget(field.message)
        layout.addSpacing(6)

    def values(self) -> tuple[dict[str, Any], list[str]]:
        """What the form holds, and what is wrong with it."""
        values: dict[str, Any] = {}
        problems: list[str] = []
        for field in self.fields:
            given, value = field.value()
            field.mark()
            message = getattr(field, "message", None)
            if field.error and field.error != "needed":
                problems.append(f"{field.name}: {field.error}")
                if message is not None:
                    message.setText(f"✗ {field.error}")
                    message.show()
            elif message is not None:
                message.hide()
            if field.error == "needed":
                problems.append(f"{field.name} is needed")
            if given:
                values[field.name] = value
        return values, problems

    def command_line(self) -> str:
        from ..core import replay

        values, _ = self.values()
        return replay.command(self.task or "", self.model, values)

    def _changed(self) -> None:
        if self.task is None:
            return
        values, problems = self.values()
        from ..core import replay

        self.command.setText(replay.command(self.task, self.model, values))
        self.problems.setText("  ·  ".join(problems))
        self.problems.setVisible(bool(problems))
        self.run_button.setEnabled(not problems)
        self.plan_button.setEnabled(not problems)

    def run(self, plan: bool = False) -> tuple[bool, str]:
        line = self.command_line() + (" --plan" if plan else "")
        return self._send(line)

    def _send(self, line: str) -> tuple[bool, str]:
        QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.CursorShape.WaitCursor)
        try:
            ok, printed = self.runner(line)
        finally:
            QtWidgets.QApplication.restoreOverrideCursor()
        if self.task is not None and not line.startswith("help "):
            mark = "✓ done" if ok else "✗ it did not work"
            self.outcome.setPlainText(f"{mark}\n{printed.rstrip()}")
            self.outcome.show()
        return ok, printed


def _plural(count: int, word: str) -> str:
    return f"{count} {word}" + ("" if count == 1 else "s")


def _heading(text: str) -> QtWidgets.QLabel:
    label = QtWidgets.QLabel(text.upper())
    label.setObjectName("param")
    return label


def _missing_note(entry: TaskEntry) -> str:
    words = {"planned": "Not in harwood yet.", "partial": "Partly in harwood.",
             "replaced": "Done another way in harwood.",
             "wontport": "Not carried over."}
    head = words.get(entry.status, "Not in harwood.")
    return f"{head} {entry.note}".strip()


def _clear(layout: QtWidgets.QLayout) -> None:
    while layout.count():
        item = layout.takeAt(0)
        widget = item.widget()
        if widget is not None:
            # Off the screen now; deleteLater alone leaves it painted until
            # the event loop comes round, which a quick click can see.
            widget.hide()
            widget.setParent(None)
            widget.deleteLater()
        elif item.layout() is not None:
            _clear(item.layout())
