"""Text tables as harwood and IRAF write them, and whether several of them
can be edited as one.

Editing a batch of files at once is only safe when they have the same shape.
"Delete column 4" means the magnitude in one file and its error in another
if the two were written by different versions of a task, and nothing about
the result would say so. So before an edit touches more than one file, each
one is read for its shape -- format, columns, their names, which hold numbers
-- and every difference that matters to *that* edit is said in words, with
what to do about it. The person can still go ahead: they may know that the
difference is harmless. They just cannot go ahead without having been told.

The formats met in a reduction:

``ecsv``    harwood's own catalogues. A YAML header declares every column, so
            a column edit goes through astropy and the header stays true; an
            edit made to the text alone would leave it declaring a column
            that is not there, and the file would no longer read.
``iraf``    #K/#N/#U/#F headers, as apphot and daofind write them. The #N,
            #U and #F lines are edited with the columns they describe. phot
            writes each star over several lines ending in a backslash; there a
            "column" is a column of one of those lines, not of the star, and
            column edits leave such a file alone.
``plain``   whitespace-separated columns with # comments: lists, coordinate
            files, extinction tables. A first line of names over rows of
            numbers is taken as a header.
``csv``     the same with commas.

Everything here works on text and returns text: the editor puts the result
in the buffer, where it can be looked at, undone, or saved.
"""

from __future__ import annotations

import csv
import io
import re
import shlex
from collections import Counter
from dataclasses import dataclass, field
from typing import Literal

from . import advice
from .advice import Advice

#: IRAF's undefined value, which stands in a column of numbers.
INDEF = {"INDEF", "indef", "INDEFR", "INDEFI", "nan", "NaN"}

Kind = Literal["replace", "delete_rows", "delete_columns", "keep_columns",
               "insert_column", "set_column", "arithmetic", "sort"]

#: What each operation is called, for the menu and the record of what was done.
TITLES: dict[str, str] = {
    "replace": "Find and replace",
    "delete_rows": "Delete rows containing",
    "delete_columns": "Delete columns",
    "keep_columns": "Keep only columns",
    "insert_column": "Insert a column",
    "set_column": "Set a column to a value",
    "arithmetic": "Arithmetic on a column",
    "sort": "Sort rows by a column",
}


@dataclass
class Layout:
    """The shape of one file."""

    name: str
    format: str = "plain"               #: ecsv, iraf, plain or csv
    names: list[str] = field(default_factory=list)
    counts: Counter = field(default_factory=Counter)   #: columns -> rows with them
    numeric: list[bool] = field(default_factory=list)  #: per column, of the usual rows
    odd: list[int] = field(default_factory=list)       #: lines with an unusual width
    continued: bool = False             #: records continued over lines with \
    unreadable: str = ""                #: why it could not be read, when it could not

    @property
    def rows(self) -> int:
        return sum(self.counts.values())

    @property
    def width(self) -> int:
        """The number of columns most rows have."""
        return self.counts.most_common(1)[0][0] if self.counts else 0

    def summary(self) -> str:
        if self.unreadable:
            return self.unreadable
        text = f"{self.format}, {self.width} column{'s' if self.width != 1 else ''}, " \
               f"{self.rows} row{'s' if self.rows != 1 else ''}"
        if self.names:
            text += ": " + " ".join(self.names)
        return text


@dataclass(frozen=True)
class Operation:
    """One edit, to be made to every file of a batch. Columns count from 1."""

    kind: Kind
    columns: tuple[int, ...] = ()
    value: str = ""                     #: insert, set; the number for arithmetic
    operator: str = "+"                 #: arithmetic: + - * /
    find: str = ""
    regex: bool = False
    reverse: bool = False               #: sort descending
    name: str = ""                      #: insert: the new column's name

    def touches_columns(self) -> bool:
        return self.kind not in ("replace", "delete_rows")

    def needs_numbers(self) -> bool:
        return self.kind == "arithmetic"

    def describe(self) -> str:
        cols = ", ".join(str(c) for c in self.columns)
        return {
            "replace": f"replace '{self.find}' with '{self.value}'"
                       + (" (a regular expression)" if self.regex else ""),
            "delete_rows": f"delete the rows containing '{self.find}'",
            "delete_columns": f"delete column{'s' if len(self.columns) > 1 else ''} {cols}",
            "keep_columns": f"keep only column{'s' if len(self.columns) > 1 else ''} {cols}",
            "insert_column": f"insert '{self.value}' as column {cols}"
                             + (f", named {self.name}" if self.name else ""),
            "set_column": f"set column {cols} to '{self.value}'",
            "arithmetic": f"column {cols} {self.operator}= {self.value}",
            "sort": f"sort the rows by column {cols}"
                    + (", largest first" if self.reverse else ""),
        }[self.kind]


@dataclass
class Change:
    """What an edit did to one file."""

    name: str
    text: str
    lines: int = 0                      #: lines changed, added or removed
    skipped: str = ""                   #: why the file, or some of it, was left alone


# ------------------------------------------------------------------- reading


def layout(text: str, name: str = "") -> Layout:
    """Read the shape of a file's text."""
    if "\x00" in text[:4096]:
        return Layout(name, unreadable="not a text file")
    shape = Layout(name, format=_format(text))
    widths: Counter = Counter()
    seen_rows: list[tuple[int, list[str]]] = []
    for number, cells in _rows(text, shape):
        widths[len(cells)] += 1
        seen_rows.append((number, cells))
    shape.counts = widths
    if not widths:
        shape.unreadable = "no rows of data"
        return shape
    usual = shape.width
    shape.odd = [number for number, cells in seen_rows if len(cells) != usual]
    shape.numeric = [all(_number(cells[i]) for _, cells in seen_rows if len(cells) == usual)
                     for i in range(usual)]
    return shape


def _format(text: str) -> str:
    lines = text.splitlines()
    if lines and lines[0].startswith("# %ECSV"):
        return "ecsv"
    if any(line.startswith("#N") for line in lines):
        return "iraf"
    data = [line for line in lines if line.strip() and not line.lstrip().startswith("#")]
    if data and sum("," in line for line in data) > len(data) / 2 and \
            sum(len(line.split()) == 1 for line in data) > len(data) / 2:
        return "csv"
    return "plain"


def _rows(text: str, shape: Layout):
    """(line number, cells) for each row of data, filling in shape.names and
    shape.continued as it goes."""
    lines = text.splitlines()
    if shape.format == "ecsv":
        delimiter = "," if re.search(r"^# delimiter: ','", text, re.M) else " "
        header_done = False
        for number, line in enumerate(lines, start=1):
            if line.startswith("#") or not line.strip():
                continue
            cells = _split(line, delimiter)
            if not header_done:
                shape.names, header_done = cells, True
                continue
            yield number, cells
        return

    if shape.format == "iraf":
        for line in lines:
            if line.startswith("#N"):
                shape.names += [c for c in line[2:].split() if c != "\\"]
    first = True
    for number, line in enumerate(lines, start=1):
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if stripped.endswith("\\"):
            shape.continued = True
        cells = _split(line, "," if shape.format == "csv" else " ")
        if first and shape.format in ("plain", "csv") and _is_header(cells, lines, number):
            shape.names = cells
            first = False
            continue
        first = False
        yield number, cells


def _is_header(cells: list[str], lines: list[str], number: int) -> bool:
    """A first row of names over rows that hold numbers where the names are."""
    if any(_number(cell) for cell in cells):
        return False
    for line in lines[number:]:
        if line.strip() and not line.lstrip().startswith("#"):
            return any(_number(cell) for cell in line.split(","
                                                            if "," in line else None))
    return False


def _split(line: str, delimiter: str) -> list[str]:
    if delimiter == ",":
        return next(csv.reader([line]), [])
    try:
        return shlex.split(line, comments=False, posix=True)
    except ValueError:
        return line.split()


def _number(text: str) -> bool:
    if text in INDEF:
        return True
    try:
        float(text)
    except ValueError:
        return False
    return True


# ------------------------------------------------------------------ checking


def check(layouts: list[Layout], operation: Operation) -> list[Advice]:
    """Everything about these files that could make `operation` do something
    other than what it says, in words. Empty means they are as alike as the
    edit needs them to be."""
    found: list[Advice] = []
    readable = [shape for shape in layouts if not shape.unreadable]
    for shape in layouts:
        if shape.unreadable:
            found.append(advice.problem(
                f"{shape.name}: {shape.unreadable}", "it will be left as it is; take it "
                "out of the batch, or open the right file"))
    if not operation.touches_columns() or not readable:
        return found

    formats = Counter(shape.format for shape in readable)
    if len(formats) > 1:
        found.append(advice.warn(
            "the files are not all written the same way: " + _grouped(
                readable, lambda s: s.format),
            "a column number counts the same way in each, but check that it is the "
            "same quantity: compare one row of each file"))

    widths = Counter(shape.width for shape in readable)
    if len(widths) > 1:
        found.append(advice.warn(
            "the files do not have the same number of columns: " + _grouped(
                readable, lambda s: f"{s.width} columns"),
            "a column number may point at different quantities in different files; "
            "open them side by side and compare a row of each"))

    named = [shape for shape in readable if shape.names]
    if len(named) > 1:
        for column in sorted(set(operation.columns)):
            labels = {shape.name: (shape.names[column - 1]
                                   if column <= len(shape.names) else "(none)")
                      for shape in named}
            if len(set(labels.values())) > 1:
                found.append(advice.warn(
                    f"column {column} is not the same quantity everywhere: "
                    + ", ".join(f"{label} in {name}" for name, label in labels.items()),
                    "the files' columns are in a different order; edit them in groups "
                    "that agree, or by name, one file at a time"))
        if not operation.columns and len({tuple(s.names) for s in named}) > 1:
            found.append(advice.warn("the files' column names differ",
                                     "compare their headers before editing them together"))

    for shape in readable:
        if shape.odd:
            lines = ", ".join(str(n) for n in shape.odd[:8]) + \
                ("..." if len(shape.odd) > 8 else "")
            found.append(advice.warn(
                f"{shape.name}: not every row has {shape.width} columns "
                f"(line{'s' if len(shape.odd) > 1 else ''} {lines})",
                "those rows are edited by the same column number, which may not be "
                "the same quantity there; look at them first"))
        beyond = [c for c in operation.columns
                  if c > shape.width + (1 if operation.kind == "insert_column" else 0)]
        if beyond:
            found.append(advice.warn(
                f"{shape.name} has only {shape.width} columns, so column "
                f"{', '.join(map(str, beyond))} is not there",
                "going ahead leaves its rows without that column as they are"))
        if shape.continued:
            found.append(advice.warn(
                f"{shape.name} writes each record over several lines ending in \\, "
                "as phot does; a column of one line is not a column of the record",
                "column edits leave this file as it is even if you go ahead; dump the "
                "fields you need to a one-row-per-star table first"))
        if operation.needs_numbers():
            for column in operation.columns:
                if column <= len(shape.numeric) and not shape.numeric[column - 1]:
                    found.append(advice.warn(
                        f"{shape.name}: column {column} does not hold only numbers",
                        "the rows where it is text are left as they are; arithmetic "
                        "is only done where there is a number"))
    return found


def _grouped(layouts: list[Layout], key) -> str:
    groups: dict[str, list[str]] = {}
    for shape in layouts:
        groups.setdefault(key(shape), []).append(shape.name)
    return "; ".join(f"{label} in {', '.join(names)}" for label, names in groups.items())


# ------------------------------------------------------------------- editing


def apply(text: str, operation: Operation, name: str = "") -> Change:
    """Make the edit to one file's text."""
    shape = layout(text, name)
    if operation.touches_columns():
        if shape.unreadable:
            return Change(name, text, skipped=shape.unreadable)
        if shape.continued:
            return Change(name, text, skipped="records continue over lines")
        if shape.format == "ecsv":
            return _apply_table(text, operation, shape, name)
    return _apply_text(text, operation, shape, name)


def _apply_text(text: str, op: Operation, shape: Layout, name: str) -> Change:
    newline = "\r\n" if "\r\n" in text else "\n"
    lines = text.splitlines()
    data, header = _roles(lines, shape)
    out: list[str] = []
    skipped = 0

    if op.kind == "sort":
        return _sort(text, lines, data, op, shape, name, newline)

    for index, line in enumerate(lines):
        role = "data" if index in data else ("header" if index in header else "other")
        if op.kind == "replace":
            if shape.format == "ecsv" and role != "data":
                out.append(line)
                continue
            out.append(re.sub(op.find, op.value, line) if op.regex
                       else line.replace(op.find, op.value))
            continue
        if op.kind == "delete_rows":
            if role == "data" and op.find in line:
                continue
            out.append(line)
            continue
        if role == "other":
            out.append(line)
            continue
        edited, ok = _edit_line(line, op, role, shape.format)
        skipped += not ok
        out.append(edited)

    changed = _changed(lines, out)
    result = newline.join(out) + (newline if text.endswith(("\n", "\r\n")) else "")
    note = f"{skipped} row{'s' if skipped != 1 else ''} left as they were" if skipped else ""
    return Change(name, result, changed, note)


def _roles(lines: list[str], shape: Layout) -> tuple[set[int], set[int]]:
    """Which lines are rows of data, and which are headers edited with them."""
    data, header = set(), set()
    names_row_done = False
    for index, line in enumerate(lines):
        stripped = line.strip()
        if shape.format == "iraf" and line[:2] in ("#N", "#U", "#F"):
            header.add(index)
            continue
        if not stripped or stripped.startswith("#"):
            continue
        if shape.names and not names_row_done and shape.format in ("plain", "csv", "ecsv"):
            names_row_done = True
            header.add(index)
            continue
        data.add(index)
    return data, header


def _edit_line(line: str, op: Operation, role: str, form: str) -> tuple[str, bool]:
    """One row (or a header row) with the column edit made. False when the
    row lacked the column and was left alone."""
    if form == "csv":
        return _edit_csv(line, op, role)
    spans = _spans(line, role, form)

    if op.kind in ("delete_columns", "keep_columns"):
        complete = max(op.columns) <= len(spans)
        wanted = set(op.columns) if op.kind == "delete_columns" else \
            set(range(1, len(spans) + 1)) - set(op.columns)
        wanted = {c for c in wanted if c <= len(spans)}
        if op.kind == "delete_columns" and not wanted:
            return line, False
        for column in sorted(wanted, reverse=True):
            # Measured afresh each time: what was removed moved the rest.
            spans = _spans(line, role, form)
            start, end = spans[column - 1]
            if column > 1:
                start = spans[column - 2][1]      # with the space before it
            elif len(spans) > 1:
                end = spans[1][0]                 # the first: with the space after
            line = line[:start] + line[end:]
        return line, complete

    column = op.columns[0]
    if op.kind == "insert_column":
        if column > len(spans) + 1:
            return line, False
        cell = op.value if role == "data" else _header_cell(line, op, column)
        if column == len(spans) + 1:
            at = spans[-1][1] if spans else len(line)
            return line[:at] + " " + cell + line[at:], True
        at = spans[column - 1][0]
        return line[:at] + cell + " " + line[at:], True

    if column > len(spans):
        return line, False
    start, end = spans[column - 1]
    old = line[start:end]
    if role == "header":
        return line, True
    if op.kind == "set_column":
        new = op.value
    else:                                   # arithmetic
        new = _arithmetic(old, op.operator, op.value)
        if new is None:
            return line, False
    return line[:start] + _fit(new, old, line, start) + line[end:], True


def _spans(line: str, role: str, form: str) -> list[tuple[int, int]]:
    """Where each cell of a row is. An IRAF header's cells start after its
    #N (or #U, #F), and a trailing \\ is not a cell."""
    spans = [m.span() for m in re.finditer(r"\S+", line)]
    if role == "header" and form == "iraf":
        spans = spans[1:]
        if spans and line[spans[-1][0]:spans[-1][1]] == "\\":
            spans = spans[:-1]
    return spans


def _header_cell(line: str, op: Operation, column: int) -> str:
    if line.startswith("#U"):
        return "##"
    if line.startswith("#F"):
        return "%-12s"
    return op.name or f"col{column}"


def _fit(new: str, old: str, line: str, start: int) -> str:
    """Keep the columns lined up: a value as wide as the one it replaces,
    padded on the side the old one was."""
    if len(new) >= len(old):
        return new
    right_aligned = start > 0 and line[start - 1] == " " and \
        (start < 2 or line[start - 2] == " ")
    return new.rjust(len(old)) if right_aligned else new.ljust(len(old))


def _arithmetic(old: str, operator: str, value: str) -> str | None:
    if old in INDEF:
        return old
    try:
        a, b = float(old), float(value)
    except ValueError:
        return None
    if operator == "/" and b == 0:
        return None
    result = {"+": a + b, "-": a - b, "*": a * b, "/": a / b if b else 0.0}[operator]
    integral = re.fullmatch(r"[+-]?\d+", old) and re.fullmatch(r"[+-]?\d+", value) \
        and operator != "/"
    if integral:
        return str(int(result))
    if "e" in old.lower():
        return f"{result:.{_decimals(old.lower().split('e')[0])}e}"
    return f"{result:.{max(_decimals(old), _decimals(value))}f}"


def _decimals(text: str) -> int:
    return len(text.split(".", 1)[1]) if "." in text else 0


def _edit_csv(line: str, op: Operation, role: str) -> tuple[str, bool]:
    cells = next(csv.reader([line]), [])
    column = op.columns[0] if op.columns else 0
    if op.kind == "delete_columns":
        if max(op.columns) > len(cells):
            return line, False
        cells = [c for i, c in enumerate(cells, start=1) if i not in op.columns]
    elif op.kind == "keep_columns":
        cells = [c for i, c in enumerate(cells, start=1) if i in op.columns]
    elif op.kind == "insert_column":
        if column > len(cells) + 1:
            return line, False
        cells.insert(column - 1, op.value if role == "data" else (op.name or f"col{column}"))
    elif role == "header":
        return line, True
    elif column > len(cells):
        return line, False
    elif op.kind == "set_column":
        cells[column - 1] = op.value
    else:
        new = _arithmetic(cells[column - 1].strip(), op.operator, op.value)
        if new is None:
            return line, False
        cells[column - 1] = new
    buffer = io.StringIO()
    csv.writer(buffer, lineterminator="").writerow(cells)
    return buffer.getvalue(), True


def _sort(text, lines, data, op, shape, name, newline) -> Change:
    column = op.columns[0]
    slots = sorted(data)
    rows = [lines[i] for i in slots]
    delimiter = "," if shape.format == "csv" else " "
    numeric = column <= len(shape.numeric) and shape.numeric[column - 1]
    short = [row for row in rows if len(_split(row, delimiter)) < column]

    def key(row: str):
        cells = _split(row, delimiter)
        if len(cells) < column:
            return (1, 0.0, "")              # rows without the column go last
        cell = cells[column - 1]
        if numeric:
            number = float("inf") if cell in INDEF else float(cell)
            return (0, number, "")
        return (0, 0.0, cell)

    ordered = sorted(rows, key=key, reverse=op.reverse)
    if op.reverse and short:                 # still last, whichever way
        ordered = [r for r in ordered if r not in short] + short
    out = list(lines)
    for slot, row in zip(slots, ordered, strict=True):
        out[slot] = row
    changed = _changed(lines, out)
    result = newline.join(out) + (newline if text.endswith(("\n", "\r\n")) else "")
    note = (f"{len(short)} row{'s' if len(short) != 1 else ''} without column "
            f"{column} put last") if short else ""
    return Change(name, result, changed, note)


def _apply_table(text: str, op: Operation, shape: Layout, name: str) -> Change:
    """An ECSV file, edited through its table so that its header stays true."""
    import numpy as np
    from astropy.table import Column, Table

    try:
        table = Table.read(text, format="ascii.ecsv")
    except Exception as error:
        return Change(name, text, skipped=f"astropy could not read it ({error})")
    names = list(table.colnames)
    missing = [c for c in op.columns if c > len(names) + (op.kind == "insert_column")]
    if missing:
        return Change(name, text, skipped=f"no column {', '.join(map(str, missing))}")
    picked = [names[c - 1] for c in op.columns if c <= len(names)]
    if op.kind == "delete_columns":
        table.remove_columns(picked)
    elif op.kind == "keep_columns":
        table.keep_columns(picked)
    elif op.kind == "insert_column":
        new = op.name or f"col{op.columns[0]}"
        value = float(op.value) if _number(op.value) and op.value not in INDEF else op.value
        table.add_column(Column([value] * len(table), name=new), index=op.columns[0] - 1)
    elif op.kind == "set_column":
        value = float(op.value) if _number(op.value) and op.value not in INDEF else op.value
        table.replace_column(picked[0], Column([value] * len(table), name=picked[0]))
    elif op.kind == "arithmetic":
        column = table[picked[0]]
        if column.dtype.kind not in "iuf":
            return Change(name, text, skipped=f"column {picked[0]} is not numbers")
        number = float(op.value)
        with np.errstate(divide="ignore", invalid="ignore"):
            table[picked[0]] = {"+": column + number, "-": column - number,
                                "*": column * number, "/": column / number}[op.operator]
    elif op.kind == "sort":
        table.sort(picked[0], reverse=op.reverse)
    buffer = io.StringIO()
    table.write(buffer, format="ascii.ecsv")
    result = buffer.getvalue()
    return Change(name, result, _changed(text.splitlines(), result.splitlines()))


def _changed(before: list[str], after: list[str]) -> int:
    """Lines changed, added or removed."""
    import difflib

    return sum(max(i2 - i1, j2 - j1) for tag, i1, i2, j1, j2
               in difflib.SequenceMatcher(None, before, after, autojunk=False).get_opcodes()
               if tag != "equal")
