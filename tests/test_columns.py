"""Several text files edited as one, and the check that comes first.

The edits are held to keeping a file what it was -- an ECSV header that
still declares its columns, IRAF's #N/#U/#F lines in step with the data,
alignment and comments where they were. The check is held to the promise
the editor makes: every difference that could make one edit mean different
things in different files is said, with what to do, before anything is
changed; and files that agree pass without a word.
"""

import numpy as np
import pytest
from astropy.table import Table

from harwood.core import catalog
from harwood.core import columns as C

IRAF = """\
#K IRAF       = NOAO/IRAFV2.18    version    %-23s
#N XCENTER   YCENTER   MAG      ID         \\
#U pixels    pixels    #        #          \\
#F %-13.3f   %-10.3f   %-9.3f   %-6d       \\
#
   40.313    45.687    -0.687   1
  110.557    52.293     0.557   2
  160.228   140.864    -0.772   3
"""

PLAIN = "# a coordinate file\n40.313 45.687 -0.687\n110.557 52.293 0.557\n"
NAMED = "x y merr mag\n40.313 45.687 0.01 -0.687\n110.557 52.293 0.02 0.557\n"
PHOT = "  40.313  45.687 \\\n  13.703  0.004\n  110.557  52.293 \\\n  13.290  0.004\n"


def _ecsv(tmp_path):
    path = tmp_path / "h.coo"
    catalog.write(Table({"x": [1.5, 2.5], "y": [10.0, 20.0], "mag": [14.1, 15.2]}), path)
    return path.read_text()


# ----------------------------------------------------------------- reading


def test_each_format_is_read_for_its_shape(tmp_path):
    iraf = C.layout(IRAF, "a.coo")
    assert (iraf.format, iraf.width, iraf.rows) == ("iraf", 4, 3)
    assert iraf.names == ["XCENTER", "YCENTER", "MAG", "ID"]
    assert iraf.numeric == [True, True, True, True]
    named = C.layout(NAMED, "b.coo")
    assert (named.format, named.width, named.rows, named.names) == \
        ("plain", 4, 2, ["x", "y", "merr", "mag"]), "a first row of names is a header"
    ecsv = C.layout(_ecsv(tmp_path), "h.coo")
    assert (ecsv.format, ecsv.width, ecsv.names) == ("ecsv", 3, ["x", "y", "mag"])
    assert C.layout("a,b\n1,2\n3,4\n").format == "csv"
    assert C.layout(PHOT).continued


def test_short_rows_and_what_is_not_a_table_are_seen():
    ragged = C.layout(PLAIN + "160.228 140.864\n")
    assert ragged.width == 3 and ragged.odd == [4]
    assert C.layout("# only comments\n\n").unreadable == "no rows of data"
    assert C.layout("SIMPLE\x00\x00binary").unreadable == "not a text file"
    assert C.layout("1 INDEF\n2 3.5\n").numeric == [True, True], "INDEF is a number"


# ---------------------------------------------------------------- checking


def _check(texts, operation):
    return C.check([C.layout(text, name) for name, text in texts.items()], operation)


def test_files_that_agree_pass_without_a_word():
    other = PLAIN.replace("40.313", "41.000")
    assert _check({"a.coo": PLAIN, "b.coo": other},
                  C.Operation("arithmetic", (1,), "3")) == []


def test_a_different_number_of_columns_is_said_with_what_to_do():
    found = _check({"a.coo": PLAIN, "b.coo": NAMED}, C.Operation("delete_columns", (3,)))
    widths = [f for f in found if "number of columns" in f.message]
    assert widths and "3 columns in a.coo" in widths[0].message
    assert all(flag.fix for flag in found), "every notice says what to do"


def test_columns_in_another_order_are_named():
    swapped = NAMED.replace("x y merr mag", "x y mag merr")
    found = _check({"a.coo": NAMED, "b.coo": swapped}, C.Operation("arithmetic", (4,), "1"))
    [flag] = found
    assert "column 4 is not the same quantity everywhere" in flag.message
    assert "mag in a.coo" in flag.message and "merr in b.coo" in flag.message


def test_what_a_column_edit_cannot_do_to_a_file_is_said():
    found = _check({"a.coo": PLAIN, "p.mag": PHOT, "c.coo": "x\n1 2\n"},
                   C.Operation("set_column", (3,), "0"))
    text = " ".join(f.message for f in found)
    assert "several lines ending in \\" in text
    assert "c.coo has only 2 columns, so column 3 is not there" in text


def test_arithmetic_on_text_is_flagged():
    found = _check({"a.coo": "ngc1 1.0\nngc2 2.0\n"}, C.Operation("arithmetic", (1,), "1"))
    assert any("does not hold only numbers" in f.message for f in found)


def test_a_text_edit_does_not_care_about_columns():
    assert _check({"a.coo": PLAIN, "b.coo": NAMED},
                  C.Operation("replace", find="40.313", value="40.3")) == []


# ----------------------------------------------------------------- editing


def test_arithmetic_keeps_the_decimals_and_the_alignment():
    change = C.apply(IRAF, C.Operation("arithmetic", (2,), "1.5", "-"))
    rows = change.text.splitlines()[5:]
    assert rows[0] == "   40.313    44.187    -0.687   1"
    assert rows[1] == "  110.557    50.793     0.557   2"
    assert change.lines == 3


def test_deleting_a_column_takes_its_header_cells_with_it():
    text = C.apply(IRAF, C.Operation("delete_columns", (2, 3))).text
    lines = text.splitlines()
    assert lines[1].split() == ["#N", "XCENTER", "ID", "\\"]
    assert lines[2].split() == ["#U", "pixels", "#", "\\"]
    assert lines[3].split() == ["#F", "%-13.3f", "%-6d", "\\"]
    assert lines[5].split() == ["40.313", "1"]
    assert C.layout(text).names == ["XCENTER", "ID"], "and the file still reads"


def test_deleting_the_first_columns_one_after_the_other():
    assert C.apply("a b c d\n", C.Operation("delete_columns", (1, 2))).text == "c d\n"
    assert C.apply("a b c d\n", C.Operation("keep_columns", (2, 4))).text == "b d\n"


def test_an_inserted_column_is_named_in_the_header():
    text = C.apply(NAMED, C.Operation("insert_column", (3,), "0.5", name="err")).text
    assert text.splitlines()[0] == "x y err merr mag"
    assert text.splitlines()[1] == "40.313 45.687 0.5 0.01 -0.687"
    iraf = C.apply(IRAF, C.Operation("insert_column", (5,), "0", name="FLAG")).text
    assert C.layout(iraf).names[-1] == "FLAG"


def test_an_ecsv_header_stays_true_after_a_column_edit(tmp_path):
    for operation in (C.Operation("delete_columns", (2,)),
                      C.Operation("insert_column", (1,), "7", name="id"),
                      C.Operation("arithmetic", (3,), "0.1", "+"),
                      C.Operation("sort", (3,), reverse=True)):
        change = C.apply(_ecsv(tmp_path), operation, "h.coo")
        path = tmp_path / "edited.coo"
        path.write_text(change.text)
        table = catalog.read(path)                   # it still reads, as a table
        if operation.kind == "delete_columns":
            assert table.colnames == ["x", "mag"]
        if operation.kind == "insert_column":
            assert table.colnames[0] == "id" and list(table["id"]) == [7.0, 7.0]
        if operation.kind == "arithmetic":
            assert np.allclose(table["mag"], [14.2, 15.3])
        if operation.kind == "sort":
            assert list(table["mag"]) == [15.2, 14.1]


def test_a_text_edit_to_an_ecsv_file_leaves_its_header_alone(tmp_path):
    text = _ecsv(tmp_path)
    change = C.apply(text, C.Operation("replace", find="x", value="X"))
    assert "name: x" in change.text and change.text.count("X") == 0, \
        "renaming in the header alone would make the file unreadable"


def test_sorting_keeps_comments_in_place_and_short_rows_last():
    text = "# head\n3 c\n1 a\n# middle\n2\n0 z\n"
    change = C.apply(text, C.Operation("sort", (1,)))
    assert change.text == "# head\n0 z\n1 a\n# middle\n2\n3 c\n"
    assert C.apply(text, C.Operation("sort", (2,))).skipped == \
        "1 row without column 2 put last"


def test_rows_a_column_edit_cannot_reach_are_left_and_counted():
    change = C.apply(PLAIN + "1 2\n", C.Operation("set_column", (3,), "0"))
    assert change.text.splitlines()[-1] == "1 2"
    assert change.skipped == "1 row left as they were"


def test_a_continued_file_is_left_alone_by_column_edits():
    change = C.apply(PHOT, C.Operation("delete_columns", (1,)))
    assert change.text == PHOT and change.skipped


def test_rows_containing_text_are_deleted_and_counted():
    change = C.apply(PLAIN, C.Operation("delete_rows", find="110.557"))
    assert "110.557" not in change.text and "# a coordinate file" in change.text
    assert change.lines == 1


def test_windows_line_endings_survive_an_edit():
    change = C.apply("1 2\r\n3 4\r\n", C.Operation("arithmetic", (1,), "1"))
    assert change.text == "2 2\r\n4 4\r\n"


def test_integers_stay_integers_and_indef_stays_indef():
    change = C.apply("1 INDEF\n2 7\n", C.Operation("arithmetic", (2,), "3"))
    assert change.text == "1 INDEF\n2 10\n"


@pytest.mark.parametrize("text, wanted, problem", [
    ("3", (3,), ""), ("2,4", (2, 4), ""), ("2-4", (2, 3, 4), ""), ("4 2 2", (2, 4), ""),
    ("", (), "say which column"), ("0", (), "count from 1"), ("a", (), "not a column"),
    ("5-2", (), "low to high")])
def test_columns_are_written_as_people_write_them(text, wanted, problem):
    pytest.importorskip("PySide6")
    from harwood.viewer.files import parse_columns

    found, said = parse_columns(text)
    assert found == wanted and problem in said


# -------------------------------------------------------------- the command


def test_edit_check_compares_files_from_the_shell(tmp_path, monkeypatch):
    from click.testing import CliRunner

    from harwood.cli.main import app

    monkeypatch.chdir(tmp_path)
    (tmp_path / "a.coo").write_text(PLAIN)
    (tmp_path / "b.coo").write_text(PLAIN)
    (tmp_path / "c.coo").write_text(NAMED)
    alike = CliRunner().invoke(app, ["edit", "--check", "a.coo", "b.coo"])
    assert "alike" in alike.output
    differ = CliRunner().invoke(app, ["edit", "--check", "a.coo", "c.coo"])
    assert "number of columns" in differ.output and "warning" in differ.output


# --------------------------------------------------------------- the window


@pytest.fixture
def window(tmp_path, monkeypatch):
    pytest.importorskip("PySide6")
    from harwood.viewer import _qt, theme
    from harwood.viewer.window import HarwoodWindow

    monkeypatch.chdir(tmp_path)
    (tmp_path / "a.coo").write_text(IRAF)
    (tmp_path / "b.coo").write_text(IRAF.replace("40.313", "41.000"))
    (tmp_path / "c.coo").write_text(NAMED)
    _qt.application()
    w = HarwoodWindow()
    yield w
    for tab in w.files.files():
        tab.document().setModified(False)       # nothing left to ask about
    _qt.set_window(None)
    w.close()
    theme.forget(w._restyle)


def _dialog(window, kind, columns="", value="", find=""):
    from harwood.viewer.files import TogetherDialog

    dialog = TogetherDialog(window.files)
    kinds = [dialog.kind.itemData(i) for i in range(dialog.kind.count())]
    dialog.kind.setCurrentIndex(kinds.index(kind))
    dialog.columns_edit.setText(columns)
    dialog.value_edit.setText(value)
    dialog.find_edit.setText(find)
    return dialog


def test_edit_at_the_prompt_opens_the_files_in_tabs(window, tmp_path):
    fits_file = tmp_path / "img.fits"
    fits_file.write_bytes(b"SIMPLE  =                    T" + b" " * 2850)
    window.run_line("edit a.coo c.coo img.fits")
    assert window.files_dock.isVisibleTo(window)
    assert [tab.path.name for tab in window.files.files()] == ["a.coo", "c.coo"]
    assert "img.fits is a FITS image; `display img.fits` shows it" in \
        window.session.toPlainText()
    window.run_line("edit a.coo")
    assert len(window.files.files()) == 2, "an open file is gone back to, not reopened"


def test_the_bar_says_which_column_the_cursor_is_in(window):
    from PySide6 import QtGui

    window.edit_files(["a.coo"])
    tab = window.files.files()[0]
    cursor = tab.textCursor()
    cursor.movePosition(QtGui.QTextCursor.MoveOperation.Down, n=6)
    cursor.movePosition(QtGui.QTextCursor.MoveOperation.Right, n=22)
    tab.setTextCursor(cursor)
    assert "column 3 of 4: MAG" in window.files.where.text()


def test_files_that_agree_are_edited_without_a_question(window):
    window.edit_files(["a.coo", "b.coo"])
    dialog = _dialog(window, "arithmetic", "1", "3")
    asked = []
    dialog.confirm = lambda flags, operation: asked.append(flags) or True
    assert dialog.flags() == []
    assert "ready" in dialog.notes.toPlainText()
    assert dialog.apply() and asked == []
    a, b = window.files.files()
    assert "43.313" in a.toPlainText() and "44.000" in b.toPlainText()
    assert a.modified and window.files.tabs.tabText(0) == "● a.coo"
    assert a.path.read_text() == IRAF, "not saved yet"
    a.undo()
    assert a.toPlainText() == IRAF, "one Ctrl+Z takes the whole edit back"


def test_files_that_differ_are_asked_about_and_left_alone_on_no(window):
    window.edit_files(["a.coo", "c.coo"])
    dialog = _dialog(window, "delete_columns", "3")
    asked = []
    dialog.confirm = lambda flags, operation: asked.append(flags) or False
    assert "column 3 is not the same quantity everywhere" in dialog.notes.toPlainText()
    assert not dialog.apply()
    assert asked and all(flag.fix for flag in asked[0])
    assert window.files.files()[0].toPlainText() == IRAF, "nothing was touched"


def test_going_ahead_anyway_is_said_in_the_session(window):
    window.edit_files(["a.coo", "c.coo"])
    dialog = _dialog(window, "delete_columns", "3")
    dialog.confirm = lambda flags, operation: True
    assert dialog.apply()
    session = window.session.toPlainText()
    assert "edited together anyway: delete column 3" in session
    assert "Save all" in session
    window.files.save_all()
    assert "MAG" not in (window.files.files()[0].path.read_text().splitlines()[1])


def test_the_confirmation_defaults_to_going_back(window):
    from harwood.core.advice import warn
    from harwood.viewer.files import force_box

    box, force = force_box(window, [warn("a differs", "look")],
                           C.Operation("delete_columns", (1,)))
    assert box.defaultButton() is not force and box.escapeButton() is not force
    assert "a differs" in box.informativeText() and "look" in box.informativeText()


def test_a_bad_column_stops_the_edit_with_a_word(window):
    window.edit_files(["a.coo"])
    dialog = _dialog(window, "arithmetic", "x", "3")
    assert not dialog.apply_button.isEnabled()
    assert "not a column number" in dialog.notes.toPlainText()
