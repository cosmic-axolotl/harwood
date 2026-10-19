"""The command line: several commands on a line, Tab, and the history.

What the window's prompt and the terminal session have in common lives in
cli/line.py and cli/history.py, so it is tested once, without Qt; then each of
the two is held to using it -- the same chain runs the same way at hw> in a
terminal and in the window, and a line typed in one is found by Up in the
other.
"""

import numpy as np
import pytest

from harwood.cli import history
from harwood.cli import line as command_line

# ----------------------------------------------------------- one line, several


def _split(text):
    return [(step.connector, step.command) for step in command_line.split(text)]


def test_a_line_splits_at_its_connectors():
    assert _split("imstatistics a.fits") == [("", "imstatistics a.fits")]
    assert _split("a && b; c || d") == [("", "a"), ("&&", "b"), (";", "c"), ("||", "d")]
    assert _split("a;b&&c") == [("", "a"), (";", "b"), ("&&", "c")]


def test_a_quoted_connector_is_a_character():
    """hselect's expressions are written with && and ||, as IRAF's are."""
    assert _split("hselect *.fits $I \"imagetyp=='zero' && exptime>0\"") == [
        ("", "hselect *.fits $I \"imagetyp=='zero' && exptime>0\"")]
    assert _split("hedit a.fits title 'a; b' && list") == [
        ("", "hedit a.fits title 'a; b'"), ("&&", "list")]
    assert _split(r"echo a\;b") == [("", r"echo a\;b")]


def test_a_stray_semicolon_is_harmless():
    assert _split("a;") == [("", "a")]
    assert _split("a ;; b") == [("", "a"), (";", "b")]
    assert _split("") == []


@pytest.mark.parametrize("text, said", [
    ("a &&", "nothing after '&&'"),
    ("a ||", "nothing after '||'"),
    ("&& b", "nothing before '&&'"),
    ("a && && b", "nothing before '&&'"),
    ("imcombine @bias.lis Zero & list", "background"),
    ("imstatistics a.fits | grep mean", "pipe"),
])
def test_a_line_that_cannot_mean_anything_says_what_to_write(text, said):
    with pytest.raises(ValueError) as raised:
        command_line.split(text)
    assert said in str(raised.value)
    assert raised.value.fix, "and what to write instead"


def _chain(text, failing=()):
    ran, notes = [], []

    def run_one(command):
        ran.append(command)
        return command not in failing

    ok = command_line.run(text, run_one, notes.append)
    return ok, ran, notes


def test_and_runs_the_next_only_after_success():
    assert _chain("a && b") == (True, ["a", "b"], [])
    ok, ran, notes = _chain("a && b && c", failing={"a"})
    assert (ok, ran) == (False, ["a"])
    assert notes == ["skipped: b -- the command before it failed",
                     "skipped: c -- the command before it failed"]


def test_semicolon_runs_the_next_whatever_happened():
    ok, ran, _ = _chain("a && b; c", failing={"a"})
    assert ran == ["a", "c"] and ok, "c runs, and its status is the line's"


def test_or_runs_the_next_only_after_failure():
    assert _chain("a || b", failing={"a"})[1] == ["a", "b"]
    ok, ran, notes = _chain("a || b")
    assert ran == ["a"] and ok
    assert notes == ["skipped: b -- the command before it worked"]


# ---------------------------------------------------------------- Tab


def _values(text):
    return [item.value for item in command_line.complete(text).items]


def test_tab_completes_task_names_with_what_each_does():
    found = command_line.complete("imcomb")
    assert found.start == 0
    assert [item.value for item in found.items] == ["imcombine"]
    assert found.items[0].help.startswith("Combine")
    assert "imstatistics" in _values("ims")


def test_tab_completes_options_and_the_values_they_allow():
    assert _values("imcombine --rej") == ["--reject"]
    assert _values("imcombine --combine ") == ["average", "median", "lmedian", "sum"]
    assert _values("imcombine --reject cc") == ["ccdclip"]


def test_an_option_is_offered_once_in_the_spelling_being_typed():
    """--exposure-key and --exposure_key are one option (section 3.1)."""
    assert "--exposure_key" not in _values("imcombine --exp")
    assert "--exposure-key" in _values("imcombine --exp")
    assert _values("imcombine --exposure_") == ["--exposure_key"]


def test_tab_completes_the_names_help_and_lpar_take():
    assert _values("help daof") == ["daofind"]
    assert _values("lpar imstat") == ["imstatistics"]
    assert "immatch" in _values("list imm")


def test_tab_completes_files_lists_and_directories(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    for name in ("obj001.fits", "obj002.fits", "bias.lis", ".hidden"):
        (tmp_path / name).write_text("")
    (tmp_path / "raw night").mkdir()
    assert _values("imstatistics obj") == ["obj001.fits", "obj002.fits"]
    assert command_line.complete("imstatistics obj").common() == "obj00"
    assert _values("imcombine @bi") == ["@bias.lis"], "an IRAF list keeps its @"
    assert _values("imstatistics raw") == [r"raw\ night/"], "a space stays in the word"
    assert ".hidden" not in _values("imstatistics ")
    assert _values("ccdproc --zero bi") == ["bias.lis"], "an option's value too"
    assert _values("help bi") == [], "help takes a task, never a file"


def test_tab_completes_the_word_under_the_cursor_of_the_last_command():
    text = "imstatistics a.fits && imcomb"
    found = command_line.complete(text)
    assert found.start == text.index("imcomb")
    assert [item.value for item in found.items] == ["imcombine"]
    assert _values("display a.fits; :rad") == [":radius"]


def test_tab_completes_imexamine_colon_commands_harwood_accepts():
    assert ":ncstat" in _values(":nc")
    assert ":title" not in _values(":t"), "styling IRAF's graphs is not offered"


@pytest.mark.parametrize("text", ["nosuchtask --x", "imstatistics 'unbalanced",
                                  "imcombine --nlow notanumber ", "\\", "'", "&& ;",
                                  "--", "harwood"])
def test_tab_never_raises(text):
    command_line.complete(text)


# ------------------------------------------------------------- history


def test_history_keeps_what_was_typed_once_in_a_row():
    for line in ("list", "list", "  help phot  ", "", "list"):
        history.add(line)
    assert history.load() == ["list", "help phot", "list"]


def test_history_is_trimmed_to_its_limit(monkeypatch):
    monkeypatch.setattr(history, "LIMIT", 8)
    for number in range(30):
        history.add(f"display {number}")
    kept = history.load()
    assert len(kept) <= 10 and kept[-1] == "display 29"


def test_a_history_that_cannot_be_written_does_not_stop_a_command(tmp_path, monkeypatch):
    blocker = tmp_path / "file"
    blocker.write_text("")
    monkeypatch.setenv("XDG_STATE_HOME", str(blocker))     # harwood/ cannot go in a file
    history.add("list")
    assert history.load() == []


def test_the_history_command_numbers_what_was_typed():
    from click.testing import CliRunner

    from harwood.cli.main import app

    for line in ("imcombine @bias.lis Zero.fits", "imstatistics Zero.fits", "list"):
        history.add(line)
    result = CliRunner().invoke(app, ["history"])
    assert "1  imcombine @bias.lis Zero.fits" in result.output
    found = CliRunner().invoke(app, ["history", "--find", "Zero"])
    assert "list" not in found.output and "2  imstatistics Zero.fits" in found.output
    assert "Nothing typed contains" in CliRunner().invoke(app, ["history", "-f", "x"]).output


# ------------------------------------------------------ the terminal session


def test_the_session_runs_a_chain_and_keeps_its_history(monkeypatch, capsys):
    from harwood.cli import session

    typed = iter(["imstatistics no_such_file.fits && help phot; lpar daofind",
                  "list &&", "exit"])
    monkeypatch.setattr(session.console, "input", lambda prompt="": next(typed))
    session.start()
    out = capsys.readouterr()
    text = out.out + out.err
    assert "skipped: help phot -- the command before it failed" in text
    assert "daofind" in text, "lpar ran after the ;"
    assert "nothing after '&&'" in text
    assert history.load()[0].startswith("imstatistics no_such_file.fits &&")


def test_exit_inside_a_chain_ends_the_session(monkeypatch):
    from harwood.cli import session

    typed = iter(["list; exit", "list"])
    seen = []

    def ask(prompt=""):
        line = next(typed)
        seen.append(line)
        return line

    monkeypatch.setattr(session.console, "input", ask)
    session.start()
    assert seen == ["list; exit"]


# ------------------------------------------------------------- the window


@pytest.fixture
def window(tmp_path, monkeypatch):
    pytest.importorskip("PySide6")
    from astropy.io import fits

    from harwood.viewer import _qt, theme
    from harwood.viewer.window import HarwoodWindow

    monkeypatch.chdir(tmp_path)
    fits.writeto(tmp_path / "a.fits", np.arange(16, dtype=np.float32).reshape(4, 4))
    _qt.application()
    w = HarwoodWindow()
    yield w
    _qt.set_window(None)
    w.close()
    theme.forget(w._restyle)


def _keys(widget, *keys):
    from PySide6.QtTest import QTest

    for key in keys:
        if isinstance(key, str):
            QTest.keyClicks(widget, key)
        else:
            QTest.keyClick(widget, *key) if isinstance(key, tuple) else \
                QTest.keyClick(widget, key)


def test_a_chain_typed_in_the_window_runs_like_the_terminals(window):
    ok, printed = window.run_line("imstatistics a.fits --fields mean && "
                                  "imstatistics missing.fits && help phot; "
                                  "imstatistics a.fits --fields mean")
    assert ok, "the last command worked"
    session = window.session.toPlainText()
    assert "▸ imstatistics missing.fits" in session
    assert "skipped: help phot -- the command before it failed" in session
    assert session.count("7.5") == 2, "both statistics ran"


def test_a_line_that_cannot_mean_anything_is_answered_in_the_window(window):
    ok, printed = window.run_line("imstatistics a.fits &")
    assert not ok and "background" in printed and "&&" in printed


def test_tab_in_the_window_writes_in_one_candidate_and_lists_several(window):
    from PySide6 import QtCore

    prompt = window.prompt
    _keys(prompt, "imcomb", QtCore.Qt.Key.Key_Tab)
    assert prompt.text() == "imcombine "
    _keys(prompt, "--combine ", QtCore.Qt.Key.Key_Tab)
    rows = [prompt._model.item(i).text() for i in range(prompt._model.rowCount())]
    assert rows == ["average", "median", "lmedian", "sum"]
    prompt._chosen(prompt._model.index(1, 0))
    assert prompt.text() == "imcombine --combine median "


def test_enter_on_a_listed_candidate_chooses_it_and_runs_nothing(window):
    from PySide6 import QtCore

    window.show()
    prompt = window.prompt
    ran = []
    prompt.returnPressed.connect(lambda: ran.append(prompt.text()))
    _keys(prompt, "imc", QtCore.Qt.Key.Key_Tab)
    popup = prompt.completer_.popup()
    assert popup.isVisible()
    _keys(popup, QtCore.Qt.Key.Key_Return)
    assert prompt.text() == "imcentroid " and ran == []


def test_tab_on_an_empty_prompt_still_moves_the_focus(window):
    """Otherwise a keyboard that walks into the prompt can never walk out."""
    from PySide6 import QtCore, QtGui

    event = QtGui.QKeyEvent(QtCore.QEvent.Type.KeyPress, QtCore.Qt.Key.Key_Tab,
                            QtCore.Qt.KeyboardModifier.NoModifier)
    window.prompt.setText("imcomb")
    window.prompt.setCursorPosition(6)
    assert window.prompt.event(event) and window.prompt.text() == "imcombine "
    window.prompt.clear()
    window.prompt.event(event)
    assert window.prompt.text() == "", "nothing was completed into it"


def test_up_finds_what_starts_with_what_is_typed(window):
    from PySide6 import QtCore

    prompt = window.prompt
    for line in ("imstatistics a.fits", "help phot", "imstatistics a.fits --fields mean"):
        prompt.setText(line)
        window._run_command()
    _keys(prompt, "ims", QtCore.Qt.Key.Key_Up)
    assert prompt.text() == "imstatistics a.fits --fields mean"
    _keys(prompt, QtCore.Qt.Key.Key_Up)
    assert prompt.text() == "imstatistics a.fits", "help phot is passed over"
    _keys(prompt, QtCore.Qt.Key.Key_Down, QtCore.Qt.Key.Key_Down)
    assert prompt.text() == "ims", "and what was typed comes back"


def test_ctrl_r_searches_the_history_shared_with_the_terminal(window):
    from PySide6 import QtCore

    history.add("imcombine @bias.lis Zero.fits")       # typed in a terminal session
    from harwood.viewer.prompt import CommandLine

    prompt = CommandLine()
    prompt.setText("bias")
    _keys(prompt, (QtCore.Qt.Key.Key_R, QtCore.Qt.KeyboardModifier.ControlModifier))
    assert prompt._model.item(0).text() == "imcombine @bias.lis Zero.fits"
    prompt._chosen(prompt._model.index(0, 0))
    assert prompt.text() == "imcombine @bias.lis Zero.fits"

    window.prompt.setText("list")
    window._run_command()
    assert history.load()[-1] == "list", "and the window's lines reach the terminal's"


def test_the_command_line_is_a_panel_that_comes_back(window):
    window.session_dock.hide()
    window.go_to_prompt()
    assert window.session_dock.isVisibleTo(window)
    names = [action.text() for action in window.menuBar().actions()
             if action.text() == "&Panels"][0:1]
    assert names, "the Panels menu is there"
