"""What has to hold for harwood to run on Windows as it does on Linux, checked on any machine:
text is read and written as UTF-8 whatever the code page, the terminal's too, and a FITS file
that may be written again is not left mapped, which Windows does not allow."""

import ast
import io
from pathlib import Path

import harwood

SOURCE = Path(harwood.__file__).parent


def _calls():
    for path in sorted(SOURCE.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                yield path.relative_to(SOURCE), node


def _mode(node: ast.Call, at: int):
    if len(node.args) > at and isinstance(node.args[at], ast.Constant):
        return node.args[at].value
    for keyword in node.keywords:
        if keyword.arg == "mode" and isinstance(keyword.value, ast.Constant):
            return keyword.value.value
    return None


def test_text_is_read_and_written_as_utf8():
    """Without an encoding, Windows reads and writes in its code page, cp1252, and a name with
    an accent in a header or a list is garbled, or stops the task."""
    left = []
    for path, node in _calls():
        function = node.func
        name = getattr(function, "attr", None) or getattr(function, "id", None)
        given = {keyword.arg for keyword in node.keywords}
        if "encoding" in given:
            continue
        attribute = isinstance(function, ast.Attribute)
        owner = getattr(getattr(function, "value", None), "id", None)
        if (name == "read_text" and attribute and not node.args) or (
                name == "write_text" and attribute and len(node.args) < 2):
            left.append(f"{path}:{node.lineno}")
        elif name == "open" and not attribute:
            mode = _mode(node, 1)
            if not (mode and "b" in mode):
                left.append(f"{path}:{node.lineno}")
        elif name == "open" and owner not in ("fits", "gzip", "bz2", "zipfile", "tarfile",
                                              "os", "webbrowser", "Image"):
            # A path's open takes its mode first, or nothing; other things called open, as a
            # session's or a panel's, take what they open
            first = node.args[0] if node.args else None
            if first is None or isinstance(first, ast.Constant):
                mode = _mode(node, 0)
                if not (mode and "b" in mode):
                    left.append(f"{path}:{node.lineno}")
    assert not left, "give encoding='utf-8' at " + ", ".join(left)


def test_a_fits_file_read_is_not_left_mapped():
    """astropy maps a FITS file by default, and Windows will not overwrite or rename a file
    that is mapped: reading an image and writing it again would fail there. A file opened to
    be updated is the exception, as that is what it is for."""
    left = []
    for path, node in _calls():
        function = node.func
        if not (isinstance(function, ast.Attribute) and getattr(function.value, "id", None)
                == "fits" and function.attr in ("getdata", "open")):
            continue
        given = {keyword.arg: keyword.value for keyword in node.keywords}
        if "memmap" in given:
            continue
        mode = given.get("mode")
        if isinstance(mode, ast.Constant) and mode.value == "update":
            continue
        if mode is not None and not isinstance(mode, ast.Constant):
            continue
        left.append(f"{path}:{node.lineno}")
    assert not left, "give memmap=False at " + ", ".join(left)


def test_the_terminal_is_utf8_whatever_the_code_page(monkeypatch):
    """A redirected output on Windows is cp1252, where a table's lines and a warning's mark
    do not exist: the task stopped half way through its answer."""
    import sys

    from harwood.cli.main import utf8_streams

    page = io.TextIOWrapper(io.BytesIO(), encoding="cp1252")
    monkeypatch.setattr(sys, "stdout", page)
    monkeypatch.setattr(sys, "stderr", io.TextIOWrapper(io.BytesIO(), encoding="cp1252"))
    utf8_streams()
    print("✖ │ ·", file=sys.stdout)
    sys.stdout.flush()
    assert page.buffer.getvalue().decode("utf-8").strip() == "✖ │ ·"


def test_an_editor_with_spaces_and_backslashes_in_its_path_on_windows(monkeypatch, tmp_path):
    """On Windows the command line is Windows', given as it is: a POSIX split would take the
    backslashes of C:\\Program Files\\... for escapes."""
    import subprocess

    from harwood.cli import main

    seen = []
    monkeypatch.setattr(subprocess, "run", lambda command, check=False: seen.append(command))
    editor = r'"C:\Program Files\Notepad++\notepad++.exe" -multiInst'
    main.run_editor(editor, tmp_path / "a.par", windows=True)
    assert seen == [f'{editor} "{tmp_path / "a.par"}"']
    main.run_editor("vi -n", tmp_path / "a.par", windows=False)
    assert seen[1] == ["vi", "-n", str(tmp_path / "a.par")]
