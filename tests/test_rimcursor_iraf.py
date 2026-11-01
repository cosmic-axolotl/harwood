"""rimcursor against IRAF 2.18, the lines it prints for a file of cursor reads.

tests/data/iraf_rimcursor holds them (run.py, rimcursor_cases.py).
"""

import json
import sys
from pathlib import Path

import pytest

from harwood.core.registry import get_task

HERE = Path(__file__).parent / "data" / "iraf_rimcursor"
GOLD = json.loads((HERE / "rimcursor.json").read_text())
sys.path.insert(0, str(HERE))
from rimcursor_cases import CURSOR  # noqa: E402
from rimcursor_inputs import write  # noqa: E402


@pytest.fixture
def work(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    write(tmp_path)


@pytest.mark.parametrize("case", list(GOLD["cases"]))
def test_a_read_prints_what_iraf_prints(work, case):
    expected = GOLD["cases"][case]
    Path("cur").write_text(CURSOR)
    if expected["error"]:
        with pytest.raises(ValueError, match="coordinate system not defined"):
            get_task("rimcursor").run(cursor="cur", **expected["params"])
        return
    result = get_task("rimcursor").run(cursor="cur", **expected["params"])
    theirs = expected["text"].split("\n")[:-1]
    assert result.stdout == theirs


@pytest.mark.parametrize("case", list(GOLD["bad"]))
def test_a_bad_read_stops_it(work, case):
    expected = GOLD["bad"][case]
    Path("cur").write_text(expected["cursor"])
    with pytest.raises(ValueError, match="bad cursor read"):
        get_task("rimcursor").run(cursor="cur")
    assert "bad cursor read" in expected["error"]


def test_the_window_s_keys_are_reads(work, monkeypatch):
    import os

    pytest.importorskip("PySide6")
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6 import QtCore

    from harwood.core import display
    from harwood.viewer._qt import application
    from harwood.viewer.window import HarwoodWindow

    display.reset()
    monkeypatch.setenv("HARWOOD_DISPLAY", "native")
    application()
    window = HarwoodWindow()
    window.show()

    def press():
        window._cursor = (20.0, 10.0)
        window._key("a")
        window._key(" ")
        window.run_line(":mark it")
        window.examine_done.emit()

    waiting = window.wait_for_examine

    def wait_then_press():
        QtCore.QTimer.singleShot(0, press)
        waiting()

    monkeypatch.setattr(window, "wait_for_examine", wait_then_press)
    try:
        result = get_task("rimcursor").run(image="tan.fits", wcs="physical")
    finally:
        display.reset()
    assert result.stdout == ["20. 10. 101 a", "20. 10. 101 \\040", "20. 10. 101 :mark it"]
    assert window.key_hook is None
