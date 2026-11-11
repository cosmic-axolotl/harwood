"""splot's window: the keys pressed over the graph reach splot as a cursor file's do.

The panel runs core.splot.Splot in a thread that reads the keys it is sent; here the
keys are pressed (panel.key) where the mouse is (panel._cursor), and what splot
prints, logs and draws is what the same keys in a cursor file give. A question --
s's box -- is answered on the command line.
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import pytest

from harwood.core import splot

HERE = Path(__file__).parent / "data" / "iraf_splot"
sys.path.insert(0, str(HERE))
from sp_inputs import write  # noqa: E402

KEYS = [((6450.0, 1100.0), " "), ((6420.0, 1060.0), "m"), ((6500.0, 1080.0), "m"),
        ((6530.0, 1065.0), "e"), ((6600.0, 1080.0), "e"), ((6535.0, 1060.0), "d"),
        ((6600.0, 1080.0), "d"), ((6548.0, 1500.0), "g"), ((6563.0, 3000.0), "g"),
        ((6583.0, 1800.0), "g"), ((0.0, 0.0), "q"), ((0.0, 0.0), "a"), ((0.0, 0.0), "a"),
        ((0.0, 0.0), "y"), ((0.0, 0.0), "b"), ((0.0, 0.0), "+"), ((0.0, 0.0), "q"),
        ((0.0, 0.0), "q"), ((6500.0, 0.0), "a"), ((6620.0, 0.0), "a"), ((0.0, 0.0), "s")]


def _wait(panel, app):
    for _ in range(500):
        app.processEvents()
        if panel._idle.is_set():
            return
        time.sleep(0.01)
    raise AssertionError("splot did not come back for a key")


@pytest.fixture
def frames(tmp_path, monkeypatch):
    write(tmp_path)
    monkeypatch.chdir(tmp_path)
    return tmp_path


def test_the_window_sends_keys_as_a_cursor_file_does(frames):
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from harwood.viewer._qt import application
    from harwood.viewer.splot_panel import SplotPanel

    app = application()
    panel = SplotPanel({"save_file": ""}, ["one.fits"])
    _wait(panel, app)
    for (x, y), key in KEYS:
        panel._cursor = (x, y)
        assert panel.key(key)
        _wait(panel, app)
    assert panel._asking.is_set() and panel.session.asking == "boxsize"
    panel.command.setText("3")
    panel._run_command()
    _wait(panel, app)
    panel._cursor = (6563.0, 0.0)
    panel.key(" ")
    _wait(panel, app)
    panel.key("q")
    for _ in range(300):
        app.processEvents()
        if not panel._thread.is_alive():
            break
        time.sleep(0.01)
    assert not panel._thread.is_alive()

    text = "".join(f"{x} {y} 1 {'\\040' if k == ' ' else k}\n" for (x, y), k in KEYS)
    text += "6563 0 1 \\040\n0 0 1 q\n"
    reference = splot.Splot({"save_file": ""}, iter(["3"]))
    reference.run(["one.fits"], splot.cursor_reads(text))
    assert panel.session.out == reference.out
    assert panel.session.log == reference.log
    assert panel.session.window == reference.window
    panel._poll()
    assert panel.plot.getPlotItem().titleLabel.text
