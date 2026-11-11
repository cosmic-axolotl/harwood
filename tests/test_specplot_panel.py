"""specplot's window: the keys pressed over the graph reach specplot as a cursor
file's do, and a's question is answered on the command line."""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

from harwood.core import specplot, splot

HERE = Path(__file__).parent / "data" / "iraf_splot"
sys.path.insert(0, str(HERE))
from sp_inputs import write  # noqa: E402

KEYS = [((6563.0, 3500.0), "d"), ((6563.0, 3500.0), "e"), ((0.0, 0.0), "x"),
        ((0.0, 0.0), "z"), ((6563.0, 1000.0), "a")]


def _wait(panel, app):
    for _ in range(500):
        app.processEvents()
        if panel._idle.is_set():
            return
        time.sleep(0.01)
    raise AssertionError("specplot did not come back for a key")


def test_the_window_sends_keys_as_a_cursor_file_does(tmp_path, monkeypatch):
    write(tmp_path)
    monkeypatch.chdir(tmp_path)
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from harwood.viewer._qt import application
    from harwood.viewer.specplot_panel import SpecplotPanel

    app = application()
    panel = SpecplotPanel({}, ["ms.fits"])
    _wait(panel, app)
    for (x, y), key in KEYS:
        panel._cursor = (x, y)
        assert panel.key(key)
        _wait(panel, app)
    assert panel._asking.is_set()
    panel._poll()
    assert "Spectrum:" in panel.command.placeholderText()
    panel.command.setText("one")
    panel._run_command()
    _wait(panel, app)
    panel.command.setText(":show")
    panel._run_command()
    _wait(panel, app)
    panel.key("q")
    for _ in range(300):
        app.processEvents()
        if not panel._thread.is_alive():
            break
        time.sleep(0.01)
    assert not panel._thread.is_alive()
    assert len(panel.session.sps) == 4

    text = "".join(f"{x} {y} 1 {k}\n" for (x, y), k in KEYS) + ":show\n0 0 1 q\n"
    reference = specplot.Specplot({}, iter(["one"]))
    reference.run(["ms.fits"], splot.cursor_reads(text))
    strip = specplot.sysid
    assert panel.session.out.replace(strip(), "") == reference.out.replace(strip(), "")
    assert len(panel.session.graphs) == len(reference.graphs)


def test_the_task_runs_a_cursor_file_and_writes_the_log(tmp_path, monkeypatch):
    from harwood.tasks.specplot import Specplot, SpecplotParams

    write(tmp_path)
    monkeypatch.chdir(tmp_path)
    (tmp_path / "keys").write_text(":show\n0 0 1 q\n", encoding="utf-8")
    result = Specplot().execute(SpecplotParams(spectra=["ms,one"], gcommands="keys",
                                               logfile="sp.log"))
    assert result.data["spectra"] == 4
    log = (tmp_path / "sp.log").read_text(encoding="utf-8")
    assert "Image Name" in log and "V(km/s)" in log
