"""tvmark: the list marked, commands replayed, objects added and deleted,
and the snapshot -- the frame's grey levels with the marks in them."""

from __future__ import annotations

import numpy as np
import pytest

import harwood
from harwood.core import display, render, tvmark

COO = """#K IRAF       = NOAO/IRAFV2.18        version    %-23s
#N XCENTER   YCENTER   MAG      SHARPNESS
#U pixels    pixels    #        #
  10.000    12.000    -1.234   0.5
  30.500    20.000    -2.000   0.6
  45.000    40.000    -0.500   0.4
"""


@pytest.fixture
def night(tmp_path, monkeypatch):
    from astropy.io import fits

    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("HARWOOD_DISPLAY", "headless")
    display.reset()
    yy, xx = np.indices((50, 60))
    data = (100 + xx + 0.5 * yy).astype(np.float32)
    fits.PrimaryHDU(data).writeto("field.fits")
    (tmp_path / "field.coo").write_text(COO)
    yield tmp_path
    display.reset()


def test_the_list_is_marked_and_snapped(night):
    from astropy.io import fits

    result = harwood.run("tvmark", frame=1, coords="field.coo", image="field.fits",
                         mark="point", color=204, outimage="snap.fits")
    snap = fits.getdata(night / "snap.fits")
    data = fits.getdata(night / "field.fits")
    z1, z2 = render.zscale(data)
    levels = render.device_levels(data, z1, z2)
    # every object has a 3x3 dot at colour 204 ...
    for x, y in ((10, 12), (31, 20), (45, 40)):
        assert (snap[y - 2:y + 1, x - 2:x + 1] == 204).all(), (x, y)
    # ... and everything else is the frame as display shows it, so
    # `display snap.fits --ztrans none` shows the same picture
    untouched = snap != 204
    assert np.array_equal(snap[untouched], levels[untouched])
    assert result.data["marked"] == 3


def test_numbers_and_circles(night):
    from astropy.io import fits

    harwood.run("tvmark", frame=1, coords="field.coo", image="field.fits",
                mark="circle", radii="3,6", number=True, color=205, outimage="c.fits")
    snap = fits.getdata(night / "c.fits")
    # a ring of radius 6 around (45, 40), not its centre
    assert snap[39, 44] != 205
    assert snap[39, 44 + 6] == 205 and snap[39 + 6, 44] == 205
    # the number 1 is drawn with its lower left corner at the object
    assert (snap[11:18, 9:15] == 205).any()


def test_a_logfile_replays_to_the_same_marks(night):
    """Commands go to the logfile in the form --commands reads, so a session
    can be redone without the cursor. With commands, a snapshot is taken
    only by :write, as in IRAF."""
    from astropy.io import fits

    commands = night / "cmds"
    commands.write_text("20 20 1 c\n25 25 1 +\n1 1 1 :color 204\n30 30 1 x\n"
                        "5 5 1 s\n15 5 1 s\n1 1 1 :write a.fits\n")
    harwood.run("tvmark", frame=1, coords="", image="field.fits", commands=str(commands),
                radii="4", autolog=True, logfile="log1")
    logged = (night / "log1").read_text()
    assert logged.splitlines()[0] == "20 20 1 c"
    (night / "log2").write_text(logged.replace(":write a.fits", ":write b.fits"))
    harwood.run("tvmark", frame=1, coords="", image="field.fits", commands="log2",
                radii="4")
    first, again = fits.getdata(night / "a.fits"), fits.getdata(night / "b.fits")
    assert np.array_equal(first, again)
    assert (first == 204).any() and (first == 255).any(), "the colour changed midway"


def test_objects_added_and_deleted_keep_the_rest_of_the_list(night):
    commands = night / "cmds"
    commands.write_text("30 20.5 1 d\n52 8 1 a\n")
    result = harwood.run("tvmark", frame=1, coords="field.coo", image="field.fits",
                         commands=str(commands), deletions="del")
    lines = (night / "field.coo").read_text().splitlines()
    assert lines[:3] == COO.splitlines()[:3], "the daofind header stays"
    assert not any(line.startswith("  30.500") for line in lines)
    assert lines[-1] == "52.000 8.000"
    assert (night / "field.coo.del").read_text().startswith("30.500 20.000 -2.000")
    assert result.data["marked"] >= 2


def test_colon_commands_set_what_they_name():
    engine = tvmark.Engine(tvmark.Settings())
    assert engine.colon(":mark circle") == "mark = circle"
    assert engine.colon(":radii 5,10") == "radii = 5,10"
    assert engine.colon(":lengths 10,20 0.5") == "lengths = 10,20 0.5"
    assert engine.colon(":number yes") == "number = yes"
    assert "not a tvmark command" in engine.colon(":frobnicate")
    engine.items = [tvmark.Item(5, 5), tvmark.Item(9, 9), tvmark.Item(12, 3)]
    engine.colon(":next 2 3")
    assert len([m for m in engine.marks if m.shape == "circle"]) == 4
    assert [m.label for m in engine.marks if m.shape == "text"] == ["2", "3"]


def test_a_frame_with_nothing_in_it_says_so(night):
    with pytest.raises(Exception, match="holds no image"):
        harwood.run("tvmark", frame=3, coords="field.coo")


def test_marking_with_the_cursor_in_the_window(night, monkeypatch):
    """The interactive path: keys at the cursor go to the same engine, the
    marks appear on the frame, a is added to the list and q ends it."""
    import os

    pytest.importorskip("PySide6")
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6 import QtCore

    from harwood.viewer._qt import application
    from harwood.viewer.window import HarwoodWindow

    display.reset()
    monkeypatch.setenv("HARWOOD_DISPLAY", "native")
    application()
    window = HarwoodWindow()
    window.show()

    def press():
        window._cursor = (20.0, 20.0)
        window._key("c")
        window.run_line(":number yes")
        window._cursor = (52.0, 8.0)
        window._key("a")
        window._key("q")

    waiting = window.wait_for_examine

    def wait_then_press():
        QtCore.QTimer.singleShot(0, press)
        waiting()

    monkeypatch.setattr(window, "wait_for_examine", wait_then_press)
    try:
        result = harwood.run("tvmark", frame=1, coords="field.coo", image="field.fits",
                             interactive=True, radii="5")
        marks = [m for m in window.frames[1].markers if m.tag == "tvmark"]
    finally:
        display.reset()
    assert any(m.shape == "circle" and m.radius == 5 for m in marks)
    assert [m.label for m in marks if m.shape == "text"] == ["4"], "numbered after :number"
    assert result.data["marked"] == len(marks)
    assert (night / "field.coo").read_text().splitlines()[-1] == "52.000 8.000"
    assert window.key_hook is None, "the keys go back to imexamine"
