"""cosmicrays' IRAF algorithm against IRAF 2.18: the pixels, the mask, the header's card, what
training saves and what the graph's keys do.

tests/data/iraf_cosmicrays holds them (run.py, crf_cases.py, crf_inputs.py, crf_collect.py).
"""

import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np
import pytest
from astropy.io import fits

from harwood.core import crfind

HERE = Path(__file__).parent / "data" / "iraf_cosmicrays"
GOLD = json.loads((HERE / "crf.json").read_text())
sys.path.insert(0, str(HERE))
from crf_cases import CASES, SESSIONS  # noqa: E402
from crf_inputs import write  # noqa: E402
from crf_m92_collect import M92  # noqa: E402

ALL = {**CASES, **{"s-" + name: dict(interactive="yes", cursor="keys.cur", keys=keys)
                   for name, keys in SESSIONS.items()}}


@pytest.fixture(scope="module")
def frames(tmp_path_factory):
    folder = tmp_path_factory.mktemp("frames")
    write(folder)
    return folder


def crcor(threshold, fluxratio, removed):
    return f"Threshold={threshold:5.1f}, fluxratio={fluxratio:6.2f}, removed={removed}"


def _run(frames, name):
    case = ALL[name]
    source = frames / case.get("input", "f.fits")
    data = fits.getdata(source)
    objects = None
    if case.get("objects"):
        objects = list(crfind.reads((frames / case["objects"]).read_text()))
    examine = None
    if "keys" in case:
        cursor = crfind.reads(case["keys"].replace("\\\\", "\\"))

        def examine(session):
            crfind.examine_reads(session, cursor)

    pixtype = data.dtype.newbyteorder("=")
    result = crfind.run(data, case.get("threshold", 25.0), case.get("fluxratio", 2.0),
                        case.get("npasses", 5), int(case.get("window", "5")), objects, examine,
                        pixtype)
    return data, case, result, examine


@pytest.mark.parametrize("name", list(GOLD))
def test_the_pixels_the_mask_and_the_card_are_iraf_s(frames, name):
    data, case, result, _ = _run(frames, name)
    expected = GOLD[name]
    changed = np.nonzero(data != result.image)
    ours = [[int(x) + 1, int(y) + 1, float(np.float32(result.image[y, x]))]
            for y, x in zip(*changed, strict=True)]
    assert ours == expected["changed"]
    mask = [[int(x) + 1, int(y) + 1] for y, x in zip(*np.nonzero(result.mask), strict=True)]
    assert mask == expected["mask"]
    assert crcor(case.get("threshold", 25.0), result.fluxratio, result.replaced) == \
        expected["crcor"]


@pytest.mark.parametrize("name", [name for name in GOLD if GOLD[name]["saved"]])
def test_the_training_is_saved_as_iraf_saves_it(frames, name):
    assert "".join(_run(frames, name)[2].saved) == GOLD[name]["saved"]


@pytest.mark.parametrize("name", [name for name in GOLD if GOLD[name]["printed"]])
def test_the_space_bar_finds_iraf_s_candidate(frames, name):
    assert _run(frames, name)[2].printed == GOLD[name]["printed"]


@pytest.mark.parametrize("name", list(GOLD))
def test_the_task_writes_what_iraf_writes(frames, tmp_path, monkeypatch, name):
    from harwood.core.registry import get_task

    monkeypatch.chdir(tmp_path)
    case = ALL[name]
    source = case.get("input", "f.fits")
    kept = fits.getdata(frames / source)
    fits.writeto(source, kept)
    options = dict(algorithm="iraf", output="cr_", gain=1.0, rdnoise=0.0,
                   threshold=case.get("threshold", 25.0), fluxratio=case.get("fluxratio", 2.0),
                   npasses=case.get("npasses", 5), window=int(case.get("window", "5")))
    if "keys" in case:
        Path("keys.cur").write_text(case["keys"].replace("\\\\", "\\"))
        options["cursor"] = "keys.cur"
    if case.get("objects"):
        Path("objects.cur").write_text((frames / case["objects"]).read_text())
        options.update(train=True, objects="objects.cur", savefile="saved.txt")
    result = get_task("cosmicrays").run(input=[source], **options)
    expected = GOLD[name]
    with fits.open(f"cr_{source}") as written:
        out = written["SCI"].data if "SCI" in written else written[0].data
        card = written["SCI" if "SCI" in written else 0].header.get("CRCOR")
        if card is None:
            card = written[0].header["CRCOR"]
    assert out.dtype.kind == kept.dtype.kind and out.dtype.itemsize == kept.dtype.itemsize
    theirs = kept.copy()
    for column, line, value in expected["changed"]:
        theirs[line - 1, column - 1] = value
    assert np.array_equal(out, theirs)
    assert card == expected["crcor"]
    assert (result.stdout or []) == expected["printed"]
    if expected["saved"]:
        assert Path("saved.txt").read_text() == expected["saved"]


def test_two_frames_share_the_reads_and_the_threshold(frames, tmp_path, monkeypatch):
    from harwood.core.registry import get_task

    monkeypatch.chdir(tmp_path)
    for name in ("a.fits", "b.fits"):
        fits.writeto(name, fits.getdata(frames / "f.fits"))
    Path("keys.cur").write_text("1000 50 1 t\n1 1 1 q\n1000 1500 1 \\040\n1 1 1 q\n")
    result = get_task("cosmicrays").run(input=["a.fits", "b.fits"], algorithm="iraf",
                                        output="cr_", gain=1.0, rdnoise=0.0, cursor="keys.cur")
    assert len(result.stdout) == 1
    for name in ("a.fits", "b.fits"):
        assert fits.getheader(f"cr_{name}")["CRCOR"] == GOLD["s-threshold-all"]["crcor"]


def test_the_options_of_iraf_s_algorithm_need_it(frames, tmp_path, monkeypatch):
    from harwood.core.registry import get_task

    monkeypatch.chdir(tmp_path)
    fits.writeto("a.fits", fits.getdata(frames / "f.fits"))
    with pytest.raises(ValueError, match="--train"):
        get_task("cosmicrays").run(input=["a.fits"], train=True, gain=1.0, rdnoise=0.0)


def test_the_panel_is_the_session(frames):
    pytest.importorskip("PySide6")
    import os

    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from harwood.viewer._qt import application
    from harwood.viewer.crgraph_panel import CrGraphPanel

    application()
    data = fits.getdata(frames / "f.fits")
    found, _ = crfind.first_pass(data, 25.0, 5)
    crfind.flags(found, 2.0)
    session = crfind.Session(found, 2.0)
    panel = CrGraphPanel(session, data)
    ended = []
    panel.finished.connect(lambda: ended.append(True))
    panel._cursor = (1000.0, 50.0)
    assert panel.key("t")
    assert session.fluxratio == 50.0
    panel._cursor = (1500.0, 17.0)
    assert panel.key(" ")
    assert session.printed == ["42 49"]
    assert "42 49" in panel.status.text()
    assert panel.key("s")
    assert panel._section is not None
    assert panel.key("x")
    assert panel._section is None
    assert not panel.key("w")
    assert panel.key("q")
    assert ended == [True]


def test_the_training_reads_the_window_s_keys(frames, tmp_path, monkeypatch):
    pytest.importorskip("PySide6")
    import os

    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6 import QtCore

    from harwood.core import display
    from harwood.core.registry import get_task
    from harwood.viewer._qt import application
    from harwood.viewer.window import HarwoodWindow

    monkeypatch.chdir(tmp_path)
    fits.writeto("a.fits", fits.getdata(frames / "f.fits"))
    display.reset()
    monkeypatch.setenv("HARWOOD_DISPLAY", "native")
    application()
    window = HarwoodWindow()
    window.show()

    def press():
        for at, key in (((12.0, 15.0), "c"), ((50.0, 20.0), "c"), ((61.0, 38.0), "s"),
                        ((43.0, 49.0), "s"), ((1.0, 1.0), "q")):
            window._cursor = at
            window._key(key)

    waiting = window.wait_for_examine

    def wait_then_press():
        QtCore.QTimer.singleShot(0, press)
        waiting()

    monkeypatch.setattr(window, "wait_for_examine", wait_then_press)
    try:
        get_task("cosmicrays").run(input=["a.fits"], algorithm="iraf", output="cr_", gain=1.0,
                                   rdnoise=0.0, train=True, savefile="saved.txt")
    finally:
        display.reset()
    assert Path("saved.txt").read_text() == ("12. 15. 1 c\n50. 20. 1 c\n"
                                             "61. 38. 1 s\n43. 49. 1 s\n")
    assert window.key_hook is None


M92_FOLDER = Path(os.environ.get("HARWOOD_TEST_DATA", "~/iraf/exercises")).expanduser()
M92_FOLDER = M92_FOLDER / "phot"


@pytest.mark.skipif(not (M92_FOLDER / "m92010.fits").exists(),
                    reason="needs the tutorial data")
@pytest.mark.parametrize("tag", list(M92))
def test_real_frames_of_m92_come_out_the_same(tag):
    source, threshold, fluxratio, window = M92[tag]
    data = fits.getdata(M92_FOLDER / source)
    result = crfind.run(data, threshold, fluxratio, 5, window)
    expected = json.loads((HERE / "crf_m92.json").read_text())[tag]
    image = result.image.astype(">f4")
    assert hashlib.sha256(image.tobytes()).hexdigest() == expected["image"]
    assert hashlib.sha256(result.mask.tobytes()).hexdigest() == expected["mask"]
    assert crcor(threshold, result.fluxratio, result.replaced) == expected["crcor"]
