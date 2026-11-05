"""standard against IRAF 2.18's own std files.

tests/data/iraf_standard holds what IRAF's standard wrote for the spectra
of tests/iraf_field.standard_cases(): the calibration table's bands and
bands of its own width and spacing, a spectrum already corrected for
extinction (EX-FLAG 0), airmasses worked out from RA, DEC and HA or ST at
Kitt Peak, beam-switched object and sky pairs (one sky on another
dispersion, two objects left without one), three apertures of one frame
(all, some, each its own star), bands added and deleted from a cursor file
(carried to the aperture's next spectrum), another fnuzero, blackbodies in
V and in J (microns), and exposure time and airmass asked for. The task
writes IRAF's format here, so the comparison is of the text, line by line.

Two differences are IRAF's mistakes and are put back to compare: a band
marked right to left is written by IRAF with a negative width (harwood
keeps it positive), and getbbcal reads params.dat's colours two columns to
the right (harwood reads the band's own; stdstar.colour_column).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from astropy.io import fits
from iraf_field import standard_cases

from harwood.core import spectrum, stdfile, stdstar
from harwood.core.registry import get_task

DATA = Path(__file__).parent / "data"
REF = DATA / "iraf_standard"
BB = Path(stdstar.__file__).resolve().parent.parent / "data" / "iraf" / "onedstds" / \
    "blackbody"

CASES = {
    "basic": ("std1.fits,std2.fits", {}),
    "bandwidth": ("std1.fits", {"bandwidth": 30, "bandsep": 25}),
    "bandwidth2": ("std2.fits", {"bandwidth": 7.5, "bandsep": 40}),
    "exflag": ("ex1.fits", {}),
    "noam": ("noam.fits,noamst.fits", {}),
    "beam": ("b1.fits,b2.fits,b3.fits,b4.fits,b5.fits,b6.fits", {"beam_switch": True}),
    "beamrebin": ("b7.fits,b8.fits", {"beam_switch": True}),
    "ms3": ("ms3.fits", {}),
    "ms3aps": ("ms3.fits", {"apertures": "1,3"}),
    "ms3nosame": ("ms3.fits", {"samestar": False, "apertures": "2-3"}),
    "ms3int": ("ms3.fits,std1.fits", {"interact": True, "answer": "YES", "cursor": "cur2"}),
    "int": ("std1.fits,std2.fits", {"interact": True, "answer": "YES!", "cursor": "cur1"}),
    "intno": ("std1.fits,std2.fits", {"interact": True, "answer": "no", "cursor": "cur1"}),
    "fnuzero": ("std1.fits", {"fnuzero": 3.6e-20}),
    "ask": ("ask.fits", {"exptime": 450, "airmass": 1.42}),
    "bb": ("std1.fits", {"caldir": str(BB), "star_name": "V", "mag": 10.3, "magband": "V",
                         "teff": "A0V"}),
    "bbx": ("std1.fits", {"caldir": str(BB), "star_name": "V", "mag": 10.3, "magband": "V",
                          "teff": "B5"}),
    "bbj": ("ir1.fits", {"caldir": str(BB), "star_name": "J", "mag": 10.3, "magband": "V",
                         "teff": "K5V"}),
    "bbj2": ("ir1.fits", {"caldir": str(BB), "star_name": "J", "mag": 10.3, "magband": "V",
                          "teff": "4400"}),
    "bbj3": ("ir1.fits", {"caldir": str(BB), "star_name": "J", "mag": 10.3, "magband": "J",
                          "teff": "G2"}),
}

#: What IRAF printed beside the std file, where it says something.
SAID = {
    "beam": ["b5.fits[1]: Sky spectrum not found", "b6.fits[1]: Sky spectrum not found"],
    "bb": ["Blackbody: V = 10.30, Teff = 9480"],
    "bbx": ["WARNING: Effective temperature for B5 not found - using B5V",
            "Blackbody: V = 10.30, Teff = 15400"],
    "bbj": ["Blackbody: V = 10.30, J = 10.19, Teff = 4340"],
    "bbj2": ["WARNING: Assuming spectral type of K5V",
             "Blackbody: V = 10.30, J = 10.19, Teff = 4400"],
    "bbj3": ["WARNING: Effective temperature for G2 not found - using G2V",
             "Blackbody: J = 10.30, Teff = 5830"],
    "int": ["Two cursor positions required"],
}


def write_spectra(where: Path) -> None:
    for name, (apertures, header) in standard_cases().items():
        made = []
        for number, w1, dw, counts, title in apertures:
            n = counts.size
            made.append(spectrum.Aperture(number=number, centre=float(number), low=-1.0,
                                          high=1.0, pixel=np.arange(1.0, n + 1.0),
                                          flux=counts.astype(float),
                                          wavelength=w1 + dw * np.arange(n),
                                          title=title if len(apertures) > 1 else ""))
        spectrum.write(where / f"{name}.fits", made, header=fits.Header(header))


@pytest.fixture(scope="module")
def spectra(tmp_path_factory):
    where = tmp_path_factory.mktemp("standard")
    write_spectra(where)
    return where


@pytest.mark.parametrize("case", list(CASES))
def test_standard_writes_what_iraf_writes(case, spectra, tmp_path, monkeypatch):
    import shutil

    for path in spectra.iterdir():
        shutil.copy(path, tmp_path / path.name)
    for name in ("cur1", "cur2"):
        shutil.copy(REF / name, tmp_path / name)
    monkeypatch.chdir(tmp_path)
    if case.startswith("bbj"):
        monkeypatch.setattr(stdstar, "colour_column", lambda bands, band:
                            bands.index(band) + 2)
    images, extra = CASES[case]
    values = {"caldir": str(REF / "mycal"), "star_name": "fakestar",
              "extinction": str(DATA / "kpnoextinct.dat"), "observatory": "kpno",
              "format": "iraf", **extra}
    result = get_task("standard").run(input=images, output=tmp_path / "std", **values)
    ours = (tmp_path / "std").read_text(encoding="utf-8").splitlines()
    theirs = [line.replace("  -20.000", "   20.000")
              for line in (REF / f"{case}.std").read_text(encoding="utf-8").splitlines()]
    assert ours == theirs
    said = result.data.get("text", [])
    for line in SAID.get(case, []):
        assert line in said
    if case == "ask":
        header = fits.getheader(tmp_path / "ask.fits")
        assert (header["EXPTIME"], header["AIRMASS"]) == (450.0, 1.42)


#: The cursor file and how many frames IRAF drew: curw and curc are w's keys and the
#: /commands, a band added and deleted in a window flipped, transposed, on elog.
GRAPHS = {"int": ("cur1", 2), "redraw": ("cur3", 8), "curw": ("curw", 9), "curc": ("curc", 7)}
GKI_UNITS = 2.5


@pytest.mark.parametrize("case", list(GRAPHS))
def test_the_graph_is_iraf_s(case, spectra, tmp_path, monkeypatch):
    """IRAF's metacode, where a frame is drawn: the start of each spectrum, r, and
    what w and the /commands redraw. What a and d put over the frame after that comes
    after the frame's boxes. Where IRAF's std file is kept, the bands are its."""
    import shutil

    for path in spectra.iterdir():
        shutil.copy(path, tmp_path / path.name)
    cursor, count = GRAPHS[case]
    shutil.copy(REF / cursor, tmp_path / cursor)
    monkeypatch.chdir(tmp_path)
    drawn = []

    class Recording(stdstar.Editor):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            drawn.append(self.graph())

        def key(self, x, key, command="", y=0.0):
            super().key(x, key, command, y)
            if self.newgraph:
                drawn.append(self.graph())

    monkeypatch.setattr(stdstar, "Editor", Recording)
    get_task("standard").run(input="std1.fits,std2.fits", output=tmp_path / "std",
                             caldir=str(REF / "mycal"), star_name="fakestar",
                             extinction=str(DATA / "kpnoextinct.dat"), observatory="kpno",
                             interact=True, answer="YES", cursor=cursor, format="iraf")
    if (REF / "graphs" / f"{case}.std").exists():
        assert (tmp_path / "std").read_text(encoding="utf-8") == (
            REF / "graphs" / f"{case}.std").read_text(encoding="utf-8")
    frames = json.loads((REF / "graphs" / f"{case}.json").read_text(encoding="utf-8"))
    assert len(frames) == len(drawn) == count
    for number, (graph, frame) in enumerate(zip(drawn, frames, strict=True)):
        where = f"frame {number}"
        wx1, wx2, wy1, wy2, sx1, sx2, sy1, sy2 = frame["windows"][0]
        assert graph.window[:2] == pytest.approx((wx1, wx2), abs=1e-4 * abs(wx2 - wx1)), where
        assert graph.window[2:] == pytest.approx((wy1, wy2), abs=1e-4 * abs(wy2 - wy1)), where
        for text in (graph.image, graph.title, graph.xlabel, graph.ylabel):
            assert not text or text in frame["text"], (where, text)
        logs = frame.get("logs", [[False, False]])[0]
        assert tuple(graph.logs) == tuple(logs), where
        if any(logs):
            continue                    # the boxes, gmark's, are sized in NDC
        scale_x = (sx2 - sx1) * 32767 / (wx2 - wx1)
        scale_y = (sy2 - sy1) * 32767 / (wy2 - wy1)
        mine = np.c_[sx1 * 32767 + (graph.x - wx1) * scale_x,
                     sy1 * 32767 + (graph.y - wy1) * scale_y]
        if graph.kind == "line":
            # The spectrum, cut by the window: what is well inside it, of pieces too
            # long to be an axis's ticks or a box the edge cut.
            pieces = [p for _, p in frame["lines"] if len(p) >= 30]
            drawn_points = np.array([q for p in pieces for q in p]).reshape(-1, 2)
            inside = drawn_points[(drawn_points[:, 0] > sx1 * 32767 + 1000)
                                  & (drawn_points[:, 0] < sx2 * 32767 - 1000)
                                  & (drawn_points[:, 1] > sy1 * 32767 + 1000)
                                  & (drawn_points[:, 1] < sy2 * 32767 - 1000)]
            assert len(inside), where
            for point in inside:
                assert np.abs(mine - point).max(axis=1).min() <= GKI_UNITS, where
        else:
            marks = np.array([m[1:3] for m in frame["marks"] if m[0] == graph.mark])
            inside = mine[(mine[:, 0] > sx1 * 32767 + 400) & (mine[:, 0] < sx2 * 32767 - 400)
                          & (mine[:, 1] > sy1 * 32767 + 400) & (mine[:, 1] < sy2 * 32767 - 400)]
            for point in inside:
                assert np.abs(marks - point).max(axis=1).min() <= GKI_UNITS, where
        boxes = [m for m in frame["marks"] if m[0] == "box"]
        whole = 0
        for wave, dwave, level in graph.boxes:
            at = np.array([sx1 * 32767 + (wave - wx1) * scale_x,
                           sy1 * 32767 + (level - wy1) * scale_y])
            width, height = abs(dwave * scale_x), abs(graph.box_height * scale_y)
            if not (sx1 * 32767 + width / 2 + 2 < at[0] < sx2 * 32767 - width / 2 - 2
                    and sy1 * 32767 + height / 2 + 2 < at[1] < sy2 * 32767 - height / 2 - 2):
                continue                # GIO cut it at the edge of the viewport
            whole += 1
            # Bands of one centre and two widths overlap: the box of its own width.
            _, bx, by, bw, bh = min(boxes, key=lambda m, at=at, width=width: max(
                np.abs(np.array(m[1:3]) - at).max(), abs(m[3] - width)))
            assert tuple(at) == pytest.approx((bx, by), abs=GKI_UNITS), where
            assert width == pytest.approx(bw, abs=2 * GKI_UNITS), where
            assert height == pytest.approx(bh, abs=5 * GKI_UNITS), where
        assert whole <= len(boxes), where


def test_harwood_s_file_holds_the_same_bands(spectra, tmp_path, monkeypatch):
    """The CSV harwood writes by default is read back to the bands IRAF's
    file gives: counts per second per Angstrom, apertures and images kept."""
    import shutil

    for path in spectra.iterdir():
        shutil.copy(path, tmp_path / path.name)
    monkeypatch.chdir(tmp_path)
    get_task("standard").run(input="ms3.fits", output=tmp_path / "std",
                             caldir=str(REF / "mycal"), star_name="fakestar",
                             extinction=str(DATA / "kpnoextinct.dat"))
    ours = stdfile.read(tmp_path / "std").measurements
    theirs = stdfile.read(REF / "ms3.std").measurements
    assert len(ours) == len(theirs) == 39
    for a, b in zip(ours, theirs, strict=True):
        assert (a.aperture, a.image, a.wavelength, a.width) == \
            (b.aperture, b.image, b.wavelength, b.width)
        assert a.counts == pytest.approx(b.counts, rel=5e-5)
        assert a.flux == pytest.approx(b.flux, rel=5e-5)


def test_an_existing_file_keeps_its_format(spectra, tmp_path, monkeypatch):
    import shutil

    for path in spectra.iterdir():
        shutil.copy(path, tmp_path / path.name)
    monkeypatch.chdir(tmp_path)
    common = {"caldir": str(REF / "mycal"), "star_name": "fakestar",
              "extinction": str(DATA / "kpnoextinct.dat")}
    get_task("standard").run(input="std1.fits", output=tmp_path / "std", format="iraf",
                             **common)
    result = get_task("standard").run(input="std2.fits", output=tmp_path / "std", **common)
    assert (tmp_path / "std").read_text(encoding="utf-8").splitlines() == \
        (REF / "basic.std").read_text(encoding="utf-8").splitlines()
    assert any("already in iraf" in w.message for w in result.warnings)


def test_harwood_converts_the_blackbody_s_band_through_its_own_colour():
    """J - V of a K5V is -2.13 in params.dat; IRAF takes K - H, -0.11."""
    cal = stdstar.read_calibration(str(BB), "J", 10.3, "V", "K5V")
    assert "Blackbody: V = 10.30, J = 8.17, Teff = 4340" in cal.notes


def test_a_band_marked_right_to_left_keeps_a_positive_width():
    cases = standard_cases()
    (_, w1, dw, counts, _), = cases["std1"][0]
    spec = stdstar.Spectrum("std1.fits", 1, "fakestar", w1 + dw * np.arange(counts.size),
                            counts, 1.15, 300.0)
    cal = stdstar.read_calibration(str(REF / "mycal"), "fakestar")
    bands = stdstar.initial_bands(cal, spec)
    editor = stdstar.Editor(spec, bands, stdstar.Extinction(None))
    editor.key(4120, "a")
    editor.key(4100, "a")
    added = bands.waves.index(np.float32(4110))
    assert bands.dwaves[added] == np.float32(20) and bands.fluxes[added] > 0


def test_cases_cover_every_path():
    """Every frozen IRAF file has its case here, and the other way round."""
    frozen = {path.stem for path in REF.glob("*.std")}
    assert frozen == set(CASES)


def test_the_window_drives_the_same_editor():
    from harwood.viewer._qt import application
    from harwood.viewer.standard_panel import StandardPanel

    application()
    (_, w1, dw, counts, _), = standard_cases()["std1"][0]
    spec = stdstar.Spectrum("std1.fits", 1, "fakestar", w1 + dw * np.arange(counts.size),
                            counts, 1.15, 300.0)
    cal = stdstar.read_calibration(str(REF / "mycal"), "fakestar")
    bands = stdstar.initial_bands(cal, spec)
    editor = stdstar.Editor(spec, bands, stdstar.Extinction(None))
    stdstar.measure(spec, bands, stdstar.Extinction(None))
    panel = StandardPanel(editor)
    assert len(panel.editor.graph().boxes) == 13
    panel._x = 4101.0
    panel.key("d")
    assert np.float32(4100) not in bands.waves
    panel._x = 4150.0
    panel.key("a")
    panel._x = 4190.0
    panel.key("a")
    assert np.float32(4170) in bands.waves and len(panel.editor.graph().boxes) == 13
    box = panel.plot.getViewBox()
    panel._x, panel._y = 4150.0, 140000.0
    panel.key("w")
    panel.key("x")
    (x0, x1), _ = box.viewRange()
    assert x0 < 4150 < x1 and x1 - x0 == pytest.approx(
        editor.window[1] - editor.window[0], rel=1e-6)
    panel.command.setText(":/xflip yes")
    panel._run_command()
    panel.key("r")
    assert box.xInverted() and editor.window[0] > editor.window[1]
    panel.command.setText(":/ytransform log")
    panel._run_command()
    panel.key("r")
    assert editor.gt.logs() == (False, True)
    panel.command.setText(":.zero")
    panel._run_command()
    assert "cursor-mode command harwood does not have" in panel.status.text()
    done = []
    panel.finished.connect(lambda: done.append(True))
    panel.key("q")
    assert done
