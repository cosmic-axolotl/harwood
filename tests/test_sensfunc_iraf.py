"""sensfunc against IRAF 2.18's own output, bit for bit.

tests/data/iraf_sensfunc holds IRAF std files (two stars at airmasses 1.15
and 1.85; three apertures of one frame; both together; two band sets) and
what IRAF's sensfunc made of them: the sensitivity spectra, the logfile,
the :stats and :vstats files and the revised extinction table. In batch:
every aperture, apertures chosen, ignoreaps, four functions. From a cursor
file, on the three stars of aperture 1: points, stars and wavelengths
deleted and restored, moved, added and reweighted, the stars shifted and
made a composite, a residual extinction applied and cancelled, the original
data back, the order and function changed, and three graphs. What each cursor
file draws is in iraf_sensfunc/graphs, from IRAF's graphics metacode.

IRAF reads the new weight (w) and the answer to the residual extinction
(e) from the terminal; the cursor files here give them as the next line's
':' text, as harwood reads them. IRAF's icfit takes the keys between e and
its q; harwood fits that curve as IRAF does by default.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import pytest

from harwood.core import flux
from harwood.core.registry import get_task

DATA = Path(__file__).parent / "data"
REF = DATA / "iraf_sensfunc"
SENS = np.load(REF / "sens.npz")

BATCH = {
    "a": ("std_two", "sa", {}),
    "b": ("std_ms3", "sb", {}),
    "c": ("std_ms3", "sc", {"ignoreaps": True, "function": "legendre", "order": 4}),
    "d": ("std_mix", "sd", {"apertures": "1,3", "function": "chebyshev", "order": 3}),
    "e": ("std_bw", "se", {"function": "spline1", "order": 3}),
}
CURSOR = ["del", "delw", "shift", "comp", "move", "add", "wts", "ext", "order", "reset",
          "graphs", "ecancel"]


def lines(path: Path) -> list[str]:
    """A logfile without its first line of each block, which names the system."""
    return [line for line in path.read_text().splitlines()
            if not line.startswith(("Community IRAF", "harwood "))]


@pytest.mark.parametrize("case", list(BATCH))
def test_batch_is_iraf_s(case, tmp_path, monkeypatch):
    standards, root, extra = BATCH[case]
    monkeypatch.chdir(tmp_path)
    get_task("sensfunc").run(standards=REF / standards, sensitivity=root,
                             extinction=str(DATA / "kpnoextinct.dat"),
                             logfile=f"log_{case}", **extra)
    names = [n for n in SENS.files if n.split("_")[0] == root and not n.endswith("_w")]
    assert names
    for name in names:
        file = name.replace("_", ".") + ".fits"
        ours = flux.SensSpectrum.read(tmp_path / file)
        w1, dw = SENS[name + "_w"]
        assert np.float32(ours.w1) == np.float32(w1) and np.float32(ours.dw) == np.float32(dw)
        assert np.array_equal(ours.values, SENS[name]), file
    assert lines(tmp_path / f"log_{case}") == lines(REF / f"log_{case}")


@pytest.mark.parametrize("case", CURSOR)
def test_the_keys_do_what_iraf_s_do(case, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    shutil.copy(REF / "cursor" / case, tmp_path / "cursor")
    result = get_task("sensfunc").run(
        standards=REF / "std_mix", sensitivity=f"i_{case}", apertures="1",
        extinction=str(DATA / "kpnoextinct.dat"), interactive=True, answer="YES",
        cursor=tmp_path / "cursor", logfile=f"log_{case}", newextinction=f"ext_{case}")
    name = f"i_{case}_0001"
    ours = flux.SensSpectrum.read(tmp_path / f"i_{case}.0001.fits")
    assert np.array_equal(ours.values, SENS[name])
    assert lines(tmp_path / f"log_{case}") == lines(REF / f"log_{case}")
    stats = REF / f"st_{case}"
    if stats.exists():
        assert lines(tmp_path / f"st_{case}") == lines(stats)
    if case == "ext":
        assert (tmp_path / "ext_ext").read_text() == (REF / "ext_ext").read_text()
    if case in ("ext", "ecancel"):
        assert any("Significance =  1.7 sigma" in line for line in result.data["text"])
    if case == "del":
        assert "ms3.fits - Delete p(oint), s(tar), or w(avelength):" in result.data["text"]


GRAPHS = sorted(path.stem for path in (REF / "graphs").glob("*.json"))
GKI_UNITS = 2.5


def _on_screen(window, x, y):
    wx1, wx2, wy1, wy2, sx1, sx2, sy1, sy2 = window
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    return np.c_[(sx1 + (x - wx1) / (wx2 - wx1) * (sx2 - sx1)) * 32767,
                 (sy1 + (y - wy1) / (wy2 - wy1) * (sy2 - sy1)) * 32767]


def _on_plot(window, point) -> bool:
    return (window[4] * 32767 - 1 <= point[0] <= window[5] * 32767 + 1
            and window[6] * 32767 - 1 <= point[1] <= window[7] * 32767 + 1)


def _write_standards(where) -> None:
    """The spectra of std_two, as harwood has them, for the graphs i and l."""
    from astropy.io import fits
    from iraf_field import standard_cases

    from harwood.core import spectrum

    for name in ("std1", "std2"):
        ((number, w1, dw, counts, _),), header = standard_cases()[name]
        n = counts.size
        aperture = spectrum.Aperture(number=number, centre=1.0, low=-1.0, high=1.0,
                                     pixel=np.arange(1.0, n + 1.0), flux=counts.astype(float),
                                     wavelength=w1 + dw * np.arange(n), title="")
        spectrum.write(where / f"{name}.fits", [aperture], header=fits.Header(header))


@pytest.mark.parametrize("case", GRAPHS)
def test_the_graphs_are_iraf_s(case, tmp_path, monkeypatch):
    """IRAF's metacode (iraf_sensfunc/run_graphs.py): every frame's windows, titles
    and curves, and every mark in the order drawn, through the same cursor file. IRAF
    clips what falls outside a window, and what d and u put over a frame comes after."""
    from harwood.core import sensfunc as sf

    drawn = []

    class Recording(sf.Session):
        def draw(self):
            super().draw()
            drawn.append((self.title(), self.drawn()))

    monkeypatch.setattr(sf, "Session", Recording)
    monkeypatch.chdir(tmp_path)
    shutil.copy(REF / "cursor" / case, tmp_path / "cursor")
    standards = "std_two" if case.startswith("im") else "std_mix"
    if case.startswith("im"):
        _write_standards(tmp_path)
    get_task("sensfunc").run(
        standards=REF / standards, sensitivity=f"i_{case}", apertures="1",
        extinction=str(DATA / "kpnoextinct.dat"), interactive=True, answer="YES",
        cursor=tmp_path / "cursor", logfile=f"log_{case}", newextinction=f"ext_{case}")
    # The frames of icfit, in e, have no title of sensfunc's.
    frames = [frame for frame in json.loads((REF / "graphs" / f"{case}.json").read_text())
              if any(text.startswith("Aperture=") for text in frame["text"])]
    assert len(frames) == len(drawn)
    for number, (frame, (title, graphs)) in enumerate(zip(frames, drawn, strict=True)):
        where = f"frame {number}"
        assert title in frame["text"], where
        lines = np.array([p for _, points in frame["lines"] for p in points]).reshape(-1, 2)
        assert sum(style == 2 for style, _ in frame["lines"]) == sum(
            dashed for graph in graphs for _, _, dashed in graph.lines), where
        marks = iter(frame["marks"])
        for graph, window in zip(graphs, frame["windows"], strict=False):
            assert graph.title in frame["text"], where
            width, height = abs(window[1] - window[0]), abs(window[3] - window[2])
            assert [float(v) for v in graph.window[:2]] == pytest.approx(
                window[:2], abs=1e-4 * width), where
            assert [float(v) for v in graph.window[2:]] == pytest.approx(
                window[2:4], abs=1e-4 * height), where
            for kind, x, y in graph.marks:
                point = _on_screen(window, [x], [y])[0]
                if _on_plot(window, point):
                    theirs = next(marks)
                    assert theirs[0] == kind, where
                    assert np.abs(point - theirs[1:3]).max() <= GKI_UNITS, where
            for x, y, _ in graph.lines:
                for point in _on_screen(window, x, y):
                    if _on_plot(window, point):
                        assert np.abs(lines - point).max(axis=1).min() <= GKI_UNITS, where
            if graph.widths is not None:
                x, y, width = graph.widths
                for edge in (x - width / 2, x + width / 2):
                    for point in _on_screen(window, edge, y):
                        if _on_plot(window, point):
                            nearest = np.abs(lines - point).max(axis=1).min()
                            assert nearest <= 2 * GKI_UNITS, where
            if graph.bars is not None:
                x, y, error = graph.bars
                for end in (y - error / 2, y + error / 2):
                    for point in _on_screen(window, x, end):
                        assert np.abs(lines - point).max(axis=1).min() <= GKI_UNITS, where


def test_the_panel_draws_every_graph_there_is():
    pytest.importorskip("PySide6")
    pytest.importorskip("pyqtgraph")
    from harwood.core import sensfunc as sf
    from harwood.viewer import _qt
    from harwood.viewer.sensfunc_panel import SensfuncPanel

    _qt.application()
    stars = sf.read_iraf((REF / "std_mix").read_text(), [1], False)
    curve = flux.read_extinction(DATA / "kpnoextinct.dat")
    session = sf.Session(stars, curve, "spline3", 6, graphs="sace")
    panel = SensfuncPanel(session)
    assert [plot.titleLabel.text for plot in panel.plots] == [
        "Sensitivity vs Wavelength", "Sensitivity Residuals vs Airmass",
        "Composite Points vs Wavelength", "Extinction vs Wavelength"]
    assert panel.status.text().startswith("Aperture=1  Function=spline3  Order=6")


def test_harwood_s_own_std_file_gives_iraf_s_fit_to_its_precision(tmp_path, monkeypatch):
    """harwood's CSV keeps seven figures where IRAF's file keeps five, so a
    fit from it is IRAF's to the figures IRAF's file holds."""
    from harwood.core import stdfile

    monkeypatch.chdir(tmp_path)
    measured = stdfile.read(REF / "std_ms3").measurements
    stdfile.append(tmp_path / "std", measured, str(DATA / "kpnoextinct.dat"))
    get_task("sensfunc").run(standards=tmp_path / "std", sensitivity="sb")
    for ap in (1, 2, 3):
        ours = flux.SensSpectrum.read(tmp_path / f"sb.{ap:04d}.fits")
        assert np.allclose(ours.values, SENS[f"sb_{ap:04d}"], atol=2e-4)


def test_the_window_drives_the_same_session():
    """The panel hands the keys to the session under the mouse's graph: a
    point deleted in the residuals graph, a shift, a refit, q."""
    from harwood.core import ranges
    from harwood.core import sensfunc as sf
    from harwood.viewer._qt import application
    from harwood.viewer.sensfunc_panel import SensfuncPanel

    application()
    curve = flux.read_extinction(str(DATA / "kpnoextinct.dat"))
    stars = sf.read_iraf((REF / "std_mix").read_text(), ranges.parse("1"), False)
    session = sf.Session(stars, curve, "spline3", 6)
    panel = SensfuncPanel(session)
    assert len(panel.plots) == 2
    ms3 = next(s for s in stars if s.image == "ms3.fits")
    j = int(np.flatnonzero(ms3.waves == np.float32(4120))[0])
    panel._cursor = (4120.0, float(ms3.sens[j] - ms3.fit[j]), 2)
    panel.key("d")
    panel.key("p")
    assert ms3.wts[j] == 0
    panel.key("s")
    panel.key("f")
    assert session.shifted and session.npts == 38
    done = []
    panel.finished.connect(lambda: done.append(True))
    panel.key("q")
    assert done
