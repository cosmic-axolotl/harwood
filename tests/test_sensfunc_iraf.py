"""sensfunc against IRAF 2.18's own output, bit for bit.

tests/data/iraf_sensfunc holds IRAF std files (two stars at airmasses 1.15
and 1.85; three apertures of one frame; both together; two band sets) and
what IRAF's sensfunc made of them: the sensitivity spectra, the logfile,
the :stats and :vstats files and the revised extinction table. In batch:
every aperture, apertures chosen, ignoreaps, four functions. From a cursor
file, on the three stars of aperture 1: points, stars and wavelengths
deleted and restored, moved, added and reweighted, the stars shifted and
made a composite, a residual extinction applied and cancelled, the original
data back, the order and function changed, and three graphs.

IRAF reads the new weight (w) and the answer to the residual extinction
(e) from the terminal; the cursor files here give them as the next line's
':' text, as harwood reads them. IRAF's icfit takes the keys between e and
its q; harwood fits that curve as IRAF does by default.
"""

from __future__ import annotations

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
