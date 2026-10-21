"""dispcor on multispec spectra, against IRAF's own dispcor.

tests/data/iraf_dispcor_multispec holds what IRAF 2.18.1's dispcor wrote for
the stars of tests/iraf_field.multispec_night(), three apertures each, with
the solutions IRAF's reidentify left for arc3 and arc3b: one grid per
aperture and one for all (samedisp), grids taken over several spectra
(global), two arcs weighted, an aperture no arc has (ignoreaps), a
wavelength table, no linearizing, the sinc interpolant, a log grid,
listonly, a spectrum already dispersion corrected, and the error IRAF stops
with when an aperture has no reference.

The log has to be IRAF's line for line and the grids the same. The pixels
are IRAF's but for a few: its inverse of a two-arc solution stops at 0.0001
px, so where a pixel edge lands a step the other way the integral moves by
1e-4 of a pixel's worth.
"""

import json
import shutil
from pathlib import Path

import numpy as np
import pytest
from astropy.io import fits
from iraf_field import multispec_night

from harwood.core import spectrum
from harwood.core.registry import get_task

DATA = Path(__file__).parent / "data"
REFERENCE = json.loads((DATA / "iraf_dispcor_multispec" / "cases.json").read_text())
SPECTRA = np.load(DATA / "iraf_dispcor_multispec" / "spectra.npz")


@pytest.fixture
def night(tmp_path, monkeypatch):
    stars, _ = multispec_night()
    for name, (apertures, header) in stars.items():
        spectrum.write(tmp_path / f"{name}.fits",
                       [spectrum.Aperture(number=ap, centre=centre, low=-5.0, high=5.0,
                                          pixel=np.arange(1.0, flux.size + 1.0),
                                          flux=flux.astype(float), title=header["OBJECT"])
                        for ap, centre, flux in apertures],
                       header=fits.Header(header))
    shutil.copytree(DATA / "iraf_dispcor_multispec" / "database", tmp_path / "database")
    (tmp_path / "tbl").write_text(REFERENCE["table_text"])
    monkeypatch.chdir(tmp_path)
    return tmp_path


def run(case):
    entry = REFERENCE["cases"][case]
    if entry["input"] == "d1":
        run("d1")
    return get_task("dispcor").run(input=entry["input"], output=entry["outputs"],
                                   check=False, **entry["params"])


@pytest.mark.parametrize("case", list(REFERENCE["cases"]))
def test_dispcor_corrects_multispec_as_iraf_does(night, case):
    entry = REFERENCE["cases"][case]
    result = run(case)
    assert result.data["text"] == entry["log"]

    for output, grids in entry["grids"].items():
        for aperture in spectrum.read(f"{output}.fits"):
            dcflag, w1, dw, nw = grids[str(aperture.number)]
            assert aperture.flux.size == nw
            if entry["params"].get("linearize", True):
                wave = np.log10(aperture.wavelength) if dcflag == 1 else aperture.wavelength
                assert wave[0] == pytest.approx(w1, rel=1e-9)
                assert np.diff(wave).mean() == pytest.approx(dw, rel=1e-9)
            theirs = SPECTRA[f"{output}_{aperture.number}"].astype(float)
            off = np.abs(aperture.flux - theirs) / np.maximum(np.abs(theirs), 1.0)
            assert (off > 1e-5).sum() <= 3
            assert off.max() < 2e-4


def test_an_aperture_without_a_reference_stops_as_iraf_stops(night):
    case = REFERENCE["d13"]
    with pytest.raises(RuntimeError, match=case["error"]):
        get_task("dispcor").run(input=case["input"], output="d13", check=False,
                                **case["params"])
