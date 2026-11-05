"""rspectext against IRAF 2.18's: each dispersion type, a calibrated flux, a header
in front (wspectext's, from IRAF), and an output that exists.

tests/data/iraf_rspectext holds the text and the cases (rs_cases.py), the header text
IRAF's wspectext wrote (hdr.txt), and what IRAF's rspectext made of each (rs.json):
listpixels' world coordinates and pixels, and the title and flags. A log dispersion's
world coordinate is its log10.
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pytest
from astropy.io import fits

from harwood.core import spectrum
from harwood.core.registry import get_task

HERE = Path(__file__).parent / "data" / "iraf_rspectext"
GOLD = json.loads((HERE / "rs.json").read_text(encoding="utf-8"))
sys.path.insert(0, str(HERE))
from rs_cases import CASES, text  # noqa: E402


@pytest.fixture
def start(tmp_path, monkeypatch):
    (tmp_path / "spec.txt").write_text(text(), encoding="utf-8")
    shutil.copy(HERE / "hdr.txt", tmp_path / "hdr.txt")
    monkeypatch.chdir(tmp_path)
    return tmp_path


@pytest.mark.parametrize("case", list(CASES))
def test_rspectext_reads_as_iraf_does(start, case):
    source, params = CASES[case]
    get_task("rspectext").run(input=source, output=f"o_{case}", **params)
    theirs = GOLD[case]
    aperture, = spectrum.read(f"o_{case}.fits")
    header = fits.getheader(f"o_{case}.fits")
    world = np.asarray(theirs["world"])
    if params.get("dtype") == "log":
        world = 10 ** world
    if aperture.wavelength is None:
        assert world == pytest.approx(aperture.pixel)
    else:
        assert aperture.wavelength == pytest.approx(world, rel=1e-9)
    # listpixels prints a pixel with %g's seven figures.
    assert aperture.flux == pytest.approx(theirs["flux"], rel=5e-7)
    title = theirs["keys"][0].strip('"')
    assert header.get("OBJECT", "") == title
    flags = theirs["keys"][1:]
    if params.get("flux"):
        assert header["CA-FLAG"] == 0 and header["EX-FLAG"] == 0


def test_an_output_that_exists_stops_it(start):
    get_task("rspectext").run(input="spec.txt", output="o_linear", **CASES["linear"][1])
    with pytest.raises(FileExistsError,
                       match=r"Operation would overwrite existing image \(o_linear.fits\)"):
        get_task("rspectext").run(input="spec.txt", output="o_linear")
    assert "o_linear.fits" in GOLD["exists"]["said"][-1]
