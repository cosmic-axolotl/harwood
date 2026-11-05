"""sapertures against IRAF 2.18's: what it says on both streams, and each aperture's
beam, dispersion, redshift, limits and title after it.

tests/data/iraf_sapertures holds the cases and the table (sa_cases.py) and what IRAF
said and left in the world coordinates (sa.json), on ../iraf_slist's spectrum. The
wavelengths are held to the dispersion IRAF's attributes describe, over 1+z.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest
from astropy.io import fits

from harwood.core import spectrum
from harwood.core.registry import get_task

HERE = Path(__file__).parent / "data"
GOLD = json.loads((HERE / "iraf_sapertures" / "sa.json").read_text(encoding="utf-8"))
sys.path.insert(0, str(HERE / "iraf_slist"))
sys.path.insert(0, str(HERE / "iraf_sapertures"))
from sa_cases import CASES, TABLE  # noqa: E402
from sl_inputs import HEADER, N, spectra  # noqa: E402


@pytest.fixture
def start(tmp_path, monkeypatch):
    header = fits.Header()
    for key, value in HEADER.items():
        header[key] = value
    pixel = np.arange(1.0, N + 1.0)
    spectrum.write(tmp_path / "in.fits",
                   [spectrum.Aperture(number=ap, centre=10.0 * ap, low=-3.0, high=3.0,
                                      pixel=pixel, flux=flux.astype(float),
                                      wavelength=wave, title=title, beam=beam)
                    for ap, beam, _, _, _, title, flux, wave in spectra("in")],
                   header=header)
    (tmp_path / "apid.dat").write_text(TABLE, encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    return tmp_path


@pytest.mark.parametrize("case", list(CASES))
def test_sapertures_changes_as_iraf_does(start, case):
    result = get_task("sapertures").run(input="in", verbose=True, **CASES[case])
    said = GOLD[case]["said"]
    theirs_errors = [line for line in said if line.startswith("Negative")]
    assert result.data["errors"] == theirs_errors
    assert result.data["stdout"] == [line for line in said if line not in theirs_errors]
    pixel = np.arange(N, dtype=float)
    for line, aperture in enumerate(spectrum.read("in.fits"), 1):
        theirs = GOLD[case]["lines"][str(line)]
        assert (aperture.beam or aperture.number) == theirs["beam"]
        assert aperture.centre + aperture.low == pytest.approx(theirs["aplow"])
        assert aperture.centre + aperture.high == pytest.approx(theirs["aphigh"])
        assert (aperture.title or HEADER["OBJECT"]) == theirs["title"]
        if theirs["dtype"] == -1:
            assert aperture.wavelength is None
            continue
        world = theirs["w1"] + theirs["dw"] * pixel
        world = 10 ** world if theirs["dtype"] == 1 else world
        assert aperture.wavelength == pytest.approx(world / (1 + theirs["z"]), rel=1e-11)
