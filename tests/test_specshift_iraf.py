"""specshift against IRAF 2.18's: what it says and where it leaves each aperture.

tests/data/iraf_specshift holds the cases and IRAF's runs of them (run.py) on the
spectrum of ../iraf_dopcor (three apertures with linear wavelengths), one of them
after a dopcor, and what they left (specshift.json). The said starts of a
dispersion are double %g's, and IRAF's dtoc prints 4010 as 4009.999999999999:
iraffmt does the same, and the lines are held to IRAF's to the character.
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
GOLD = json.loads((HERE / "iraf_specshift" / "specshift.json").read_text(encoding="utf-8"))
sys.path.insert(0, str(HERE / "iraf_dopcor"))
sys.path.insert(0, str(HERE / "iraf_specshift"))
from dop_inputs import HEADER, N, spectra  # noqa: E402
from ss_cases import CASES  # noqa: E402


@pytest.fixture
def start(tmp_path, monkeypatch):
    header = fits.Header()
    for key, value in HEADER.items():
        header[key] = value
    pixel = np.arange(1.0, N + 1.0)
    spectrum.write(tmp_path / "in.fits",
                   [spectrum.Aperture(number=ap, centre=10.0 * ap, low=-3.0, high=3.0,
                                      pixel=pixel, flux=flux.astype(float),
                                      wavelength=w1 + dw * (pixel - 1))
                    for ap, w1, dw, flux in spectra()], header=header)
    monkeypatch.chdir(tmp_path)
    return tmp_path


@pytest.mark.parametrize("case", list(CASES))
def test_specshift_shifts_as_iraf_does(start, case):
    z, params = CASES[case]
    if z is not None:
        get_task("dopcor").run(input="in", output="in", redshift=str(z))
    params = {k: (v == "yes") if v in ("yes", "no") else v for k, v in params.items()}
    result = get_task("specshift").run(spectra="in", **params)
    theirs = GOLD[case]
    assert result.data["text"] == theirs["said"]
    pixel = np.arange(N, dtype=float)
    for aperture in spectrum.read("in.fits"):
        iraf_s = theirs["apertures"][str(aperture.number)]
        wave = (iraf_s["w1"] + iraf_s["dw"] * pixel) / (1 + iraf_s["z"])
        assert aperture.wavelength == pytest.approx(wave, rel=1e-12)
        assert aperture.redshift == pytest.approx(iraf_s["z"], rel=1e-9)
