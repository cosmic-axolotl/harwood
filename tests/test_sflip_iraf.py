"""sflip against IRAF 2.18's: each line's pixels, wavelengths and positions after
either flip, both, neither, and in place.

tests/data/iraf_sflip holds the cases and IRAF's runs of them (run.py) on the spectrum
of ../iraf_dopcor, with listpixels' world and physical coordinates of each line
(sflip.json): those are harwood's wavelength and pixel columns.
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
GOLD = json.loads((HERE / "iraf_sflip" / "sflip.json").read_text(encoding="utf-8"))
sys.path.insert(0, str(HERE / "iraf_dopcor"))
sys.path.insert(0, str(HERE / "iraf_sflip"))
from dop_inputs import HEADER, N, spectra  # noqa: E402
from sf_cases import CASES  # noqa: E402


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
def test_sflip_flips_as_iraf_does(start, case):
    params = dict(CASES[case])
    target = params.pop("output", "o")
    get_task("sflip").run(input="in", output=target, **params)
    for aperture in spectrum.read(f"{target}.fits"):
        theirs = GOLD[case][str(aperture.number)]
        assert aperture.wavelength == pytest.approx(theirs["world"], rel=1e-11)
        assert aperture.pixel == pytest.approx(theirs["physical"], abs=1e-9)
        # listpixels prints a pixel with %g's seven figures.
        assert aperture.flux == pytest.approx(theirs["flux"], rel=6e-7)
