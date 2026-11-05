"""sbands against IRAF 2.18's, line for line: fluxes normalized and not, magnitudes,
a band with a filter, one with one continuum and one with two, one narrower than a
pixel, one that starts before the spectrum, a line of continua written without ids,
and some apertures without the header.

tests/data/iraf_sbands holds the band and filter files and the cases (sb_cases.py)
and what IRAF wrote (sb.json), on ../iraf_dopcor's spectrum. The banner's system
line is the run's own.
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
GOLD = json.loads((HERE / "iraf_sbands" / "sb.json").read_text(encoding="utf-8"))
sys.path.insert(0, str(HERE / "iraf_dopcor"))
sys.path.insert(0, str(HERE / "iraf_sbands"))
from dop_inputs import HEADER, N, spectra  # noqa: E402
from sb_cases import BANDS, CASES, FILTER  # noqa: E402


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
    (tmp_path / "bands.dat").write_text(BANDS, encoding="utf-8")
    (tmp_path / "filt.dat").write_text(FILTER, encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    return tmp_path


def _masked(lines):
    return ["# SBANDS: (system)" if line.startswith("# SBANDS: ") else line
            for line in lines]


@pytest.mark.parametrize("case", list(CASES))
def test_sbands_measures_as_iraf_does(start, case):
    get_task("sbands").run(input="in", output=f"{case}.out", bands="bands.dat",
                           **CASES[case])
    ours = Path(f"{case}.out").read_text(encoding="utf-8").splitlines()
    assert _masked(ours) == _masked(GOLD[case])
