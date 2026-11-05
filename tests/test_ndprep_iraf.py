"""ndprep against IRAF 2.18's: a filter's correction in linear wavelengths, in log10
with the flux conserved and not, and as frames of rows and of columns.

tests/data/iraf_ndprep holds the cases (nd_cases.py), on the CTIO filter curves harwood
carries, and what IRAF made (nd.json: the pixels' bits and the dispersion axis's world
coordinates).
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import pytest
from astropy.io import fits

from harwood.core import spectrum
from harwood.core.registry import get_task

HERE = Path(__file__).parent / "data" / "iraf_ndprep"
GOLD = json.loads((HERE / "nd.json").read_text(encoding="utf-8"))
sys.path.insert(0, str(HERE))
from nd_cases import CASES  # noqa: E402


@pytest.mark.parametrize("case", list(CASES))
def test_ndprep_makes_what_iraf_makes(tmp_path, monkeypatch, case):
    monkeypatch.chdir(tmp_path)
    get_task("ndprep").run(output=case, **CASES[case])
    theirs = np.array(GOLD[case]["bits"], np.uint32).view(np.float32).reshape(
        GOLD[case]["shape"])
    world = np.asarray(GOLD[case]["world"])
    if CASES[case].get("nspace"):
        mine = fits.getdata(f"{case}.fits")
        header = fits.getheader(f"{case}.fits")
        axis = CASES[case]["dispaxis"]
        n = mine.shape[2 - axis]
        ours = header[f"CRVAL{axis}"] + header[f"CDELT{axis}"] * np.arange(n)
        assert ours == pytest.approx(world, rel=1e-11)
        assert mine.shape == theirs.shape
        assert np.array_equal(mine, theirs)
        return
    aperture, = spectrum.read(f"{case}.fits")
    if CASES[case].get("logarithm"):
        world = 10 ** world
    assert aperture.wavelength == pytest.approx(world, rel=1e-10)
    mine = aperture.flux.astype(np.float32)
    # Linear, to the bit; resampled in log, a unit of the last place at most.
    assert mine == pytest.approx(theirs, rel=2.5e-7)
    assert math.isfinite(float(mine.sum()))
