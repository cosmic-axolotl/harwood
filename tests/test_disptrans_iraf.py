"""disptrans against IRAF 2.18's: to nanometers, microns, inverse centimeters (fitted,
and resampled), a velocity, hertz (resampled), log Angstroms, air to vacuum and back,
and units IRAF does not know.

tests/data/iraf_disptrans holds the cases (dt_cases.py) and what IRAF said and left
(dt.json: each line's world coordinates and the pixels' bits), on ../iraf_dopcor's
spectrum. A log unit's world coordinate is its log10.
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
GOLD = json.loads((HERE / "iraf_disptrans" / "dt.json").read_text(encoding="utf-8"))
sys.path.insert(0, str(HERE / "iraf_dopcor"))
sys.path.insert(0, str(HERE / "iraf_disptrans"))
from dop_inputs import HEADER, N, spectra  # noqa: E402
from dt_cases import CASES  # noqa: E402


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
def test_disptrans_transforms_as_iraf_does(start, case):
    theirs = GOLD[case]
    if theirs["said"][0].startswith("ERROR"):
        with pytest.raises(ValueError, match="Cannot convert to specified units"):
            get_task("disptrans").run(input="in", output="o", **CASES[case])
        return
    result = get_task("disptrans").run(input="in", output="o", **CASES[case])
    assert result.data["text"] == theirs["said"]
    for line, aperture in enumerate(spectrum.read("o.fits"), 1):
        world = np.asarray(theirs[str(line)]["world"])
        assert aperture.wavelength == pytest.approx(world, rel=1e-11)
        assert aperture.units == CASES[case]["units"]
        iraf_s = np.array(theirs[str(line)]["bits"], np.uint32).view(np.float32)
        # Copied, to the bit; resampled, within a unit of the last place.
        assert aperture.flux.astype(np.float32) == pytest.approx(iraf_s, rel=2.5e-7)
