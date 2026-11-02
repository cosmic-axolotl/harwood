"""apall against IRAF's own output.

tests/data/iraf_apall.json and .npz hold what IRAF 2.18.1's apall wrote for
tests/iraf_field.spectra_frame() -- two curved spectra on a sloping sky with
emission lines and cosmic rays -- under fourteen sets of parameters: every
background (none, average, median, minimum, fit, with and without skybox),
variance weighting with and without cleaning, both dispersion axes, a spline
trace, a median profile, fixed limits. The apertures IRAF found, sized and
wrote to its database, and every band of every spectrum, must come out of
harwood's apall the same, to single precision.
"""

import json
from pathlib import Path

import numpy as np
import pytest
from astropy.io import fits
from iraf_field import spectra_frame

from harwood.core import spectrum
from harwood.core.registry import get_task

DATA = Path(__file__).parent / "data"
REFERENCE = json.loads((DATA / "iraf_apall.json").read_text(encoding="utf-8"))
SPECTRA = np.load(DATA / "iraf_apall.npz")
BANDS = {"spectrum": "flux", "raw": "raw", "background": "sky", "sigma": "error"}


@pytest.mark.parametrize("case", list(REFERENCE["cases"]))
def test_apall_extracts_what_iraf_extracts(tmp_path, case):
    params = dict(REFERENCE["cases"][case])
    dispaxis = params.pop("dispaxis")
    data = spectra_frame() if dispaxis == 2 else spectra_frame().T.copy()
    fits.writeto(tmp_path / f"{case}.fits", data, fits.Header({"DISPAXIS": dispaxis}))
    get_task("apall").run(input=str(tmp_path / f"{case}.fits"),
                          output=tmp_path / f"{case}.ms.fits", nfind=2,
                          database=tmp_path, **params)
    ours = spectrum.read(tmp_path / f"{case}.ms.fits")

    axis = 0 if dispaxis == 2 else 1
    for aperture, iraf in zip(ours, REFERENCE["apertures"][case], strict=True):
        # IRAF's database keeps seven figures, harwood's header four decimals.
        assert aperture.centre == pytest.approx(iraf["center"][axis], abs=1e-4)
        assert aperture.low == pytest.approx(iraf["low"][axis], abs=1e-4)
        assert aperture.high == pytest.approx(iraf["high"][axis], abs=1e-4)

    bands = [key.split("__")[1] for key in SPECTRA.files if key.startswith(case + "__")]
    for band in bands:
        theirs = SPECTRA[f"{case}__{band}"].astype(float)
        mine = np.array([getattr(a, BANDS[band]) for a in ours], dtype=float)
        scale = np.maximum(np.abs(theirs), 1.0)
        assert np.max(np.abs(mine - theirs) / scale) < 5e-6, band
    for aperture in ours:
        for band, column in BANDS.items():
            if band not in bands:
                assert getattr(aperture, column) is None, \
                    f"IRAF writes no {band} band for this case"
