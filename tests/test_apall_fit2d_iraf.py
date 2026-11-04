"""apall's fit2d profile (Marsh 1989) against IRAF's own.

tests/data/iraf_apall_fit2d.json and .npz hold what IRAF 2.18.1's apall
extracted with pfit=fit2d from tests/iraf_field.spectra_frame(), whose spectra
drift across the pixel columns so that the aperture starts at a different
pixel from line to line -- the case Marsh's overlapping polynomials are for:
with variance weights, a fitted background, cleaning, along either axis, and
with fewer, wider polynomials.

With polysep 0.7 and polyorder 4 the normal equations are well conditioned
and harwood's spectrum is IRAF's to 1e-4. With the defaults, 0.9 and 10, they
are a hundred unknowns at a condition number near 1e11, in single precision:
hfti's rank cut at 0.01 then falls among values that are rounding, and the
answer depends on the last bit of every input -- the aperture limits too,
which agree with IRAF's to the seventh figure. There harwood and IRAF agree
as well as either agrees with itself: to a fifth of the uncertainty of each
point at the median and three of it at worst, the total flux to 3e-5 and the
uncertainty itself to 1e-3.
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
REFERENCE = json.loads((DATA / "iraf_apall_fit2d.json").read_text(encoding="utf-8"))
SPECTRA = np.load(DATA / "iraf_apall_fit2d.npz")


@pytest.mark.parametrize("case", list(REFERENCE["cases"]))
def test_marshs_profile_extracts_what_iraf_extracts(tmp_path, case):
    entry = REFERENCE["cases"][case]
    image = entry["image"]
    data = spectra_frame() if image == "A" else spectra_frame().T.copy()
    fits.writeto(tmp_path / f"{image}.fits", data,
                 fits.Header({"DISPAXIS": 2 if image == "A" else 1}))
    get_task("apall").run(input=str(tmp_path / f"{image}.fits"),
                          output=tmp_path / "out.ms.fits", nfind=2, database=tmp_path,
                          **entry["params"])
    ours = spectrum.read(tmp_path / "out.ms.fits")
    flux = np.array([a.flux for a in ours])
    error = np.array([a.error for a in ours])
    raw = np.array([a.raw for a in ours])
    theirs, sigma = SPECTRA[f"{case}__spectrum"], SPECTRA[f"{case}__sigma"]

    assert np.max(np.abs(raw - SPECTRA[f"{case}__raw"]) / np.abs(SPECTRA[f"{case}__raw"])) \
        < 5e-6, "the unweighted sum does not depend on the profile"
    if "polyorder" in entry["params"]:
        assert np.max(np.abs(flux - theirs) / np.abs(theirs)) < 2e-4
        return
    z = np.abs(flux - theirs) / sigma
    assert np.median(z) < 0.3 and np.max(z) < 3.0
    assert np.all(np.abs(flux.sum(axis=1) / theirs.sum(axis=1) - 1) < 3e-5)
    assert np.all(np.abs(np.median(error / sigma, axis=1) - 1) < 1e-3)
