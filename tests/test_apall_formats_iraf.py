"""apall's IRAF formats -- multispec, echelle, onedspec, strip -- against IRAF's own files.

tests/data/iraf_apall_formats.json and .npz hold every file IRAF 2.18.1's
apall wrote in each format, with and without the extra bands, with
subapertures, along either dispersion axis, from a trimmed frame and from one
with a wavelength-like WCS of its own. harwood must write the same files, by
the same names, with the same header cards in the same order -- the frame's
own, the BANDIDs and APNUMs, and IRAF's equispec world coordinates -- and the
same pixels: to single precision, and a strip, an interpolation across the
profile at the aperture's edges, to 2e-5 of its brightest pixel (the edges
agree with IRAF's to the seventh figure of its database).
"""

import json
from pathlib import Path

import numpy as np
import pytest
from astropy.io import fits
from iraf_field import spectra_frame

from harwood.core.registry import get_task

DATA = Path(__file__).parent / "data"
REFERENCE = json.loads((DATA / "iraf_apall_formats.json").read_text(encoding="utf-8"))
PIXELS = np.load(DATA / "iraf_apall_formats.npz")
SKIP = {"SIMPLE", "ORIGIN", "DATE", "IRAF-TLM", "", "HARWOOD", "HRWDDATE", "HISTORY"}


def _frame(name):
    """The frames the .json describes."""
    data = spectra_frame()
    if name == "A":
        return data, {"DISPAXIS": 2, "OBJECT": "two stars", "EXPTIME": 300.0, "GAIN": 2.0}
    if name == "T":
        return data.T.copy(), {"DISPAXIS": 1, "OBJECT": "two stars", "EXPTIME": 300.0}
    if name == "L":
        return data, {"DISPAXIS": 2, "OBJECT": "two stars", "EXPTIME": 300.0,
                      "LTV1": -10.0, "LTV2": -20.0, "LTM1_1": 1.0, "LTM2_2": 1.0}
    return data, {"DISPAXIS": 2, "OBJECT": "two stars", "CTYPE1": "LINEAR",
                  "CTYPE2": "LINEAR", "CRVAL1": 1.0, "CRPIX1": 1.0, "CDELT1": 1.0,
                  "CRVAL2": 4000.0, "CRPIX2": 1.0, "CDELT2": 2.0, "CD1_1": 1.0,
                  "CD2_2": 2.0, "WCSDIM": 2}


@pytest.mark.parametrize("case", list(REFERENCE["cases"]))
def test_irafs_formats_are_written_as_iraf_writes_them(tmp_path, monkeypatch, case):
    entry = REFERENCE["cases"][case]
    image = entry["image"]
    data, cards = _frame(image)
    fits.writeto(tmp_path / f"{image}.fits", data, fits.Header(cards))
    monkeypatch.chdir(tmp_path)
    params = {"nfind": 2, **entry["params"]}
    result = get_task("apall").run(input=image, database=tmp_path, **params)
    written = {Path(p).name: Path(p) for p in result.outputs if str(p).endswith(".fits")}
    assert sorted(written) == sorted(entry["files"])

    for name, theirs in entry["files"].items():
        header = fits.getheader(written[name])
        mine = [[c.keyword, c.value] for c in header.cards if c.keyword not in SKIP]
        assert mine == theirs, name
        iraf = PIXELS[f"{case}__{name}"].astype(float)
        ours = fits.getdata(written[name]).astype(float)
        assert ours.shape == iraf.shape
        if params["format"] == "strip":
            assert np.max(np.abs(ours - iraf)) / np.max(np.abs(iraf)) < 2e-5, name
        else:
            assert np.max(np.abs(ours - iraf) / np.maximum(np.abs(iraf), 1.0)) < 2e-5, name


def test_strips_of_weighted_subapertures_are_refused(tmp_path):
    """IRAF builds every part's strip from the first part's spectrum."""
    fits.writeto(tmp_path / "A.fits", spectra_frame(), fits.Header({"DISPAXIS": 2}))
    with pytest.raises(ValueError, match="first part"):
        get_task("apall").run(input=str(tmp_path / "A.fits"), output=tmp_path / "A",
                              nfind=2, database=tmp_path, format="strip", nsubaps=2,
                              weights="variance")
