"""apall with apertures from another frame, against IRAF's own apall.

tests/data/iraf_apall_references.json and .npz hold what IRAF 2.18.1's
apall wrote taking the apertures of tests/iraf_field.spectra_frame() for an
arc with no star in it (arc_frame(), extracted as it is), and for the same
stars 1.3 px along the slit (spectra_frame(1.3)), recentred every way
ap_recenter does it: all by the median move, each by its own, by the
brightest only, by the moves of the apertures aprecenter names -- and then
resized and traced afresh. The RECENTER line of IRAF's log, the apertures
it wrote and every band it extracted must come out the same.
"""

import json
from pathlib import Path

import numpy as np
import pytest
from astropy.io import fits
from iraf_field import arc_frame, spectra_frame

from harwood.core import spectrum
from harwood.core.registry import get_task

DATA = Path(__file__).parent / "data"
REFERENCE = json.loads((DATA / "iraf_apall_references.json").read_text(encoding="utf-8"))
SPECTRA = np.load(DATA / "iraf_apall_references.npz")
BANDS = {"spectrum": "flux", "raw": "raw", "background": "sky", "sigma": "error"}


@pytest.mark.parametrize("case", list(REFERENCE["cases"]))
def test_apall_takes_apertures_from_a_reference_as_iraf_does(tmp_path, monkeypatch, case):
    entry = REFERENCE["cases"][case]
    header = fits.Header({"DISPAXIS": 2})
    fits.writeto(tmp_path / "A.fits", spectra_frame(), header)
    fits.writeto(tmp_path / "B.fits", spectra_frame(1.3), header)
    fits.writeto(tmp_path / "arc.fits", arc_frame(), header)
    monkeypatch.chdir(tmp_path)
    get_task("apall").run(input="A", output="A.ms.fits", nfind=2)

    image = entry["image"]
    result = get_task("apall").run(input=image, output=f"{image}.ms.fits", nfind=2,
                                   references="A", **entry["params"])
    assert result.data.get("recenter") == entry["log"]

    ours = spectrum.read(f"{image}.ms.fits")
    for aperture, (centre, low, high) in zip(ours, entry["apertures"], strict=True):
        assert aperture.centre == pytest.approx(centre, abs=1e-4)
        assert aperture.low == pytest.approx(low, abs=1e-4)
        assert aperture.high == pytest.approx(high, abs=1e-4)
    for key in (k for k in SPECTRA.files if k.startswith(case + "__")):
        theirs = SPECTRA[key].astype(float)
        mine = np.array([getattr(a, BANDS[key.split("__")[1]]) for a in ours], dtype=float)
        assert np.max(np.abs(mine - theirs) / np.maximum(np.abs(theirs), 1.0)) < 5e-6


def test_a_reference_with_no_apertures_says_which_to_run_first(tmp_path, monkeypatch):
    fits.writeto(tmp_path / "arc.fits", arc_frame(), fits.Header({"DISPAXIS": 2}))
    monkeypatch.chdir(tmp_path)
    with pytest.raises(FileNotFoundError) as raised:
        get_task("apall").run(input="arc", output="arc.ms.fits", references="A")
    assert "apall on A first" in raised.value.fix


def test_a_frame_that_has_apertures_reuses_them(tmp_path, monkeypatch):
    """As IRAF's apall: the second run starts from database/ap<frame> and
    recentres, rather than finding the spectra again. Recentring measures
    from the centre plus the trace at the middle line, and a straight trace
    of these curved spectra is 0.83 and -1.41 px off there, so even on the
    same frame the apertures move by the median, 0.29 -- as IRAF's do."""
    fits.writeto(tmp_path / "A.fits", spectra_frame(), fits.Header({"DISPAXIS": 2}))
    monkeypatch.chdir(tmp_path)
    get_task("apall").run(input="A", output="one.ms.fits", nfind=2)
    second = get_task("apall").run(input="A", output="two.ms.fits", nfind=1)
    assert second.data["apertures"] == 2, "the database's two, not a fresh find of one"
    assert second.data["recenter"] == "RECENTER  - 2 apertures shifted by 0.29 for A."
