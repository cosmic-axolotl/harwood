"""apall's subapertures (nsubaps) against IRAF's own.

tests/data/iraf_apall_subaps.json and .npz hold what IRAF 2.18.1's apall wrote
cutting each of tests/iraf_field.spectra_frame()'s two apertures across the
slit into two or three parts: a plain sum, a fitted, median and average
background, both dispersion axes, variance weights and cleaning. Each part is
a spectrum of its own, numbered as IRAF numbers it (aperture n's part p is
n + 1000 (p - 1)), and every band of every part must come out the same.

To 2e-5 rather than apall's 5e-6: harwood's aperture limits agree with IRAF's
to the seventh figure, its database's, and a boundary between two parts sits
on the peak of the profile, where a few millionths of a pixel move light from
one part to the next.

One difference is deliberate. With variance weights and a gain other than 1,
IRAF works in photons and divides the result back by the gain over the first
part only (apextract.x: adivkr over nb points, not nb times nsubaps), so its
other parts come out in photons, the gain times the counts of the first.
harwood divides them all: the parts of one aperture are in the same units.
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
REFERENCE = json.loads((DATA / "iraf_apall_subaps.json").read_text(encoding="utf-8"))
SPECTRA = np.load(DATA / "iraf_apall_subaps.npz")
BANDS = {"spectrum": "flux", "raw": "raw", "background": "sky", "sigma": "error"}


@pytest.mark.parametrize("case", list(REFERENCE["cases"]))
def test_subapertures_are_irafs(tmp_path, case):
    entry = REFERENCE["cases"][case]
    image = entry["image"]
    data = spectra_frame() if image == "A" else spectra_frame().T.copy()
    fits.writeto(tmp_path / f"{image}.fits", data,
                 fits.Header({"DISPAXIS": 2 if image == "A" else 1}))
    params = entry["params"]
    get_task("apall").run(input=str(tmp_path / f"{image}.fits"),
                          output=tmp_path / "out.ms.fits", nfind=2, database=tmp_path,
                          **params)
    ours = spectrum.read(tmp_path / "out.ms.fits")

    assert [a.number for a in ours] == [int(card.split()[0]) for card in entry["apnum"]]
    for aperture, card in zip(ours, entry["apnum"], strict=True):
        low, high = (float(value) for value in card.split()[2:4])
        assert aperture.centre + aperture.low == pytest.approx(low, abs=0.006)
        assert aperture.centre + aperture.high == pytest.approx(high, abs=0.006)

    with fits.open(tmp_path / "out.ms.fits") as hdul:
        assert [hdu.header["SUBAP"] for hdu in hdul[1:]] == \
            [a.number // 1000 + 1 for a in ours], "SUBAP says which part, as IRAF's"

    nsubaps = params["nsubaps"]
    weighted = params.get("weights") == "variance" or params.get("clean")
    gain = params.get("gain", 1.0) if weighted else 1.0
    for key in (k for k in SPECTRA.files if k.startswith(case + "__")):
        theirs = SPECTRA[key].astype(float)
        theirs = np.array([line / (gain if index % nsubaps else 1.0)
                           for index, line in enumerate(theirs)])
        mine = np.array([getattr(a, BANDS[key.split("__")[1]]) for a in ours], dtype=float)
        assert np.max(np.abs(mine - theirs) / np.maximum(np.abs(theirs), 1.0)) < 2e-5, key


def test_the_parts_add_up_to_the_whole(tmp_path):
    fits.writeto(tmp_path / "A.fits", spectra_frame(), fits.Header({"DISPAXIS": 2}))
    get_task("apall").run(input=str(tmp_path / "A.fits"), output=tmp_path / "w.fits",
                          nfind=2, database=tmp_path / "w")
    get_task("apall").run(input=str(tmp_path / "A.fits"), output=tmp_path / "p.fits",
                          nfind=2, database=tmp_path / "p", nsubaps=3)
    one = spectrum.read(tmp_path / "w.fits")
    parts = spectrum.read(tmp_path / "p.fits")
    for k, aperture in enumerate(one):
        summed = sum(part.flux for part in parts[3 * k:3 * k + 3])
        assert np.allclose(summed, aperture.flux, rtol=1e-5)
