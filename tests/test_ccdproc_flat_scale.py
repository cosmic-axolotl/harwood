"""How ccdproc scales by a flat and an illumination that are ready made, as IRAF's
set_flat, set_illum and ccdmean do: nothing raised to minreplace when dividing --
a normalized flat has half its pixels below 1 -- and the CCDMEAN believed only when
it is not older than the image (a CCDMEANT copied from another image, as response
and illumination copy their input's, is: the mean is computed again).
"""

from __future__ import annotations

import time

import numpy as np
from astropy.io import fits

import harwood
from harwood.core import ccdsections
from harwood.tasks.ccdred import _real_card, iraf_clock


def _frames(where, flat_cards: dict):
    y, x = np.indices((20, 30))
    obj = (1000 + 5 * x + 3 * y).astype(np.float32)
    flat = (1 + 0.3 * np.sin(x / 3.0) * np.cos(y / 4.0)).astype(np.float32)
    h = fits.Header({"IMAGETYP": "object", "CCDPROC": "done"})
    fits.PrimaryHDU(obj, h).writeto(where / "obj.fits")
    fh = fits.Header({"IMAGETYP": "flat", "CCDPROC": "done", **flat_cards})
    fits.PrimaryHDU(flat, fh).writeto(where / "flat.fits")
    return obj, flat


def _run(where):
    harwood.run("ccdproc", input=[str(where / "obj.fits")], output=str(where / "out.fits"),
                overscan=False, trim=False, zerocor=False, darkcor=False, flatcor=True,
                flat=str(where / "flat.fits"))
    from harwood.core import images

    return images.read(str(where / "out.fits"))[0]


def test_a_normalized_flat_is_divided_as_it_is(tmp_path):
    now = iraf_clock(time.time()) + 5
    obj, flat = _frames(tmp_path, {"CCDMEAN": 1.0, "CCDMEANT": now})
    assert (flat < 1).mean() > 0.3
    out = _run(tmp_path)
    assert np.array_equal(out, ((obj * np.float32(1.0)).astype(np.float32) / flat))


def test_an_out_of_date_ccdmean_is_computed_again(tmp_path):
    obj, flat = _frames(tmp_path, {"CCDMEAN": 1.0, "CCDMEANT": 1000,
                                   "IRAF-TLM": "2026-11-06T20:51:00"})
    # ccdmean writes the mean to the CCDMEAN card and set_flat reads the card back.
    mean = np.float32(_real_card(ccdsections.mean_of(flat)))
    out = _run(tmp_path)
    assert np.array_equal(out, ((obj * mean).astype(np.float32) / flat))


def test_a_ccdmean_without_ccdmeant_is_believed(tmp_path):
    obj, flat = _frames(tmp_path, {"CCDMEAN": 2.0})
    out = _run(tmp_path)
    assert np.array_equal(out, ((obj * np.float32(2.0)).astype(np.float32) / flat))
