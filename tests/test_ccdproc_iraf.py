"""ccdproc against IRAF's own ccdproc, on the raw 1987 M92 frames.

tests/data/iraf_ccdproc_m92.json holds what IRAF 2.18 wrote, reducing the
tape files exactly as below: 300 sampled pixels of the bias, the flat and an
object frame, their CCDMEAN, and the line-by-line level of the bias under six
overscan fits (chebyshev, spline3 with rejection, naverage, a median of 5, a
sample with a gap, spline1). harwood has to land on the same numbers.

The tolerance is IRAF's own arithmetic: it fits and stores in single
precision, which leaves about 1e-3 counts on a bias level of 500, and it
sums CCDMEAN in single precision too (4e-6 of the mean).

Skipped without the tutorial data, like tests/test_m92.py.
"""

import json
import os
from pathlib import Path

import numpy as np
import pytest
from astropy.io import fits

from harwood.core.registry import get_task

DATA = Path(os.environ.get("HARWOOD_TEST_DATA", "~/iraf/exercises")).expanduser() / "phot"
REFERENCE = json.loads((Path(__file__).parent / "data" / "iraf_ccdproc_m92.json").read_text())

pytestmark = pytest.mark.skipif(not (DATA / "fm920001").exists(),
                                reason=f"IRAF tutorial data not found under {DATA}")

FIT = dict(function="legendre", order=1, niterate=1)
OVERSCAN = {"cheb3": dict(function="chebyshev", order=3, niterate=1),
            "spl3": dict(function="spline3", order=4, niterate=2,
                         low_reject=2.5, high_reject=2.5),
            "avg": dict(function="legendre", order=2, naverage=4, niterate=1),
            "med": dict(function="legendre", order=2, naverage=-5, niterate=0),
            "samp": dict(function="legendre", order=2, sample="20:250,300:480",
                         niterate=1),
            "spl1": dict(function="spline1", order=3, niterate=1)}


@pytest.fixture(scope="module")
def reduced(tmp_path_factory):
    work = tmp_path_factory.mktemp("ccdproc_iraf")
    for tape, name in (("fm920001", "bias"), ("fm920003", "flat"), ("fm920004", "obj")):
        (work / f"{name}.fits").write_bytes((DATA / tape).read_bytes())
    run = get_task("ccdproc").run
    run(input=[str(work / "bias.fits")], output=work / "Zero.fits",
        overscan=True, trim=True, **FIT)
    run(input=[str(work / "flat.fits")], output=work / "Flat.fits", overscan=True,
        trim=True, zerocor=True, zero=work / "Zero.fits", **FIT)
    run(input=[str(work / "obj.fits")], output=work / "Obj.fits", overscan=True,
        trim=True, zerocor=True, zero=work / "Zero.fits", flatcor=True,
        flat=work / "Flat.fits", **FIT)
    for name, fit in OVERSCAN.items():
        run(input=[str(work / "bias.fits")], output=work / f"Z_{name}.fits",
            overscan=True, trim=True, **fit)
    return work


@pytest.mark.parametrize("name, tolerance", [("Zero", 1e-4), ("Flat", 1e-3),
                                             ("Obj", 3e-4)])
def test_each_stage_is_what_iraf_wrote(reduced, name, tolerance):
    ours = fits.getdata(reduced / f"{name}.fits").astype(float)
    rows, cols = np.array(REFERENCE["pixels"]).T - 1
    theirs = np.array(REFERENCE[name]["values"])
    # The object is divided by the flat, so its tolerance is relative.
    scale = np.maximum(np.abs(theirs), 1.0) if name == "Obj" else 1.0
    assert np.max(np.abs(ours[rows, cols] - theirs) / scale) < tolerance
    if name != "Obj":
        assert fits.getheader(reduced / f"{name}.fits")["CCDMEAN"] == pytest.approx(
            REFERENCE[name]["ccdmean"], rel=1e-5, abs=1e-4)


def test_where_the_flat_is_below_minreplace_harwood_does_not_divide_by_it(reduced):
    """The one deliberate difference. The 1987 flat has 25 pixels below 1 --
    down to -4.4, on the chip's last line -- and its header does not say it
    is a flat, so IRAF processed it as an object and kept them; dividing by
    them puts values of 50,000 into the object frame, and into its CCDMEAN
    (48.86 against 48.61). harwood raises them to minreplace before dividing,
    which is what IRAF itself does to any image it knows is a flat."""
    flat = fits.getdata(reduced / "Flat.fits")
    obj = fits.getdata(reduced / "Obj.fits")
    assert (flat < 1).sum() == 25
    assert np.abs(obj).max() < 50000


@pytest.mark.parametrize("name", list(OVERSCAN))
def test_every_overscan_fit_is_iraf_s(reduced, name):
    ours = fits.getdata(reduced / f"Z_{name}.fits").astype(float).mean(axis=1)
    assert np.max(np.abs(ours - np.array(REFERENCE[f"overscan_{name}"]))) < 2e-3
