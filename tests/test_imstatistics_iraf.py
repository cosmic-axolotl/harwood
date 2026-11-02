"""imstatistics against IRAF's, field by field.

tests/data/iraf_imstatistics.txt is what IRAF 2.18.1 printed (format=no)
for every field -- npix, mean, midpt, mode, stddev, skew, kurtosis, min,
max -- on three images and a section, plain, clipped twice over, between
limits, and with a coarser histogram. midpt and mode are IRAF's histogram
estimates, not a true median, so they only agree if the histogram is IRAF's.

The raw M92 frame needs the tutorial data; the two synthetic frames do not.
"""

import os
from pathlib import Path

import numpy as np
import pytest
from iraf_field import combine_stack, field

from harwood.tasks.imutil import imstat

DATA = Path(__file__).parent / "data" / "iraf_imstatistics.txt"
TAPE = Path(os.environ.get("HARWOOD_TEST_DATA", "~/iraf/exercises")).expanduser() \
    / "phot" / "fm920004"
FIELDS = ["npix", "mean", "midpt", "mode", "stddev", "skew", "kurtosis", "min", "max"]
CASES = {"all": {}, "clip": dict(nclip=3),
         "clip1": dict(nclip=5, lsigma=2.0, usigma=2.5),
         "limits": dict(lower=50.0, upper=3000.0), "bin": dict(binwidth=0.5)}


def _image(name):
    from astropy.io import fits

    if name.startswith("raw"):
        if not TAPE.exists():
            pytest.skip("IRAF tutorial data not found")
        data = fits.getdata(TAPE).astype(float)
        if "[" in name:
            data = data[49:400, 99:200]          # [100:200,50:400]
        return data
    if name == "field.fits":
        return field()
    return combine_stack()[3][0]


def _rows():
    for line in DATA.read_text(encoding="utf-8").splitlines():
        image, case, values = line.split("|")
        yield image, case, [float(v) for v in values.split()[1:]]


@pytest.mark.parametrize("image, case, iraf", list(_rows()),
                         ids=lambda v: v if isinstance(v, str) else "")
def test_imstatistics_prints_what_iraf_prints(image, case, iraf):
    ours = imstat(_image(image), FIELDS, **CASES[case])
    for name, theirs in zip(FIELDS, iraf, strict=True):
        assert ours[name] == pytest.approx(theirs, rel=2e-6, abs=1e-6), name


def test_a_constant_image_has_no_spread_and_no_histogram():
    stats = imstat(np.full((5, 5), 7.0), FIELDS)
    assert stats["stddev"] == 0.0 and stats["mean"] == 7.0
    assert np.isnan(stats["midpt"]) and np.isnan(stats["skew"])
