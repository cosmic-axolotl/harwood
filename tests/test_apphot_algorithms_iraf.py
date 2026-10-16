"""Every centring and sky algorithm of apphot against IRAF's own phot.

tests/data/iraf_apphot_algorithms.json holds what IRAF 2.18.1's phot
printed for the six stars of tests/iraf_field.field() and of dirty_field()
-- the same field with a hot pixel in each centring box and cosmic-ray
blobs and a faint neighbour in each sky annulus -- under 38 sets of
parameters: gauss and ofilter centring, the symmetry clean with both noise
models, and every sky algorithm with and without region growing,
percentile clips, histogram smoothing and an INDEF rejection limit.

Every column must agree to the precision IRAF prints: half a unit in the
last place of the centres and errors (three decimals) and of the sky's
seven significant figures. It does -- including IRAF's own quirks, such as
mean with rgrow counting only the last region it grows.
"""

import json
from pathlib import Path

import numpy as np
import pytest
from iraf_field import CURSOR, dirty_field, field

from harwood.core import apphot

REFERENCE = json.loads((Path(__file__).parent / "data"
                        / "iraf_apphot_algorithms.json").read_text())["cases"]
IMAGES = {"field": field, "dirty": dirty_field}


def _value(v):
    if v == "INDEF":
        return None
    return {"yes": True, "no": False}.get(v, v) if isinstance(v, str) else v


def _pars(case):
    def given(key):
        return {k: _value(v) for k, v in case.get(key, {}).items()}

    return (apphot.DataPars(epadu=2.0, **given("data")), apphot.CenterPars(**given("center")),
            apphot.SkyPars(**given("sky")))


def _same(ours, theirs, tolerance):
    if theirs is None:
        return np.isnan(ours)
    return abs(ours - theirs) <= tolerance


@pytest.mark.parametrize("case", list(REFERENCE))
def test_phot_matches_iraf(case):
    dpars, cpars, spars = _pars(REFERENCE[case])
    data = IMAGES[REFERENCE[case]["image"]]()
    for (x, y), row in zip(CURSOR, REFERENCE[case]["rows"], strict=True):
        c = apphot.center(data, x, y, cpars, dpars)
        s = apphot.fitsky(data, c.x, c.y, spars, dpars)
        a = apphot.apertures(data, c.x, c.y, [3.0, 5.0, 8.0], s, 25.0, dpars)
        xc, yc, _, _, xerr, yerr, cier, msky, stdev, sskew, nsky, nsrej, sier = row[:13]
        where = f"star at ({x}, {y})"
        assert c.ier == cier, where
        for ours, theirs in ((c.x, xc), (c.y, yc), (c.xerr, xerr), (c.yerr, yerr)):
            assert _same(ours, theirs, 5.01e-4), where
        assert (s.nsky, s.nreject, s.ier) == (nsky, nsrej, sier), where
        for ours, theirs in ((s.value, msky), (s.sigma, stdev), (s.skew, sskew)):
            assert _same(ours, theirs, 5.01e-7 * abs(theirs or 0) + 1e-6), where
        for ours, theirs in zip(a.mag, row[13 + 9:13 + 12], strict=True):
            assert _same(ours, theirs, 5.01e-4), where
