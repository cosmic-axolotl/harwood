"""daofind against IRAF's own output.

tests/data/iraf_daofind.json holds what IRAF 2.18.1's apphot.daofind wrote
for the two synthetic fields of tests/iraf_field.py -- the six-star field
and a crowded one with close pairs, stars over the edges, cosmic rays, a
galaxy, a hot column and a gradient -- under several sets of parameters, and
for the raw M92 frame of the phot exercise when it is installed.

The star lists must agree star for star, in IRAF's order, and every column
to the three decimals IRAF writes; three values in the 3300 go past the
rounding, by at most 1.4e-4.
"""

import json
import os
from pathlib import Path

import numpy as np
import pytest
from astropy.io import fits
from iraf_field import crowded_field, field

from harwood.core import catalog, daofind
from harwood.core.registry import get_task

REFERENCE = json.loads((Path(__file__).parent / "data" / "iraf_daofind.json").read_text())
REFERENCE.pop("_comment")
COLUMNS = ("x", "y", "mag", "sharpness", "sround", "ground")
#: Three decimals in the file, plus IRAF summing the marginals in single
#: precision: on a detection at the threshold (mag -0.02 in c_lenient) the
#: centre moves by a further 1.4e-4.
TOLERANCE = 1e-3

PHOT = Path(os.environ.get("HARWOOD_TEST_DATA", "~/iraf/exercises")).expanduser() / "phot"


def image(name):
    if name == "field":
        return field()
    if name == "crowded":
        return crowded_field()
    tape = PHOT / "fm920004"
    if not tape.exists():
        pytest.skip(f"IRAF tutorial data not found under {PHOT}")
    return fits.getdata(tape)


def iraf(case):
    return np.array([[float(v) for v in row.split()] for row in REFERENCE[case]["rows"]])


@pytest.mark.parametrize("case", list(REFERENCE))
def test_daofind_finds_what_iraf_finds(case):
    params = dict(REFERENCE[case]["params"])
    fwhmpsf, sigma = params.pop("fwhmpsf"), params.pop("sigma")
    found = daofind.find(image(REFERENCE[case]["image"]), fwhmpsf, sigma, **params)
    ours = np.column_stack([found[name] for name in COLUMNS])
    theirs = iraf(case)
    assert ours.shape == theirs.shape
    assert np.abs(ours - theirs).max() < TOLERANCE


def test_the_task_writes_iraf_s_stars_brightest_first(tmp_path):
    """The task over the crowded field: the same stars, reordered by
    magnitude, with nothing dropped at the edge."""
    path = tmp_path / "crowded.fits"
    fits.writeto(path, crowded_field())
    result = get_task("daofind").run(image=str(path), output=tmp_path / "c.coo",
                                     fwhmpsf=3.06, sigma=6.0)
    written = catalog.read(tmp_path / "c.coo")
    assert list(written.colnames) == ["id", *COLUMNS]
    assert np.all(np.diff(np.asarray(written["mag"])) >= 0)
    assert result.data["dropped_at_border"] == 0

    theirs = iraf("c_default")
    ours = np.column_stack([np.asarray(written[name]) for name in COLUMNS])
    ours = ours[np.lexsort((ours[:, 0], ours[:, 1]))]
    theirs = theirs[np.lexsort((theirs[:, 0], theirs[:, 1]))]
    assert np.abs(ours - theirs).max() < TOLERANCE
    # Five stars sit on the edges of this field; IRAF keeps them.
    assert (ours[:, 0] < 3).any() and (ours[:, 1] > 157).any()


def test_border_drops_what_is_near_the_edge(tmp_path):
    path = tmp_path / "crowded.fits"
    fits.writeto(path, crowded_field())
    result = get_task("daofind").run(image=str(path), output=tmp_path / "c.coo",
                                     fwhmpsf=3.06, sigma=6.0, border=5)
    written = catalog.read(tmp_path / "c.coo")
    x, y = np.asarray(written["x"]), np.asarray(written["y"])
    assert x.min() > 5.5 and y.min() > 5.5 and x.max() < 175.5 and y.max() < 155.5
    assert result.data["dropped_at_border"] == len(iraf("c_default")) - len(written)
    assert result.data["dropped_at_border"] > 0


def test_sigma_is_measured_when_not_given(tmp_path):
    """IRAF, with sigma INDEF, detects at a threshold of zero. harwood
    measures the noise instead -- the field's is 5 -- and finds the six."""
    path = tmp_path / "field.fits"
    fits.writeto(path, field())
    result = get_task("daofind").run(image=str(path), output=tmp_path / "f.coo")
    assert result.data["sigma"] == pytest.approx(5.0, rel=0.05)
    assert result.data["sources"] == 6
