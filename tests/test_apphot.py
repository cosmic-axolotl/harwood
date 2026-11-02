"""apphot against IRAF: centring, sky fitting and aperture sums.

tests/iraf_field.py builds the pixels; tests/data/iraf_phot_<salgorithm>.txt
is what IRAF 2.18's own phot printed for them (txdump of the .mag), with the
defaults of every pset except epadu = 2 and apertures = 3,5,8, and each
star's cursor a pixel or so off -- so the centring has work to do.

Everything matches to the precision IRAF prints. That is the point of the
port: a magnitude from harwood is the magnitude IRAF gives.
"""

import numpy as np
import pytest
from iraf_field import CURSOR, field

from harwood.core import apphot

COLUMNS = ["xcenter", "ycenter", "xshift", "yshift", "xerr", "yerr", "cier",
           "msky", "stdev", "sskew", "nsky", "nsrej", "sier"]
APERTURES = [3.0, 5.0, 8.0]


def _iraf(salgorithm):
    from pathlib import Path

    path = Path(__file__).parent / "data" / f"iraf_phot_{salgorithm}.txt"
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        values = [float(v) if v != "INDEF" else float("nan") for v in line.split()]
        row = dict(zip(COLUMNS, values[:13], strict=True))
        n = len(APERTURES)
        for name, start in (("sum", 13), ("area", 13 + n), ("flux", 13 + 2 * n),
                            ("mag", 13 + 3 * n), ("merr", 13 + 4 * n),
                            ("pier", 13 + 5 * n)):
            row[name] = np.array(values[start:start + n])
        rows.append(row)
    return rows


@pytest.mark.parametrize("salgorithm", ["centroid", "mode", "median", "mean"])
def test_phot_prints_what_iraf_prints(salgorithm):
    data = field()
    dpars = apphot.DataPars(epadu=2.0)
    for (x, y), iraf in zip(CURSOR, _iraf(salgorithm), strict=True):
        c = apphot.center(data, x, y, apphot.CenterPars(), dpars)
        s = apphot.fitsky(data, c.x, c.y, apphot.SkyPars(salgorithm=salgorithm), dpars)
        a = apphot.apertures(data, c.x, c.y, APERTURES, s, 25.0, dpars)

        assert c.x == pytest.approx(iraf["xcenter"], abs=6e-4)
        assert c.y == pytest.approx(iraf["ycenter"], abs=6e-4)
        assert c.xshift == pytest.approx(iraf["xshift"], abs=6e-4)
        assert c.xerr == pytest.approx(iraf["xerr"], abs=6e-4)
        assert c.ier == iraf["cier"]

        assert s.value == pytest.approx(iraf["msky"], rel=1e-6)
        assert s.sigma == pytest.approx(iraf["stdev"], rel=1e-5)
        assert s.skew == pytest.approx(iraf["sskew"], rel=1e-4)
        assert s.nsky == iraf["nsky"]
        assert s.nreject == iraf["nsrej"]
        assert s.ier == iraf["sier"]

        assert a.sums == pytest.approx(iraf["sum"], rel=1e-6)
        assert a.areas == pytest.approx(iraf["area"], rel=1e-6)
        assert a.flux == pytest.approx(iraf["flux"], rel=1e-5)
        assert a.mag == pytest.approx(iraf["mag"], abs=6e-4)
        assert a.merr == pytest.approx(iraf["merr"], abs=6e-4)
        assert a.ier == iraf["pier"][0]


def test_a_star_off_the_box_is_flagged_and_kept():
    """maxshift is a flag, not a veto: IRAF keeps the centre it found and
    says that it moved too far, so the user can decide."""
    data = field()
    far = apphot.center(data, 42.5, 44.0, apphot.CenterPars(maxshift=0.5))
    assert far.ier == apphot.CTR_BADSHIFT
    assert far.x == pytest.approx(40.31, abs=0.05)


def test_algorithms_without_a_batch_meaning_fail_as_iraf_flags_them():
    """histplot and radplot without a graphics cursor, and file without a
    sky file, give IRAF's own errors (NoGraph 208, NoFile 209); an unknown
    algorithm is refused by name."""
    data = field()
    assert apphot.fitsky(data, 40, 45, apphot.SkyPars(salgorithm="histplot")).ier == 208
    assert apphot.fitsky(data, 40, 45, apphot.SkyPars(salgorithm="radplot")).ier == 208
    assert apphot.fitsky(data, 40, 45, apphot.SkyPars(salgorithm="file")).ier == 209
    with pytest.raises(ValueError) as raised:
        apphot.fitsky(data, 40, 45, apphot.SkyPars(salgorithm="guess"))
    assert "crosscor" in raised.value.fix
    with pytest.raises(ValueError) as raised:
        apphot.center(data, 40, 45, apphot.CenterPars(calgorithm="moffat"))
    assert "ofilter" in raised.value.fix


@pytest.mark.parametrize("case, datamax, maxshift", [("edge", 9000.0, 0.8),
                                                     ("flags", 5000.0, 0.6)])
def test_the_flags_and_the_threshold_are_iraf_s(case, datamax, maxshift):
    """The paths the defaults never take: a cthreshold with a sky sigma
    (ap_ctr1d rather than ap_mctr1d), a datamax two stars saturate, and a
    maxshift three of them exceed. IRAF keeps the centre it found and flags
    it (cier 107, 108); an aperture holding a saturated pixel gets no
    magnitude at all (pier 305)."""
    data = field()
    dpars = apphot.DataPars(epadu=2.0, sigma=5.0, datamax=datamax)
    cpars = apphot.CenterPars(cthreshold=1.0, maxshift=maxshift)
    for (x, y), iraf in zip(CURSOR, _iraf(case), strict=True):
        c = apphot.center(data, x, y, cpars, dpars)
        s = apphot.fitsky(data, c.x, c.y, apphot.SkyPars(salgorithm="mode"), dpars)
        a = apphot.apertures(data, c.x, c.y, APERTURES, s, 25.0, dpars)
        assert c.x == pytest.approx(iraf["xcenter"], abs=6e-4)
        assert c.y == pytest.approx(iraf["ycenter"], abs=6e-4)
        assert c.xerr == pytest.approx(iraf["xerr"], abs=6e-4)
        assert c.ier == iraf["cier"]
        assert s.value == pytest.approx(iraf["msky"], rel=1e-6)
        assert s.nreject == iraf["nsrej"]
        assert np.allclose(a.mag, iraf["mag"], atol=6e-4, equal_nan=True)
        assert a.ier == iraf["pier"][0]
