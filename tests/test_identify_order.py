"""A line given the wrong wavelength makes a solution that is not monotonic. IRAF
says "Coordinate solution is not monotonic" (identify) and then, in reidentify,
"Too many features lost" at every step; harwood says which lines are out of order
and what to do -- beside IRAF's text in identify, and before starting in
reidentify."""

from __future__ import annotations

import pytest

from harwood.core.identify import Feature, disorder, disorder_advice

# exercise 7.2 as it went: 6677.28 and 6666.36 Angstroms swapped
SWAPPED = [Feature(58.55, 6416.31, 6416.31), Feature(86.63, 6483.08, 6483.08),
           Feature(319.28, 6677.28, 6677.28), Feature(386.67, 6666.36, 6666.36),
           Feature(492.14, 6937.66, 6937.66)]


def test_the_pair_out_of_order_is_found():
    (a, b), = disorder(SWAPPED)
    assert (a.user, b.user) == (6677.28, 6666.36)
    said = disorder_advice(SWAPPED, "angstroms")
    assert "6677.28 Å at pixel 319.3 comes before 6666.36 Å at pixel 386.7" in said
    assert "d to delete" in said and "m to mark" in said


def test_lines_in_order_and_blue_to_the_right_are_fine():
    assert not disorder(SWAPPED[:3] + SWAPPED[4:])
    falling = [Feature(f.pix, -f.user, -f.user) for f in SWAPPED[:3]]
    assert not disorder(falling) and disorder_advice(falling) == ""
    deleted = [Feature(f.pix, f.fit, f.user, 0.0 if f.user == 6666.36 else 1.0)
               for f in SWAPPED]
    assert not disorder(deleted)              # a deleted line does not count


RECORD = """# harwood dispersion solution.
task: identify
image: arc.fits
section: '[20,*]'
aperture: null
units: Angstroms
coordlist: ''
fit:
  function: spline3
  order: 2
  rms: 0.0
npts: 550
xrange:
- 1.0
- 550.0
niterate: 1
low_reject: 3.0
high_reject: 3.0
sample: '*'
grow: 0.0
shift: 0.0

features:
#  pixel      wavelength    fitted       used
  - [   58.546665,    6416.31000000,    6416.31000000, true]
  - [   86.632401,    6483.08250000,    6483.08250000, true]
  - [  319.284790,    6677.28170000,    6677.28170000, true]
  - [  386.665039,    6666.35880000,    6666.35880000, true]
  - [  492.141357,    6937.66420000,    6937.66420000, true]
"""


def test_reidentify_stops_on_a_reference_out_of_order(tmp_path, monkeypatch):
    import numpy as np
    from astropy.io import fits

    import harwood

    monkeypatch.chdir(tmp_path)
    fits.writeto("arc.fits", np.full((550, 40), 100.0, np.float32), overwrite=True)
    (tmp_path / "database").mkdir()
    (tmp_path / "database" / "idarc").write_text(RECORD, encoding="utf-8")
    with pytest.raises(Exception, match=r"not monotonic.*6677\.28 Å at pixel 319\.3"):
        harwood.run("reidentify", reference="arc.fits", images="arc.fits",
                    section="column 20", interactive="no")


def test_too_few_lines_for_the_order_is_said():
    """Her 7.2 arc: 5 lines and spline3 order 2 (5 coefficients) go through every
    line, RMS 0 -- the advice says to press l; with the list found it says nothing."""
    import numpy as np

    from harwood.core import icfit
    from harwood.core.identify import too_few_lines

    pix = np.array([58.5, 86.6, 254.8, 398.9, 506.9])
    lam = np.array([6384.72, 6416.31, 6604.85, 6766.61, 6888.17])
    spec = icfit.FitSpec(function="spline3", order=2)
    fit = icfit.fit(pix, lam, spec, xrange=(1, 550))
    said = too_few_lines(fit)
    assert "5" in said and " l " in said
    many = np.linspace(30, 530, 29)
    fit = icfit.fit(many, 6320 + 1.12 * many, icfit.FitSpec(function="spline3", order=1),
                    xrange=(1, 550))
    assert too_few_lines(fit) == ""


def test_fitcoords_says_five_lines_cannot_carry_yorder_6(tmp_path, monkeypatch):
    import numpy as np

    from harwood.core import fitcoords as fc
    from harwood.core import igsfit
    from harwood.tasks.longslit import _too_few_lines

    rows = []
    for col in range(10, 250, 10):
        for k, lam in enumerate((6384.7, 6416.3, 6604.9, 6766.6, 6888.2)):
            rows.append((k, col, 50 + 100 * k, lam, 1.0, 0.0, 0.0))
    pts = np.array(rows, dtype=np.float32).T
    coords = fc.Coords(2, 1, 258, 1, 550, pts, "angstroms")
    assert _too_few_lines(coords, igsfit.Common("chebyshev", 6, 6)) is not None
    assert _too_few_lines(coords, igsfit.Common("chebyshev", 6, 4)) is None
