"""autoidentify (and identify's automatic path) against IRAF's own output.

tests/data/iraf_autoidentify.json holds what IRAF 2.18.1's autoidentify wrote
to its database for tests/iraf_field.arc_spectrum() under seven sets of
parameters -- one of them finding no solution -- and for the real comparison
arc of the IRAF spectroscopy exercise, when it is installed. Every feature
must come out the same: the same lines, centred to IRAF's printed
precision, the same fit, and the same ones rejected.
"""

import json
import os
from pathlib import Path

import numpy as np
import pytest
from astropy.io import fits
from iraf_field import arc_spectrum

from harwood.core import arc, database
from harwood.core.registry import get_task

DATA = Path(__file__).parent / "data"
REFERENCE = json.loads((DATA / "iraf_autoidentify.json").read_text())
SPEC = Path(os.environ.get("HARWOOD_TEST_DATA", "~/iraf/exercises")).expanduser() / "spec"


def compare(solution, iraf):
    ours = np.column_stack([solution.pixel, solution.fitted, solution.wavelength,
                            solution.used.astype(float)])
    theirs = np.array(iraf, dtype=float)
    assert ours.shape == theirs.shape, "a different set of lines"
    assert np.abs(ours[:, 0] - theirs[:, 0]).max() <= 0.0051      # IRAF prints %.2f
    assert np.abs(ours[:, 1] - theirs[:, 1]).max() < 1e-5
    assert np.abs(ours[:, 2] - theirs[:, 2]).max() < 1e-5
    assert (ours[:, 3] == theirs[:, 3]).all(), "other lines rejected"


@pytest.mark.parametrize("case", list(REFERENCE["synthetic"]))
def test_autoidentify_identifies_what_iraf_identifies(tmp_path, case):
    params = dict(REFERENCE["synthetic"][case]["params"])
    image = params.pop("image")
    fits.writeto(tmp_path / "arc.fits", arc_spectrum(flip=image == "arcflip"))
    run = dict(image=str(tmp_path / "arc.fits"), coordlist=str(DATA / "arc_lines.dat"),
               database=tmp_path, **params)
    expected = REFERENCE["synthetic"][case]["features"]
    if expected is None:
        with pytest.raises(RuntimeError, match="No solution found"):
            get_task("autoidentify").run(**run)
        return
    get_task("autoidentify").run(**run)
    compare(database.read(str(tmp_path / "arc.fits"), tmp_path), expected)


@pytest.mark.parametrize("case", list(REFERENCE["course"]))
def test_autoidentify_on_the_course_arc(tmp_path, case):
    if not (SPEC / "fsp0018").exists():
        pytest.skip(f"IRAF tutorial spectra not found under {SPEC}")
    if not any(d.is_dir() for d in arc.IRAF_LINELISTS):
        pytest.skip("IRAF line lists are not installed")
    (tmp_path / "arc18.fits").write_bytes((SPEC / "fsp0018").read_bytes())
    get_task("autoidentify").run(image=str(tmp_path / "arc18.fits"),
                                 coordlist="linelists$thorium.dat", database=tmp_path,
                                 **REFERENCE["course"][case]["params"])
    compare(database.read(str(tmp_path / "arc18.fits"), tmp_path),
            REFERENCE["course"][case]["features"])


def test_the_course_arc_is_not_where_its_object_name_says(tmp_path):
    """'comp 6707' is the grating setting; the lamp's lines are at 4445-4680 A.
    Searching at 6707 finds nothing, and the message says what to check."""
    if not (SPEC / "fsp0018").exists():
        pytest.skip(f"IRAF tutorial spectra not found under {SPEC}")
    if not any(d.is_dir() for d in arc.IRAF_LINELISTS):
        pytest.skip("IRAF line lists are not installed")
    (tmp_path / "arc18.fits").write_bytes((SPEC / "fsp0018").read_bytes())
    with pytest.raises(RuntimeError, match="No solution found") as raised:
        get_task("autoidentify").run(image=str(tmp_path / "arc18.fits"),
                                     section="middle column", crval=6707.0, cdelt=-0.15,
                                     coordlist="linelists$thorium.dat", database=tmp_path)
    assert "grating" in raised.value.fix


def test_section_middle_line_is_a_spectrum_along_x():
    from harwood.core.identify import arc_vector

    frame = np.arange(5 * 7, dtype=np.float32).reshape(5, 7)
    vector, where = arc_vector(frame, "middle line", nsum=1)
    np.testing.assert_array_equal(vector, frame[2])
    assert where == "[*,3]"
    vector, where = arc_vector(frame, "middle column", nsum=3)
    np.testing.assert_array_equal(vector, frame[:, 2:5].sum(axis=1))
    assert where == "[4,*]"
