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
import re
from pathlib import Path

import numpy as np
import pytest
from astropy.io import fits
from iraf_field import crowded_field, field

import harwood
from harwood.core import catalog, daofind, daofindsession, photsession
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


KEYS = Path(__file__).parent / "data" / "iraf_daofindkeys"
PARS = dict(fwhmpsf=2.5, sigma=6.0, threshold=4.0, nsigma=1.5, ratio=1.0, theta=0.0,
            sharplo=0.2, sharphi=1.0, roundlo=-1.0, roundhi=1.0, epadu=1.0, readnoise=0.0,
            scale=1.0)
ROW = re.compile(r"\s*-?\d+\.\d\d\s+-?\d+\.\d\d\s+-?\d+\.\d{3}\s")


def _session(case):
    crowded = case.startswith("c_")
    session = daofindsession.Session(
        (crowded_field() if crowded else field()).astype(np.float32),
        "crowded.fits" if crowded else "field.fits", f"{case}.coo", dict(PARS))
    for x, y, key in photsession.read_commands(KEYS / f"{case}.cur"):
        session.colon(key) if key.startswith(":") else session.key(key, x, y)
        if session.done:
            break
    return session


def _coo(path):
    rows = [line.split()[:6] for line in path.read_text().splitlines()
            if line.strip() and not line.startswith("#")]
    return np.array(rows, dtype=float).reshape(-1, 6)


@pytest.mark.parametrize("case", sorted(path.stem for path in KEYS.glob("*.cur")))
def test_daofind_by_the_keys_says_and_finds_what_iraf_does(case):
    """Each case is a cursor file IRAF 2.18 ran (iraf_daofindkeys/run_iraf.py):
    every line printed, each star in them, and each list written."""
    session = _session(case)
    theirs = (KEYS / f"{case}.out").read_text().replace("\a", "").splitlines()
    assert (KEYS / f"{case}.err").read_text() == ""
    assert len(session.said) == len(theirs)
    for ours, line in zip(session.said, theirs, strict=True):
        if ROW.match(line):
            assert np.allclose(np.array(ours.split(), dtype=float),
                               np.array(line.split(), dtype=float), atol=1.1e-3, rtol=0)
        elif line.startswith("Output file:"):
            # IRAF names the third list after the image.
            assert ours.startswith("Output file:")
        elif "    output: " in line:
            assert "    output: " in ours
        else:
            assert ours == line
    for i, (_, found, _) in enumerate(session.written):
        written = KEYS / (f"{case}.coo" if i == 0 else f"{case}.coo.{i + 1}")
        ours = np.column_stack([found[name] for name in COLUMNS])
        assert ours.shape == _coo(written).shape
        assert np.abs(ours - _coo(written)).max() < TOLERANCE


def test_finding_again_on_a_kernel_counts_one_pixel_too_few():
    """What makes IRAF's second find differ from its first: the sharpness moves,
    nothing else does, and a kernel parameter puts it back."""
    data = field().astype(np.float32)
    first = daofind.find(data, 2.5, 6.0)
    again = daofind.find(data, 2.5, 6.0, refit=True)
    assert (first["sharpness"] != again["sharpness"]).all()
    for name in ("x", "y", "mag", "sround", "ground"):
        assert (first[name] == again[name]).all()
    session = _session("refitcut")
    assert [refit for _, _, refit in session.written] == [True]


def _keys(tmp_path, *lines, **params):
    cursor = tmp_path / "keys.cur"
    cursor.write_text("".join(f"0 0 1 {line}\n" for line in lines))
    image = tmp_path / "field.fits"
    fits.writeto(image, field(), overwrite=True)
    return harwood.run("daofind", image=str(image), output=tmp_path / "out.coo",
                       icommands=str(cursor), sigma=6.0, fwhmpsf=2.5, **params)


def test_the_task_writes_a_file_for_each_list(tmp_path):
    result = _keys(tmp_path, "\\040", ":threshold 6", "\\040", "q")
    assert [path.name for path in result.outputs] == ["out.coo", "out.coo.2"]
    assert len(catalog.read(result.outputs[0])) == 6
    assert any("one pixel too few" in str(warning) for warning in result.warnings)
    assert any("Output file:" in line for line in result.data["text"])


def test_the_task_says_when_only_f_was_pressed(tmp_path):
    result = _keys(tmp_path, "f", "q")
    assert result.outputs == []
    assert any("nothing was written" in str(warning) for warning in result.warnings)
    assert not (tmp_path / "out.coo").exists()


def test_the_keys_need_a_file_that_is_there(tmp_path):
    image = tmp_path / "field.fits"
    fits.writeto(image, field())
    with pytest.raises(FileNotFoundError):
        harwood.run("daofind", image=str(image), icommands="nosuch.cur", sigma=6.0)


def test_what_is_not_ported_says_so():
    session = daofindsession.Session(field().astype(np.float32), "f", "o", dict(PARS))
    session.key("v")
    session.colon(":datamin 0")
    assert "not ported" in session.said[0] and "not ported" in session.said[1]
    session.colon(":sh")
    assert "Kernel Parameters" in session.said
