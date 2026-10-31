"""imshift, imcentroid and imalign against IRAF's own output.

tests/data/iraf_imalign.json holds what IRAF 2.18.1 did with the frames of
tests/iraf_field.py: imshift over every interpolant, boundary and kind of
shift (as sha256 of the output -- harwood's must be the same bytes); what
imcentroid printed under five sets of parameters, one of them on the raw
M92 tapes; and imalign's output images. spline3's output is kept as numbers, in
iraf_imshift_spline3.npz; it is the same to the last bit.
"""

import hashlib
import json
import os
import re
from pathlib import Path

import numpy as np
import pytest
from astropy.io import fits
from iraf_field import align_frames, combine_stack

from harwood.core import imcentroid as ic
from harwood.core.imshift import shift
from harwood.core.registry import get_task

DATA = Path(__file__).parent / "data"
REFERENCE = json.loads((DATA / "iraf_imalign.json").read_text())
SPLINE3 = np.load(DATA / "iraf_imshift_spline3.npz")
PHOT = Path(os.environ.get("HARWOOD_TEST_DATA", "~/iraf/exercises")).expanduser() / "phot"
M92 = {"m92010": "fm920004", "m92011": "fm920005", "m92014": "fm920006",
       "m92015": "fm920007"}


def sha(array):
    return hashlib.sha256(np.asarray(array, ">f4").tobytes()).hexdigest()


# ------------------------------------------------------------------ imshift

@pytest.mark.parametrize("case", list(REFERENCE["imshift"]))
def test_imshift_is_iraf_s(case):
    interp, boundary, xs, ys, digest = REFERENCE["imshift"][case]
    ours = shift(combine_stack()[0][0], xs, ys, interp, boundary, -5.0)
    if digest is not None:
        assert sha(ours) == digest, "not the bytes IRAF wrote"
    elif case in SPLINE3:
        assert np.array_equal(ours, SPLINE3[case]), "not the numbers IRAF wrote"


def test_imshift_moves_the_wcs_as_iraf_does(tmp_path):
    """IRAF, for this header: CRPIX1 52.3, LTV1 2.3 after shifting by
    (2.3, -1.7); 47.3 and -2.7 after imcopy of [6:60,3:50]."""
    header = fits.Header({"CTYPE1": "RA---TAN", "CTYPE2": "DEC--TAN", "CRPIX1": 50.0,
                          "CRPIX2": 40.0, "CRVAL1": 10.0, "CRVAL2": 20.0})
    fits.writeto(tmp_path / "w.fits", np.zeros((60, 70), np.float32), header)
    get_task("imshift").run(input=[str(tmp_path / "w.fits")],
                            output=str(tmp_path / "ws.fits"), xshift=2.3, yshift=-1.7)
    moved = fits.getheader(tmp_path / "ws.fits")
    assert moved["CRPIX1"] == pytest.approx(52.3, abs=1e-6)
    assert moved["CRPIX2"] == pytest.approx(38.3, abs=1e-6)
    assert moved["LTV1"] == pytest.approx(2.3, abs=1e-6)

    from harwood.tasks.immatch import _cut_header
    _cut_header(moved, 6, 3)
    assert moved["CRPIX1"] == pytest.approx(47.3, abs=1e-6)
    assert moved["LTV2"] == pytest.approx(-3.7, abs=1e-6)


def test_imshift_wants_a_shift():
    with pytest.raises(ValueError, match="xshift"):
        get_task("imshift").run(input=["nothing.fits"], xshift=1.0)


# --------------------------------------------------------------- imcentroid

def printed(text):
    """IRAF's printed table: centres per image, reference centres, shifts."""
    stars, ref, shifts, trim, block = {}, {}, {}, None, None
    for line in text.splitlines():
        words = line.split()
        if line.startswith("#Coords"):
            block = "c"
        elif line.startswith("#Refcoords"):
            block = "r"
        elif line.startswith("#Shifts"):
            block = "s"
        elif line.startswith("#Trim"):
            trim = tuple(int(v) for v in re.findall(r"\d+", line))
        elif words and block in ("c", "r"):
            values = [float(v.strip("()")) for v in words[1:]]
            (stars if block == "c" else ref).setdefault(words[0], {})[int(values[4])] = \
                values[:4]
        elif words and block == "s":
            rest = line[line.index(words[0]) + len(words[0]):]
            values = re.findall(r"-?\d+\.\d+|INDEF|\d+", rest)
            shifts[words[0]] = values
    return stars, ref, shifts, trim


def frames_for(case):
    if case == "m92":
        if not all((PHOT / tape).exists() for tape in M92.values()):
            pytest.skip(f"IRAF tutorial data not found under {PHOT}")
        frames = [(f"{name}.fits", fits.getdata(PHOT / tape).astype(np.float32))
                  for name, tape in M92.items()]
        return frames, [tuple(c) for c in REFERENCE["m92_coords"]]
    images, _ = align_frames()
    return ([(f"a{k}.fits", image) for k, image in enumerate(images)],
            [tuple(c) for c in REFERENCE["coords"]])


@pytest.mark.parametrize("case", list(REFERENCE["imcentroid"]))
def test_imcentroid_is_iraf_s(case):
    run = REFERENCE["imcentroid"][case]
    params = dict(run["params"])
    shifts = REFERENCE["initial_shifts"] if params.pop("shifts", False) else None
    frames, coords = frames_for(case)
    result = ic.measure(frames, frames[0], coords, ic.Settings(**params), shifts)
    stars, ref, shift_rows, trim = printed(run["printed"])

    for image in [*result.images, result.reference]:
        table = ref[image.name] if image is result.reference else stars.get(image.name, {})
        ours = {j: [s.x, s.xerr, s.y, s.yerr]
                for j, s in enumerate(image.stars, 1) if not s.rejected}
        assert set(ours) == set(table), f"{image.name}: other stars accepted"
        for j, values in ours.items():
            assert np.abs(np.array(values) - table[j]).max() < 6e-4, (image.name, j)

    for image in result.images:
        row = shift_rows[image.name]
        assert [image.xshift, image.xerr, image.yshift, image.yerr] == \
            pytest.approx([float(v) for v in row[:4]], abs=6e-4)
        assert image.nstars == int(row[4])
        assert [image.xinternal, image.yinternal] == \
            pytest.approx([float(v) for v in row[5:7]], abs=6e-4)
    assert result.trim == trim

    for warning in re.findall(r"failed to converge near \((\d+),(\d+)\)", run["stderr"]):
        reasons = [s.reason for i in [*result.images, result.reference] for s in i.stars]
        assert f"failed to converge near ({warning[0]},{warning[1]})" in reasons


def test_imcentroid_task_prints_iraf_s_table(tmp_path):
    images, coords = align_frames()
    paths = []
    for k, image in enumerate(images):
        paths.append(str(tmp_path / f"a{k}.fits"))
        fits.writeto(paths[-1], image)
    (tmp_path / "a.coo").write_text("x y\n" + "".join(f"{x} {y}\n" for x, y in coords))
    result = get_task("imcentroid").run(input=paths, reference=paths[0],
                                        coords=tmp_path / "a.coo")
    _, _, theirs, trim = printed(REFERENCE["imcentroid"]["default"]["printed"])
    rows = {Path(row["image"]).name: row for row in result.data["shifts"]}
    for name, values in theirs.items():
        assert rows[name]["xshift"] == pytest.approx(float(values[0]), abs=6e-4)
        assert rows[name]["yshift"] == pytest.approx(float(values[2]), abs=6e-4)
    assert result.data["trim"] == "[{}:{},{}:{}]".format(*trim)
    assert any(line.startswith("#Shifts") for line in result.data["text"])
    assert any("failed to converge" in str(w) for w in result.warnings)


# ------------------------------------------------------------------ imalign

@pytest.mark.parametrize("interp", list(REFERENCE["imalign"]))
def test_imalign_writes_what_iraf_writes(tmp_path, interp):
    images, coords = align_frames()
    paths = []
    for k, image in enumerate(images):
        paths.append(str(tmp_path / f"a{k}.fits"))
        fits.writeto(paths[-1], image)
    (tmp_path / "a.coo").write_text("x y\n" + "".join(f"{x} {y}\n" for x, y in coords))
    expected = REFERENCE["imalign"][interp]
    result = get_task("imalign").run(input=paths, reference=paths[0],
                                     coords=tmp_path / "a.coo",
                                     output=str(tmp_path / "out"), interp_type=interp)
    assert result.data["trim"] == expected["trim"]
    for k, digest in enumerate(expected["sha256"]):
        ours = fits.getdata(tmp_path / "out" / f"a{k}.fits")
        assert list(ours.shape) == expected["shape"]
        if digest is not None:
            assert sha(ours) == digest, f"a{k}: not the bytes IRAF wrote"


def test_imalign_stops_when_an_image_has_no_shift(tmp_path):
    images, coords = align_frames()
    paths = []
    for k, image in enumerate([images[0], np.full_like(images[0], 100.0)]):
        paths.append(str(tmp_path / f"a{k}.fits"))
        fits.writeto(paths[-1], image)
    (tmp_path / "a.coo").write_text("x y\n" + "".join(f"{x} {y}\n" for x, y in coords))
    with pytest.raises(RuntimeError, match="a1.fits"):
        get_task("imalign").run(input=paths, reference=paths[0], coords=tmp_path / "a.coo",
                                output=str(tmp_path / "out"), shifts=None,
                                bigbox=11)
