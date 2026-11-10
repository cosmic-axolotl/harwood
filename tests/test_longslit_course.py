"""response, illumination and ccdproc on the course's long-slit exercises, against
the files IRAF 2.18 wrote for them (6.1 and 7.1 of the IRAF course, in
~/iraf/exercises or HARWOOD_TEST_DATA): skipped when they are not there.

6.1: Resp is response's defaults (spline3, order 1 -- the session that made it
kept them), Illum the course's fit (chebyshev 2, high_reject 1, grow 2,
niterate 2), SuperFlat their product. 7.1: flat_response is spline3 order 5 with
3-sigma rejection; flat_illum three bins, 1:222 109:220 330:440, the first two
overlapping, fitted with orders 8, 8 and 6 by hand; and ccdproc's flat and
illumination correction of the galaxy and the lamp with them is IRAF's btpf, the
flat's and the illumination's means computed again as IRAF computes them.
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path

import numpy as np
import pytest

import harwood
from harwood.core import images

DATA = Path(os.environ.get("HARWOOD_TEST_DATA", "~/iraf/exercises")).expanduser()
SPEC, LAMP = DATA / "spec", DATA / "7.1"


def _same(a: str, b: Path) -> bool:
    mine, _ = images.read(a)
    theirs, _ = images.read(str(b))
    return np.array_equal(mine, theirs)


@pytest.mark.skipif(not (SPEC / "Resp.fits").exists(), reason=f"no 6.1 files under {SPEC}")
def test_the_course_61_response_illumination_and_superflat(tmp_path, monkeypatch):
    for name in ("Flat", "SkyFlat"):
        shutil.copy(SPEC / f"{name}.fits", tmp_path)
    monkeypatch.chdir(tmp_path)
    harwood.run("response", calibration=["Flat.fits"], normalization=["Flat.fits"],
                response=["Resp.fits"], dispaxis=2)
    harwood.run("illumination", images=["SkyFlat.fits"], illuminations=["Illum.fits"],
                dispaxis=2, function="chebyshev", order=2, high_reject=1.0, grow=2.0,
                niterate=2)
    harwood.run("imarith", operand1="Resp.fits", op="*", operand2="Illum.fits",
                result="SuperFlat.fits")
    for name in ("Resp", "Illum", "SuperFlat"):
        assert _same(f"{name}.fits", SPEC / f"{name}.fits"), name


@pytest.mark.skipif(not (LAMP / "flat_illum.fits").exists(),
                    reason=f"no 7.1 files under {LAMP}")
def test_the_course_71_flat_illumination_and_ccdproc(tmp_path, monkeypatch):
    for name in ("flat_btp", "ngc660_1800_btp", "ngc660_900_btp", "cuar_btp"):
        shutil.copy(LAMP / f"{name}.fits", tmp_path)
    monkeypatch.chdir(tmp_path)
    fit = dict(function="spline3", low_reject=3.0, high_reject=3.0, dispaxis=2)
    harwood.run("response", calibration=["flat_btp.fits"], normalization=["flat_btp.fits"],
                response=["flat_response.fits"], order=5, **fit)
    Path("bins.cur").write_text("q\n" + "".join(f":order {o}\nf\nq\n" for o in (8, 8, 6)),
                                encoding="utf-8")
    harwood.run("illumination", images=["flat_response.fits"],
                illuminations=["flat_illum.fits"], bins="1:222 109:220 330:440",
                cursor="bins.cur", **fit)
    assert _same("flat_response.fits", LAMP / "flat_response.fits")
    assert _same("flat_illum.fits", LAMP / "flat_illum.fits")
    frames = ("ngc660_1800", "ngc660_900", "cuar")
    harwood.run("ccdproc", input=[",".join(f"{f}_btp.fits" for f in frames)],
                output=",".join(f"{f}_btpf.fits" for f in frames), overscan=False,
                trim=False, flatcor=True, flat="flat_response.fits", illumcor=True,
                illum="flat_illum.fits")
    for f in frames:
        assert _same(f"{f}_btpf.fits", LAMP / f"{f}_btpf.fits"), f
