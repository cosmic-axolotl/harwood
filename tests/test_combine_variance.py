"""The variance imcombine and imarith write, against the scatter it claims.

Known truth, as section 12 asks for what IRAF never had: each frame is a
constant plus independent noise of a known variance, so every pixel of the
result is an independent draw, and the scatter over 40000 of them must be
what the VAR plane says. One run per case, no repetition needed.

The median is where it is easy to be wrong: its variance is the mean's times
an inefficiency that reaches pi/2 only for many frames. With three it is
1.35, and the test would fail by 16 per cent with the asymptote.
"""

import numpy as np
import pytest
from astropy.io import fits

from harwood.core import dq, mef
from harwood.core.registry import get_task

SHAPE = (200, 200)


def _frame(path, level, sigma, rng, exptime=1.0, quality=None):
    sci = level + rng.normal(0.0, sigma, SHAPE)
    header = fits.Header({"EXPTIME": exptime, "OBJECT": "flat field"})
    mef.write(path, mef.Image(header, [mef.Plane(sci, np.full(SHAPE, sigma ** 2),
                                                  quality)]))
    return path


def _ratio(path):
    plane = mef.read(path).plane
    return float(np.var(plane.sci) / np.mean(plane.var))


@pytest.mark.parametrize("combine", ["median", "lmedian", "average", "sum"])
def test_combining_three_frames_scatters_as_the_variance_says(tmp_path, combine):
    rng = np.random.default_rng(7)
    frames = [str(_frame(tmp_path / f"f{k}.fits", 100.0, sigma, rng))
              for k, sigma in enumerate((3.0, 4.0, 5.0))]
    get_task("imcombine").run(input=frames, output=tmp_path / "c.fits", combine=combine)
    # 40000 pixels: the sample variance is good to 0.7 per cent.
    assert _ratio(tmp_path / "c.fits") == pytest.approx(1.0, abs=0.03)


def test_the_median_of_three_is_not_the_asymptote(tmp_path):
    rng = np.random.default_rng(8)
    frames = [str(_frame(tmp_path / f"f{k}.fits", 100.0, 4.0, rng)) for k in range(3)]
    get_task("imcombine").run(input=frames, output=tmp_path / "c.fits", combine="median")
    plane = mef.read(tmp_path / "c.fits").plane
    assert np.mean(plane.var) == pytest.approx(1.3485 * 16.0 / 3, rel=1e-6)


def test_a_weighted_average_of_scaled_frames(tmp_path):
    """Exposures of 1, 2 and 4 seconds: scaled to the first by exposure and
    weighted by it, as IRAF does."""
    rng = np.random.default_rng(9)
    frames = [str(_frame(tmp_path / f"f{k}.fits", 100.0 * t, 3.0 * np.sqrt(t), rng, t))
              for k, t in enumerate((1.0, 2.0, 4.0))]
    get_task("imcombine").run(input=frames, output=tmp_path / "c.fits",
                              scale="exposure", weight="exposure")
    assert _ratio(tmp_path / "c.fits") == pytest.approx(1.0, abs=0.03)


def test_a_flagged_pixel_is_left_out_and_one_flagged_everywhere_says_so(tmp_path):
    rng = np.random.default_rng(10)
    frames = []
    for k in range(3):
        quality = dq.empty(SHAPE)
        quality[5, 5] = dq.SATURATED if k == 0 else 0      # one frame saturated there
        quality[9, 9] = dq.BAD_PIXEL | (dq.COSMIC_RAY if k else 0)   # all bad there
        path = _frame(tmp_path / f"f{k}.fits", 100.0, 2.0, rng, quality=quality)
        if k == 0:
            with fits.open(path, mode="update") as hdul:
                hdul["SCI"].data[5, 5] = 65535.0
        frames.append(str(path))
    get_task("imcombine").run(input=frames, output=tmp_path / "c.fits")
    plane = mef.read(tmp_path / "c.fits").plane
    assert abs(plane.sci[5, 5] - 100.0) < 10, "the saturated value is not averaged in"
    assert plane.dq[5, 5] == 0, "two good values made a good pixel"
    assert plane.var[5, 5] == pytest.approx(4.0 / 2)
    assert plane.dq[9, 9] == dq.BAD_PIXEL, "what every frame said, and only that"

    get_task("imcombine").run(input=frames, output=tmp_path / "n.fits", masktype="none")
    assert mef.read(tmp_path / "n.fits").plane.sci[5, 5] > 10000, "as IRAF, by request"


def test_imarith_propagates_through_a_quotient(tmp_path):
    rng = np.random.default_rng(11)
    _frame(tmp_path / "a.fits", 1000.0, 30.0, rng)
    _frame(tmp_path / "b.fits", 2.0, 0.02, rng)
    get_task("imarith").run(operand1=str(tmp_path / "a.fits"), op="/",
                            operand2=str(tmp_path / "b.fits"),
                            result=str(tmp_path / "q.fits"))
    assert _ratio(tmp_path / "q.fits") == pytest.approx(1.0, abs=0.03)
    get_task("imarith").run(operand1=str(tmp_path / "a.fits"), op="*", operand2="3",
                            result=str(tmp_path / "t.fits"))
    assert np.mean(mef.read(tmp_path / "t.fits").plane.var) == pytest.approx(9 * 900.0)


def test_a_bias_made_by_harwood_carries_its_own_variance(tmp_path, monkeypatch):
    """ccdproc gives each bias read noise only; zerocombine averages them;
    the next ccdproc takes the master's VAR instead of estimating one."""
    monkeypatch.chdir(tmp_path)
    rng = np.random.default_rng(12)
    gain, readnoise = 2.0, 6.0
    for k in range(8):
        raw = 500.0 + rng.normal(0, readnoise / gain, (40, 48))
        fits.PrimaryHDU(raw.astype(np.float32), fits.Header(
            {"IMAGETYP": "zero", "GAIN": gain, "RDNOISE": readnoise})).writeto(f"b{k}.fits")
    get_task("ccdproc").run(input=[f"b{k}.fits" for k in range(8)], output="p",
                            overscan=True, trim=True, biassec="[41:48,*]",
                            trimsec="[1:40,*]", ccdtype="zero", logfile="")
    one = mef.read("p/b0.fits").plane
    assert np.allclose(one.var, (readnoise / gain) ** 2), "read noise, and no photons"
    get_task("zerocombine").run(input=["p/b*.fits"], output="Zero.fits")
    zero = mef.read("Zero.fits").plane
    # zerocombine is IRAF's: minmax drops the highest of the 8, so 7 go in.
    assert np.mean(zero.var) == pytest.approx((readnoise / gain) ** 2 / 7, rel=1e-6)
