"""apall's errors from the frame's variance, against the scatter they claim.

A long slit with the same star on every one of its 1600 lines -- a straight
Gaussian profile on a flat sky -- so each extracted line is an independent
draw of one flux, and the scatter along the spectrum must be what the error
column says. For a plain sum, which IRAF gives no sigma at all, and for
variance weighting, both with the sky fitted and subtracted: the sky level's
own uncertainty is part of the error, and here, a faint star on a bright
sky measured from a narrow sample, a large part of it.
"""

import numpy as np
import pytest
from astropy.io import fits

from harwood.core import dq, mef, spectrum
from harwood.core.registry import get_task

GAIN, READNOISE = 2.0, 5.0
NY, NX = 1600, 60


def _slit(path, rng, quality=None):
    x = np.arange(1, NX + 1, dtype=float)
    profile = np.exp(-0.5 * ((x - 30.3) / 2.0) ** 2)
    electrons = 500.0 + 2000.0 * profile / profile.sum()     # the sky dominates
    detected = rng.poisson(np.tile(electrons, (NY, 1))) + rng.normal(0, READNOISE, (NY, NX))
    sci = detected / GAIN
    var = np.maximum(sci, 0) / GAIN + (READNOISE / GAIN) ** 2
    mef.write(path, mef.Image(fits.Header({"OBJECT": "star", "DISPAXIS": 2}),
                              [mef.Plane(sci, var, quality)]))


@pytest.mark.parametrize("weights", ["none", "variance"])
def test_the_extracted_errors_describe_the_scatter(tmp_path, weights):
    _slit(tmp_path / "slit.fits", np.random.default_rng(21))
    get_task("apall").run(input=str(tmp_path / "slit.fits"),
                          output=tmp_path / "slit.ms.fits", database=tmp_path,
                          background="fit", b_sample="-14:-10,10:14", weights=weights,
                          gain=GAIN, readnoise=READNOISE, lower=-6.0, upper=6.0,
                          resize=False, t_step=20)
    star = spectrum.read(tmp_path / "slit.ms.fits")[0]
    inner = slice(20, NY - 20)                 # away from the trace's ends
    ratio = float(np.std(star.flux[inner], ddof=1) / np.median(star.error[inner]))
    # 1560 lines: the sample scatter is good to 2 per cent. The sky fit is
    # IRAF's default, a constant through medians of three interpolated
    # points, propagated as skyeval and icfit do it; both come out at 0.98.
    # Left out, the sky level's own uncertainty would put this near 1.4, and
    # taking the sample as a plain mean of its pixels near 1.1 -- icfit drops
    # the last two points of each five-point range.
    assert ratio == pytest.approx(1.0, abs=0.06)


def test_a_flagged_pixel_shows_in_its_line(tmp_path):
    quality = dq.empty((NY, NX))
    quality[100, 30] = dq.SATURATED
    quality[200, 2] = dq.BAD_PIXEL                 # outside the aperture
    _slit(tmp_path / "slit.fits", np.random.default_rng(22), quality)
    get_task("apall").run(input=str(tmp_path / "slit.fits"),
                          output=tmp_path / "slit.ms.fits", database=tmp_path,
                          lower=-6.0, upper=6.0, resize=False, t_step=20)
    star = spectrum.read(tmp_path / "slit.ms.fits")[0]
    assert star.dq[100] == dq.SATURATED
    assert int(np.count_nonzero(star.dq)) == 1, "the pixel outside is not the star's"


def test_dispcor_carries_the_dq_onto_its_grid():
    """Each new pixel takes the bits of the old pixels its interval
    overlaps; one beyond the old range is no data."""
    from harwood.tasks.onedspec import _resampled_dq

    before = np.arange(1.0, 11.0)                   # wavelengths of ten pixels
    quality = np.zeros(10, dtype=np.int32)
    quality[4] = dq.COSMIC_RAY                     # the pixel at 5.0
    after = np.arange(4.0, 12.0, 0.5)
    out = dict(zip(after, _resampled_dq(quality, before, after), strict=True))
    assert out[5.0] == dq.COSMIC_RAY
    assert out[4.5] == dq.COSMIC_RAY and out[5.5] == dq.COSMIC_RAY  # straddle it
    assert out[4.0] == 0 and out[7.0] == 0
    assert out[11.5] == dq.NO_DATA
