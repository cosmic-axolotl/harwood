"""The variance ccdproc writes, against the scatter it claims to describe.

IRAF has no variance to compare with, so this is the known-truth check
section 12 asks for. A night is simulated 120 times over -- a master bias
and a master flat combined from fresh frames each time, as IRAF's would be,
and an object frame of 200 to 20000 electrons behind a dusty flat -- with a
gain of 2 e-/ADU and 6 e- of read noise. Every time ccdproc reduces it and
writes a variance; across the 120 the reduced pixels actually scatter. The
two must agree, pixel by pixel and on average.

The overscan fit's own noise is not in the variance (a level fitted to 192
pixels adds under one per cent of the smallest pixel variance here); the
tolerance allows for that.
"""

import numpy as np
from astropy.io import fits

from harwood.core import dq, mef
from harwood.core.registry import get_task

GAIN, READNOISE, BIAS = 2.0, 6.0, 500.0
NY, NX, OVERSCAN = 24, 32, 8
REALIZATIONS = 120


def _scene():
    y, x = np.mgrid[0:NY, 0:NX].astype(float)
    electrons = 200.0 * (100.0 ** (x / (NX - 1)))           # 200 to 20000 e-
    response = 1.0 + 0.15 * np.sin(x / 3.0) * np.cos(y / 4.0)
    response[5, 7] = 0.3                                       # a speck of dust
    pattern = 2.0 * np.sin(x / 5.0)                            # fixed bias structure, ADU
    return electrons, response, pattern


def _night(rng, directory):
    electrons, response, pattern = _scene()
    noise = READNOISE / GAIN

    def marked(header, *flags):
        for flag in flags:
            header[flag] = "simulated"
        return header

    biases = [pattern + rng.normal(0, noise, (NY, NX)) for _ in range(5)]
    fits.PrimaryHDU(np.mean(biases, axis=0).astype(np.float32),
                    marked(fits.Header({"IMAGETYP": "zero", "NCOMBINE": 5}),
                           "OVERSCAN", "TRIM")).writeto(directory / "Zero.fits")
    flats = [rng.poisson(response * 30000.0) / GAIN + rng.normal(0, noise, (NY, NX))
             for _ in range(4)]
    fits.PrimaryHDU(np.mean(flats, axis=0).astype(np.float32),
                    marked(fits.Header({"IMAGETYP": "flat", "NCOMBINE": 4}),
                           "OVERSCAN", "TRIM", "ZEROCOR")).writeto(directory / "Flat.fits")
    raw = np.empty((NY, NX + OVERSCAN))
    raw[:, :NX] = (BIAS + pattern + rng.poisson(electrons * response) / GAIN
                   + rng.normal(0, noise, (NY, NX)))
    raw[:, NX:] = BIAS + rng.normal(0, noise, (NY, OVERSCAN))
    raw[2, 3] = 40000.0                                        # one pixel over the top
    fits.PrimaryHDU(raw.astype(np.float32), fits.Header(
        {"IMAGETYP": "object", "GAIN": GAIN, "RDNOISE": READNOISE})).writeto(
        directory / "obj.fits")


def test_the_variance_describes_the_scatter(tmp_path):
    rng = np.random.default_rng(2026)
    sci, var = [], []
    for k in range(REALIZATIONS):
        directory = tmp_path / f"n{k}"
        directory.mkdir()
        _night(rng, directory)
        get_task("ccdproc").run(
            input=[str(directory / "obj.fits")], output=str(directory / "red.fits"),
            overscan=True, trim=True, zerocor=True, flatcor=True,
            biassec=f"[{NX + 1}:{NX + OVERSCAN},*]", trimsec=f"[1:{NX},*]",
            zero=directory / "Zero.fits", flat=directory / "Flat.fits",
            saturation=30000.0, logfile="")
        plane = mef.read(directory / "red.fits").plane
        sci.append(plane.sci)
        var.append(plane.var)
        if k == 0:
            assert plane.dq[2, 3] & dq.SATURATED, "the pixel over the top is flagged"
    sci, var = np.array(sci, dtype=float), np.array(var, dtype=float)

    good = np.ones((NY, NX), dtype=bool)
    good[2, 3] = False
    ratio = sci.var(axis=0, ddof=1) / var.mean(axis=0)
    # On average, to 3 per cent (the sample variances of 120 realizations
    # pooled over 767 pixels are good to half a per cent; the rest is the
    # overscan fit the variance leaves out).
    assert abs(float(np.mean(ratio[good])) - 1.0) < 0.03
    # Pixel by pixel, within what 119 degrees of freedom allow (13 per cent
    # one sigma), across the whole range of signal: faint and bright alike.
    for columns in (slice(0, 4), slice(14, 18), slice(28, 32)):
        assert abs(float(np.median(ratio[:, columns])) - 1.0) < 0.06
    assert abs(float(ratio[5, 7]) - 1.0) < 0.4, "behind the dust too"


def test_without_a_gain_there_is_no_variance_and_the_run_says_why(tmp_path):
    rng = np.random.default_rng(1)
    _night(rng, tmp_path)
    with fits.open(tmp_path / "obj.fits", mode="update") as hdul:
        del hdul[0].header["GAIN"]
    result = get_task("ccdproc").run(
        input=[str(tmp_path / "obj.fits")], output=str(tmp_path / "red.fits"),
        overscan=True, trim=True, biassec=f"[{NX + 1}:{NX + OVERSCAN},*]",
        trimsec=f"[1:{NX},*]", logfile="")
    assert mef.read(tmp_path / "red.fits").plane.var is None
    assert any("no gain" in str(w) and "--gain" in w.fix for w in result.warnings)


def test_format_iraf_gives_iraf_its_image_and_says_what_stayed_behind(tmp_path):
    rng = np.random.default_rng(3)
    _night(rng, tmp_path)
    result = get_task("ccdproc").run(
        input=[str(tmp_path / "obj.fits")], output=str(tmp_path / "red.fits"),
        overscan=True, trim=True, biassec=f"[{NX + 1}:{NX + OVERSCAN},*]",
        trimsec=f"[1:{NX},*]", saturation=30000.0, format="iraf", logfile="")
    with fits.open(tmp_path / "red.fits") as hdul:
        assert len(hdul) == 1 and hdul[0].header["BPM"] == "red.bpm.fits"
        assert "OVERSCAN" in hdul[0].header
    assert (tmp_path / "red.bpm.fits") in result.outputs
    assert any("without the variance" in str(w) for w in result.warnings)
