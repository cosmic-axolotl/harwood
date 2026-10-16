"""Regression against real data: the M92 frames from the IRAF tutorial.

KPNO, 1 September 1987, courtesy of George Jacoby -- the dataset behind
Massey & Davis, *A User's Guide to Stellar CCD Photometry with IRAF*. Raw
frames with a real overscan, a real dome flat, and a real globular cluster.

Synthetic data proves the arithmetic; this proves harwood survives contact
with a 1987 header, an int16 big-endian array, a crowded field and a telescope
that drifted two pixels between exposures.

Set HARWOOD_TEST_DATA to the exercises directory to run these; they skip
otherwise, so the suite still passes on a machine without the tutorial.

The frames are read from the ``fm92000N`` tape files, not the ``m92NNN.fits``
beside them. The exercise has you run IRAF's ccdproc in place, which
overwrites those with reduced frames -- and every assertion here about a raw
overscan then fails for a reason that has nothing to do with harwood. The tape
files are what ``rfits`` reads, so no step of the exercise touches them.
"""

import os
from pathlib import Path

import numpy as np
import pytest

from harwood.core import catalog
from harwood.core.registry import get_task

DATA = Path(os.environ.get("HARWOOD_TEST_DATA", "~/iraf/exercises")).expanduser()
PHOT = DATA / "phot"

#: Tape file -> the name the exercise gives it after ``rfits``.
TAPE = {"fm920001": "m92001", "fm920002": "m92006", "fm920003": "m92007",
        "fm920004": "m92010", "fm920005": "m92011", "fm920006": "m92014",
        "fm920007": "m92015"}

pytestmark = pytest.mark.skipif(
    not all((PHOT / tape).exists() for tape in TAPE),
    reason=f"IRAF tutorial data not found under {DATA}",
)

#: From the exercise text: gain 14 e-/ADU, FWHM about 3 pixels, 10 pixel aperture.
EPADU = 14.0
FWHM = 3.0


@pytest.fixture(scope="module")
def raw(tmp_path_factory):
    """The seven raw frames under their exercise names, untouched by any reduction."""
    where = tmp_path_factory.mktemp("m92raw")
    for tape, name in TAPE.items():
        (where / f"{name}.fits").write_bytes((PHOT / tape).read_bytes())
    return where


@pytest.fixture(scope="module")
def reduced(tmp_path_factory, raw):
    """Bias, flat and two V frames, reduced the way exercise 2 does it."""
    work = tmp_path_factory.mktemp("m92")
    ccdproc = get_task("ccdproc")

    ccdproc.run(input=[str(raw / "m92001.fits")], output=work / "Zero.fits",
                overscan=True, trim=True)
    ccdproc.run(input=[str(raw / "m92007.fits")], output=work / "FlatV.fits",
                overscan=True, trim=True, zerocor=True, zero=work / "Zero.fits")
    for name in ("m92010", "m92011"):
        ccdproc.run(input=[str(raw / f"{name}.fits")], output=work / f"{name}p.fits",
                    overscan=True, trim=True, zerocor=True, flatcor=True,
                    zero=work / "Zero.fits", flat=work / "FlatV.fits")
    return work


def measure(work, name):
    get_task("daofind").run(image=str(work / f"{name}p.fits"),
                            output=work / f"{name}.coo", fwhmpsf=FWHM, threshold=4.0)
    get_task("phot").run(image=str(work / f"{name}p.fits"), coords=work / f"{name}.coo",
                         output=work / f"{name}.mag", apertures=[10.0], annulus=15.0,
                         dannulus=10.0, epadu=EPADU, zmag=25.0, exposure="EXPTIME")
    return catalog.read(work / f"{name}.mag")


def cross_match(a, b, tolerance=1.2):
    """Match two catalogues after estimating the shift between the frames."""
    ax, ay = np.asarray(a["x"]), np.asarray(a["y"])
    bx, by = np.asarray(b["x"]), np.asarray(b["y"])
    dx = (bx[None, :] - ax[:, None]).ravel()
    dy = (by[None, :] - ay[:, None]).ravel()
    near = (np.abs(dx) < 20) & (np.abs(dy) < 20)
    hx, ex = np.histogram(dx[near], 80, (-20, 20))
    hy, ey = np.histogram(dy[near], 80, (-20, 20))
    sx = 0.5 * (ex[hx.argmax()] + ex[hx.argmax() + 1])
    sy = 0.5 * (ey[hy.argmax()] + ey[hy.argmax() + 1])
    distance = np.hypot((ax + sx)[:, None] - bx[None, :], (ay + sy)[:, None] - by[None, :])
    index = distance.argmin(axis=1)
    keep = distance.min(axis=1) < tolerance
    return np.flatnonzero(keep), index[keep]


def test_the_1987_headers_are_understood(reduced):
    """BIASSEC and TRIMSEC come from the header, and the frame ends up trimmed."""
    from astropy.io import fits

    data, header = fits.getdata(reduced / "m92010p.fits", header=True)
    assert data.shape == (512, 320)          # 352 columns minus the 32 of overscan
    assert "OVERSCAN" in header and "TRIM" in header and "ZEROCOR" in header
    assert "FLATCOR" in header


def test_the_measured_seeing_matches_the_exercise(reduced):
    """The tutorial says the FWHM is about 3 pixels. imexamine's r must agree."""
    from astropy.io import fits

    from harwood.viewer import analysis

    data = fits.getdata(reduced / "m92010p.fits").astype(float)
    stars = measure(reduced, "m92010")
    brightest = stars[np.argsort(np.asarray(stars["mag"]))[:12]]

    widths = []
    for row in brightest:
        profile = analysis.radial_profile(data, float(row["x"]), float(row["y"]),
                                          radius=12, buffer=4, width=8)
        if profile.converged and 1.0 < profile.fwhm < 8.0:
            widths.append(profile.fwhm)
    assert len(widths) >= 5
    assert np.median(widths) == pytest.approx(3.0, abs=0.5)


def test_the_two_v_frames_agree(reduced):
    """Two 60-second exposures of one field, minutes apart.

    The zero point between them must be zero, and the scatter of the bright
    stars must be a few millimagnitudes. This is the strongest end-to-end check
    there is: it exercises the overscan fit, the bias, the flat, the detection
    and the photometry, and nothing about it can be tuned.
    """
    first, second = measure(reduced, "m92010"), measure(reduced, "m92011")
    i, j = cross_match(first, second)
    assert len(i) > 50, "the two frames overlap almost completely"

    difference = np.asarray(first["mag"])[i] - np.asarray(second["mag"])[j]
    magnitude = np.asarray(first["mag"])[i]
    finite = np.isfinite(difference)

    assert abs(np.median(difference[finite])) < 0.05, "equal exposures, equal scale"

    # Isolated ones: no other detection within the 10 px aperture plus two
    # FWHM. daofind finds the faint companions a few pixels from bright stars,
    # as IRAF's does, and a 10 px aperture over a pair measures how the light
    # of one falls in the aperture of the other -- which moves with the centre,
    # not with the photometry.
    x, y = np.asarray(first["x"]), np.asarray(first["y"])
    separation = np.hypot(x[:, None] - x[None, :], y[:, None] - y[None, :])
    np.fill_diagonal(separation, np.inf)
    isolated = separation.min(axis=1)[i] > 10.0 + 2 * FWHM

    bright = finite & (magnitude < 19.0) & isolated
    assert bright.sum() >= 5
    assert np.std(difference[bright]) < 0.05, "bright stars must repeat to a few mmag"


def test_the_error_model_predicts_the_scatter_it_gets(reduced):
    """IRAF's apphot error, checked against reality rather than against itself.

    The predicted error uses the *measured* sky scatter; if it used the
    theoretical Poisson sky instead it would understate the real scatter of the
    faint stars by a factor of seven, which is how this test was found.
    """
    first, second = measure(reduced, "m92010"), measure(reduced, "m92011")
    i, j = cross_match(first, second)
    difference = np.asarray(first["mag"])[i] - np.asarray(second["mag"])[j]
    predicted = np.asarray(first["merr"])[i] * np.sqrt(2)
    finite = np.isfinite(difference) & np.isfinite(predicted) & (predicted > 0)

    faint = finite & (np.asarray(first["mag"])[i] > 20.0)
    assert faint.sum() > 20
    ratio = np.std(difference[faint]) / np.median(predicted[faint])
    assert 0.5 < ratio < 2.0, f"predicted and measured scatter differ by {ratio:.1f}x"


def test_detections_do_not_come_from_the_truncated_edge(reduced):
    stars = measure(reduced, "m92010")
    x, y = np.asarray(stars["x"]), np.asarray(stars["y"])
    assert x.min() > 2 * FWHM - 0.5
    assert y.min() > 2 * FWHM - 0.5


def test_the_overscan_of_a_real_frame_fits_and_the_gradient_is_removed(raw):
    """The 1987 overscan has a real gradient of about one ADU across the chip."""
    from astropy.io import fits

    from harwood.core import icfit, pixels

    frame = fits.getdata(raw / "m92010.fits").astype(float)
    strip = frame[pixels.parse_section("[321:352,1:512]", frame.shape)]
    profile = strip.mean(axis=1)
    position = np.arange(1.0, profile.size + 1.0)

    flat = icfit.fit(position, profile, icfit.FitSpec(order=1))
    sloped = icfit.fit(position, profile, icfit.FitSpec(order=4, niterate=3))
    assert sloped.rms < flat.rms, "a real overscan is not flat"
    assert np.ptp(sloped.values) > 0.5, "the gradient is about an ADU"


def _xyxymatch(first, second, **params):
    """IRAF's xyxymatch (core.xyxymatch) of two daofind catalogues: the
    result and the transformation that carries the second onto the first."""
    from harwood.core import xyxymatch

    def padded(table, name):
        return np.concatenate(([0.0], np.asarray(table[name]))).astype(np.float32)

    rx, ry, ix, iy = (padded(first, "x"), padded(first, "y"), padded(second, "x"),
                      padded(second, "y"))
    result = xyxymatch.xyxymatch(rx, ry, ix, iy, **params)
    coeff = result.matched if result.matched is not None else result.initial
    pairs = np.array(result.pairs, dtype=int).reshape(-1, 2)
    xt, yt = xyxymatch.apply(coeff, ix[pairs[:, 1]], iy[pairs[:, 1]])
    rms = float(np.sqrt(np.mean((xt - rx[pairs[:, 0]]) ** 2 + (yt - ry[pairs[:, 0]]) ** 2)))
    return result, coeff, rms


def test_the_frames_are_matched_and_the_shift_is_recovered(reduced):
    """The shift that forced this whole domain into the MVP.

    Two V frames minutes apart moved by about two pixels. IRAF's xyxymatch
    pairs them -- and tests/data/iraf_xyxymatch_m92.out is what IRAF wrote
    for these very lists: its triangles, made from the 30 lowest stars of
    each list, find no transformation here (168 and 120 detections do not
    share their lowest 30), so it falls back to the initial one and pairs
    by tolerance, which a 2-pixel shift survives. harwood writes the same
    file.
    """
    from harwood.core import xyxymatch

    first, second = measure(reduced, "m92010"), measure(reduced, "m92011")
    for name, table in (("m92010", first), ("m92011", second)):
        (reduced / f"{name}.xy").write_text(
            "".join(f"{x:.3f} {y:.3f}\n" for x, y in zip(table["x"], table["y"],
                                                          strict=True)))
    xr, yr, rl = xyxymatch.read_list(reduced / "m92010.xy")
    xi, yi, il = xyxymatch.read_list(reduced / "m92011.xy")
    result = xyxymatch.xyxymatch(xr, yr, xi, yi)
    written = xyxymatch.format_output(result, xr, yr, rl, xi, yi, il, "m92011.xy",
                                      "m92010.xy")
    iraf = (Path(__file__).parent / "data" / "iraf_xyxymatch_m92.out").read_text()
    assert written == iraf

    pairs = np.array(result.pairs)
    assert len(pairs) > 60
    dx, dy = xi[pairs[:, 1]] - xr[pairs[:, 0]], yi[pairs[:, 1]] - yr[pairs[:, 0]]
    assert 0.5 < np.hypot(np.median(dx), np.median(dy)) < 5.0
    assert np.std(dx) < 0.5 and np.std(dy) < 0.5, "a fraction of a pixel, as centroids"


def test_imcentroid_and_xyxymatch_agree_on_the_same_frames(reduced):
    """Two independent routes to the same number: marginal centroids of chosen
    stars, and a transform fitted to every detection. They must agree."""
    from harwood.core import catalog
    from harwood.core.registry import get_task

    first = measure(reduced, "m92010")
    reference = first["id", "x", "y"][:25]
    catalog.write(reference, reduced / "ref.coo")

    centroid = get_task("imcentroid").run(
        input=[str(reduced / "m92011p.fits")], reference=str(reduced / "m92010p.fits"),
        coords=reduced / "ref.coo", boxsize=7)
    row = centroid.data["shifts"][0]
    assert row["nstars"] >= 20
    assert row["xinternal"] < 0.1 and row["yinternal"] < 0.1

    second = measure(reduced, "m92011")
    result, _, _ = _xyxymatch(first, second)
    pairs = np.array(result.pairs)
    dx = np.asarray(second["x"])[pairs[:, 1] - 1] - np.asarray(first["x"])[pairs[:, 0] - 1]
    dy = np.asarray(second["y"])[pairs[:, 1] - 1] - np.asarray(first["y"])[pairs[:, 0] - 1]
    # imcentroid's shift is reference minus image.
    assert row["xshift"] == pytest.approx(-np.median(dx), abs=0.3)
    assert row["yshift"] == pytest.approx(-np.median(dy), abs=0.3)


def test_imalign_closes_the_loop(reduced):
    """Align, detect again, match again: the residual shift must be near zero."""
    from harwood.core import catalog
    from harwood.core.registry import get_task

    first = measure(reduced, "m92010")
    catalog.write(first["id", "x", "y"][:25], reduced / "ref.coo")

    result = get_task("imalign").run(
        input=[str(reduced / "m92010p.fits"), str(reduced / "m92011p.fits")],
        reference=str(reduced / "m92010p.fits"), output=str(reduced / "aligned"),
        coords=reduced / "ref.coo", interp_type="linear")
    assert len(result.outputs) == 2
    assert any("correlates the noise" in w for w in result.warnings), \
        "interpolating a fractional shift must be said out loud, not documented"

    finder = get_task("daofind")
    catalogues = []
    for name in ("m92010p", "m92011p"):
        finder.run(image=str(reduced / "aligned" / f"{name}.fits"),
                   output=reduced / f"a_{name}.coo", fwhmpsf=FWHM, threshold=4.0)
        catalogues.append(catalog.read(reduced / f"a_{name}.coo"))

    result, _, _ = _xyxymatch(catalogues[0], catalogues[1], matching="tolerance",
                              tolerance=2.0)
    pairs = np.array(result.pairs)
    assert len(pairs) > 50, "aligned frames match without being told a shift"
    ref, moved = catalogues
    dx = np.asarray(moved["x"])[pairs[:, 1] - 1] - np.asarray(ref["x"])[pairs[:, 0] - 1]
    dy = np.asarray(moved["y"])[pairs[:, 1] - 1] - np.asarray(ref["y"])[pairs[:, 0] - 1]
    assert abs(np.median(dx)) < 0.5 and abs(np.median(dy)) < 0.5


def test_exercise_two_reproduces_its_own_numbers(tmp_path, raw):
    """Path 1 of the course's exercise 2, run end to end.

    The exercise states two numbers a reader can check: the trimmed frames are
    318x509, and the modes of the two flats are 1313 and 1468. Both come out of
    choices the text leaves to the reader -- which overscan columns, which trim
    -- so reproducing them is a real check that harwood does the same
    arithmetic in the same order, not just similar arithmetic.
    """
    from astropy.io import fits

    from harwood.core.registry import get_task
    from harwood.tasks.imutil import mode

    for name in ("m92001", "m92006", "m92007", "m92010"):
        (tmp_path / f"{name}.fits").write_bytes((raw / f"{name}.fits").read_bytes())

    # colbias: overscan over the columns the text picks, then trim.
    get_task("ccdproc").run(
        input=[str(tmp_path / "m92*.fits")], output="%m%tr%",
        overscan=True, trim=True, biassec="[335:350,2:510]",
        trimsec="[1:318,2:510]", function="chebyshev", order=1, niterate=1)

    trimmed = fits.getdata(tmp_path / "tr92006.fits")
    assert trimmed.shape == (509, 318), "the text says 318 columns by 509 lines"

    # imarith @zlist - tr92001 @zlist: subtract the bias from everything else.
    get_task("imarith").run(
        operand1=str(tmp_path / "tr9200[67].fits"), op="-",
        operand2=str(tmp_path / "tr92001.fits"), result="%tr%tr%")

    modes = {}
    for name, claimed in (("tr92006", 1313), ("tr92007", 1468)):
        data = fits.getdata(tmp_path / f"{name}.fits").astype(float)
        modes[name] = mode(data)
        assert modes[name] == pytest.approx(claimed, abs=1.0), (
            f"{name}: the exercise quotes {claimed}, harwood gets {modes[name]:.2f}"
        )

    # And the normalisation those numbers are for: a flat centred on one.
    get_task("imarith").run(operand1=str(tmp_path / "tr92007.fits"), op="/",
                            operand2=str(round(modes["tr92007"])),
                            result=str(tmp_path / "Vflat.fits"))
    flat = fits.getdata(tmp_path / "Vflat.fits").astype(float)
    assert mode(flat) == pytest.approx(1.0, abs=0.01)
