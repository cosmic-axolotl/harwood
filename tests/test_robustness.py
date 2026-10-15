"""Every measurement, fed the images a night actually produces.

Not the well-behaved star of the other tests: frames 3 pixels wide, constant,
all zero, negative, integer, with blank (NaN) pixels, with the star on the
corner; positions inside, on the edge and off the image. The rule checked is
the one a user depends on -- a measurement either comes back, comes back
flagged, or fails with a ValueError that says what to do. Never a bare
exception from deep inside numpy.

hypothesis drives it; on a first run, before these were fixed, it found blank
pixels crashing the centring, the sky and the profile.
"""

import warnings

import numpy as np
import pytest

hypothesis = pytest.importorskip("hypothesis")
from hypothesis import HealthCheck, given, settings  # noqa: E402
from hypothesis import strategies as st  # noqa: E402

from harwood.core import apphot, examine, fixpix  # noqa: E402
from harwood.viewer import analysis  # noqa: E402

KINDS = ["noise", "star", "const", "zero", "neg", "int", "nan", "corner"]


@st.composite
def frames(draw):
    ny, nx = draw(st.integers(3, 40)), draw(st.integers(3, 40))
    kind = draw(st.sampled_from(KINDS))
    rng = np.random.default_rng(draw(st.integers(0, 2 ** 31)))
    image = rng.normal(100, 5, (ny, nx))
    if kind in ("star", "nan", "int", "corner"):
        x0 = 1.0 if kind == "corner" else draw(st.floats(0.5, nx + 0.5))
        y0 = 1.0 if kind == "corner" else draw(st.floats(0.5, ny + 0.5))
        yy, xx = np.mgrid[1:ny + 1, 1:nx + 1]
        image += 3000 * np.exp(-((xx - x0) ** 2 + (yy - y0) ** 2) / (2 * 1.3 ** 2))
    if kind == "const":
        image[:] = 42.0
    elif kind == "zero":
        image[:] = 0.0
    elif kind == "neg":
        image -= 500
    elif kind == "int":
        image = np.round(image).astype(np.int16)
    elif kind == "nan":
        image[rng.random(image.shape) < 0.05] = np.nan
    x = draw(st.floats(-5, nx + 5))
    y = draw(st.floats(-5, ny + 5))
    return image, x, y


def _acceptable(error: Exception) -> bool:
    return isinstance(error, (ValueError, FileNotFoundError)) and bool(
        getattr(error, "fix", ""))


@settings(max_examples=60, deadline=None, suppress_health_check=list(HealthCheck))
@given(frames())
def test_no_measurement_fails_without_saying_what_to_do(frame):
    image, x, y = frame
    data = image.astype(float)
    annulus = apphot.SkyPars(annulus=3, dannulus=4)
    calls = {
        "center": lambda: apphot.center(data, x, y),
        "fitsky": lambda: [apphot.fitsky(data, x, y, apphot.SkyPars(a, 3, 4))
                           for a in ("centroid", "mode", "median", "mean")],
        "apertures": lambda: apphot.apertures(
            data, x, y, [1.5, 3.0], apphot.fitsky(data, x, y, annulus)),
        "radial_profile": lambda: analysis.radial_profile(data, x, y),
        **{f"imexamine {key}": (lambda key=key: examine.examine(data, key, x, y))
           for key in examine.KEYS},
    }
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for name, call in calls.items():
            try:
                call()
            except Exception as error:
                assert _acceptable(error), f"{name} at ({x:.2f}, {y:.2f}): {error!r}"


@settings(max_examples=30, deadline=None, suppress_health_check=list(HealthCheck))
@given(frames(), st.sampled_from([0.0, 0.1, 0.5, 1.0]))
def test_fixpix_changes_only_the_bad_pixels_and_keeps_the_type(frame, fraction):
    image, _, _ = frame
    mask = (np.random.default_rng(1).random(image.shape) < fraction).astype(int)
    fixed, _ = fixpix.fix(image, mask)
    assert fixed.shape == image.shape and fixed.dtype == image.dtype
    good = mask == 0
    assert np.array_equal(fixed[good], image[good], equal_nan=True)
