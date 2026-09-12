"""The pixel convention is the one thing that must never drift.

Section 2.2 fixes 1-based (x, y) on every visible surface. These tests are
permanent: a failure here means every coordinate harwood reports is wrong by
a pixel or half a pixel, silently, in a way that looks plausible.
"""

import numpy as np
import pytest

from harwood.core import pixels


def test_round_trip_numpy():
    row, col = pixels.to_numpy(612.03, 539.91)
    back = pixels.from_numpy(row, col)
    assert back.x == pytest.approx(612.03)
    assert back.y == pytest.approx(539.91)


def test_first_pixel_is_one_not_zero():
    row, col = pixels.to_numpy(1.0, 1.0)
    assert (row, col) == (0.0, 0.0)


def test_scene_offset_is_half_a_pixel():
    # Section 8.4: the centre of IRAF pixel 1 lands at scene 0.5.
    sx, sy = pixels.to_scene(1.0, 1.0)
    assert (sx, sy) == (0.5, 0.5)
    back = pixels.from_scene(sx, sy)
    assert back == (1.0, 1.0)


def test_section_is_x_first_and_inclusive():
    data = np.arange(100).reshape(10, 10)          # shape (y=10, x=10)
    got = data[pixels.parse_section("[2:4,*]", data.shape)]
    assert got.shape == (10, 3)                    # 3 columns, every row
    np.testing.assert_array_equal(got[0], [1, 2, 3])


def test_section_wildcard_and_stride():
    data = np.zeros((20, 30))
    assert data[pixels.parse_section("[*,5:10]", data.shape)].shape == (6, 30)
    assert data[pixels.parse_section("[1:30:2,*]", data.shape)].shape == (20, 15)


def test_section_out_of_range_says_so():
    data = np.zeros((10, 10))
    with pytest.raises(ValueError, match="11"):
        pixels.parse_section("[1:11,*]", data.shape)


def test_section_reversed_axis():
    data = np.arange(10).reshape(1, 10)
    got = data[pixels.parse_section("[-*,*]", data.shape)]
    np.testing.assert_array_equal(got[0], np.arange(10)[::-1])


def test_synthetic_star_lands_where_it_was_injected():
    """A Gaussian injected at an exact 1-based position must read back there.

    This is the test that catches an off-by-one or off-by-half anywhere in
    the chain, because it goes all the way through numpy's own indexing.
    """
    true = pixels.Position(612.03, 539.91)
    ny, nx = 700, 800
    row, col = pixels.to_numpy(true.x, true.y)
    yy, xx = np.mgrid[0:ny, 0:nx]
    sigma = 3.40 / 2.3548
    image = 4530.0 * np.exp(-((xx - col) ** 2 + (yy - row) ** 2) / (2 * sigma**2))

    total = image.sum()
    cy = (image * yy).sum() / total
    cx = (image * xx).sum() / total
    measured = pixels.from_numpy(cy, cx)

    assert measured.x == pytest.approx(true.x, abs=0.01)
    assert measured.y == pytest.approx(true.y, abs=0.01)
