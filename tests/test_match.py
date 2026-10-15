"""Matching frames of one field: the step every multi-frame reduction needs.

Two frames are never on the same pixels. These tests build point sets with a
known transformation and check it comes back -- and, just as importantly,
check that a match which cannot be trusted is refused rather than returned.
"""

import numpy as np
import pytest

from harwood.core import match


def field(n=60, seed=3):
    rng = np.random.default_rng(seed)
    return (rng.uniform(1, 500, n), rng.uniform(1, 500, n),
            10 ** rng.uniform(3, 5, n), rng)


def transform(x, y, rotation=0.0, scale=1.0, shift=(0.0, 0.0)):
    angle = np.radians(rotation)
    return (scale * (np.cos(angle) * x - np.sin(angle) * y) + shift[0],
            scale * (np.sin(angle) * x + np.cos(angle) * y) + shift[1])


@pytest.mark.parametrize("rotation,scale,shift", [
    (0.0, 1.0, (1.75, 0.75)),        # what the M92 frames actually did
    (0.4, 1.003, (37.2, -18.6)),
    (12.0, 0.98, (-40.0, 60.0)),
    (90.0, 1.0, (0.0, 0.0)),         # the camera was rotated
])
def test_triangles_find_the_transformation_with_no_prior(rotation, scale, shift):
    rx, ry, flux, rng = field()
    ix, iy = transform(rx, ry, rotation, scale, shift)
    keep = rng.permutation(len(rx))[:45]           # the input misses some stars
    ix = ix[keep] + rng.normal(0, 0.05, 45)
    iy = iy[keep] + rng.normal(0, 0.05, 45)

    found = match.by_triangles(rx, ry, ix, iy, nmatch=25, tolerance=2.0,
                               rflux=flux, iflux=flux[keep])
    assert found, found.message
    assert len(found) == 45, "every star present in both frames should pair up"

    model = match.fit_transform(rx[found.reference], ry[found.reference],
                                ix[found.input], iy[found.input])
    assert model["rotation"] == pytest.approx(rotation, abs=0.02)
    assert model["xscale"] == pytest.approx(scale, rel=1e-4)
    assert model["xshift"] == pytest.approx(shift[0], abs=0.2)
    assert model["rms"] < 0.2


def test_the_brightest_stars_are_the_ones_both_frames_agree_on():
    """Without fluxes, `nmatch` takes the first N, which in two differently
    ordered lists are different stars -- and the triangles then have nothing
    in common. This is the failure mode the flux argument exists to prevent."""
    rx, ry, flux, rng = field(n=40)
    ix, iy = transform(rx, ry, 0.4, 1.0, (12.0, -7.0))
    order = rng.permutation(len(ix))

    blind = match.by_triangles(rx, ry, ix[order], iy[order], nmatch=15)
    sighted = match.by_triangles(rx, ry, ix[order], iy[order], nmatch=15,
                                 rflux=flux, iflux=flux[order])
    assert len(sighted) > len(blind)


def test_a_star_with_no_flux_is_the_faintest_not_the_brightest():
    """phot writes NaN for a star whose aperture sums to less than its sky.
    A reversed argsort puts NaN first, and the pattern search then ran on the
    unmeasurable stars -- on M92 that sank the whole match."""
    rx, ry, flux, rng = field(n=40)
    ix, iy = transform(rx, ry, 0.4, 1.0, (12.0, -7.0))
    rflux, iflux = flux.copy(), flux.copy()
    rflux[rng.permutation(40)[:5]] = np.nan
    iflux[rng.permutation(40)[:5]] = np.nan
    assert np.isfinite(rflux[match._brightest(rflux, 40, 35)]).all()   # 35 are finite

    found = match.by_triangles(rx, ry, ix, iy, nmatch=15, rflux=rflux, iflux=iflux)
    assert found, found.message
    assert len(found) == 40


def test_a_match_that_does_not_fit_a_transform_is_refused():
    """A set of voted pairs with no consistent transform behind it is a guess.
    Returning it would be a silent wrong answer, which is worse than nothing."""
    rng = np.random.default_rng(1)
    rx, ry = rng.uniform(1, 500, 40), rng.uniform(1, 500, 40)
    ix, iy = rng.uniform(1, 500, 40), rng.uniform(1, 500, 40)   # an unrelated field

    found = match.by_triangles(rx, ry, ix, iy, nmatch=20, tolerance=1.0)
    assert not found
    assert found.message


def test_tolerance_matching_estimates_the_shift_when_none_is_given():
    rx, ry, flux, rng = field(n=50)
    ix, iy = rx + 4.75, ry - 2.17                 # the B-to-V shift on M92
    found = match.by_tolerance(rx, ry, ix, iy, tolerance=1.0)
    assert len(found) == 50
    assert found.shift[0] == pytest.approx(4.75, abs=0.1)


def test_tolerance_matching_with_a_wrong_prior_finds_nothing_rather_than_rubbish():
    rx, ry, flux, rng = field(n=50)
    ix, iy = rx + 40.0, ry + 40.0
    found = match.by_tolerance(rx, ry, ix, iy, tolerance=2.0, xin=0.0, yin=0.0)
    assert len(found) == 0


def test_one_input_star_is_never_claimed_by_two_reference_stars():
    reference_x = np.array([10.0, 10.4, 200.0])   # two reference stars, one input
    reference_y = np.array([10.0, 10.0, 200.0])
    found = match.by_tolerance(reference_x, reference_y,
                               np.array([10.2, 200.0]), np.array([10.0, 200.0]),
                               tolerance=1.0, xin=0.0, yin=0.0)
    assert len(set(found.input.tolist())) == len(found)


def test_the_transform_fit_rejects_a_mismatched_pair():
    rx, ry, flux, rng = field(n=30)
    ix, iy = transform(rx, ry, 0.0, 1.0, (5.0, 5.0))
    ix[7] += 90.0                                  # one wrong pair

    loose = match.fit_transform(rx, ry, ix, iy)
    tight = match.fit_transform(rx, ry, ix, iy, niterate=3, reject=2.5)
    assert tight["nrejected"] >= 1
    assert tight["rms"] < loose["rms"] / 10
    assert tight["xshift"] == pytest.approx(5.0, abs=0.01)


def test_rejection_uses_a_robust_sigma_and_does_not_cascade():
    """With the plain standard deviation, each pass shrinks sigma and rejects
    more; on the real M92 frames that threw away 40% of good pairs and reported
    an rms far better than centroids can be. The MAD does not shrink that way."""
    rx, ry, flux, rng = field(n=80)
    ix = rx + 3.0 + rng.normal(0, 0.15, 80)
    iy = ry - 1.0 + rng.normal(0, 0.15, 80)

    model = match.fit_transform(rx, ry, ix, iy, niterate=5, reject=2.5)
    assert model["nrejected"] < 12, "well-behaved pairs must not be eaten"
    assert model["rms"] == pytest.approx(0.21, abs=0.08)


def test_estimate_shift_works_when_the_lists_have_different_depths():
    rx, ry, flux, rng = field(n=80)
    ix, iy = rx[:25] + 12.5, ry[:25] - 6.25       # the input reaches far fewer stars
    dx, dy = match.estimate_shift(rx, ry, ix, iy)
    assert dx == pytest.approx(12.5, abs=0.3)
    assert dy == pytest.approx(-6.25, abs=0.3)
