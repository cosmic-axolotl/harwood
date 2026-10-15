"""A star field built the same way every time, for checking against IRAF.

`tests/data/imexam_iraf_*.log` hold what IRAF 2.18's own imexamine printed
for this field (see `make_reference`). The test rebuilds the field from this
function, so the comparison needs no data files beyond those logs -- and
float32 with a fixed seed makes the pixels identical bit for bit.
"""

import numpy as np

#: x, y (1-based), peak, FWHM, beta, axis ratio, angle (degrees).
STARS = [
    (40.3, 45.7, 9000.0, 1.6, 3.0, 1.0, 0.0),     # narrow: the correction table
    (110.6, 52.2, 6000.0, 2.5, 2.5, 1.0, 0.0),
    (160.2, 140.9, 4000.0, 3.5, 3.5, 1.0, 0.0),
    (60.8, 150.4, 2500.0, 5.0, 2.5, 1.0, 0.0),    # wide: a large aperture
    (130.1, 100.6, 5000.0, 3.0, 3.0, 0.6, 30.0),  # elongated
    (95.5, 170.5, 600.0, 2.8, 3.0, 1.0, 0.0),     # faint
]

#: Where the cursor is put for each star: a pixel or so off, as a hand would.
CURSOR = [(41.0, 45.0), (110.0, 53.0), (161.0, 141.0), (60.0, 151.0),
          (131.0, 100.0), (96.0, 170.0)]


def field() -> np.ndarray:
    ny, nx = 200, 200
    yy, xx = np.mgrid[1:ny + 1, 1:nx + 1].astype(float)
    image = np.full((ny, nx), 100.0)
    for x, y, peak, fwhm, beta, ratio, angle in STARS:
        alpha = fwhm / (2 * np.sqrt(2 ** (1 / beta) - 1))
        t = np.radians(angle)
        u = (xx - x) * np.cos(t) + (yy - y) * np.sin(t)
        v = -(xx - x) * np.sin(t) + (yy - y) * np.cos(t)
        image += peak * (1 + (u / alpha) ** 2 + (v / (alpha * ratio)) ** 2) ** (-beta)
    image += np.random.default_rng(7).normal(0, 5.0, image.shape)
    return image.astype(np.float32)


def make_reference(directory) -> None:
    """Write the field and the cursor file IRAF is run on."""
    from pathlib import Path

    from astropy.io import fits

    directory = Path(directory)
    fits.writeto(directory / "field.fits", field(), overwrite=True)
    (directory / "cursor").write_text(
        "".join(f"{x} {y} 1 a\n" for x, y in CURSOR))


def fixpix_case():
    """An image and a mask with every shape fixpix decides differently on.

    Mask values: 1 for the ordinary cases; 5 forces a region along lines and
    7 along columns, for linterp=5 cinterp=7.
    """
    rng = np.random.default_rng(11)
    ny, nx = 40, 50
    yy, xx = np.mgrid[1:ny + 1, 1:nx + 1].astype(float)
    image = (200 + 3 * xx + 2 * yy + rng.normal(0, 4, (ny, nx))).astype(np.float32)
    mask = np.zeros((ny, nx), np.int16)
    mask[5:35, 10] = 1            # a bad column: fixed along lines
    mask[20, 15:30] = 1           # a bad row segment: fixed along columns
    mask[8:11, 30:35] = 1         # a 5 wide, 3 tall block
    mask[25, 40] = 1              # one hot pixel: a tie, so along the line
    mask[0:4, 45] = 1             # a column stub at the bottom edge
    mask[30, 0:3] = 1             # a row stub at the left edge
    mask[12:16, 20] = 1           # an L: a column ...
    mask[15, 20:26] = 1           # ... and a row sharing a pixel
    mask[33:36, 30:38] = 5        # wide block, forced along lines
    mask[2:4, 2:8] = 7            # wide block, forced along columns
    return image, mask


def combine_stack():
    """Seven frames of one field for imcombine: different sky levels and
    exposure times, a star, cosmic rays in single frames, a few low pixels.
    Returns [(data, exptime), ...]."""
    rng = np.random.default_rng(23)
    ny, nx = 40, 50
    yy, xx = np.mgrid[1:ny + 1, 1:nx + 1].astype(float)
    star = 2000 * np.exp(-((xx - 25.3) ** 2 + (yy - 20.6) ** 2) / (2 * 1.6 ** 2))
    frames = []
    for k, exptime in enumerate([10.0, 12.0, 10.0, 15.0, 10.0, 20.0, 11.0]):
        level = 300.0 + 40 * k
        gain = exptime / 10.0
        image = (level + star) * gain + rng.normal(0, 8, (ny, nx))
        image[rng.integers(0, ny, 6), rng.integers(0, nx, 6)] += rng.uniform(800, 5000, 6)
        image[rng.integers(0, ny, 2), rng.integers(0, nx, 2)] -= 400
        frames.append((image.astype(np.float32), exptime))
    return frames


def crowded_field():
    """A field for daofind: a hundred gaussian stars, pairs closer than a
    FWHM and a few FWHM apart, stars on and over the edges, cosmic rays (too
    sharp), a galaxy (too broad), a hot column and a sky gradient."""
    rng = np.random.default_rng(31)
    ny, nx = 160, 180
    yy, xx = np.mgrid[1:ny + 1, 1:nx + 1].astype(float)
    image = 200.0 + 0.3 * xx - 0.2 * yy

    def star(x, y, peak, sigma=1.3):
        return peak * np.exp(-((xx - x) ** 2 + (yy - y) ** 2) / (2 * sigma ** 2))

    for _ in range(100):
        image += star(rng.uniform(-2, nx + 3), rng.uniform(-2, ny + 3),
                      10 ** rng.uniform(1.5, 4.0))
    for x, y, dx, dy in ((40.2, 30.7, 2.1, 0.4), (90.6, 120.3, 4.3, -3.2),
                         (140.1, 60.5, 0.0, 6.0)):          # pairs
        image += star(x, y, 3000.0) + star(x + dx, y + dy, 1200.0)
    for x, y in ((1.2, 80.4), (180.0, 20.6), (70.3, 1.0), (120.7, 159.6),
                 (2.0, 2.0)):                                # on the edges
        image += star(x, y, 2500.0)
    image += star(100.4, 40.6, 600.0, sigma=6.0)             # a galaxy
    image[rng.integers(5, ny - 5, 8), rng.integers(5, nx - 5, 8)] += \
        rng.uniform(500, 4000, 8)                            # cosmic rays
    image[:, 150] += 150.0                                   # a hot column
    image += rng.normal(0, 6.0, image.shape)
    return image.astype(np.float32)


#: Where the frames of align_frames() sit relative to the first, (x, y).
ALIGN_SHIFTS = [(0.0, 0.0), (2.37, -1.64), (-4.71, 3.18), (0.42, 0.93)]


def align_frames():
    """Four exposures of one field for imcentroid and imalign: thirty stars
    and a sky that changes, each frame moved by ALIGN_SHIFTS and with its own
    noise. Returns (frames, coords): the stars' positions in the first frame,
    a hand's worth off -- plus one that leaves the image in the third frame,
    one on blank sky and one on a pair."""
    rng = np.random.default_rng(41)
    ny, nx = 120, 140
    yy, xx = np.mgrid[1:ny + 1, 1:nx + 1].astype(float)
    stars = [(rng.uniform(8, nx - 8), rng.uniform(8, ny - 8), 10 ** rng.uniform(2.5, 3.8))
             for _ in range(30)]
    stars += [(70.4, 60.2, 3000.0), (72.6, 61.0, 1500.0)]      # a pair
    stars += [(3.5, 100.3, 2500.0)]                           # leaves frame 3
    frames = []
    for k, (dx, dy) in enumerate(ALIGN_SHIFTS):
        image = 150.0 + 20 * k + 0.1 * xx
        for x, y, peak in stars:
            image = image + peak * np.exp(-((xx - x - dx) ** 2 + (yy - y - dy) ** 2)
                                          / (2 * 1.4 ** 2))
        image += np.random.default_rng(50 + k).normal(0, 5.0, image.shape)
        frames.append(image.astype(np.float32))
    picks = sorted(range(30), key=lambda i: -stars[i][2])[:12]
    coords = [(round(stars[i][0] + rng.uniform(-1, 1), 1),
               round(stars[i][1] + rng.uniform(-1, 1), 1)) for i in picks]
    coords += [(70.0, 60.0), (3.0, 100.0), (25.0, 115.0)]    # pair, edge, sky
    return frames, coords
