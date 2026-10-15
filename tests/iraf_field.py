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
