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
