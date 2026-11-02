"""What the artdata and precess cases read, in IRAF and in the tests."""

from pathlib import Path

import numpy as np
from astropy.io import fits

COORDS = """12:30:10.12 10:18:27.5
12:30 10:18
12:30 -20 1900
12:30 -20 1900 2000
0 0
23:59:59.9 89:59:59
5.5 -45.25
5.5 -45.25 2000
5.5 -45.25 2000 2000
18:00 66:33 1950.0 2026.5
bad line
12:30
# a comment
12:30, 10
"""
CARDS = """OBJECT  = 'from a card file'
EXPTIME =                 12.5 / with a comment
this is not a card
   RDNOISE =                  5.0
COMMENT a comment card
badkey=1
"""


COSRAYS = """# cosmic rays
5.3 6.7 500
12.1 3.2 1000
300 3 50
9.9 9.9 2000
"""
STARS = """# stars
10.3 12.7 8.
25.5 5.25 7.
40.1 30.9 6.5
5. 35. 9.
300 300 5.
25.5 5.25 7.5
45.6 3.3 6.
"""
GALAXIES = """# stars and galaxies
12.3 14.7 8.
30.5 20.25 7. expdisk 4. 0.6 30.
15.1 30.9 6.5 devauc 3. 0.8 110.
40.2 40.2 7.5 sersic1.5 2. 1. 0.
40.2 10.2 7.5 sersic2.5 2. 0.5 45.
5.2 45.2 7.5 nosuch 2. 0.5 45.
"""


def write(where):
    where = Path(where)
    where.mkdir(parents=True, exist_ok=True)
    (where / "coords.txt").write_text(COORDS, encoding="utf-8")
    (where / "cards.dat").write_text(CARDS, encoding="utf-8")
    (where / "rays.txt").write_text(COSRAYS, encoding="utf-8")
    (where / "stars.txt").write_text(STARS, encoding="utf-8")
    (where / "galaxies.txt").write_text(GALAXIES, encoding="utf-8")
    # Profiles and functions from a file
    profile = [np.exp(-0.5 * (i / 3.0) ** 2) for i in range(16)]
    (where / "prof.txt").write_text("".join(f"{value:.6f}\n" for value in profile),
                                    encoding="utf-8")
    growth = [1.0 - np.exp(-0.5 * (i / 3.0) ** 2) for i in range(16)]
    (where / "grow.txt").write_text("".join(f"{value:.6f}\n" for value in growth),
                                    encoding="utf-8")
    (where / "protos.txt").write_text(STARS + "30.2 20.6 13.0 prof.txt 6.0 0.7 25.0\n"
                                      "12.4 33.8 13.5 grow.txt 4.0 1.0 0.0\n",
                                              encoding="utf-8")
    y, x = np.mgrid[-10:11, -12:13].astype(np.float64)
    star = np.exp(-(x * x / 8.0 + y * y / 5.0)) + 0.002 * (x + 12)
    fits.PrimaryHDU(star.astype(np.float32)).writeto(where / "psfimg.fits", overwrite=True)
    disk = np.exp(-np.hypot(x, y * 1.3) / 3.0)
    fits.PrimaryHDU(disk.astype(np.float32)).writeto(where / "galimg.fits", overwrite=True)
    (where / "imgals.txt").write_text(STARS + "30.2 20.6 13.0 galimg.fits 6.0 0.7 0.0\n"
                                      "12.4 33.8 13.5 galimg.fits 4.0 1.0 35.0\n",
                                              encoding="utf-8")
    (where / "density.txt").write_text("".join(f"{r} {np.exp(-r / 80.0):.5f}\n"
                                               for r in range(0, 400, 10)), encoding="utf-8")
    (where / "lum.txt").write_text("".join(f"{m} {10 ** (0.3 * (m - 5)):.5f}\n"
                                           for m in range(5, 21)), encoding="utf-8")
    a = (np.arange(48).reshape(6, 8) * 1.5 + 3).astype(np.float32)
    h = fits.Header({"OBJECT": "base", "EXPTIME": 3.0})
    fits.PrimaryHDU(a, h).writeto(where / "base.fits", overwrite=True)
    fits.PrimaryHDU(np.round(a).astype(np.int16), h).writeto(
        where / "bases.fits", overwrite=True
    )
    fits.PrimaryHDU(a.astype(np.float64), h).writeto(where / "based.fits", overwrite=True)
