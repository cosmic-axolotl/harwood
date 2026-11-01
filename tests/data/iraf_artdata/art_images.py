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
    (where / "coords.txt").write_text(COORDS)
    (where / "cards.dat").write_text(CARDS)
    (where / "rays.txt").write_text(COSRAYS)
    (where / "stars.txt").write_text(STARS)
    (where / "galaxies.txt").write_text(GALAXIES)
    a = (np.arange(48).reshape(6, 8) * 1.5 + 3).astype(np.float32)
    h = fits.Header({"OBJECT": "base", "EXPTIME": 3.0})
    fits.PrimaryHDU(a, h).writeto(where / "base.fits", overwrite=True)
    fits.PrimaryHDU(np.round(a).astype(np.int16), h).writeto(
        where / "bases.fits", overwrite=True
    )
    fits.PrimaryHDU(a.astype(np.float64), h).writeto(where / "based.fits", overwrite=True)
