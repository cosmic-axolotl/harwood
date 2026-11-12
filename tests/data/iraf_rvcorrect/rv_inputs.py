"""The inputs of the rvcorrect gabarito: python rv_inputs.py <folder> (harwood's venv).

Observation lists (date, UT, RA, Dec, with and without the epoch and the observed
velocity, a line that is not one) and images whose headers carry the date as
yyyy-mm-dd, dd/mm/yy and with its time, an OBSERVAT, a VOBS, and one without an
EPOCH."""

import sys
from pathlib import Path

import numpy as np
from astropy.io import fits

OBS = """\
1985 12 19 3.35 1:37:41.3 13:08:44 1950 780
2026 3 1 22.5 12:00:00 -30:00:00
1990 6 15 0.0 18:36:56.3 38:47:01 2000 -13.2
# a comment
2001 1 1
1999 12 31 23:59:59 5:35:17.3 -5:23:28 2000 25.0
2010 7 4 6.0 23.9 89.5 1950.0 -100.0
"""


def write(where: Path) -> None:
    where.mkdir(parents=True, exist_ok=True)
    (where / "obs.txt").write_text(OBS, encoding="utf-8")
    (where / "more.txt").write_text("1975 2 28 12.0 0:00:00 0:00:00 1950\n"
                                    "2020 9 22 18.25 14:15:39.7 19:10:57 2000 -5.2\n",
                                    encoding="utf-8")
    data = np.zeros((4, 4), np.float32)
    cards = [
        ("im1", {"DATE-OBS": "1985-12-19", "UT": "03:21:00", "RA": "01:37:41.3",
                 "DEC": "13:08:44", "EPOCH": 1950.0, "VOBS": 780.0}),
        ("im2", {"DATE-OBS": "19/12/85", "UT": "03:21:00", "RA": "01:37:41.3",
                 "DEC": "13:08:44", "EPOCH": 1950.0, "OBSERVAT": "CTIO"}),
        ("im3", {"DATE-OBS": "2005-08-12T04:15:30", "UT": "01:00:00", "RA": "21:29:58.4",
                 "DEC": "12:10:01", "EPOCH": 2000.0, "VOBS": -105.5}),
        ("im4", {"DATE-OBS": "2005-08-12", "UT": "2005-08-12T04:15:30", "RA": "21:29:58.4",
                 "DEC": "12:10:01", "EPOCH": 2000.0, "OBSERVAT": "lasilla"}),
        ("noepoch", {"DATE-OBS": "2005-08-12", "UT": "04:15:30", "RA": "21:29:58.4",
                     "DEC": "12:10:01"}),
    ]
    for name, h in cards:
        header = fits.Header()
        header.update(h)
        fits.writeto(where / f"{name}.fits", data, header, overwrite=True)


if __name__ == "__main__":
    write(Path(sys.argv[1]))
