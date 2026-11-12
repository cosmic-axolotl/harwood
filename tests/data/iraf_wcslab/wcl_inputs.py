"""The images of the wcslab gabarito: only their size and WCS matter to wcslab, so
they are made here (a Gaussian in the middle, for contour to have something to
draw), each with the header keys IRAF reads.

    python wcl_inputs.py <folder>
"""

import math
import sys
from pathlib import Path

import numpy as np
from astropy.io import fits


def _tan(ra, dec, crpix, scale, rot=0.0, flip=False, proj="TAN"):
    a = math.radians(rot)
    s = scale / 3600.0
    cards = {"CTYPE1": f"RA---{proj}", "CTYPE2": f"DEC--{proj}", "CRVAL1": ra,
             "CRVAL2": dec, "CRPIX1": crpix[0], "CRPIX2": crpix[1],
             "CD1_1": -s * math.cos(a), "CD1_2": s * math.sin(a),
             "CD2_1": s * math.sin(a), "CD2_2": s * math.cos(a)}
    if flip:
        cards = {"CTYPE1": f"DEC--{proj}", "CTYPE2": f"RA---{proj}", "CRVAL1": dec,
                 "CRVAL2": ra, "CRPIX1": crpix[0], "CRPIX2": crpix[1],
                 "CD1_1": 0.0, "CD1_2": s, "CD2_1": -s, "CD2_2": 0.0}
    return cards


IMAGES = {
    "wpix.fits": ((512, 512), {"OBJECT": "m51  B  600s", "EPOCH": 0.0, "CRPIX1": 257.75,
                               "CRPIX2": 258.93, "CRVAL1": 201.94541667302,
                               "CRVAL2": 47.45444, "CDELT1": -2.1277777e-4,
                               "CDELT2": 2.1277777e-4, "CTYPE1": "RA---TAN",
                               "CTYPE2": "DEC--TAN"}),
    "rot.fits": ((300, 200), {"OBJECT": "rotated", **_tan(75.0, 20.0, (150, 100), 1.0,
                                                          30.0)}),
    "south.fits": ((400, 400), {"OBJECT": "across 0h", **_tan(0.02, -35.0, (200, 200),
                                                              1.0)}),
    "polar.fits": ((512, 512), {"OBJECT": "pole", **_tan(40.0, 89.95, (256, 256), 2.0)}),
    "southpolar.fits": ((512, 512), {"OBJECT": "south pole",
                                     **_tan(300.0, -89.95, (240, 270), 2.0)}),
    "nearpolar.fits": ((512, 512), {"OBJECT": "near the pole",
                                    **_tan(120.0, 89.80, (256, 256), 2.0)}),
    "wide.fits": ((500, 400), {"OBJECT": "wide", **_tan(150.0, 30.0, (250, 200), 72.0)}),
    "tiny.fits": ((100, 100), {"OBJECT": "tiny", **_tan(200.0, 10.0, (50, 50), 0.002)}),
    "small.fits": ((200, 200), {"OBJECT": "small", **_tan(200.0, -10.0, (100, 100),
                                                          0.05)}),
    "linear.fits": ((300, 200), {"OBJECT": "linear", "CTYPE1": "LINEAR",
                                 "CTYPE2": "LINEAR", "CRVAL1": 100.0, "CDELT1": 2.5,
                                 "CRPIX1": 1.0, "CRVAL2": -3.0, "CDELT2": 0.01,
                                 "CRPIX2": 1.0}),
    "linrot.fits": ((300, 200), {"OBJECT": "linear, rotated", "CTYPE1": "LINEAR",
                                 "CTYPE2": "LINEAR", "CRVAL1": 100.0, "CRVAL2": -3.0,
                                 "CRPIX1": 150.0, "CRPIX2": 100.0, "CD1_1": 0.94,
                                 "CD1_2": -0.34, "CD2_1": 0.34, "CD2_2": 0.94}),
    "nowcs.fits": ((120, 80), {"OBJECT": "no wcs"}),
    "flip.fits": ((300, 300), {"OBJECT": "transposed",
                               **_tan(210.0, 40.0, (150, 150), 1.0, flip=True)}),
    "arc.fits": ((300, 300), {"OBJECT": "arc", **_tan(80.0, -60.0, (150, 150), 3.0,
                                                      proj="ARC")}),
    "zea.fits": ((300, 300), {"OBJECT": "zea", **_tan(80.0, 30.0, (150, 150), 3.0,
                                                      proj="ZEA")}),
    "rect.fits": ((600, 150), {"OBJECT": "rectangle", **_tan(10.0, 5.0, (300, 75),
                                                             1.0)}),
}


def write(folder) -> None:
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    for name, ((nx, ny), cards) in IMAGES.items():
        header = fits.Header()
        for key, value in cards.items():
            header[key] = value
        y, x = np.mgrid[:ny, :nx]
        blob = 1000 * np.exp(-((x - nx / 2) ** 2 + (y - ny / 2) ** 2) / (0.02 * nx * ny))
        fits.PrimaryHDU(blob.astype(np.int16), header).writeto(folder / name,
                                                               overwrite=True)


if __name__ == "__main__":
    write(sys.argv[1])
