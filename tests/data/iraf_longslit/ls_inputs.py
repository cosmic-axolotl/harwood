"""The images of IRAF's response and illumination gabarito: python ls_inputs.py
<folder> (harwood's venv, for astropy).

flat2: a dome flat dispersed along the lines (DISPAXIS=2, as the course's), 120
lines by 30 columns -- a lamp spectrum along the lines, an illumination that
falls along the slit, pixel-to-pixel noise. sky2: a twilight sky the same way,
with a star in it. flat1 and sky1: the same transposed, DISPAXIS=1. nodisp: flat2
without DISPAXIS. pre: an existing response image of ones and twos, for a
response written into a section of it."""

import sys
from pathlib import Path

import numpy as np
from astropy.io import fits


def _frames():
    rng = np.random.default_rng(20261110)
    y, x = np.indices((120, 30)).astype(float)
    lamp = 800 + 600 * np.exp(-((y - 70) / 35) ** 2) + 3 * y
    slit = 1.0 - 0.004 * (x - 15) ** 2 / 15 + 0.006 * (x - 15)
    flat = lamp * slit * (1 + 0.02 * rng.standard_normal(y.shape))
    sky = (400 + 2 * y) * (slit + 0.01 * (y / 120)) * (1 + 0.01 * rng.standard_normal(y.shape))
    sky += 900 * np.exp(-0.5 * ((x - 21.3) / 1.2) ** 2)
    return flat.astype(np.float32), sky.astype(np.float32)


def write(where: Path) -> None:
    where.mkdir(parents=True, exist_ok=True)
    flat, sky = _frames()
    for name, data, axis in (("flat2", flat, 2), ("sky2", sky, 2), ("flat1", flat.T, 1),
                             ("sky1", sky.T, 1), ("nodisp", flat, None)):
        hdu = fits.PrimaryHDU(np.ascontiguousarray(data))
        hdu.header["OBJECT"] = name
        if axis:
            hdu.header["DISPAXIS"] = axis
        hdu.writeto(where / f"{name}.fits", overwrite=True)
    pre = np.where(np.indices(flat.shape)[1] % 2, 2.0, 1.0).astype(np.float32)
    hdu = fits.PrimaryHDU(pre)
    hdu.header["DISPAXIS"] = 2
    hdu.writeto(where / "pre.fits", overwrite=True)


if __name__ == "__main__":
    write(Path(sys.argv[1]))
