"""The images imexpr is run on, in IRAF and in the tests."""

from pathlib import Path

import numpy as np
from astropy.io import fits

KEYS = {"EXPTIME": 30.5, "NCOMB": 7, "OBJECT": "M92 field", "FLAG": True, "OTHER": False,
        "CRPIX1": 2.0, "CRPIX2": 1.5, "CRVAL1": 100.0, "CRVAL2": 200.0, "CD1_1": 0.5,
        "CD2_2": 0.25, "CTYPE1": "LINEAR", "CTYPE2": "LINEAR"}


def base():
    yy, xx = np.indices((3, 4))
    return (10 + 3 * xx - 7 * yy + 0.37 * ((xx * 7 + yy * 13) % 11)).astype(np.float32)


def other():
    yy, xx = np.indices((3, 4))
    a = (2 + xx - 2 * yy + 0.11 * ((xx * 5 + yy * 3) % 7)).astype(np.float32)
    a[1, 2] = 0.0
    return a


def write(where):
    where = Path(where)
    where.mkdir(parents=True, exist_ok=True)

    def w(name, data, header=None):
        fits.PrimaryHDU(data, fits.Header(header) if header else None).writeto(
            where / f"{name}.fits", overwrite=True)

    a, b = base(), other()
    w("ra", a, KEYS)
    w("rb", b)
    w("rc", a[::-1] * 0.5 + 1)
    w("rd", b + a)
    w("re", a - b)
    w("sa", np.round(a * 1000).astype(np.int16), {"OBJECT": "short"})
    w("sb", np.round(b * 3).astype(np.int16))
    w("ia", np.round(a * 100000).astype(np.int32))
    w("ib", np.round(b * 7).astype(np.int32))
    w("da", a.astype(np.float64) + 1e-9)
    w("db", b.astype(np.float64) * 1.000000001)
    fits.PrimaryHDU((np.round(a * 100) + 3000).astype(np.int32).clip(0, 65535)
                    .astype(np.uint16)).writeto(where / "ua.fits", overwrite=True)
    w("la", np.round(a * 1000).astype(np.int64))
    h = fits.PrimaryHDU(np.round(a).astype(np.int16))
    h.header["BSCALE"], h.header["BZERO"] = 0.5, 10.0
    h.writeto(where / "sc.fits", overwrite=True)
    w("oned", a[0])
    w("onedi", np.round(a[1]).astype(np.int16))
    w("long1", np.arange(1.0, 7.0, dtype=np.float32) * 1.5)
    w("cube", np.stack([a, a + 100, a + 200]))
    w("tall", np.vstack([a, a + 50]), {"OBJECT": "tall"})
    w("wide", np.hstack([a, a + 50]))
    w("sorted", np.sort(a.ravel()).reshape(3, 4) * 1.1)
    w("rough", np.where(a > 12, a, a * 1000).astype(np.float32))
    w("target", np.full((3, 4), -1.0, np.float32), {"OBJECT": "target"})
    w("target2", np.full((3, 4), -1, np.int16))
