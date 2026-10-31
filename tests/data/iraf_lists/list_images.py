"""The images minmax and listpixels are run on, in IRAF and in the tests."""

from pathlib import Path

import numpy as np
from astropy.io import fits

TAN = {"CTYPE1": "RA---TAN", "CTYPE2": "DEC--TAN", "CRPIX1": 2.0, "CRPIX2": 1.5,
       "CRVAL1": 150.0, "CRVAL2": -20.0, "CD1_1": -1e-4, "CD1_2": 0.0, "CD2_1": 0.0,
       "CD2_2": 1e-4, "LTV1": -3.0, "LTV2": -4.0, "LTM1_1": 1.0, "LTM2_2": 1.0}
TILT = {"CTYPE1": "RA---TAN", "CTYPE2": "DEC--TAN", "CRPIX1": 12.5, "CRPIX2": -3.25,
        "CRVAL1": 359.9, "CRVAL2": 62.0, "CD1_1": -2.1e-4, "CD1_2": 3.3e-5,
        "CD2_1": 3.1e-5, "CD2_2": 2.2e-4}
LINEAR = {"CTYPE1": "LINEAR", "CTYPE2": "LINEAR", "CRPIX1": 1.0, "CRPIX2": 1.0,
          "CRVAL1": 4000.0, "CRVAL2": 10.0, "CDELT1": 2.5, "CDELT2": 0.5,
          "CD1_1": 2.5, "CD2_2": 0.5}
SHEAR = {"CTYPE1": "LINEAR", "CTYPE2": "LINEAR", "CRPIX1": 2.0, "CRPIX2": 2.0,
         "CRVAL1": 100.0, "CRVAL2": 200.0, "CD1_1": 2.0, "CD1_2": 0.5, "CD2_1": -0.25,
         "CD2_2": 3.0}
CDELT = {"CRPIX1": 2.0, "CRVAL1": 7.0, "CDELT1": 0.5}
PHYSICAL = {"LTV1": -3.0, "LTV2": -4.0, "LTM1_1": 0.5, "LTM2_2": 0.5}
SPECTRUM = {"CTYPE1": "LINEAR", "CRPIX1": 1.0, "CRVAL1": 4000.0, "CDELT1": 2.5,
            "CD1_1": 2.5, "DC-FLAG": 0, "WCSDIM": 1, "WAT0_001": "system=equispec",
            "WAT1_001": "wtype=linear label=Wavelength units=angstroms"}


def small():
    yy, xx = np.indices((3, 4))
    return (100 + 3 * xx + 2 * yy + 0.25 * ((xx * 7 + yy * 13) % 11)).astype(np.float32)


def frame():
    """Repeated extremes, to see which one is reported."""
    a = small()
    a = np.block([[a, a + 1], [a + 2, a + 3]]).astype(np.float32)
    a[3, 2], a[4, 5], a[1, 4], a[0, 1] = -4.5, -4.5, 1234.5, 1234.5
    return a


def write(where):
    where = Path(where)
    where.mkdir(parents=True, exist_ok=True)

    def w(name, data, header=None, **kw):
        fits.PrimaryHDU(data, fits.Header(header) if header else None).writeto(
            where / f"{name}.fits", overwrite=True)

    a, f = small(), frame()
    w("real", a)
    w("short", a.astype(np.int16))
    w("double", a.astype(np.float64) + 1e-9)
    w("long", a.astype(np.int32) * 10000000)
    w("oned", a[0])
    w("cube", np.stack([a, a + 1]))
    w("big", a * 1e7 + 0.1)
    w("tiny", a * 3.3e-9)
    w("tan", a, TAN)
    w("tilt", a, TILT)
    w("linear", a, LINEAR)
    w("shear", a, SHEAR)
    w("cdelt", a[0], CDELT)
    w("phys", a, PHYSICAL)
    w("spec", a[0], SPECTRUM)

    w("f", f)
    w("fs", f.astype(np.int16))
    w("fd", f.astype(np.float64) + 1e-9)
    w("fl", f.astype(np.int32) * 100000)
    w("fb", np.clip(f, 0, 255).astype(np.uint8))
    w("f1", f[0])
    w("f3", np.stack([f, f - 5, f + 5]))
    w("flat", np.full((3, 3), 7.0, np.float32))
    w("fine", (f + 0.1234567).astype(np.float32))
    w("huge", f.astype(np.float64) * 1e20)
    u = (f.astype(np.int32) + 40000).clip(0, 65535).astype(np.uint16)
    fits.PrimaryHDU(u).writeto(where / "fu.fits", overwrite=True)
    h = fits.PrimaryHDU(f.astype(np.int16))
    h.header["BSCALE"], h.header["BZERO"] = 0.5, 10.0
    h.writeto(where / "fscale.fits", overwrite=True)
    w("limits", f, {"DATAMIN": -1.0, "DATAMAX": 50.0})
    w("lmin", f, {"DATAMIN": -1.0})
    w("lmax", f, {"DATAMAX": 9.0})
    w("lfine", f, {"DATAMIN": 0.1, "DATAMAX": 1234.5678})
    w("lzero", f, {"DATAMIN": 0.0})
    w("lequal", f, {"DATAMIN": 5.0, "DATAMAX": 5.0})
    w("tanwat", small(), {**TAN, "WAT1_001": "wtype=tan axtype=ra format=%H",
                          "WAT2_001": "wtype=tan axtype=dec format=%.2h"})
    w("specwat", small()[0], {**SPECTRUM, "WAT1_001": "wtype=linear label=Wavelength "
                                                      "units=angstroms format=%8.2f"})
    mef = fits.HDUList([fits.PrimaryHDU(), fits.ImageHDU(f, name="SCI"),
                        fits.ImageHDU(f * 2, name="ERR")])
    mef.writeto(where / "mef.fits", overwrite=True)
    fits.PrimaryHDU().writeto(where / "empty.fits", overwrite=True)
