"""The files imextensions and mimstatistics are run on, in IRAF and in the tests."""

from pathlib import Path

import numpy as np
from astropy.io import fits


def frame():
    yy, xx = np.indices((20, 30))
    a = 100 + 3 * xx - 2 * yy + 7.3 * ((xx * 7 + yy * 13) % 11)
    a = a.astype(np.float32)
    a[5, 5], a[12, 20], a[3, 28] = 9000.0, -500.0, 4000.0
    return a


def mask():
    m = np.zeros((20, 30), np.int16)
    m[5, 5] = 1
    m[3:6, 26:30] = 1
    m[10:15, 18:23] = 3
    return m


def write(where):
    where = Path(where)
    where.mkdir(parents=True, exist_ok=True)
    a = frame()

    def im(d, name=None, ver=None):
        hdu = fits.ImageHDU(d, ver=ver)
        if name:
            hdu.header["EXTNAME"] = name
        return hdu

    fits.HDUList([fits.PrimaryHDU(), im(a, "SCI", 1), im(a + 1, "SCI", 2), im(a + 2, "ERR", 1),
                  im(a + 3), im(a + 4, "joy"), im(a + 5, "nojoy"), im(a + 6, "joyfull"),
                  im(a + 7, "Joy")]).writeto(where / "mef.fits", overwrite=True)
    fits.HDUList([fits.PrimaryHDU(a), im(a + 1, "SCI")]).writeto(where / "withdata.fits",
                                                                 overwrite=True)
    table = fits.BinTableHDU.from_columns([fits.Column("x", "E", array=np.arange(3.0))],
                                          name="TAB")
    ascii_table = fits.TableHDU.from_columns(
        [fits.Column("y", "E12.4", ascii=True, array=np.arange(3.0))], name="ATAB")
    fits.HDUList([fits.PrimaryHDU(), im(a, "A"), table, im(a, "B"), ascii_table,
                  im(a, "C")]).writeto(where / "tables.fits", overwrite=True)
    fits.PrimaryHDU(a).writeto(where / "simple.fits", overwrite=True)
    fits.PrimaryHDU().writeto(where / "empty.fits", overwrite=True)
    (where / "notfits.txt").write_text("hello\n")
    fits.PrimaryHDU(mask()).writeto(where / "m.fits", overwrite=True)
    fits.PrimaryHDU(mask() * 0).writeto(where / "m0.fits", overwrite=True)
    fits.PrimaryHDU(a + 5).writeto(where / "b.fits", overwrite=True)
    fits.PrimaryHDU(np.round(a).astype(np.int16)).writeto(where / "s.fits", overwrite=True)
    fits.PrimaryHDU(mask()[::-1]).writeto(where / "m2.fits", overwrite=True)
