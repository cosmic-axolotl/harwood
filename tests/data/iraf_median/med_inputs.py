"""The images of IRAF's median gabarito: python med_inputs.py <folder> (harwood's
venv, for astropy). Values that tie often, a gradient for the boundaries to show,
spikes for the rejection; images narrower and shorter than the windows, 1-D, one
column, short, and a NaN."""

import sys
from pathlib import Path

import numpy as np
from astropy.io import fits


def _field(ny, nx, seed):
    y, x = np.indices((ny, nx))
    v = ((x * 7 + y * 13 + seed) % 11) * 3.0 + 0.5 * x - 0.25 * y
    v[(x * 5 + y * 3 + seed) % 17 == 0] += 500.0
    v[(x * 3 + y * 7 + seed) % 19 == 0] -= 400.0
    return v


def images():
    yield "a", _field(17, 23, 1).astype(np.float32)
    yield "s", _field(11, 15, 2).astype(np.int16)
    yield "narrow", _field(9, 4, 3).astype(np.float32)
    yield "tiny", _field(2, 3, 4).astype(np.float32)
    yield "one", _field(1, 30, 5)[0].astype(np.float32)
    yield "col", _field(12, 1, 6).astype(np.float32)
    nan = _field(8, 9, 7).astype(np.float32)
    nan[3, 4] = np.nan
    yield "nan", nan
    yield "d", _field(10, 12, 8).astype(np.float64)


def write(where: Path) -> None:
    where.mkdir(parents=True, exist_ok=True)
    for name, data in images():
        hdu = fits.PrimaryHDU(data)
        hdu.header["OBJECT"] = name
        hdu.writeto(where / f"{name}.fits", overwrite=True)


if __name__ == "__main__":
    write(Path(sys.argv[1]))
