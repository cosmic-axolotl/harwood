"""The images of IRAF's imsum gabarito: python ims_inputs.py <folder> (harwood's venv,
for astropy). Twenty short and twenty real frames whose values tie often and whose
sums leave a short's range, one of each other pixel type IRAF reads from FITS, 1-D
images, cubes and one of another size; EXPTIME and AIRMASS for hparams."""

import sys
from pathlib import Path

import numpy as np
from astropy.io import fits

NX, NY = 7, 5


def _grid(k):
    y, x = np.indices((NY, NX))
    return ((x * 3 + y * 5 + k * 7) % 9).astype(np.float64)


def images():
    for k in range(1, 21):
        real = _grid(k) - 3.5 + (k % 3) * 0.5
        if k == 3:
            real[2, 3] = np.nan
            real[0, 0] = -0.0
        yield f"r{k:02d}", real.astype(np.float32), k
        yield f"s{k:02d}", (_grid(k) * 1000 + k * 100 - 4000).astype(np.int16), k
    yield "u01", (_grid(1) * 4000 + 30000).astype(np.uint16), 1
    yield "u02", (_grid(2) * 4000 + 30000).astype(np.uint16), 2
    yield "i01", (_grid(4) * 1000003 - 4000000).astype(np.int32), 4
    yield "d01", (_grid(5) / 3.0 - 1.1).astype(np.float64), 5
    yield "b01", (_grid(6) * 25).astype(np.uint8), 6
    for k in (1, 2, 3):
        yield f"one{k}", (_grid(k)[1] * 1.25 - 2).astype(np.float32), k
        yield f"cube{k}", np.stack([_grid(k), _grid(k + 1), _grid(k + 2)]
                                   ).astype(np.int16) * 7 - 20, k
    yield "wide", np.zeros((NY, NX + 2), dtype=np.float32), 1


def write(where: Path) -> None:
    where.mkdir(parents=True, exist_ok=True)
    for name, data, k in images():
        hdu = fits.PrimaryHDU(data)
        hdu.header["OBJECT"] = f"frame {name}"
        hdu.header["EXPTIME"] = 10.5 * k
        hdu.header["AIRMASS"] = round(1.0 + 0.013 * k, 4)
        hdu.writeto(where / f"{name}.fits", overwrite=True)


if __name__ == "__main__":
    write(Path(sys.argv[1]))
