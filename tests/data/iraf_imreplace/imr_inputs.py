"""The images of IRAF's imreplace gabarito: python imr_inputs.py <folder> (harwood's
venv, for astropy). One of each pixel type IRAF reads from FITS, on a pattern that
crosses the limits the cases use from both sides, integral and not."""

import sys
from pathlib import Path

import numpy as np
from astropy.io import fits

NX, NY = 11, 9


def pattern():
    y, x = np.indices((NY, NX))
    return ((x * 37 + y * 101) % 451 - 225).astype(np.float64)


def images():
    base = pattern()
    shorts = base.astype(np.int16)
    shorts[0, 0], shorts[0, 1], shorts[NY - 1, NX - 1] = -32768, -32767, 32767
    floats = base + np.where((base.astype(int) % 2) == 0, 0.5, -0.3)
    floats[4, 5] = -0.0
    yield "s16", shorts, {}
    yield "u16", (base + 300).astype(np.uint16), {}
    yield "i32", (base * 1000).astype(np.int32), {}
    yield "r32", floats.astype(np.float32), {}
    yield "r64", (floats + 1e-9 * base).astype(np.float64), {}
    yield "b8", (base + 225).clip(0, 255).astype(np.uint8), {}
    yield "one", floats[3].astype(np.float32), {}
    cube = np.stack([shorts, shorts[::-1], -shorts]).astype(np.int16)
    cube[1, 0, 0] = 5
    yield "cube", cube, {}
    yield "scaled", shorts, {"BSCALE": 0.5, "BZERO": 10.0}
    # What IRAF's IEEE input maps: NaN and infinities, zeros of either sign and
    # numbers too small for the type's normal range.
    odd = np.array([np.nan, np.inf, -np.inf, 1e-40, -0.0, 5.0, -1e-42, 3.0, 1e-310,
                    -4.0, 4.4])
    rows = np.vstack([odd, odd[::-1], np.roll(odd, 3)])
    with np.errstate(over="ignore", under="ignore"):
        yield "odd32", rows.astype(np.float32), {}
    yield "odd64", rows, {}


def write(where: Path) -> None:
    where.mkdir(parents=True, exist_ok=True)
    for name, data, scaling in images():
        hdu = fits.PrimaryHDU(data)
        hdu.header["OBJECT"] = name
        hdu.writeto(where / f"{name}.fits", overwrite=True)
        if scaling:
            # The stored integers stay as they are; IRAF reads them times BSCALE
            # plus BZERO, as a real image.
            with fits.open(where / f"{name}.fits", mode="update",
                           do_not_scale_image_data=True) as hdul:
                hdul[0].header.update(scaling)

if __name__ == "__main__":
    write(Path(sys.argv[1]))
