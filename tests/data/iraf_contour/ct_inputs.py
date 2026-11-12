"""The images of the contour gabarito: python ct_inputs.py <folder> (harwood's venv).

A galaxy-like field (a bright nucleus, a disk, a few stars, noise) of 200 by 150,
the same field with negative values, scaled to small and to large numbers, a
small one contoured at its own resolution, and a constant image."""

import sys
from pathlib import Path

import numpy as np
from astropy.io import fits


def field(nx, ny, rng):
    y, x = np.mgrid[1:ny + 1, 1:nx + 1]
    f = (2000 * np.exp(-0.5 * (((x - 0.45 * nx) / 6) ** 2 + ((y - 0.55 * ny) / 5) ** 2))
         + 300 * np.exp(-0.5 * (((x - 0.45 * nx) / 30) ** 2 + ((y - 0.55 * ny) / 18) ** 2))
         + 100 + 0.2 * x)
    for sx, sy, a in ((0.2, 0.2, 500), (0.8, 0.7, 800), (0.7, 0.2, 300)):
        f += a * np.exp(-0.5 * (((x - sx * nx) / 2) ** 2 + ((y - sy * ny) / 2) ** 2))
    return (f + rng.normal(0, 5, f.shape)).astype(np.float32)


def write(where: Path) -> None:
    where.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(17)
    f = field(200, 150, rng)
    for name, data, title in (("gal", f, "a galaxy"), ("neg", f - 600.0, "below zero"),
                              ("small", f * 1e-4, "small numbers"),
                              ("large", f * 1e3, "large numbers"),
                              ("tiny", field(40, 30, rng), "a small one"),
                              ("flat", np.full((20, 20), 7.0, np.float32), "constant")):
        h = fits.Header()
        h["OBJECT"] = title
        fits.writeto(where / f"{name}.fits", data.astype(np.float32), h, overwrite=True)


if __name__ == "__main__":
    write(Path(sys.argv[1]))
