"""A small night to try things on: the data every documentation example runs on.

Written on demand (`harwood docs --sample DIR`), never shipped as files, and
the same bytes every time -- the examples quote numbers, and the tests run
the examples. Small, so that each example takes a moment rather than a
minute; real enough that a reduction of it means something:

    bias001-003.fits   zero frames: a 500 ADU level, 5 ADU read noise
    flat001-003.fits   V dome flats: vignetting and a pixel-to-pixel pattern
    obj001-003.fits    a V field of thirty stars (FWHM 3 px) on 300 ADU of
                       sky, the telescope drifting a few pixels between them;
                       obj003 was hit by eight cosmic rays
    bias.lis, flat.lis, obj.lis     the IRAF @lists of each
    bright.coo         three bright stars on obj001, as x y (1-based)
    stars.cur          imexamine's cursor file: r and a on the two
                       brightest stars

Every frame is 120 x 100 with an overscan strip in columns 105-120 (BIASSEC)
and the data in [1:100,1:100] (TRIMSEC), as a CCD of the 1980s wrote them,
with IMAGETYP, FILTERS, EXPTIME, GAIN, RDNOISE, the time and the pointing in
the header.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

NX, NY = 120, 100                   #: the whole frame, overscan included
DATA = (100, 100)                   #: the part with light on it
BIASSEC, TRIMSEC = "[105:120,1:100]", "[1:100,1:100]"
LEVEL, READ_NOISE, GAIN = 500.0, 5.0, 2.0
SKY, FWHM = 300.0, 3.0
#: Where the telescope pointed for each object frame, in pixels from obj001.
DRIFT = [(0.0, 0.0), (2.3, -1.4), (-1.1, 3.2)]
#: The cosmic rays on obj003: (x, y) of the hit pixel, 1-based, and its counts
#: above the sky. Each also lights one neighbour, as a track at an angle does.
COSMIC_RAYS = [(8, 52, 2600.0), (17, 93, 4100.0), (31, 6, 1800.0), (52, 27, 3300.0),
               (64, 95, 5200.0), (71, 3, 2200.0), (93, 61, 3900.0), (97, 86, 2900.0)]


def write(directory: Path | str) -> list[Path]:
    """Write the night into `directory` (made if needed). Returns the files."""
    from astropy.io import fits

    where = Path(directory)
    where.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(1987)
    # Fixed for the night: the shape of the illumination and of the pixels.
    y, x = np.mgrid[1:DATA[1] + 1, 1:DATA[0] + 1]
    vignetting = 1.0 - 0.08 * (((x - 50.5) / 50) ** 2 + ((y - 50.5) / 50) ** 2)
    response = 1.0 + 0.01 * rng.standard_normal(DATA[::-1])
    stars = _stars(rng)

    written: list[Path] = []
    lists: dict[str, list[str]] = {"bias": [], "flat": [], "obj": []}

    def frame(name: str, light: np.ndarray, kind: str, exptime: float, hour: int,
              minute: int, title: str) -> None:
        image = np.empty((NY, NX))
        image[:, :DATA[0]] = LEVEL + light
        image[:, DATA[0]:] = LEVEL
        # Where the light is, its photon noise; everywhere, the read noise.
        image[:, :DATA[0]] += rng.normal(0, 1, DATA[::-1]) * np.sqrt(np.maximum(light, 0)
                                                                        / GAIN)
        image += rng.normal(0, READ_NOISE, image.shape)
        header = fits.Header()
        for key, value in (("OBJECT", title), ("IMAGETYP", kind), ("FILTERS", "V"),
                           ("EXPTIME", exptime), ("DARKTIME", exptime),
                           ("BIASSEC", BIASSEC), ("TRIMSEC", TRIMSEC),
                           ("DATASEC", TRIMSEC), ("GAIN", GAIN), ("RDNOISE", READ_NOISE),
                           ("DATE-OBS", "2026-09-13"), ("UT", f"{hour:02d}:{minute:02d}:00"),
                           ("RA", "17:17:07.4"), ("DEC", "43:08:09"), ("EPOCH", 2000.0),
                           ("OBSERVAT", "kpno")):
            header[key] = value
        path = where / f"{name}.fits"
        fits.PrimaryHDU(np.round(image).astype(np.int16), header).writeto(path,
                                                                         overwrite=True)
        written.append(path)
        lists[name.rstrip("0123456789")].append(path.name)

    for number in range(1, 4):
        frame(f"bias{number:03d}", np.zeros(DATA[::-1]), "zero", 0.0, 1, number, "bias")
    for number in range(1, 4):
        frame(f"flat{number:03d}", 20000.0 * vignetting * response, "flat", 5.0, 1,
              10 + number, "dome flat V")
    for number, (dx, dy) in enumerate(DRIFT, start=1):
        light = (SKY + _render(stars, dx, dy)) * vignetting * response
        if number == 3:
            # Not light: a cosmic ray deposits charge, unvignetted and unflat.
            for x, y, counts in COSMIC_RAYS:
                light[y - 1, x - 1] += counts
                light[y - 1, x - 2] += 0.4 * counts
        frame(f"obj{number:03d}", light, "object", 60.0, 2, 5 * number, "field 1 V")

    for name, members in lists.items():
        path = where / f"{name}.lis"
        path.write_text("\n".join(members) + "\n", encoding="utf-8")
        written.append(path)
    bright = sorted(stars, key=lambda s: -s[2])[:3]
    path = where / "bright.coo"
    path.write_text("# three bright stars on obj001, 1-based as IRAF and DS9 count\n"
                    "x y\n" + "".join(f"{sx:.2f} {sy:.2f}\n" for sx, sy, _ in bright),
                    encoding="utf-8")
    written.append(path)
    path = where / "stars.cur"
    path.write_text("# imexamine's cursor: x y frame key, one key press a line\n"
                    + "".join(f"{sx:.1f} {sy:.1f} 1 {key}\n"
                              for sx, sy, _ in bright[:2] for key in "ra"), encoding="utf-8")
    written.append(path)
    return written


def _stars(rng) -> list[tuple[float, float, float]]:
    """Thirty stars, kept away from the edges so that a drift of a few pixels
    never takes one off the frame: (x, y, total counts), 1-based."""
    count = 30
    x = rng.uniform(12, DATA[0] - 12, count)
    y = rng.uniform(12, DATA[1] - 12, count)
    flux = 10 ** rng.uniform(3.3, 5.3, count)
    return list(zip(x.tolist(), y.tolist(), flux.tolist(), strict=True))


def _render(stars, dx: float, dy: float) -> np.ndarray:
    """The stars as Gaussians of FWHM 3 px, shifted by the drift."""
    y, x = np.mgrid[1:DATA[1] + 1, 1:DATA[0] + 1].astype(float)
    sigma = FWHM / 2.3548
    image = np.zeros(DATA[::-1])
    for sx, sy, flux in stars:
        image += flux / (2 * np.pi * sigma ** 2) * np.exp(
            -((x - sx - dx) ** 2 + (y - sy - dy) ** 2) / (2 * sigma ** 2))
    return image
