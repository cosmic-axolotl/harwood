"""An image as it is shown: limits, stretch and colours, without Qt.

The window, the headless display and the `impicture` task all draw through
here, so a PNG saved from the window and one written by `harwood impicture`
from the same image with the same choices are the same pixels.

    limits(data, "zscale")              -> (low, high)
    stretch(data, low, high, "asinh")   -> 0..1
    lut("cividis")                      -> 256 x 3 colours
    rgb(data, ...)                      -> rows top down, ready for a file

Colour maps: grey, and the perceptually uniform viridis, cividis (made to
read the same with colour-vision deficiencies) and magma, which change
lightness steadily so that a brighter colour always means a brighter pixel;
plus harwood's earlier heat, cool and sepia. A map that runs through red and
green (a rainbow) is deliberately not offered.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

SCALES = ("zscale", "minmax", "99.5%")
STRETCHES = ("linear", "sqrt", "log", "asinh", "none", "user")

#: Stops, evenly spaced from 0 to 1, linearly interpolated between.
#: viridis, cividis and magma are sampled from their published tables
#: (van der Walt and Smith; Nunez, Anderton and Renslow 2018), CC0.
COLORMAPS: dict[str, list[tuple[int, int, int]]] = {
    "grey": [(0, 0, 0), (255, 255, 255)],
    "viridis": [(68, 1, 84), (72, 40, 120), (62, 74, 137), (49, 104, 142),
                (38, 130, 142), (31, 158, 137), (53, 183, 121), (109, 205, 89),
                (180, 222, 44), (253, 231, 37)],
    "cividis": [(0, 32, 77), (0, 48, 111), (55, 66, 108), (87, 84, 109),
                (112, 103, 115), (138, 122, 118), (166, 143, 115), (194, 164, 105),
                (224, 187, 88), (253, 234, 69)],
    "magma": [(0, 0, 4), (24, 15, 62), (69, 16, 121), (114, 31, 129), (159, 47, 127),
              (205, 64, 113), (241, 96, 93), (254, 145, 104), (254, 196, 136),
              (252, 253, 191)],
    "heat": [(0, 0, 0), (178, 34, 34), (255, 170, 0), (255, 255, 224)],
    "cool": [(6, 12, 33), (32, 84, 140), (86, 176, 186), (238, 248, 230)],
    "sepia": [(12, 9, 6), (120, 78, 46), (214, 170, 116), (255, 246, 232)],
}
#: The maps that read the same for the common colour-vision deficiencies.
COLORBLIND_SAFE = ("grey", "viridis", "cividis", "magma")


def limits(data: np.ndarray, scale: str = "zscale") -> tuple[float, float]:
    """The display range: IRAF's zscale, the extremes, or the middle 99.5%."""
    data = np.asarray(data, dtype=float)
    finite = data[np.isfinite(data)]
    if finite.size == 0:
        return 0.0, 1.0
    if scale == "minmax":
        return float(finite.min()), float(finite.max())
    if scale == "99.5%":
        low, high = np.percentile(finite, [0.25, 99.75])
        return float(low), float(high)
    if scale != "zscale":
        raise ValueError(f"scale '{scale}' is not one of {', '.join(SCALES)}")
    if finite.size < MIN_NPIXELS:
        return float(finite.min()), float(finite.max())
    return zscale(data)


# zscale, ported from IRAF's images/tv/display/zscale.x (mzscale and
# zsc_zlimits). The constants are IRAF's.
MIN_NPIXELS = 5         # smallest permissible sample
MAX_REJECT = 0.5        # max fraction of pixels to be rejected
KREJ = 2.5              # k-sigma pixel rejection factor
MAX_ITERATIONS = 5      # maximum number of fitline iterations
_GOOD, _BAD, _REJECT = 0, 1, 2


def zscale(data: np.ndarray, contrast: float = 0.25, nsample: int = 1000,
           zmask: np.ndarray | None = None,
           bpm: np.ndarray | None = None) -> tuple[float, float]:
    """IRAF display's z1 and z2: sample the image, sort the sample, fit a
    line to the middle of it, and spread the line's slope by `contrast`.

    Without `zmask` the sample is every i-th pixel of every i-th line, with
    i chosen to give about `nsample`; with one, the first `nsample` pixels it
    selects. Pixels flagged in `bpm` -- and, unlike IRAF, which has none,
    blank (NaN) ones -- are skipped. contrast=0 gives the sample's range,
    which is what zrange uses."""
    sample = _zsample(np.asarray(data), nsample, zmask, bpm)
    return zlimits(sample, contrast)


def _zsample(data: np.ndarray, maxpix: int, zmask, bpm) -> np.ndarray:
    plane = data if data.ndim <= 2 else data.reshape((-1,) + data.shape[-2:])[0]
    if zmask is None:
        if plane.ndim == 1:
            chosen = np.ones(plane.shape, dtype=bool)
        else:
            nl, nc = plane.shape
            # SPP: i = max (1., sqrt ((nc-1)*(nl-1) / real (maxpix)))
            step = int(max(np.float32(1.0), np.sqrt(np.float32((nc - 1) * (nl - 1))
                                                    / np.float32(maxpix))))
            chosen = np.zeros(plane.shape, dtype=bool)
            chosen[::step, ::step] = True
    else:
        chosen = np.asarray(zmask) > 0
        chosen = chosen if chosen.ndim <= 2 else chosen.reshape((-1,) + chosen.shape[-2:])[0]
    good = chosen & np.isfinite(plane)
    if bpm is not None:
        flags = np.asarray(bpm)
        flags = flags if flags.ndim <= 2 else flags.reshape((-1,) + flags.shape[-2:])[0]
        good &= flags == 0
    return plane[good].astype(np.float32)[:maxpix]


def zlimits(sample: np.ndarray, contrast: float = 0.25) -> tuple[float, float]:
    """zsc_zlimits: the limits from a sample of pixel values."""
    sample = np.asarray(sample, dtype=np.float32)
    npix = sample.size
    if npix < MIN_NPIXELS:
        raise ValueError(f"insufficient sample pixels found ({npix}, "
                         f"at least {MIN_NPIXELS} are needed)")
    if contrast == 0:
        return float(sample.min()), float(sample.max())
    sample = np.sort(sample)
    zmin, zmax = sample[0], sample[-1]
    center = (npix + 1) // 2
    if npix % 2 == 1:
        median = sample[center - 1]
    else:
        median = np.float32((sample[center - 1] + sample[center]) / np.float32(2))
    minpix = max(MIN_NPIXELS, int(npix * MAX_REJECT))
    ngrow = max(1, int(np.floor(npix * 0.01 + 0.5)))
    ngoodpix, _, zslope = _fit_line(sample, KREJ, ngrow, MAX_ITERATIONS)
    if ngoodpix < minpix:
        return float(zmin), float(zmax)
    if contrast > 0:
        zslope = np.float32(zslope / np.float32(contrast))
    z1 = max(zmin, np.float32(median - np.float32(center - 1) * zslope))
    z2 = min(zmax, np.float32(median + np.float32(npix - center) * zslope))
    return float(z1), float(z2)


def _fit_line(data: np.ndarray, krej: float, ngrow: int, maxiter: int):
    """zsc_fit_line: a straight line through the sorted sample, refitted
    with the points beyond krej sigma (and their neighbours) left out.
    Returns (pixels left, z at the first pixel, slope per pixel)."""
    f32 = np.float32
    npix = data.size
    if npix == 1:
        return 1, data[0], f32(0)
    xscale = f32(2.0) / f32(npix - 1)
    normx = (np.arange(npix, dtype=f32) * xscale - f32(1.0)).astype(f32)
    badpix = np.zeros(npix, dtype=np.int8)

    sumxsqr = float(np.sum((normx * normx).astype(np.float64)))
    sumxz = float(np.sum((data * normx).astype(np.float64)))
    sumx = 0.0
    sumz = float(np.sum(data.astype(np.float64)))
    z0 = f32(sumz / npix)
    dz = f32(sumxz / sumxsqr)

    ngoodpix = npix
    minpix = max(MIN_NPIXELS, int(npix * MAX_REJECT))
    for _ in range(maxiter):
        last = ngoodpix
        flat = (data - (normx * dz + z0)).astype(f32)
        sigma = _sigma(flat, badpix)
        threshold = f32(sigma * f32(krej)) if sigma is not None else None
        ngoodpix, (sumxsqr, sumxz, sumx, sumz) = _reject(
            data, flat, normx, badpix, (sumxsqr, sumxz, sumx, sumz), threshold, ngrow)
        if ngoodpix > 0:
            rowrat = sumx / sumxsqr
            z0 = f32((sumz - rowrat * sumxz) / (ngoodpix - rowrat * sumx))
            dz = f32((sumxz - float(z0) * sumx) / sumxsqr)
        if ngoodpix >= last or ngoodpix < minpix:
            break
    return ngoodpix, f32(z0 - dz), f32(dz * xscale)


def _sigma(flat: np.ndarray, badpix: np.ndarray):
    """zsc_compute_sigma: rms about the mean of the pixels still good --
    not those only marked for rejection."""
    good = flat[badpix == _GOOD]
    n = good.size
    if n < 2:
        return None          # IRAF's INDEF
    total = float(np.sum(good.astype(np.float64)))
    sumsq = float(np.sum((good * good).astype(np.float64)))
    temp = sumsq / (n - 1) - total ** 2 / (n * (n - 1))
    return np.float32(0.0) if temp < 0 else np.float32(np.sqrt(temp))


def _reject(data, flat, normx, badpix, sums, threshold, ngrow):
    """zsc_reject_pixels, in place on badpix. A pixel beyond the threshold
    takes its neighbours with it: those behind it at once, those ahead only
    marked, so they are still tested themselves -- IRAF's way of growing
    without a direction. Pixels already out are not tested again."""
    npix = data.size
    if threshold is None:
        # SPP compares with INDEF, a huge number: nothing is beyond it.
        return int(np.count_nonzero(badpix != _BAD)), sums
    out = badpix == _BAD
    hit = np.flatnonzero(~out & ((flat < -threshold) | (flat > threshold)))
    if hit.size:
        behind = np.zeros(npix + 1, dtype=np.int32)
        ahead = np.zeros(npix + 1, dtype=np.int32)
        np.add.at(behind, np.maximum(hit - ngrow, 0), 1)
        np.add.at(behind, hit + 1, -1)
        np.add.at(ahead, hit + 1, 1)
        np.add.at(ahead, np.minimum(hit + ngrow + 1, npix), -1)
        now_out = (np.cumsum(behind)[:npix] > 0) & ~out
        marked = (np.cumsum(ahead)[:npix] > 0) & ~out & ~now_out
        x = normx[now_out].astype(np.float64)
        z = data[now_out].astype(np.float64)
        sumxsqr, sumxz, sumx, sumz = sums
        sums = (sumxsqr - float(np.sum(x * x)), sumxz - float(np.sum(z * x)),
                sumx - float(np.sum(x)), sumz - float(np.sum(z)))
        badpix[marked] = _REJECT
        badpix[now_out] = _BAD
    return int(np.count_nonzero(badpix != _BAD)), sums


def stretch(data: np.ndarray, low: float, high: float, kind: str = "linear",
            table: UserLut | None = None) -> np.ndarray:
    """Normalise to 0..1 through the chosen transfer function.

    log is IRAF display's: low..high onto 1..1000, then the logarithm. none
    and user are its other two ztrans: the pixel values themselves as grey
    levels, and a table from a file (`UserLut`)."""
    if kind == "none":
        return unitary(data)
    if kind == "user":
        if table is None:
            raise ValueError("stretch 'user' needs the table from a lutfile")
        return table.apply(data)
    span = max(high - low, 1e-12)
    norm = np.clip((np.asarray(data, dtype=float) - low) / span, 0.0, 1.0)
    if kind == "sqrt":
        return np.sqrt(norm)
    if kind == "log":
        return np.log10(1 + 999 * norm) / 3
    if kind == "asinh":
        return np.arcsinh(10 * norm) / np.arcsinh(10)
    if kind != "linear":
        raise ValueError(f"stretch '{kind}' is not one of {', '.join(STRETCHES)}")
    return norm


#: IRAF's display frames hold grey levels 1 to 200 (graphcap z0#1 zr#200);
#: what is above is left for overlay colours.
DEVICE_GREY = (1, 200)


def _grey(levels: np.ndarray) -> np.ndarray:
    low, high = DEVICE_GREY
    return np.clip((levels - low) / (high - low), 0.0, 1.0)


def unitary(data: np.ndarray) -> np.ndarray:
    """ztrans=none: each value goes into the frame as it is. The frame keeps
    8 bits, so values wrap every 256 -- the "contoured" picture IRAF's
    manual describes."""
    values = np.nan_to_num(np.asarray(data, dtype=float), nan=0.0)
    return _grey(np.mod(np.trunc(values), 256))


class UserLut:
    """ztrans=user: intensity -> grey level from a two-column file, as
    images/tv/display/dsulut.x reads it. The intensities run from z1 to z2,
    the table's own extremes; grey levels are the frame's 1 to 200."""

    MAXPTS = 4096

    def __init__(self, path) -> None:
        x, y = [], []
        for line in Path(path).read_text(encoding="latin-1").splitlines():
            if line.startswith("#") or not line.strip():
                continue
            fields = line.split()
            x.append(float(fields[0]))
            y.append(float(fields[1]) if len(fields) > 1 else 0.0)
        if len(x) > self.MAXPTS:
            raise ValueError("intensity transformation table cannot exceed 4096 values")
        if len(x) < 2:
            raise ValueError(f"{path}: a lookup table needs at least two lines")
        x = np.asarray(x, dtype=np.float32)
        y = np.asarray(y, dtype=np.float32)
        self.z1, self.z2 = float(x.min()), float(x.max())
        order = np.argsort(x, kind="stable")
        x = _amap(x, self.z1, self.z2, 0, self.MAXPTS - 1)[order]
        y = y[order]
        table = np.zeros(self.MAXPTS, dtype=np.int16)
        slope = np.float32(0)
        y1 = 0
        for i in range(1, len(x)):
            slope = np.float32((y[i] - y[i - 1]) / (x[i] - x[i - 1]))
            x1, x2, y1 = int(x[i - 1]), int(x[i]), int(y[i - 1])
            j = np.arange(x1, x2 + 1)
            table[j] = (y1 + slope * (j - x1)).astype(np.int16)
        table[-1] = np.int16(y1 + slope * (self.MAXPTS - 1))
        self.table = table

    def apply(self, data: np.ndarray) -> np.ndarray:
        index = _amap(np.nan_to_num(np.asarray(data, dtype=np.float32)), *DEVICE_GREY,
                      0, self.MAXPTS - 1).astype(np.int64)
        return _grey(self.table[index].astype(float))


def _amap(a: np.ndarray, a1: float, a2: float, b1: float, b2: float) -> np.ndarray:
    """IRAF's amapr: a1..a2 onto b1..b2 in single precision, values outside
    clipped to the ends."""
    f32 = np.float32
    scalar = (f32(b2) - f32(b1)) / (f32(a2) - f32(a1)) if a2 != a1 else f32(0)
    return np.clip((np.asarray(a, dtype=f32) - f32(a1)) * f32(scalar) + f32(b1),
                   f32(min(b1, b2)), f32(max(b1, b2)))


def _amaps(a: np.ndarray, a1: int, a2: int, b1: int, b2: int) -> np.ndarray:
    """amaps, the short-integer amap: the scaled offset is truncated before
    the output offset is added."""
    scalar = (np.float32(b2) - np.float32(b1)) / (np.float32(a2) - np.float32(a1))
    pixval = np.trunc((np.asarray(a, dtype=np.int64) - a1).astype(np.float32)
                      * np.float32(scalar)).astype(np.int64)
    return np.clip(pixval + b1, min(b1, b2), max(b1, b2))


def device_levels(data: np.ndarray, z1: float, z2: float, ztrans: str = "linear",
                  table: UserLut | None = None, short: bool = False) -> np.ndarray:
    """The grey levels IRAF's display writes into the frame, 1 to 200, as
    t_display.x computes them -- including its path for 16-bit integer
    images, which works in integers and so truncates z1 and z2 first.
    Kept separate from `stretch`, whose continuous version of the same
    transfer is what is drawn; this is what is checked against IRAF."""
    f32 = np.float32
    dz1, dz2 = DEVICE_GREY
    if ztrans == "none":
        return np.mod(np.trunc(np.nan_to_num(np.asarray(data, dtype=f32))), 256).astype(int)
    z1, z2 = f32(z1), f32(z2)
    if z1 == z2:
        z1, z2 = z1 - 1, z2 + 1
    if short and ztrans != "log":
        z1_s, z2_s = int(np.trunc(z1)), int(np.trunc(z2))
        if z1_s == z2_s:
            z1_s, z2_s = z1_s - 1, z2_s + 1
        if ztrans == "user":
            return table.table[_amaps(data, z1_s, z2_s, 0, UserLut.MAXPTS - 1)].astype(int)
        if ztrans == "linear" and z1 == dz1 and z2 == dz2:
            return np.asarray(data, dtype=np.int64)
        return _amaps(data, z1_s, z2_s, dz1, dz2)
    if ztrans == "user":
        index = np.trunc(_amap(data, z1, z2, 0, UserLut.MAXPTS - 1)).astype(np.int64)
        return table.table[index].astype(int)
    if ztrans == "log":
        out = _amap(data, z1, z2, 1.0, 10.0 ** 3)
        out = _amap(np.log10(out).astype(f32), 0.0, 3.0, dz1, dz2)
    elif z1 == dz1 and z2 == dz2:
        out = np.asarray(data, dtype=f32)
    else:
        out = _amap(data, z1, z2, dz1, dz2)
    return np.trunc(out).astype(int)


def lut(name: str, n: int = 256) -> np.ndarray:
    """n colours (uint8, n x 3) along the map."""
    if name not in COLORMAPS:
        raise ValueError(f"colour map '{name}' is not one of {', '.join(COLORMAPS)}")
    stops = np.asarray(COLORMAPS[name], dtype=float)
    where = np.linspace(0.0, 1.0, len(stops))
    at = np.linspace(0.0, 1.0, n)
    table = np.column_stack([np.interp(at, where, stops[:, k]) for k in range(3)])
    return np.clip(np.round(table), 0, 255).astype(np.uint8)


def rgb(data: np.ndarray, scale: str = "zscale", kind: str = "linear",
        colormap: str = "grey", window: tuple[float, float] | None = None,
        table: UserLut | None = None) -> np.ndarray:
    """The image as shown, rows top down (FITS counts from the bottom),
    uint8 (ny, nx, 3). Blank (NaN) pixels take the map's first colour."""
    low, high = window if window is not None else limits(data, scale)
    shown = stretch(data, low, high, kind, table)
    table_rgb = lut(colormap)
    index = np.nan_to_num(shown, nan=0.0)
    index = np.clip((index * len(table_rgb)).astype(np.int64), 0, len(table_rgb) - 1)
    return table_rgb[index][::-1].copy()
