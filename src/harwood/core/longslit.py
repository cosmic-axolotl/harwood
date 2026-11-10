"""twodspec.longslit's response and illumination without the tasks:
noao/twodspec/longslit/response.x, illumination.x, ilsetbins.x and lstools.x,
as IRAF 2.18 runs them.

response divides a calibration image (a flat) by a smooth fit to the average
of a normalization image across the slit: what is left is the pixel-to-pixel
response, the lamp's colour and the spectrograph's taken out. illumination
averages the image along the dispersion in a few bins, fits each bin's slit
profile, normalizes it, and interpolates between the bins over the whole
image: the variation along the slit. Both fit with ICFIT (core/icfitbits.py).

What follows from the code, kept:

* the average across the slit is a sum in single precision, line by line or
  along each line, divided by the count (ls_aimavg);
* each bin of illumination is normalized by its value at nx/2 - 1, not the
  middle of the slit; a bin's position is (first + last) / 2 in integers;
  between two bins the image is mapped onto the interpolator's grid so that
  the first pixel past a bin's centre and the centre itself both land on the
  bin, and the interpolator falls back to linear with fewer than four bins
  (poly3, spline3) or poly3 with fewer than six (poly5);
* the default bins are the dispersion axis cut in nbins of npts / nbins
  pixels, in integers: the pixels left over at the end are in none;
* the response threshold sets to 1 both pixels whose fit is below it and
  pixels whose own value is.
"""

from __future__ import annotations

import numpy as np

from . import icfitbits, msi
from .jit import jit

f32 = np.float32


class LongslitError(ValueError):
    """What response and illumination stop on, in IRAF's words."""


def dispersion_axis(header, given: int | None) -> int:
    """get_daxis: DISPAXIS from the header, else the one given; the other one
    when the image is transposed (LTM1_1 and LTM2_2 both zero)."""
    paxis = None
    if header is not None and "DISPAXIS" in header:
        paxis = int(header["DISPAXIS"])
    elif given is not None:
        paxis = int(given)
    if paxis not in (1, 2):
        raise LongslitError("the dispersion axis is not known: the image has no "
                            "DISPAXIS")
    laxis = paxis
    if header is not None and float(header.get("LTM1_1", 1.0)) == 0.0 and \
            float(header.get("LTM2_2", 1.0)) == 0.0:
        laxis = paxis % 2 + 1
    return laxis


@jit
def _sum_lines(data, line1, line2, col1, col2):
    """aaddr of each line in turn, columns col1..col2 (1-based, inclusive)."""
    out = np.zeros(col2 - col1 + 1, dtype=np.float32)
    for line in range(line1, line2 + 1):
        for c in range(col1, col2 + 1):
            out[c - col1] = out[c - col1] + data[line - 1, c - 1]
    return out


@jit
def _sum_along(data, line1, line2, col1, col2):
    """asumr of each line's columns col1..col2."""
    out = np.zeros(line2 - line1 + 1, dtype=np.float32)
    for line in range(line1, line2 + 1):
        s = np.float32(0.0)
        for c in range(col1, col2 + 1):
            s = s + data[line - 1, c - 1]
        out[line - line1] = s
    return out


def aimavg(data: np.ndarray, axis: int, col1: int, col2: int, line1: int,
           line2: int) -> tuple[np.ndarray, np.ndarray]:
    """ls_aimavg: the positions and the average across the other axis -- along
    axis 1, columns col1..col2 averaged over lines line1..line2; along axis 2,
    lines averaged over columns."""
    data = np.asarray(data, dtype=f32)
    if axis == 1:
        x = np.arange(col1, col2 + 1, dtype=f32)
        y = _sum_lines(data, line1, line2, col1, col2) / f32(line2 - line1 + 1)
    else:
        x = np.arange(line1, line2 + 1, dtype=f32)
        y = _sum_along(data, line1, line2, col1, col2) / f32(col2 - col1 + 1)
    return x, y.astype(f32)


def _ic(spec: dict) -> icfitbits.Ic:
    return icfitbits.Ic(function=spec["function"], order=int(spec["order"]),
                        sample=spec["sample"], naverage=int(spec["naverage"]),
                        low_reject=float(spec["low_reject"]),
                        high_reject=float(spec["high_reject"]),
                        niterate=int(spec["niterate"]), grow=float(spec["grow"]))


def fitted(x, y, spec: dict, weights=None) -> tuple[np.ndarray, icfitbits.Ic]:
    """ic_fit (xmin, xmax the first and last x) and cvvector at the same x."""
    x = np.asarray(x, dtype=f32)
    y = np.asarray(y, dtype=f32)
    w = np.ones(x.size, dtype=f32) if weights is None else np.asarray(weights, f32)
    ic = _ic(spec)
    curve = icfitbits.fit(ic, x, y, w)
    return np.asarray(curve.vector_at(x), dtype=f32), ic


# ------------------------------------------------------------------ response

def normalization(norm: np.ndarray, laxis: int) -> tuple[np.ndarray, np.ndarray]:
    """The normalization image averaged across the slit (re_make)."""
    ny, nx = norm.shape
    return aimavg(norm, laxis, 1, nx, 1, ny)


def normalize(cal: np.ndarray, spectrum: np.ndarray, laxis: int,
              threshold: float | None) -> np.ndarray:
    """re_normalize: the calibration over the fitted spectrum, in single
    precision; with a threshold, 1 where the fit or the pixel is below it."""
    cal = np.asarray(cal, dtype=f32)
    spectrum = np.asarray(spectrum, dtype=f32)
    ny, nx = cal.shape
    if laxis == 1:
        norm = np.broadcast_to(spectrum[:nx], (ny, nx))
    else:
        norm = np.broadcast_to(spectrum[:ny, None], (ny, nx))
    with np.errstate(divide="ignore", invalid="ignore"):
        out = (cal / norm).astype(f32)
    if threshold is None:
        return out
    t = f32(threshold)
    return np.where((norm < t) | (cal < t), f32(1.0), out).astype(f32)


# -------------------------------------------------------------- illumination

def ranges(text: str, rmin: int, rmax: int) -> list[tuple[int, int]]:
    """rg_ranges and rg_order: 'a:b c:d', '*' for rmin:rmax, a single number as a
    range of one; each range clipped to rmin..rmax, those wholly outside dropped,
    then sorted by their start."""
    found = []
    for piece in text.replace(",", " ").split():
        if piece.startswith("@"):
            raise LongslitError("Cannot nest @files")
        if piece == "*":
            r1, r2 = rmin, rmax
        else:
            parts = piece.replace(":", " ").split()
            try:
                r1 = int(parts[0])
                r2 = int(parts[1]) if len(parts) > 1 else r1
            except (ValueError, IndexError):
                raise LongslitError("Bad range string for parameter bins") from None
        r1, r2 = min(r1, r2), max(r1, r2)
        if r2 >= rmin and r1 <= rmax:
            found.append((max(rmin, r1), min(rmax, r2)))
    for i in range(len(found) - 1):
        for j in range(i + 1, len(found)):
            if found[i][0] > found[j][0]:
                found[i], found[j] = found[j], found[i]
    return found


def default_bins(npts: int, nbins: int) -> str:
    """il_setbins: nbins of npts / nbins pixels (integer division, the rest left
    out), as the string IRAF builds."""
    dx = f32(npts // nbins)
    text = ""
    for i in range(1, nbins + 1):
        first = int(f32(f32(i - 1) * dx)) + 1
        last = int(f32(f32(i) * dx - f32(1.0))) + 1
        text += f"{first}:{last} "
    return text


def binned(data: np.ndarray, slit_axis: int, bins: list[tuple[int, int]]):
    """il_binimage: the slit positions, the bins' dispersion positions, and each
    bin's profile across the slit, averaged over the bin."""
    ny, nx = data.shape
    centres, profiles = [], []
    x = None
    for y1, y2 in bins:
        centres.append(f32((y1 + y2) // 2))
        if slit_axis == 1:
            x, z = aimavg(data, 1, 1, nx, y1, y2)
        else:
            x, z = aimavg(data, 2, y1, y2, 1, ny)
        profiles.append(z)
    return x, np.array(centres, dtype=f32), np.vstack(profiles).astype(f32)


def interpolator(name: str, nbins: int) -> str:
    """il_expand's choice: linear for poly3 and spline3 with fewer than four bins;
    poly3 for poly5 with fewer than six, linear with fewer than four."""
    if name in ("poly3", "spline3") and nbins < 4:
        return "linear"
    if name == "poly5" and nbins < 6:
        return "linear" if nbins < 4 else "poly3"
    return name


def expand(shape: tuple[int, int], slit_axis: int, x: np.ndarray, centres: np.ndarray,
           profiles: np.ndarray, interp: str) -> np.ndarray:
    """il_expand: each profile normalized at nx/2 - 1, and interpolated over the
    image -- copied, with one bin."""
    nlines, ncols = shape
    nx = x.size
    z = np.array(profiles, dtype=f32)
    at = nx // 2 - 1
    for j in range(z.shape[0]):
        z[j] = (z[j] / z[j, at - 1]).astype(f32)
    out = np.zeros(shape, dtype=f32)
    ny = z.shape[0]
    if ny == 1:
        if slit_axis == 1:
            out[:] = z[0][:ncols]
        else:
            out[:] = z[0][:nlines, None]
        return out
    kind = interpolator(interp, ny)
    surface = msi.msifit(kind, z)
    nyout = nlines if slit_axis == 1 else ncols
    yout = np.zeros(nyout, dtype=f32)
    y1, y2 = 1, int(centres[0])
    yout[y1 - 1:y2] = f32(1.0)
    for j in range(2, ny + 1):
        y1 = y2 + 1
        y2 = int(centres[j - 1])
        with np.errstate(divide="ignore"):
            dy = f32(f32(1.0) / f32(y2 - y1))
        for i in range(y1, y2 + 1):
            yout[i - 1] = f32(f32(j - 1) + f32(i - y1) * dy)
    y1 = y2 + 1
    yout[y1 - 1:nyout] = f32(ny)
    if slit_axis == 1:
        for i in range(nlines):
            out[i] = msi.msivector(surface, x[:ncols], np.full(ncols, yout[i], f32))
    else:
        for i in range(nlines):
            out[i] = msi.msivector(surface, np.full(ncols, x[i], f32), yout[:ncols])
    return out
