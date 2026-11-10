"""fmedian without the task: images/imfilter/src/fmedian.x, fmd_buf.x, fmd_hist.x
and fmd_maxmin.x, as IRAF 2.18 runs them.

The fast median: the pixels are first digitized to integers from hmin to hmax
(by default -32768 to 32767), and the window's median is kept as a running
median over a histogram of those integers, walked back and forth over the
image. What follows from the code:

* the range digitized is zmin to zmax, each the image's own minimum or
  maximum when not given -- over the image and its boundary extension, or
  DATAMIN and DATAMAX when the header has them, believed as IRAF believes
  them, with a constant boundary widening them;
* a value is mapped linearly, in single precision, clipped to hmin..hmax and
  truncated to an integer (amapr, achtri). When zmin and zmax already are
  hmin and hmax nothing is mapped, and an integer image is only clipped;
* zloreject and zhireject are mapped the same way and truncated; values
  outside stay out of the median, which is the lower of the two middle ones
  when what is left is even. The first pixel's median is amed's, which for
  one or two values is the first of them, not the lower;
* with unmap (the default) the medians are mapped back to zmin..zmax, so the
  output is quantized to the histogram's resolution: (zmax - zmin)/65535 of
  the data's range by default, and coarser with a narrower hmin..hmax;
* the window is made odd, as median makes it, and one line high on a 1-D
  image -- but the range is found with the window as given, so a 1-D image
  reads lines past its own for zmin and zmax.
"""

from __future__ import annotations

import numpy as np

from . import median
from .jit import jit

f32 = np.float32
MAX_REAL = float(np.finfo(f32).max)


class FmedianError(ValueError):
    """What IRAF's fmedian stops on, in its words."""


def amapr(values, a1: float, a2: float, b1: float, b2: float) -> np.ndarray:
    """VOPS amapr: linear from a1..a2 to b1..b2 in single precision, clipped to
    the output range."""
    a1, a2, b1, b2 = f32(a1), f32(a2), f32(b1), f32(b2)
    scalar = f32(f32(b2 - b1) / f32(a2 - a1))
    low, high = min(b1, b2), max(b1, b2)
    with np.errstate(invalid="ignore", over="ignore"):
        pixval = ((np.asarray(values, dtype=f32) - a1) * scalar).astype(f32)
        return np.maximum(low, np.minimum(high, (pixval + b1).astype(f32))).astype(f32)


def achtri(values) -> np.ndarray:
    """A real to an int, as gcc converts it: truncated; NaN and what does not
    fit, the minimum int."""
    from .imsum import convert

    return convert(np.asarray(values, dtype=f32), np.int32).astype(np.int64)


def limits(image, header, xbox: int, ybox: int, boundary: str, constant: float,
           section=None) -> tuple[float, float]:
    """fmd_maxmin: DATAMIN and DATAMAX when the header has limits (they differ),
    else the minimum and maximum of every line read with the window as given,
    boundary extension included."""
    if header is not None and header.get("DATAMIN", 0.0) != header.get("DATAMAX", 0.0):
        low, high = f32(header.get("DATAMIN", 0.0)), f32(header.get("DATAMAX", 0.0))
        if boundary == "constant":
            low, high = min(low, f32(constant)), max(high, f32(constant))
        return float(low), float(high)
    image = np.asarray(image, dtype=f32)
    plane = image.reshape(1, -1) if image.ndim == 1 else image
    cut = None
    if section is not None:
        cut = (slice(None),) + tuple(section) if image.ndim == 1 else tuple(section)
    if image.ndim == 1 and boundary == "reflect" and ybox > 1:
        # A 1-D image reflected to lines it does not have: imrbpx reads them
        # anyway, from past the end of the pixels -- the zeros that pad a FITS
        # file's last block.
        padded = np.vstack([plane, np.zeros((ybox // 2, plane.shape[1]), dtype=f32)])
        ext, _, _ = median.extended(padded, (slice(0, 1),) + (cut[1:] if cut else
                                    (slice(None),)), xbox, ybox, "reflect", constant)
        return float(ext.min()), float(ext.max())
    try:
        ext, _, _ = median.extended(plane, cut, xbox, ybox, boundary, constant)
    except median.MedianError as error:
        raise FmedianError(str(error)) from None
    return float(ext.min()), float(ext.max())


def digitize(ext: np.ndarray, mapped: bool, integer: bool, zmin: float, zmax: float,
             hmin: int, hmax: int) -> np.ndarray:
    """fmd_buf: the lines read, as the integers the histogram counts."""
    if not mapped and integer:
        lo, hi = int(np.rint(zmin)), int(np.rint(zmax))
        return np.minimum(np.maximum(np.rint(ext).astype(np.int64), lo), hi)
    if median_fp_equal(zmin, zmax):
        return achtri(np.full(ext.shape, f32(hmin), dtype=f32))
    return achtri(amapr(ext, zmin, zmax, float(hmin), float(hmax)))


def median_fp_equal(a: float, b: float) -> bool:
    from .scombine import fp_equalr

    return fp_equalr(a, b)


def filter2d(image, xwindow: int, ywindow: int, header=None, hmin: int = -32768,
             hmax: int = 32767, zmin: float | None = None, zmax: float | None = None,
             zloreject: float | None = None, zhireject: float | None = None,
             unmap: bool = True, boundary: str = "nearest", constant: float = 0.0,
             section=None, integer: bool = False) -> tuple[np.ndarray, float, float]:
    """IRAF's fmedian of `image` (read as reals), or of a section of it: the
    output in single precision, and the zmin and zmax it used. `integer` says
    the image's pixels are an integer type (fmd_buf reads those as ints when
    nothing is mapped)."""
    image = np.asarray(image, dtype=f32)
    xbox, ybox = median.boxes(xwindow, ywindow, 2)
    low, high = zmin, zmax
    if zmin is None or zmax is None:
        found = limits(image, header, xbox, ybox, boundary, constant, section)
        low = found[0] if zmin is None else float(f32(zmin))
        high = found[1] if zmax is None else float(f32(zmax))
    else:
        low, high = float(f32(zmin)), float(f32(zmax))
    if image.ndim == 1:
        ybox = 1
    mapped = not (median_fp_equal(float(hmin), low) and median_fp_equal(float(hmax), high))
    hlo = hmin if zloreject is None else int(np.trunc(
        amapr([zloreject], low, high, float(hmin), float(hmax))[0]))
    hhi = hmax if zhireject is None else int(np.trunc(
        amapr([zhireject], low, high, float(hmin), float(hmax))[0]))
    try:
        ext, ny, nx = median.extended(image, section, xbox, ybox, boundary, constant)
    except median.MedianError as error:
        raise FmedianError(str(error)) from None
    data = digitize(ext, mapped, integer, low, high, hmin, hmax)
    out = _fmdbox(data, ny, nx, xbox, ybox, hmin, hmax, hlo, hhi)
    if unmap and mapped:
        out = amapr(out, float(hmin), float(hmax), low, high)
    out = out.astype(f32)
    return (out[0] if image.ndim == 1 else out), low, high


@jit
def _amedi(values, n):
    """VOPS amedi: the first of one or two, the middle of three by comparisons,
    else the ((n+1)/2)th smallest."""
    if n <= 2:
        return values[0]
    if n == 3:
        a, b, c = values[0], values[1], values[2]
        if a < b:
            if b < c:
                return b
            if a < c:
                return c
            return a
        if b > c:
            return b
        if a < c:
            return a
        return c
    return np.sort(values[:n])[(n + 1) // 2 - 1]


@jit
def _fmdbox(data, ny, ncols, xbox, ybox, hmin, hmax, hlo, hhi):
    """fmd_medbox: fmd_medboxset and the filter line by line. data[l, c] is
    the digitized ext; hist[1..nbins] as IRAF's, with room either side."""
    nbins = hmax - hmin + 1
    hist = np.zeros(nbins + 2, dtype=np.int64)
    nx = ncols + 2 * (xbox // 2)
    out = np.zeros((ny, ncols), dtype=np.float32)
    filt = np.zeros(xbox * ybox, dtype=np.int64)
    median = 0
    nmedian = 0
    nltmedian = 0
    nhlo = 0
    nhhi = 0
    for line in range(1, ny + 1):
        base = line - 1                    # data[i, j] is data[base + j - 1, i - 1]
        # ------------------------------------------------- fmd_medboxset
        if line == 1:
            index = 0
            nhlo = 0
            nhhi = 0
            for j in range(1, ybox + 1):
                for i in range(1, xbox + 1):
                    v = data[base + j - 1, i - 1]
                    if v < hlo:
                        nhlo += 1
                        continue
                    if v > hhi:
                        nhhi += 1
                        continue
                    filt[index] = v
                    index += 1
            if index > 0 and nbins >= 2:
                dz = np.float32(np.float32(nbins - 1) / np.float32(hmax - hmin))
                for k in range(index):
                    b = int(np.float32(filt[k] - hmin) * dz) + 1
                    if b <= 0 or b > nbins:
                        continue
                    hist[b] += 1
            if index > 0:
                median = _amedi(filt.copy(), index)
            elif nhlo < nhhi:
                median = hhi
            else:
                median = hlo
            nltmedian = 0
            if index > 0:
                for k in range(index):
                    if filt[k] < median:
                        nltmedian += 1
                nmedian = (index - 1) // 2
            else:
                nmedian = 0
        else:
            if line % 2 == 0:
                c0, c1 = nx - xbox + 1, nx
            else:
                c0, c1 = 1, xbox
            for i in range(c0, c1 + 1):
                v = data[base + ybox - 1, i - 1]
                if v < hlo:
                    nhlo += 1
                    continue
                if v > hhi:
                    nhhi += 1
                    continue
                hist[v - hmin + 1] += 1
                if v < median:
                    nltmedian += 1
            nzero = xbox * ybox - nhlo - nhhi
            nmedian = (nzero - 1) // 2 if nzero > 0 else 0
            median, nltmedian = _settle(hist, median, nltmedian, nmedian, hmin, nbins)
        # ------------------------------------------------ the line's filter
        forward = line % 2 != 0
        dindex = 1 if forward else nx
        i = 1 if forward else ncols
        for _ in range(ncols - 1):
            if xbox * ybox - nhlo - nhhi > 0:
                out[line - 1, i - 1] = median
            elif nhlo < nhhi:
                out[line - 1, i - 1] = hhi
            else:
                out[line - 1, i - 1] = hlo
            for j in range(1, ybox + 1):
                v = data[base + j - 1, dindex - 1]
                if v < hlo:
                    nhlo -= 1
                    continue
                if v > hhi:
                    nhhi -= 1
                    continue
                hist[v - hmin + 1] -= 1
                if v < median:
                    nltmedian -= 1
            add = dindex + xbox if forward else dindex - xbox
            for j in range(1, ybox + 1):
                v = data[base + j - 1, add - 1]
                if v < hlo:
                    nhlo += 1
                    continue
                if v > hhi:
                    nhhi += 1
                    continue
                hist[v - hmin + 1] += 1
                if v < median:
                    nltmedian += 1
            nzero = xbox * ybox - nhlo - nhhi
            nmedian = (nzero - 1) // 2 if nzero > 0 else 0
            median, nltmedian = _settle(hist, median, nltmedian, nmedian, hmin, nbins)
            if forward:
                dindex += 1
                i += 1
            else:
                dindex -= 1
                i -= 1
        if xbox * ybox - nhlo - nhhi > 0:
            out[line - 1, i - 1] = median
        elif nhlo < nhhi:
            out[line - 1, i - 1] = hhi
        else:
            out[line - 1, i - 1] = hlo
        # The first row of the window leaves with the line.
        if forward:
            c0, c1 = nx - xbox + 1, nx
        else:
            c0, c1 = 1, xbox
        for k in range(c0, c1 + 1):
            v = data[base, k - 1]
            if v < hlo:
                nhlo -= 1
                continue
            if v > hhi:
                nhhi -= 1
                continue
            hist[v - hmin + 1] -= 1
            if v < median:
                nltmedian -= 1
        nzero = xbox * ybox - nhlo - nhhi
        nmedian = (nzero - 1) // 2 if nzero > 0 else 0
    return out


@jit
def _settle(hist, median, nltmedian, nmedian, hmin, nbins):
    """The running median moved to where nmedian values are below it."""
    if nltmedian > nmedian:
        for _ in range(nbins):
            median -= 1
            at = median - hmin + 1
            nltmedian -= hist[at] if 0 <= at < hist.shape[0] else 0
            if nltmedian <= nmedian:
                break
    else:
        for _ in range(nbins):
            at = median - hmin + 1
            h = hist[at] if 0 <= at < hist.shape[0] else 0
            if nltmedian + h > nmedian:
                break
            nltmedian += h
            median += 1
    return median, nltmedian
