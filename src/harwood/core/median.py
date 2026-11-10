"""median without the task: images/imfilter/src/median.x, as IRAF 2.18 runs it.

IRAF keeps the window as a sorted, doubly linked list over a kernel of
xwindow x ywindow values stored a column at a time, and walks the image
back and forth -- odd lines left to right, even lines right to left --
deleting the column that leaves and merging in the one that comes. Between
lines it deletes the first row of every kernel column and puts the new line
there. That assumes the line just walked rewrote every kernel column, which
it does only when the image is wider than the window: in an image as narrow
as the window or narrower, the row deleted is the one put in at the last
change of line, and the medians are not the window's. The port is literal,
so that is what harwood gives too.

What else follows from the code:

* the window is made odd by adding one (a 4x4 is a 5x5), and is one line
  high on a 1-D image; outside the image the pixels come from IMIO's
  boundary extension -- nearest, constant, reflect (about the edge pixel,
  not repeating it: no further than the image is wide), wrap -- and for an
  image section, from the pixels of the image around it, the extension
  only beyond the image's own edge;
* values below zloreject and above zhireject stay out of the median, which
  is the lower of the two middle values when what is left is an even
  number; with nothing left it is zhireject if more were above than below,
  else zloreject. A window one column wide counts a value once, and keeps
  only what it accepts; any other counts a value on each side it falls
  (only zloreject > zhireject makes that possible) and keeps them all;
* the pixels are read as reals (a double image in single precision; a NaN
  or an infinity as 0, by IRAF's IEEE input) and the result is written in
  the input's pixel type, an integer image truncating a value not whole.
"""

from __future__ import annotations

import numpy as np

from .jit import jit

f32 = np.float32
MAX_REAL = float(np.finfo(f32).max)
BOUNDARIES = ("constant", "nearest", "reflect", "wrap")


class MedianError(ValueError):
    """What IRAF's median stops on, in its words."""


def boxes(xwindow: int, ywindow: int, ndim: int) -> tuple[int, int]:
    """MED_XBOX and MED_YBOX: odd, the next one up; 1 along y for a 1-D image."""
    xbox = xwindow + 1 if xwindow % 2 == 0 else xwindow
    ybox = ywindow + 1 if ywindow % 2 == 0 else ywindow
    if ndim == 1:
        ybox = 1
    return xbox, ybox


def _btran(v: int, vmax: int, boundary: str) -> int:
    """imbtran, one axis, 1-based: where an out-of-bounds pixel is read from."""
    if boundary == "nearest":
        return 1 if v < 1 else vmax if v > vmax else v
    if boundary == "reflect":
        return 1 + (1 - v) if v < 1 else vmax - (v - vmax) if v > vmax else v
    if boundary == "wrap":
        while v < 1:
            v += vmax
        while v > vmax:
            v -= vmax
        return v
    return v


def _tdiv(a: int, b: int) -> int:
    """Fortran's integer division, toward zero."""
    q = abs(a) // abs(b)
    return q if (a >= 0) == (b > 0) else -q


def _segment(row, x1: int, totpix: int, xstep: int, flip: bool, boundary: str,
             constant: float, oob: bool) -> np.ndarray:
    """imrbpx: a line segment of totpix pixels from physical x1 (1-based) every
    xstep, some of them off the line. It is read in three regions -- left of the
    image, on it, right of it -- whose limits are 0, 1 and the line's length
    whatever the step: with a step above 1 the pixels of a region need not fall
    on the segment's own grid, and IRAF reads them where the region starts."""
    linelen = len(row)
    x2 = x1 + totpix * xstep - 1
    regions = ((x1, min(0, x2)), (max(x1, 1), min(x2, linelen)),
               (max(x1, linelen + 1), x2))
    parts = []
    for i, (xs, xe) in enumerate(regions):
        npix = _tdiv(xe - xs, xstep) + 1
        if npix <= 0:
            continue
        v1, v2 = _btran(xs, linelen, boundary), _btran(xe, linelen, boundary)
        step = xstep
        if v1 > v2:
            step, v1 = -xstep, v2
        if (i == 1 and not oob) or boundary in ("reflect", "wrap"):
            last = v1 + (npix - 1) * abs(step)
            if v1 < 1 or last > linelen:
                raise MedianError("Pixel subscript out of bounds")
            got = row[v1 - 1:last:abs(step)]
            parts.append(got[::-1] if step < 0 else got)
        else:
            value = f32(constant) if boundary == "constant" else row[v1 - 1]
            parts.append(np.full(npix, value, dtype=f32))
    out = np.concatenate(parts)[:totpix] if parts else np.zeros(0, dtype=f32)
    return out[::-1] if flip else out


def extended(image: np.ndarray, section: tuple[slice, ...] | None, xbox: int, ybox: int,
             boundary: str, constant: float) -> tuple[np.ndarray, int, int]:
    """The lines median.x reads (med_buf, imgs2r): xbox/2 columns and ybox/2
    lines beyond the image, or the section of it, on each side -- off a section
    the image's own pixels, past the image's edge its boundary extension, as
    imggsc and imrbpx give them. `constant` is already in the image's type."""
    image = np.asarray(image, dtype=f32)
    plane = image.reshape(1, -1) if image.ndim == 1 else image
    ny, nx = plane.shape
    cuts = (slice(None),) * 2 if section is None else \
        ((slice(None),) + tuple(section) if image.ndim == 1 else tuple(section))

    def span(cut, size):
        first, stop, step = cut.indices(size)
        return first + 1, step, len(range(first, stop, step))

    x0, xstep, nxs = span(cuts[1], nx)
    y0, ystep, nys = span(cuts[0], ny) if image.ndim > 1 else (1, 1, 1)
    mx, my = xbox // 2, (ybox // 2 if image.ndim > 1 else 0)
    # The logical columns col1..col2 as physical ones; a flipped section is read
    # from its lower physical end and turned round (imsslv).
    p1, p2 = x0 + (-mx) * xstep, x0 + (nxs - 1 + mx) * xstep
    flip = p1 > p2
    start = min(p1, p2)
    totpix = nxs + 2 * mx
    inside = mx == 0
    lines = []
    for logical in range(1 - my, nys + my + 1):
        py = y0 + (logical - 1) * ystep
        mapped = _btran(py, ny, boundary)
        oob = not 1 <= mapped <= ny
        if oob and boundary != "constant":
            raise MedianError("Pixel subscript out of bounds")
        row = plane[mapped - 1] if not oob else np.zeros(nx, dtype=f32)
        if inside and not oob and 1 <= logical <= nys:
            got = row[start - 1:start - 1 + (totpix - 1) * abs(xstep) + 1:abs(xstep)]
            lines.append(got[::-1] if flip else got)
        else:
            lines.append(_segment(row, start, totpix, abs(xstep), flip, boundary,
                                  constant, oob))
    return np.vstack(lines).astype(f32), nys if image.ndim > 1 else 1, nxs


def filter2d(image: np.ndarray, xwindow: int, ywindow: int, zlo: float | None = None,
             zhi: float | None = None, boundary: str = "nearest", constant: float = 0.0,
             section: tuple[slice, ...] | None = None) -> np.ndarray:
    """IRAF's median of `image` (read as reals), or of a section of it: the
    output in single precision, the shape of the section."""
    image = np.asarray(image, dtype=f32)
    xbox, ybox = boxes(xwindow, ywindow, image.ndim)
    low = f32(-MAX_REAL if zlo is None else zlo)
    high = f32(MAX_REAL if zhi is None else zhi)
    ext, ny, nx = extended(image, section, xbox, ybox, boundary, constant)
    out = _medbox(ext, ny, nx, xbox, ybox, low, high)
    return out[0] if image.ndim == 1 else out


@jit
def _gshsrt(a, index, npts):
    """An index of a[...] in ascending order (med_gshsrt's shell sort; the values
    stay where they are, and ties may come in any order)."""
    d = npts
    while d > 1:
        d = 1 if d < 5 else (5 * d - 1) // 11
        for j in range(npts - d, 0, -1):
            temp = index[j]
            k = j + d
            while k <= npts:
                if a[temp] <= a[index[k]]:
                    break
                index[k - d] = index[k]
                k += d
            index[k - d] = temp


@jit
def _ashsrt(a, npts):
    d = npts
    while d > 1:
        d = 1 if d < 5 else (5 * d - 1) // 11
        for j in range(npts - d, 0, -1):
            temp = a[j]
            k = j + d
            while k <= npts:
                if temp <= a[k]:
                    break
                a[k - d] = a[k]
                k += d
            a[k - d] = temp


@jit
def _pick(flt, right, start, npts, nlo, nhi, zlo, zhi):
    nzero = npts - nlo - nhi
    k = start
    nhalf = int((nzero - 1) / 2)
    for _ in range(nlo + nhalf):
        k = right[k]
    if nzero > 0:
        return flt[k]
    if nlo < nhi:
        return zhi
    return zlo


@jit
def _medbox(ext, ny, ncols, xbox, ybox, zlo, zhi):
    """mde_medbox: every line through mde_medboxset and its filter. Arrays are
    1-based as IRAF's, with one spare at each end."""
    nbuf = ncols + 2 * (xbox // 2)
    full = xbox * ybox
    flt = np.zeros(full + 2, dtype=np.float32)
    left = np.zeros(full + 2, dtype=np.int64)
    right = np.zeros(full + 2, dtype=np.int64)
    index = np.zeros(full + 2, dtype=np.int64)
    insert = np.zeros(full + 2, dtype=np.float32)
    out = np.zeros((ny, ncols), dtype=np.float32)
    npts = nptsp1 = mp = start = finish = nlo = nhi = 0

    for line in range(1, ny + 1):
        base = line - 1          # data[i, j] is ext[base + j - 1, i - 1]

        # ---------------------------------------------------- mde_medboxset
        if xbox == 1:
            npts = 0
            nlo = 0
            nhi = 0
            for i in range(1, ybox + 1):
                v = ext[base + i - 1, 0]
                if v < zlo:
                    nlo += 1
                    continue
                if v > zhi:
                    nhi += 1
                    continue
                npts += 1
                flt[npts] = v
            if npts > 0:
                _ashsrt(flt, npts)
            nptsp1 = npts + 1
            mp = 1
        elif line == 1 or ybox == 1:
            npts = full
            nptsp1 = npts + 1
            mp = 1
            nlo = 0
            nhi = 0
            k = 1
            for i in range(1, xbox + 1):
                for j in range(1, ybox + 1):
                    v = ext[base + j - 1, i - 1]
                    if v < zlo:
                        nlo += 1
                    if v > zhi:
                        nhi += 1
                    flt[k] = v
                    index[k] = k
                    k += 1
            _gshsrt(flt, index, npts)
            start = index[1]
            finish = index[npts]
            left[start] = 0
            for i in range(2, npts + 1):
                left[index[i]] = index[i - 1]
            for i in range(1, npts):
                right[index[i]] = index[i + 1]
            right[finish] = npts + 1
        else:
            # Xbox elements leave with the line: the first row of each column.
            for i in range(1, npts + 1, ybox):
                if flt[i] < zlo:
                    nlo -= 1
                if flt[i] > zhi:
                    nhi -= 1
                if i == start:
                    start = right[i]
                    left[right[i]] = 0
                elif i == finish:
                    finish = left[i]
                    right[left[i]] = nptsp1
                else:
                    left[right[i]] = left[i]
                    right[left[i]] = right[i]
            # The new points: the first xbox columns on an odd line (the last
            # one ran right to left), the last xbox on an even one.
            col0 = 1 if line % 2 == 1 else nbuf - xbox + 1
            for i in range(1, xbox + 1):
                v = ext[base + ybox - 1, col0 + i - 2]
                if v < zlo:
                    nlo += 1
                if v > zhi:
                    nhi += 1
                insert[i] = v
                index[i] = i
            _gshsrt(insert, index, xbox)
            if line % 2 == 1:
                mp = mp + ybox
                if mp > npts:
                    mp = 1
            j = start
            for i in range(1, xbox + 1):
                lpos = index[i]
                k = (mp + (lpos - 1) * ybox) % npts
                flt[k] = insert[lpos]
                while j != right[finish] and insert[lpos] > flt[j]:
                    j = right[j]
                if j == start:
                    left[start] = k
                    left[k] = 0
                    right[k] = start
                    start = k
                elif j == right[finish]:
                    right[finish] = k
                    left[k] = finish
                    right[k] = npts + 1
                    finish = k
                else:
                    left[k] = left[j]
                    right[k] = right[left[j]]
                    right[left[j]] = k
                    left[j] = k
            if line % 2 == 0:
                mp = mp - ybox
                if mp < 1:
                    mp = npts + mp

        # ---------------------------------------------------------- filters
        if xbox == 1:
            # mde_yefilter
            for i in range(1, ncols):
                if npts > 0:
                    out[line - 1, i - 1] = flt[(npts + 1) // 2]
                elif nlo < nhi:
                    out[line - 1, i - 1] = zhi
                else:
                    out[line - 1, i - 1] = zlo
                npts = 0
                nlo = 0
                nhi = 0
                for j in range(1, ybox + 1):
                    v = ext[base + j - 1, i]
                    if v < zlo:
                        nlo += 1
                        continue
                    if v > zhi:
                        nhi += 1
                        continue
                    npts += 1
                    flt[npts] = v
                if npts > 0:
                    _ashsrt(flt, npts)
            if npts > 0:
                out[line - 1, ncols - 1] = flt[(npts + 1) // 2]
            elif nlo < nhi:
                out[line - 1, ncols - 1] = zhi
            else:
                out[line - 1, ncols - 1] = zlo
        elif ybox == 1:
            # mde_xefilter
            for i in range(1, ncols):
                out[line - 1, i - 1] = _pick(flt, right, start, npts, nlo, nhi, zlo, zhi)
                if flt[mp] < zlo:
                    nlo -= 1
                if flt[mp] > zhi:
                    nhi -= 1
                if mp == start:
                    start = right[mp]
                    left[right[mp]] = 0
                else:
                    right[left[mp]] = right[mp]
                if mp == finish:
                    finish = left[mp]
                    right[left[mp]] = nptsp1
                else:
                    left[right[mp]] = left[mp]
                v = ext[base, i + xbox - 1]
                if v < zlo:
                    nlo += 1
                if v > zhi:
                    nhi += 1
                flt[mp] = v
                k = start
                while k != right[finish] and flt[mp] > flt[k]:
                    k = right[k]
                if k == start:
                    left[start] = mp
                    left[mp] = 0
                    right[mp] = start
                    start = mp
                elif k == right[finish]:
                    right[finish] = mp
                    left[mp] = finish
                    right[mp] = nptsp1
                    finish = mp
                else:
                    left[mp] = left[k]
                    right[mp] = right[left[k]]
                    right[left[k]] = mp
                    left[k] = mp
                mp += 1
                if mp > npts:
                    mp = 1
            out[line - 1, ncols - 1] = _pick(flt, right, start, npts, nlo, nhi, zlo, zhi)
        else:
            forward = line % 2 == 1
            col = 1 + xbox if forward else nbuf - xbox
            i = 1 if forward else ncols
            for _ in range(ncols - 1):
                out[line - 1, i - 1] = _pick(flt, right, start, npts, nlo, nhi, zlo, zhi)
                for j in range(mp, mp + ybox):
                    if flt[j] < zlo:
                        nlo -= 1
                    if flt[j] > zhi:
                        nhi -= 1
                    if j == start:
                        start = right[j]
                        left[right[j]] = 0
                    elif j == finish:
                        finish = left[j]
                        right[left[j]] = nptsp1
                    else:
                        right[left[j]] = right[j]
                        left[right[j]] = left[j]
                for j in range(1, ybox + 1):
                    v = ext[base + j - 1, col - 1]
                    if v < zlo:
                        nlo += 1
                    if v > zhi:
                        nhi += 1
                    flt[mp + j - 1] = v
                    index[j] = j
                _gshsrt(flt[mp - 1:], index, ybox)
                k = start
                for j in range(1, ybox + 1):
                    lpos = index[j] + mp - 1
                    while flt[lpos] > flt[k] and k != right[finish]:
                        k = right[k]
                    if k == start:
                        left[start] = lpos
                        left[lpos] = 0
                        right[lpos] = start
                        start = lpos
                    elif k == right[finish]:
                        right[finish] = lpos
                        left[lpos] = finish
                        right[lpos] = nptsp1
                        finish = lpos
                    else:
                        left[lpos] = left[k]
                        right[lpos] = right[left[k]]
                        right[left[k]] = lpos
                        left[k] = lpos
                if forward:
                    mp = mp + ybox
                    if mp > npts:
                        mp = 1
                    col += 1
                    i += 1
                else:
                    mp = mp - ybox
                    if mp < 1:
                        mp = mp + npts
                    col -= 1
                    i -= 1
            out[line - 1, i - 1] = _pick(flt, right, start, npts, nlo, nhi, zlo, zhi)
    return out
