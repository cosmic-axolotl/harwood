"""IRAF's imshift (images.imgeom), ported.

From iraf-community 2.18: t_imshift.x for the task, and from math/iminterp
the interpolants it uses -- msifit.x (the coefficients and, for spline3, the
B-spline fit with natural end conditions through ii_spline2d.x) and
ii_greval.x (evaluation on a grid, x first and then y, in single precision).

The output pixel (i, j) is the input interpolated at (i - xshift, j - yshift),
the input extended past its edges by the boundary rule. Like IRAF:

- a whole-pixel shift copies pixels, whatever the interpolant;
- `nearest` rounds the shift and copies;
- the coordinates are formed the way ish_gshiftxy forms them, in single
  precision on a buffer that starts a few pixels outside the shifted image,
  so the fractional offsets are IRAF's to the last bit;
- spline3 is fitted on that buffer, sixteen output lines at a time with a
  sixteen-pixel margin, which is what makes its edges what they are.

sinc and lsinc are, as imshift sets them up, one and the same: a tapered
sinc 31 points wide (sinc21 for 21, the width after the name), tabulated
once for the exact fractional shift (msisinit, ii_sinctable, ii_bilsinc).
drizzle averages each output pixel's footprint over the input pixels it
covers, the footprint shrunk by the pixel fraction of drizzle[0.5], as
ii_grdriz does.

Translated from IRAF, Copyright (c) 1986 Association of Universities for
Research in Astronomy Inc.; the notice it is distributed under is in
LICENSES/IRAF-COPYRIGHT.
"""

from __future__ import annotations

import numpy as np

from .nlfit import square

INTERPOLANTS = ("nearest", "linear", "poly3", "poly5", "spline3", "sinc", "lsinc",
                "drizzle")
NSINC = 15              # half width of the default sinc (2 * NSINC + 1 = 31)
PIXFRAC = 1.0
MIN_PIXFRAC = 0.001
HALFPI = np.float32(1.5707963267948966)
BOUNDARIES = ("nearest", "constant", "reflect", "wrap")

NYOUT = 16              # output lines per pass
NMARGIN = 3             # boundary pixels around the buffer
NMARGIN_SPLINE3 = 16

_PAD = {"nearest": "edge", "reflect": "reflect", "wrap": "wrap", "constant": "constant"}


def _fp_equal(a: float, b: float) -> bool:
    """fp_equald: equal to within the precision of a double."""
    return bool(abs(a - b) <= 2.0 * np.finfo(float).eps * max(abs(a), abs(b), 1e-300))


def shift(data: np.ndarray, xshift: float, yshift: float, interp: str = "linear",
          boundary: str = "nearest", constant: float = 0.0) -> np.ndarray:
    """Shift an image by (xshift, yshift) pixels; positive moves it to +x, +y."""
    kind = parse_interp(interp)[0]
    if kind not in INTERPOLANTS:
        raise ValueError(f"interp_type must be one of {', '.join(INTERPOLANTS)}; "
                         f"got '{interp}'")
    if boundary not in BOUNDARIES:
        raise ValueError(f"boundary_type must be one of {', '.join(BOUNDARIES)}; "
                         f"got '{boundary}'")
    data = np.asarray(data, dtype=np.float32)
    ny, nx = data.shape
    if kind == "nearest":
        return _integer(data, _nint(xshift), _nint(yshift), boundary, constant)
    if _fp_equal(xshift, float(int(xshift))) and _fp_equal(yshift, float(int(yshift))):
        return _integer(data, int(xshift), int(yshift), boundary, constant)
    return _fractional(data, float(xshift), float(yshift), interp, boundary, constant)


def parse_interp(text: str) -> tuple[str, int, float]:
    """msitype: an interpolant's name and its parameters -- sinc31 or lsinc31
    (the full width; 31 by default) and drizzle[0.5] (the pixel fraction,
    1 by default). Returns (name, half width, pixfrac)."""
    import re

    text = str(text).strip()
    if text in INTERPOLANTS:
        return text, NSINC if text in ("sinc", "lsinc") else 0, PIXFRAC
    found = re.fullmatch(r"(l?sinc)(\d+)?(\[[^]]*\])?", text)
    if found:
        width = int(found.group(2)) if found.group(2) else 2 * NSINC + 1
        return found.group(1), (width - 1) // 2, PIXFRAC
    found = re.fullmatch(r"drizzle\[([^]]*)\]", text)
    if found:
        try:
            pixfrac = float(found.group(1))
        except ValueError:
            pixfrac = PIXFRAC
        if not 0.0 <= pixfrac <= 1.0:
            pixfrac = PIXFRAC
        return "drizzle", 0, pixfrac
    return text, 0, PIXFRAC


def _nint(value: float) -> int:
    return int(np.floor(value + 0.5)) if value >= 0 else -int(np.floor(-value + 0.5))


def _extended(data, margin, boundary, constant):
    """The image with `margin` pixels of boundary extension on every side;
    pixel (x, y), 1-based, is at [y - 1 + margin, x - 1 + margin]."""
    if boundary == "constant":
        return np.pad(data, margin, mode="constant", constant_values=np.float32(constant))
    if boundary == "wrap" or boundary == "reflect":
        # numpy pads at most one image width per pass in these modes.
        out = data
        while margin > 0:
            step = min(margin, min(out.shape) - (1 if boundary == "reflect" else 0))
            out = np.pad(out, step, mode=_PAD[boundary])
            margin -= step
        return out
    return np.pad(data, margin, mode="edge")


def _integer(data, ix, iy, boundary, constant):
    """ish_ishiftxy."""
    ny, nx = data.shape
    if not (-nx <= ix <= nx) or not (-ny <= iy <= ny):
        raise ValueError("the shift is larger than the image")
    if boundary == "wrap":
        ix, iy = int(np.fmod(ix, nx)), int(np.fmod(iy, ny))
    margin = max(abs(ix), abs(iy))
    ext = _extended(data, margin, boundary, constant)
    return ext[margin - iy:margin - iy + ny, margin - ix:margin - ix + nx].copy()


def _delta(frac_abs: float, s: float) -> float:
    if _fp_equal(frac_abs, 0.0):
        return 0.0
    return 1.0 - frac_abs if s > 0 else frac_abs


def _start(c: float, frac_abs: float) -> int:
    """cin1 / lin1: int() truncates, one less below zero unless whole."""
    return int(c) - 1 if (c <= 0 and not _fp_equal(frac_abs, 0.0)) else int(c)


def _fractional(data, xshift, yshift, interp, boundary, constant):
    """ish_gshiftxy."""
    ny, nx = data.shape
    if not (-nx <= xshift <= nx) or not (-ny <= yshift <= ny):
        raise ValueError("the shift is larger than the image")
    if boundary == "wrap":
        xs, ys = float(np.fmod(xshift, nx)), float(np.fmod(yshift, ny))
    else:
        xs, ys = xshift, yshift
    dx, dy = abs(xs - int(xs)), abs(ys - int(ys))
    interp, nsinc, pixfrac = parse_interp(interp)
    margin = (NMARGIN_SPLINE3 if interp == "spline3" else
              nsinc if interp in ("sinc", "lsinc") else NMARGIN)
    table = None
    if interp in ("sinc", "lsinc"):
        fx, fy = _delta(dx, xs), _delta(dy, ys)
        table = _sinc_table(2 * nsinc + 1, np.float32(fx - _nint(fx)),
                            np.float32(fy - _nint(fy)))
    deltax = _delta(dx, xs) + margin
    deltay = _delta(dy, ys) + margin

    # The buffer coordinates, single precision as IRAF stores them; drizzle
    # takes each pixel's two edges.
    if interp == "drizzle":
        xb = np.ravel(np.column_stack((np.arange(1, nx + 1) + deltax - 0.5,
                                       np.arange(1, nx + 1) + deltax + 0.5))
                      ).astype(np.float32)
        yb = np.ravel(np.column_stack((np.arange(1, NYOUT + 1) + deltay - 0.5,
                                       np.arange(1, NYOUT + 1) + deltay + 0.5))
                      ).astype(np.float32)
    else:
        xb = (np.arange(1, nx + 1) + deltax).astype(np.float32)
        yb = (np.arange(1, NYOUT + 1) + deltay).astype(np.float32)

    cin1 = _start(1.0 - margin - xs, dx)
    cin2 = int(nx - xs + margin + 1)
    nbpix = max(int(abs(xs) + 1.0), int(abs(ys) + 1.0)) + margin
    pad = nbpix + 8
    ext = _extended(data, pad, boundary, constant)

    def window(c1, c2, l1, l2):
        # Columns c1..c2, lines l1..l2 (1-based) of the extended image.
        return ext[l1 - 1 + pad:l2 + pad, c1 - 1 + pad:c2 + pad]

    out = np.empty((ny, nx), dtype=np.float32)
    for lout1 in range(1, ny + 1, NYOUT):
        lout2 = min(lout1 + NYOUT - 1, ny)
        nyout = lout2 - lout1 + 1
        lin1 = _start(lout1 - margin - ys, dy)
        lin2 = int(lout2 - ys + margin + 1)
        buf = window(cin1, cin2, lin1, lin2)
        if interp in ("sinc", "lsinc"):
            out[lout1 - 1:lout2] = _lsinc(buf, xb, yb[:nyout], table, nsinc)
        elif interp == "drizzle":
            frac = max(MIN_PIXFRAC, min(pixfrac, 1.0))
            out[lout1 - 1:lout2] = _drizzle(buf, xb, yb[:2 * nyout], frac, frac)
        else:
            out[lout1 - 1:lout2] = _grid(buf, xb, yb[:nyout], interp)
    return out


def _sinc_table(nconv: int, xshift, yshift) -> np.ndarray:
    """ii_bisinctable for one shift: table[j, k] = fx[k] * fy[j]."""
    fx, fy = _sinc_1d(nconv, xshift), _sinc_1d(nconv, yshift)
    return (fy[:, None] * fx[None, :]).astype(np.float32)


def _sinc_1d(nconv: int, dx) -> np.ndarray:
    """ii_sinctable, one entry: the tapered sinc at offsets k - dx,
    normalised to sum to one, in single precision."""
    f32 = np.float32
    nsinc = (nconv - 1) // 2
    sconst = square(HALFPI / f32(nsinc))
    a2, a4 = f32(-0.49670), f32(0.03705)
    sign = f32(1.0) if nsinc % 2 == 0 else f32(-1.0)
    x = f32(f32(-nsinc) - f32(dx))
    out = np.zeros(nconv, dtype=f32)
    total = f32(0)
    for j in range(1, nconv + 1):
        if x == 0:
            f = f32(1.0)
        elif dx == 0:
            f = f32(0.0)
        else:
            dx2 = f32(sconst * f32((j - nsinc - 1) ** 2))
            taper = f32(f32(1.0) + a2 * dx2 + f32(a4 * dx2) * dx2)
            f = f32(f32(sign / x) * f32(taper * taper))
        out[j - 1] = f
        total = f32(total + f)
        x = f32(x + f32(1.0))
        sign = f32(-sign)
    return (out / total).astype(f32)


def _lsinc(buf, x, y, table, nsinc):
    """ii_grlsinc / ii_bilsinc on one buffer (buf[0, 0] is coeff[1, 1]):
    each point the sum, row by row, of the table times the pixels around
    its nearest one, past the edges the edge pixel."""
    f32 = np.float32
    buf = buf.astype(f32)
    ny_c, nx_c = buf.shape
    xc = np.floor(x + f32(0.5)).astype(np.int64)
    yc = np.floor(y + f32(0.5)).astype(np.int64)
    out = np.zeros((y.size, x.size), dtype=f32)
    for j in range(-nsinc, nsinc + 1):
        rows = np.clip(yc + j, 1, ny_c) - 1
        line = buf[rows]                                  # (ny, nx_c)
        total = np.zeros((y.size, x.size), dtype=f32)
        for k in range(-nsinc, nsinc + 1):
            cols = np.clip(xc + k, 1, nx_c) - 1
            total = (total + table[j + nsinc, k + nsinc] * line[:, cols]).astype(f32)
        out = (out + total).astype(f32)
    outside = (xc[None, :] < 1) | (xc[None, :] > nx_c) | (yc[:, None] < 1) | \
        (yc[:, None] > ny_c)
    out[outside] = 0
    return out


def _drizzle_coefficients(buf):
    """msifit for drizzle: one more column and row, projected linearly."""
    f32 = np.float32
    ny, nx = buf.shape
    c = np.zeros((ny + 1, nx + 1), dtype=f32)
    c[:ny, :nx] = buf
    c[:ny, nx] = f32(2) * c[:ny, nx - 1] - c[:ny, nx - 2]
    c[ny] = (f32(2) * c[ny - 1] - c[ny - 2]).astype(f32)
    return c


def _driz_row(edges, data, pixfrac, badval=0.0):
    """ii_driz1 (pixfrac 1) and ii_driz: the mean of data over each
    [a, b], 1-based data positions."""
    f32 = np.float32
    a, b = edges[0::2], edges[1::2]
    xa, xb = np.minimum(a, b), np.maximum(a, b)
    out = np.empty(xa.size, dtype=f32)
    half = f32(f32(pixfrac) / f32(2.0))
    for i in range(xa.size):
        lo, hi = xa[i], xb[i]
        na, nb = int(lo + f32(0.5)), int(hi + f32(0.5))
        accum = waccum = f32(0.0)
        if pixfrac >= 1.0:
            da, db = f32(lo - f32(na)), f32(hi - f32(nb))
            if na == nb:
                w = f32(db - da)
                accum, waccum = f32(w * data[na - 1]), w
            else:
                w = f32(f32(0.5) - da)
                accum, waccum = f32(w * data[na - 1]), w
                for j in range(na + 1, nb):
                    accum = f32(accum + data[j - 1])
                    waccum = f32(waccum + f32(1.0))
                w = f32(db + f32(0.5))
                accum = f32(accum + f32(w * data[nb - 1]))
                waccum = f32(waccum + w)
        else:
            frac = f32(pixfrac)
            if na == nb:
                w = f32(min(hi, f32(f32(nb) + half)) - max(lo, f32(f32(na) - half)))
                if w > 0:
                    accum, waccum = f32(w * data[na - 1]), w
            else:
                w = f32(f32(f32(na) + half) - max(lo, f32(f32(na) - half)))
                if w > 0:
                    accum, waccum = f32(w * data[na - 1]), w
                for j in range(na + 1, nb):
                    accum = f32(accum + f32(frac * data[j - 1]))
                    waccum = f32(waccum + frac)
                w = f32(min(hi, f32(f32(nb) + half)) - f32(f32(nb) - half))
                if w > 0:
                    accum = f32(accum + f32(w * data[nb - 1]))
                    waccum = f32(waccum + w)
        out[i] = f32(badval) if waccum == 0 else f32(accum / waccum)
    return out


def _drizzle(buf, x, y, xfrac, yfrac):
    """ii_grdriz: each line averaged across in x, then each column of that
    averaged across in y, over the output pixel's footprint."""
    f32 = np.float32
    c = _drizzle_coefficients(buf.astype(f32))
    lo, hi = min(y[0], y[1]), max(y[-2], y[-1])
    nylmin, nylmax = int(lo + f32(0.5)), int(hi + f32(0.5))
    work = np.array([_driz_row(x, c[j - 1], xfrac) for j in range(nylmin, nylmax + 1)],
                    dtype=f32)
    half = f32(f32(yfrac) / f32(2.0))
    nypts = y.size // 2
    out = np.zeros((nypts, work.shape[1]), dtype=f32)
    for j in range(nypts):
        ymin, ymax = min(y[2 * j], y[2 * j + 1]), max(y[2 * j], y[2 * j + 1])
        na, nb = int(ymin + f32(0.5)), int(ymax + f32(0.5))
        accum = np.zeros(work.shape[1], dtype=f32)
        waccum = f32(0.0)
        if na == nb:
            w = f32(min(ymax, f32(f32(nb) + half)) - max(ymin, f32(f32(na) - half)))
            if w > 0:
                accum = (accum + w * work[na - nylmin]).astype(f32)
                waccum = f32(waccum + w)
        else:
            w = f32(f32(f32(na) + half) - max(ymin, f32(f32(na) - half)))
            if w > 0:
                accum = (accum + w * work[na - nylmin]).astype(f32)
                waccum = f32(waccum + w)
            for jj in range(na + 1, nb):
                accum = (accum + f32(yfrac) * work[jj - nylmin]).astype(f32)
                waccum = f32(waccum + f32(yfrac))
            w = f32(min(ymax, f32(f32(nb) + half)) - f32(f32(nb) - half))
            if w > 0:
                accum = (accum + w * work[nb - nylmin]).astype(f32)
                waccum = f32(waccum + w)
        out[j] = 0 if waccum <= 0 else (accum / waccum).astype(f32)
    return out


def _grid(buf, x, y, interp):
    """msifit + msigrid on one buffer: buf[0, 0] is datain[1, 1]."""
    f32 = np.float32
    buf = buf.astype(f32)
    nxi = x.astype(np.int64)          # x >= 1, so int() is floor
    sx = (x - nxi.astype(f32)).astype(f32)
    tx = (f32(1) - sx).astype(f32)
    if interp == "linear":
        nymin = int(y[0])
        nymax = int(y[-1]) + 1
        rows = buf[nymin - 1:nymax]
        work = (tx * rows[:, nxi - 1] + sx * rows[:, nxi]).astype(f32)
        out = []
        for yy in y:
            n = int(yy)
            s = f32(yy - f32(n))
            t = f32(1) - s
            out.append((work[n - nymin] * t + work[n - nymin + 1] * s).astype(f32))
        return np.array(out, dtype=f32)

    if interp == "poly3":
        # Coefficient (x, y) is datain[x, y]; index 0 and nxpix+1.. come
        # from boundary projection, never reached with IRAF's margins.
        c = _project(buf, 1, 2)
        o = 1
        nymin = int(y[0]) - 1
        nymax = int(y[-1]) + 2
        rows = c[nymin - 1 + o:nymax + o]
        a, b = rows[:, nxi - 1 + o], rows[:, nxi + o]
        am1, bp1 = rows[:, nxi - 2 + o], rows[:, nxi + 1 + o]
        cd20 = ((b - f32(2) * a + am1) / f32(6)).astype(f32)
        cd21 = ((bp1 - f32(2) * b + a) / f32(6)).astype(f32)
        sx2m1 = (sx * sx - f32(1)).astype(f32)
        tx2m1 = (tx * tx - f32(1)).astype(f32)
        work = (sx * (b + sx2m1 * cd21) + tx * (a + tx2m1 * cd20)).astype(f32)
        out = []
        for yy in y:
            n = int(yy)
            s = f32(yy - f32(n))
            t = f32(1) - s
            k = n - nymin
            lm1, l0, l1, l2 = work[k - 1], work[k], work[k + 1], work[k + 2]
            cd20y = ((l1 - f32(2) * l0 + lm1) / f32(6)).astype(f32)
            cd21y = ((l2 - f32(2) * l1 + l0) / f32(6)).astype(f32)
            out.append((s * (l1 + (s * s - f32(1)) * cd21y)
                        + t * (l0 + (t * t - f32(1)) * cd20y)).astype(f32))
        return np.array(out, dtype=f32)

    if interp == "poly5":
        c = _project(buf, 2, 3)
        o = 2
        nymin = int(y[0]) - 2
        nymax = int(y[-1]) + 3
        rows = c[nymin - 1 + o:nymax + o]

        def col(k):
            return rows[:, nxi - 1 + o + k]

        m2, m1, z0, p1, p2, p3 = (col(k) for k in (-2, -1, 0, 1, 2, 3))
        cd20 = ((p1 - f32(2) * z0 + m1) / f32(6)).astype(f32)
        cd21 = ((p2 - f32(2) * p1 + z0) / f32(6)).astype(f32)
        cd40 = ((m2 - f32(4) * m1 + f32(6) * z0 - f32(4) * p1 + p2) / f32(120)).astype(f32)
        cd41 = ((m1 - f32(4) * z0 + f32(6) * p1 - f32(4) * p2 + p3) / f32(120)).astype(f32)
        sx2m1 = (sx ** 2 - f32(1)).astype(f32)
        sx2m4 = (sx2m1 - f32(3)).astype(f32)
        tx2m1 = (tx ** 2 - f32(1)).astype(f32)
        tx2m4 = (tx2m1 - f32(3)).astype(f32)
        work = (sx * (p1 + sx2m1 * (cd21 + sx2m4 * cd41))
                + tx * (z0 + tx2m1 * (cd20 + tx2m4 * cd40))).astype(f32)
        out = []
        for yy in y:
            n = int(yy)
            s = f32(yy - f32(n))
            t = f32(1) - s
            s2m1 = f32(s * s - f32(1))
            s2m4 = f32(s2m1 - f32(3))
            t2m1 = f32(t * t - f32(1))
            t2m4 = f32(t2m1 - f32(3))
            k = n - nymin
            lm2, lm1, l0, l1, l2, l3 = (work[k + d] for d in (-2, -1, 0, 1, 2, 3))
            cd20y = ((l1 - f32(2) * l0 + lm1) / f32(6)).astype(f32)
            cd21y = ((l2 - f32(2) * l1 + l0) / f32(6)).astype(f32)
            cd40y = ((lm2 - f32(4) * lm1 + f32(6) * l0 - f32(4) * l1 + l2)
                     / f32(120)).astype(f32)
            cd41y = ((lm1 - f32(4) * l0 + f32(6) * l1 - f32(4) * l2 + l3)
                     / f32(120)).astype(f32)
            out.append((s * (l1 + s2m1 * (cd21y + s2m4 * cd41y))
                        + t * (l0 + t2m1 * (cd20y + t2m4 * cd40y))).astype(f32))
        return np.array(out, dtype=f32)

    if interp == "spline3":
        c = _spline_coefficients(buf)
        nymin = int(y[0]) - 1
        nymax = int(y[-1]) + 2
        # c[y, x] with datain[1, 1] at c[1, 1]
        rows = c[nymin:nymax + 1]
        sx3 = sx * (sx * sx)
        tx3 = tx * (tx * tx)
        bs = (f32(1) + sx * (f32(3) + sx * (f32(3) - f32(3) * sx))).astype(f32)
        bt = (f32(1) + tx * (f32(3) + tx * (f32(3) - f32(3) * tx))).astype(f32)
        work = (tx3 * rows[:, nxi - 1] + bt * rows[:, nxi] + bs * rows[:, nxi + 1]
                + sx3 * rows[:, nxi + 2]).astype(f32)
        out = []
        for yy in y:
            n = int(yy)
            s = f32(yy - f32(n))
            t = f32(1) - s
            s3, t3 = s * (s * s), t * (t * t)
            bsy = f32(f32(1) + s * (f32(3) + s * (f32(3) - f32(3) * s)))
            bty = f32(f32(1) + t * (f32(3) + t * (f32(3) - f32(3) * t)))
            k = n - nymin
            out.append((t3 * work[k - 1] + bty * work[k] + bsy * work[k + 1]
                        + s3 * work[k + 2]).astype(f32))
        return np.array(out, dtype=f32)

    raise ValueError(interp)


def _project(buf, before, after):
    """msifit's boundary projection: `before` coefficients ahead of the data
    and `after` behind it on both axes, extended linearly (2 c - ...)."""
    f32 = np.float32
    ny, nx = buf.shape
    c = np.zeros((ny + before + after, nx + before + after), dtype=f32)
    c[before:before + ny, before:before + nx] = buf
    for axis_rows in (False, True):
        a = c.T if axis_rows else c
        n = ny if axis_rows else nx
        lo, hi = before, before + n - 1
        for k in range(1, before + 1):
            a[:, lo - k] = f32(2) * a[:, lo] - a[:, lo + k]
        for k in range(1, after + 1):
            a[:, hi + k] = f32(2) * a[:, hi] - a[:, hi - k]
    return c


def _spline_coefficients(buf):
    """msifit for spline3: natural end conditions, ii_spline2d along x and
    then along y. Returns c[y, x] with datain[1, 1] at c[1, 1]."""
    f32 = np.float32
    nypix, nxpix = buf.shape
    c = np.zeros((nypix + 3, nxpix + 3), dtype=f32)
    c[1:nypix + 1, 1:nxpix + 1] = buf
    tmp = _spline2d(c, nxpix)                 # spline along x of every row
    return _spline2d(tmp.T.copy(), nypix).T.copy()


def _spline2d(rows, npts):
    """ii_spline2d on every row: entries 0..npts+1 of each row (0 and npts+1
    the end second derivatives) become the npts+2 B-spline coefficients."""
    f32 = np.float32
    diag = np.zeros(npts + 2, dtype=f32)
    diag[1] = f32(-2.0)
    diag[2] = f32(0.0)
    for i in range(3, npts + 2):
        diag[i] = f32(1) / (f32(4) - diag[i - 1])
    out = rows.astype(f32).copy()
    c = out
    # 1-based index i lives at column i - 1.
    c[:, 0] = c[:, 0] / f32(6)
    c[:, 1] = (c[:, 1] - c[:, 0]) / f32(6)
    for i in range(3, npts + 2):
        c[:, i - 1] = diag[i] * (c[:, i - 1] - c[:, i - 2])
    c[:, npts + 1] = ((diag[npts] + f32(2)) * c[:, npts] - c[:, npts - 1]
                      + c[:, npts + 1] / f32(6)) / (f32(1) + diag[npts + 1]
                                                    * (diag[npts] + f32(2)))
    for i in range(npts + 1, 2, -1):
        c[:, i - 1] = c[:, i - 1] - diag[i] * c[:, i]
    c[:, 0] = c[:, 0] + f32(2) * c[:, 1] - c[:, 2]
    return out
