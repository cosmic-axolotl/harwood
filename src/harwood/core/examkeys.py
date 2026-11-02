"""imexamine's other keys: b and x y for positions, t for a section as an image, u and v
for a vector through the image, and the quick photometry of , and .

From iraf-community 2.18: pkg/images/tv/imexamine/{iegcur,iepos,ietimexam,ievimexam,
ieqrimexam}.x. See LICENSES/IRAF-COPYRIGHT.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np

from . import iraffmt

REAL = np.float32
INDEF = None
NLINES = 16
RADIAN = float(REAL(57.295779513082320877))


def text(value, fmt: str, decpl: int | None = None, width: int | None = None) -> str:
    """A number as SPP's printf has it, or INDEF to the width."""
    if value is None or (isinstance(value, float) and not np.isfinite(value)):
        return "INDEF".rjust(width or 0)
    return iraffmt.printf(float(value), fmt, decpl, width).rstrip().rjust(width or 0)


def box(x1, y1, x2, y2) -> str:
    """The b key: "%4d %4d %4d %4d", in order, of the pixels with the two positions."""
    ix1, ix2 = int(REAL(x1) + REAL(0.5)), int(REAL(x2) + REAL(0.5))
    iy1, iy2 = int(REAL(y1) + REAL(0.5)), int(REAL(y2) + REAL(0.5))
    return (f"{min(ix1, ix2):4d} {max(ix1, ix2):4d} {min(iy1, iy2):4d} {max(iy1, iy2):4d}")


def position(data, x: float, y: float, xorigin: float, yorigin: float) -> str | None:
    """The x key, with what it adds past an origin; None when it is off the image."""
    ix, iy = int(REAL(x) + REAL(0.5)), int(REAL(y) + REAL(0.5))
    ny, nx = data.shape
    if ix < 1 or ix > nx or iy < 1 or iy > ny:
        return None
    line = (f"{text(REAL(x), 'f', 2, 7)} {text(REAL(y), 'f', 2, 7)} "
            f"{text(REAL(data[iy - 1, ix - 1]), 'g', 0, 7)}")
    if xorigin != 0.0 or yorigin != 0.0:
        dx = REAL(REAL(x) - REAL(xorigin))
        dy = REAL(REAL(y) - REAL(yorigin))
        r = REAL(math.sqrt(float(REAL(dx * dx) + REAL(dy * dy))))
        angle = math.atan2(float(dy), float(dx)) * RADIAN
        t = REAL(math.fmod(360.0 + angle, 360.0))
        line += (f" {text(REAL(xorigin), 'f', 0, 7)} {text(REAL(yorigin), 'f', 2, 7)} "
                 f"{text(dx, 'f', 2, 7)} {text(dy, 'f', 2, 7)} {text(r, 'f', 2, 7)} "
                 f"{text(t, 'f', 1, 5)}")
    return line


def origin(x: float, y: float) -> str:
    return f"Origin: {text(REAL(x), 'f', 2)} {text(REAL(y), 'f', 2)}"


def section_limits(x: float, y: float, nx: int, ny: int) -> tuple[int, int, int, int]:
    """ie_timexam's section, centred on the cursor."""
    x, y = REAL(x), REAL(y)
    half = REAL(0.5)
    return (int(REAL(REAL(x - REAL((nx - 1) // 2)) + half)),
            int(REAL(REAL(x + REAL(nx // 2)) + half)),
            int(REAL(REAL(y - REAL((ny - 1) // 2)) + half)),
            int(REAL(REAL(y + REAL(ny // 2)) + half)))


def output_name(root: str, image: str) -> Path:
    """The next of root.001.fits, root.002.fits... IRAF's root, with no root, is the image."""
    base = Path(root or image)
    suffix = base.suffix if base.suffix in (".fits", ".fit", ".imh", ".pl") else ""
    stem = str(base)[:-len(suffix)] if suffix else str(base)
    for number in range(1, 10000):
        path = Path(f"{stem}.{number:03d}{suffix or '.fits'}")
        if not path.exists():
            return path
    return Path(f"{stem}.{number + 1:03d}{suffix or '.fits'}")


def write_section(data, header, x, y, nx, ny, root, image) -> tuple[Path | None, str]:
    """ie_timexam: the section as an image of reals, with the header of the image and no
    WCS. What it says, as IRAF does.

    Line by line, each line must be in the image, so there is no image when the section passes
    the bottom or the top. Past a side IRAF goes on and fills the width it has promised with
    whatever is behind its buffer in memory, which is no part of the image: this stops at the
    side. IRAF's task dies of some of these, and they only warn here."""
    from astropy.io import fits

    x1, x2, y1, y2 = section_limits(x, y, nx, ny)
    rows, columns = data.shape
    cx1, cx2, cy1, cy2 = max(1, x1), min(columns, x2), y1, y2
    if cx2 < cx1 or y1 < 1 or y2 > rows:
        return None, "Warning: Pixels out of bounds"
    name = output_name(root, image)
    keep = fits.Header()
    for card in (header.cards if header is not None else ()):
        key = card.keyword
        if key in ("WCSDIM", "LTV1", "LTV2", "LTM1_1", "LTM2_2", "LTM1_2", "LTM2_1") \
                or key.startswith(("CRPIX", "CRVAL", "CDELT", "CTYPE", "CD1_", "CD2_", "WAT")):
            continue
        if key in ("SIMPLE", "BITPIX", "NAXIS", "NAXIS1", "NAXIS2", "EXTEND", "BZERO",
                   "BSCALE"):
            continue
        keep.append(card)
    fits.writeto(name, np.asarray(data[cy1 - 1:cy2, cx1 - 1:cx2], REAL), keep, overwrite=False)
    return name, f"{image}[{cx1}:{cx2},{cy1}:{cy2}] -> {name}"


def _boundary(data, column: int, line: int, btype: str, constant: float):
    """A pixel, in or out of the image, by IMIO's boundary extensions."""
    ny, nx = data.shape

    def fold(at, n):
        if btype == "nearest":
            return min(max(at, 1), n)
        if btype == "wrap":
            return (at - 1) % n + 1
        if btype == "reflect":
            period = 2 * n
            at = (at - 1) % period
            return (at if at < n else period - 1 - at) + 1
        return at

    if btype == "project":
        at_x = column if 1 <= column <= nx else (2 - column if column < 1 else 2 * nx - column)
        at_y = line if 1 <= line <= ny else (1 - line + 1 if line < 1 else 2 * ny - line)
        at_x, at_y = min(max(at_x, 1), nx), min(max(at_y, 1), ny)
        value = data[at_y - 1, at_x - 1]
        if 1 <= column <= nx and 1 <= line <= ny:
            return value
        edge = data[min(max(line, 1), ny) - 1, min(max(column, 1), nx) - 1]
        return REAL(REAL(2) * edge - value)
    if btype == "constant" and not (1 <= column <= nx and 1 <= line <= ny):
        return REAL(constant)
    return data[fold(line, ny) - 1, fold(column, nx) - 1]


def _buffer(data, col1, col2, line1, line2, btype, constant) -> np.ndarray:
    """imgs2r over the boundary."""
    out = np.empty((line2 - line1 + 1, col2 - col1 + 1), REAL)
    for j, line in enumerate(range(line1, line2 + 1)):
        for i, column in enumerate(range(col1, col2 + 1)):
            out[j, i] = _boundary(data, column, line, btype, constant)
    return out


def _bilinear(buf, x, y) -> np.float32:
    """ii_bilinear, one point; x and y from 1 in the buffer. Past its end a pixel is the
    last one, which only ever has a weight of zero."""
    nx, ny = int(x), int(y)
    sx = REAL(REAL(x) - REAL(nx))
    tx = REAL(REAL(1) - sx)
    sy = REAL(REAL(y) - REAL(ny))
    ty = REAL(REAL(1) - sy)
    rows, columns = buf.shape

    def at(i, j):
        return buf[min(max(j, 1), rows) - 1, min(max(i, 1), columns) - 1]

    total = REAL(REAL(tx * ty) * at(nx, ny))
    total = REAL(total + REAL(REAL(sx * ty) * at(nx + 1, ny)))
    total = REAL(total + REAL(REAL(tx * sy) * at(nx, ny + 1)))
    return REAL(total + REAL(REAL(sx * sy) * at(nx + 1, ny + 1)))


def _nint(value: float) -> int:
    return int(value + 0.5) if value >= 0 else -int(-value + 0.5)


def _get_row(data, x1, y1, x2, y2, nvals, width, btype, constant):
    nedge = max(2, width // 2 + 1)
    col1 = int(min(x1, x2)) - nedge
    col2 = _nint(float(max(x1, x2))) + nedge
    line1 = int(y1) - nedge
    line2 = _nint(float(y1)) + nedge
    dx = 0.0 if nvals == 1 else float(REAL(x2 - x1)) / (nvals - 1)
    noff = float(REAL(REAL(REAL(width) - REAL(1.0)) / REAL(2.0)))
    xv = float(x1) - col1 + 1
    xs = np.empty(nvals, REAL)
    for i in range(nvals):
        xs[i] = REAL(xv)
        xv += dx
    yv = float(y1) - noff
    oys = [REAL(yv + float(j)) for j in range(width)]
    total = np.zeros(nvals, REAL)
    buf = None
    linea = lineb = 0
    values = np.empty(nvals, REAL)
    for i in range(width):
        line = int(yv)
        if buf is None or line > linea:
            linea = min(line2, line + NLINES - 1)
            lineb = max(line1, line - nedge)
            linec = min(line2, linea + nedge)
            buf = _buffer(data, col1, col2, lineb, linec, btype, constant)
        y = REAL(float(oys[i]) - lineb + 1)
        for k in range(nvals):
            values[k] = _bilinear(buf, xs[k], y)
        total = values.copy() if width == 1 else (values + total).astype(REAL)
        yv += 1.0
    if width > 1:
        total = (total / REAL(width)).astype(REAL)
    return total


def _get_col(data, x1, y1, x2, y2, nvals, width, btype, constant):
    nedge = max(2, width // 2 + 1)
    col1 = int(x1) - nedge
    col2 = _nint(float(x1)) + nedge
    line1 = int(min(y1, y2)) - nedge
    line2 = _nint(float(max(y1, y2))) + nedge
    dy = 0.0 if nvals == 1 else float(REAL(y2 - y1)) / (nvals - 1)
    noff = float(REAL(REAL(REAL(width) - REAL(1.0)) / REAL(2.0)))
    xv = float(x1) - noff
    xs = [REAL(xv + float(j + 1 - col1)) for j in range(width)]
    yv = float(y1)
    out = np.empty(nvals, REAL)
    buf = None
    linea = lineb = 0
    for i in range(nvals):
        line = int(yv)
        if dy >= 0.0 and (buf is None or line > linea):
            linea = min(line2, line + NLINES - 1)
            lineb = max(line1, line - nedge)
            linec = min(line2, linea + nedge)
            buf = _buffer(data, col1, col2, lineb, linec, btype, constant)
        elif dy < 0.0 and (buf is None or line < linea):
            linea = max(line1, line - NLINES + 1)
            lineb = max(line1, linea - nedge)
            linec = min(line2, line + nedge)
            buf = _buffer(data, col1, col2, lineb, linec, btype, constant)
        y = REAL(yv - lineb + 1)
        if width == 1:
            out[i] = _bilinear(buf, xs[0], y)
        else:
            total = REAL(0.0)
            for k in range(width):
                total = REAL(total + _bilinear(buf, xs[k], y))
            out[i] = REAL(total / REAL(width))
        yv += dy
    return out


def _get_vector(data, x1, y1, x2, y2, nvals, width, btype, constant):
    dx = float(REAL(x2 - x1)) / (nvals - 1)
    dy = float(REAL(y2 - y1)) / (nvals - 1)
    if x1 < x2:
        dpx, dpy = -dy, dx
    else:
        dpx, dpy = dy, -dx
    ratio = dx / dy
    nedge = width + 1
    noff = float(REAL(REAL(REAL(width) - REAL(1.0)) / REAL(2.0)))
    xoff, yoff = noff * dpx, noff * dpy
    col1 = int(min(x1, x2)) - nedge
    col2 = _nint(float(max(x1, x2))) + nedge
    line1 = int(min(y1, y2)) - nedge
    line2 = _nint(float(max(y2, y1))) + nedge
    xv, yv = float(x1) - xoff, float(y1) - yoff
    oxs = [REAL(float(j) * dpx) for j in range(width)]
    oys = [REAL(float(j) * dpy) for j in range(width)]
    out = np.empty(nvals, REAL)
    buf = None
    linea = lineb = colb = 0
    for i in range(nvals):
        line = int(yv)
        fresh = False
        if dy >= 0.0 and (buf is None or line > linea):
            linea = min(line2, line + NLINES - 1)
            lineb = max(line1, line - nedge)
            linec = min(line2, linea + nedge)
            lim3 = REAL(xv + float(linea - line + 1) * ratio)
            fresh = True
        elif dy < 0.0 and (buf is None or line < linea):
            linea = max(line1, line - NLINES + 1)
            lineb = max(line1, linea - nedge)
            linec = min(line2, line + nedge)
            lim3 = REAL(xv + float(linea - line - 1) * ratio)
            fresh = True
        if fresh:
            lim1 = REAL(xv)
            lim2 = REAL(float(lim1) + float(width - 1) * dpx)
            lim4 = REAL(float(lim3) + float(width - 1) * dpx)
            colb = max(col1, int(min(lim1, lim2, lim3, lim4)) - 1)
            colc = min(col2, _nint(float(max(lim1, lim2, lim3, lim4))) + 1)
            buf = _buffer(data, colb, colc, lineb, linec, btype, constant)
        shift_x = REAL(xv - colb + 1)
        shift_y = REAL(yv - lineb + 1)
        total = REAL(0.0)
        for k in range(width):
            value = _bilinear(buf, REAL(oxs[k] + shift_x), REAL(oys[k] + shift_y))
            total = value if width == 1 else REAL(total + value)
        out[i] = total if width == 1 else REAL(total / REAL(width))
        xv += dx
        yv += dy
    return out


def vector(data, key: str, x1, y1, x2, y2, naverage: int, boundary: str, constant: float):
    """ie_vimexam: (x, values, the ends) of a vector between two cursor positions; for u,
    the first is its centre."""
    data = np.asarray(data, REAL)
    ny, nx = data.shape
    x1, y1, x2, y2 = REAL(x1), REAL(y1), REAL(x2), REAL(y2)
    if key == "u":
        first_x, first_y = REAL(REAL(REAL(2) * x1) - x2), REAL(REAL(REAL(2) * y1) - y2)
    else:
        first_x, first_y = x1, y1
    start_x, start_y = first_x, first_y
    ax = max(REAL(1.0), min(first_x, REAL(nx)))
    bx = min(REAL(nx), max(REAL(1.0), x2))
    ay = max(REAL(1.0), min(first_y, REAL(ny)))
    by = min(REAL(ny), max(REAL(1.0), y2))
    ddx = REAL(bx - ax)
    ddy = REAL(by - ay)
    nvals = int(REAL(math.sqrt(float(REAL(REAL(ddx * ddx) + REAL(ddy * ddy)))))) + 1
    if ax == bx:
        values = _get_col(data, ax, ay, bx, by, nvals, naverage, boundary, constant)
    elif ay == by:
        values = _get_row(data, ax, ay, bx, by, nvals, naverage, boundary, constant)
    else:
        values = _get_vector(data, ax, ay, bx, by, nvals, naverage, boundary, constant)
    axis = np.arange(1, nvals + 1, dtype=REAL)
    if key == "u":
        cx = REAL(REAL(start_x + x2) / REAL(2))
        cy = REAL(REAL(start_y + y2) / REAL(2))
        a, b = REAL(cx - ax), REAL(cy - ay)
        zmin = REAL(math.sqrt(float(REAL(a * a) + REAL(b * b))))
        axis = (axis - zmin).astype(REAL)
    return axis, values, (float(ax), float(ay), float(bx), float(by))


def center(data, radius, xcntr, ycntr):
    """ie_center, in real: the marginals above their means, up to three times."""
    rows, columns = data.shape
    half = REAL(0.5)
    for _ in range(3):
        xlast, ylast = xcntr, ycntr
        x1 = int(REAL(REAL(xcntr - radius) + half))
        x2 = int(REAL(REAL(xcntr + radius) + half))
        y1 = int(REAL(REAL(ycntr - radius) + half))
        y2 = int(REAL(REAL(ycntr + radius) + half))
        if x2 < 1 or x1 > columns or y2 < 1 or y1 > rows:
            break
        x1, x2, y1, y2 = max(1, x1), min(columns, x2), max(1, y1), min(rows, y2)
        block = data[y1 - 1:y2, x1 - 1:x2]
        ny, nx = block.shape
        total = REAL(np.add.accumulate(block.ravel(), dtype=REAL)[-1])
        margins = np.add.accumulate(block, axis=0, dtype=REAL)[-1]
        mean = REAL(total / REAL(nx))
        sum1 = sum2 = REAL(0.0)
        for i in range(nx):
            value = REAL(margins[i] - mean)
            if value > 0:
                sum1 = REAL(sum1 + REAL(REAL(x1 + i) * value))
                sum2 = REAL(sum2 + value)
        if sum2 == 0:
            break
        xcntr = REAL(sum1 / sum2)
        margins = np.add.accumulate(block, axis=1, dtype=REAL)[:, -1]
        mean = REAL(total / REAL(ny))
        sum1 = sum2 = REAL(0.0)
        for j in range(ny):
            value = REAL(margins[j] - mean)
            if value > 0:
                sum1 = REAL(sum1 + REAL(REAL(y1 + j) * value))
                sum2 = REAL(sum2 + value)
        if sum2 == 0:
            break
        ycntr = REAL(sum1 / sum2)
        if int(xcntr) == int(xlast) and int(ycntr) == int(ylast):
            break
    return xcntr, ycntr


def quick(data, x: float, y: float, s) -> dict:
    """ie_qrimexam: the , and . keys. Centre, sky and sums as r and a, but no iterating
    of the aperture: one fit, to the pixels within the radius and rplot, and a width from
    the second moments."""
    from . import nlfit, starfocus
    from .examine import _amedr

    data = np.asarray(data, REAL)
    rows, columns = data.shape
    radius, buffer, width = REAL(s.radius), REAL(s.buffer), REAL(s.width)
    rplot = REAL(s.rplot)
    xcntr, ycntr = REAL(x), REAL(y)
    if s.center:
        xcntr, ycntr = center(data, radius, xcntr, ycntr)
    dfwhm = radius
    if not s.background:
        buffer = width = REAL(0.0)
    reach = max(rplot, REAL(REAL(radius + buffer) + width))
    x1, x2 = int(REAL(xcntr - reach)), int(REAL(xcntr + reach))
    y1, y2 = int(REAL(ycntr - reach)), int(REAL(ycntr + reach))
    if x2 < 1 or x1 > columns or y2 < 1 or y1 > rows:
        return {"error": "Pixels out of bounds"}
    x1, x2, y1, y2 = max(1, x1), min(columns, x2), max(1, y1), min(rows, y2)
    if s.background and width > 0:
        r1 = REAL(radius * radius)
        r2 = REAL(REAL(radius + buffer) * REAL(radius + buffer))
        far = REAL(REAL(radius + buffer) + width)
        r3 = REAL(far * far)
        sky = []
        for j in range(y1, y2 + 1):
            d = REAL(ycntr - REAL(j))
            dy2 = REAL(d * d)
            for i in range(x1, x2 + 1):
                d = REAL(xcntr - REAL(i))
                r = REAL(REAL(d * d) + dy2)
                if r > r1 and r >= r2 and r <= r3:
                    sky.append(data[j - 1, i - 1])
        ns = len(sky)
    else:
        sky, ns = [], 0
    bkg = REAL(_amedr(np.array(sky, REAL))) if ns else REAL(0.0)

    xs, ys, ws = [], [], []
    far_x, far_y, far_w = [], [], []
    sumo = sums = sumxx = sumyy = sumxy = 0.0
    zcntr = REAL(0.0)
    five = REAL(5.0)
    for j in range(y1, y2 + 1):
        dy = REAL(REAL(j) - ycntr)
        for i in range(x1, x2 + 1):
            dx = REAL(REAL(i) - xcntr)
            r = REAL(math.sqrt(float(REAL(REAL(dx * dx) + REAL(dy * dy)))))
            r3 = max(REAL(0.0), min(five, REAL(REAL(REAL(REAL(2) * r) / dfwhm) - REAL(1.0))))
            value = REAL(data[j - 1, i - 1] - bkg) if ns else data[j - 1, i - 1]
            weight = REAL(math.exp(-float(REAL(r3 * r3)))
                          / float(max(REAL(0.1), REAL(r * r))))
            if r <= radius:
                sumo += float(value)
                if ns:
                    sums += float(bkg)
                sumxx += float(REAL(REAL(dx * dx) * value))
                sumyy += float(REAL(REAL(dy * dy) * value))
                sumxy += float(REAL(REAL(dx * dy) * value))
                zcntr = max(value, zcntr)
                if r <= rplot:
                    xs.append(r)
                    ys.append(value)
                    ws.append(weight)
                else:
                    far_x.append(r)
                    far_y.append(value)
                    far_w.append(weight)
            elif r <= rplot:
                far_x.append(r)
                far_y.append(value)
                far_w.append(REAL(0.0))
    no = len(xs)
    plot_x = np.array(xs + far_x[::-1], REAL)
    plot_y = np.array(ys + far_y[::-1], REAL)
    if rplot <= radius:
        fit_n = no + len(far_x)
        plot_n = no
        fit_x, fit_y = plot_x[:fit_n], plot_y[:fit_n]
        fit_w = np.array(ws + far_w[::-1], REAL)[:fit_n]
    else:
        fit_n = no
        plot_n = no + len(far_x)
        fit_x, fit_y = plot_x[:no], plot_y[:no]
        fit_w = np.array(ws, REAL)

    two = math.log(2.0)
    beta = s.beta
    if s.fittype == "gaussian":
        from ..viewer.analysis import _ie_dgauss, _ie_gauss

        fit = nlfit.NLFit(_ie_gauss, _ie_dgauss,
                          [zcntr, REAL(REAL(dfwhm * dfwhm) / REAL(REAL(8) * REAL(two)))],
                          [0, 1], 0.001, 100)
    else:
        power = REAL(-3.0 if beta is None else -beta)
        root = starfocus._sqrt(starfocus._pow(2.0, REAL(REAL(-1) / power)) - 1.0)
        fit = nlfit.NLFit(starfocus.moffat, starfocus.dmoffat,
                          [zcntr, REAL(float(dfwhm) / 2.0 / root), power],
                          [0, 1, 2] if beta is None else [0, 1], 0.001, 100)
    status = fit.fit(fit_x, fit_y, fit_w) if fit_n else nlfit.NO_DEG_FREEDOM
    peak = fwhm = None
    moffat_beta = beta
    warning = ""
    if status in (nlfit.SINGULAR, nlfit.NO_DEG_FREEDOM):
        label = "Gaussian" if s.fittype == "gaussian" else "Moffat"
        warning = f"WARNING: {label} fit did not converge"
        moffat_beta = None
    else:
        p = fit.param
        if p[1] < 0:
            moffat_beta = None
        elif s.fittype == "gaussian":
            peak = float(p[0])
            fwhm = float(REAL(math.sqrt(8.0 * float(REAL(two)) * float(p[1]))))
        else:
            peak = float(p[0])
            moffat_beta = float(-p[2])
            fwhm = float(REAL(abs(float(p[1])) * 2.0
                              * starfocus._sqrt(float(starfocus._pow(2.0, REAL(REAL(-1)
                                                                          / p[2]))) - 1.0)))
            # A power that rises: IRAF's sqrt of a negative, which it prints as INDEF
            fwhm = None if math.isnan(fwhm) else fwhm
    mag = rmom = ellip = pa = None
    if sumo > 0:
        mag = float(REAL(float(REAL(s.magzero)) - 2.5 * math.log10(sumo)))
        r2 = REAL(sumxx + sumyy)
        if r2 > 0:
            if s.fittype == "gaussian":
                rmom = float(REAL(2.0 * math.sqrt(float(REAL(two)) * float(r2) / sumo)))
            elif moffat_beta is not None and moffat_beta > 2:
                factor = (moffat_beta - 2.0) * (2.0 ** (1.0 / moffat_beta) - 1)
                rmom = float(REAL(2.0 * math.sqrt(factor * float(r2) / sumo)))
            a = REAL(sumxx - sumyy)
            b = REAL(2 * sumxy)
            r1 = REAL(REAL(a * a) + REAL(b * b))
            ellip = float(REAL(math.sqrt(float(r1)) / float(r2))) if r1 > 0 else 0.0
        if ellip is None or ellip < 0.01:
            ellip = 0.0 if ellip is not None else None
        else:
            pa = float(REAL(0.5 * math.atan2(2 * sumxy, sumxx - sumyy) * RADIAN))
    no = fit_n
    line = (f"{text(xcntr, 'f', 2, 7)} {text(ycntr, 'f', 2, 7)} {text(mag, 'f', 2, 7)} "
            f"{text(sumo, 'f', 1, 8)} {text(sums / no if no else 0.0, 'f', 2, 8)} {no:3d} "
            f"{text(rmom, 'f', 2, 5)} {text(ellip, 'f', 3, 5)} {text(pa, 'f', 1, 5)} "
            f"{text(peak, 'f', 2, 8)} {text(fwhm, 'f', 2, 5)}")
    grid = fit_values = None
    if fwhm is not None:
        grid = (np.arange(51, dtype=REAL) * REAL(rplot / REAL(50))).astype(REAL)
        fit_values = np.asarray(fit.vector(grid), REAL)
    return {"line": line, "x": float(xcntr), "y": float(ycntr), "mag": mag, "flux": sumo,
            "sky": sums / no if no else 0.0, "n": no, "rmom": rmom, "e": ellip, "pa": pa,
            "peak": peak, "fwhm": fwhm, "beta": moffat_beta, "warning": warning,
            "plot_x": plot_x[:plot_n], "plot_y": plot_y[:plot_n], "grid": grid,
            "fit": fit_values}


GLABEL = "#   COL    LINE    RMAG     FLUX      SKY   N  RMOM ELLIP    PA     PEAK GFWHM"
MLABEL = "#   COL    LINE    RMAG     FLUX      SKY   N  RMOM ELLIP    PA     PEAK MFWHM"
