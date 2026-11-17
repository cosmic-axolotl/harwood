"""contour -- a contour map of an image, as IRAF's plot.contour lays it out.

What decides the map is IRAF's, ported: the image reduced to xres by yres by block
averaging or subsampling (plt_getdata, with its message), the levels NCAR's CLGEN
chooses (from the floor -- the minimum by default -- up to the ceiling, by the
interval or by a round step for about ncontours of them), REORD's choice of the
major (labelled) levels, the labels as ENCD writes them, scaled when the values
are very large or small, the high and low marks of MINMAX (nhi 0) or each point's
value (nhi 1), and the texts of the perimeter.

The curves are NCAR's, drawn by core.ncar: CONREC's tracing, DASHSMTH's splines
under tension, the dash pattern with the label in its gap, down to gio's clipping
-- the same polylines and texts, to the unit, as IRAF writes into its metacode,
on the viewport pl_map_viewport chooses (pkg/plot/vport.x).

Translated from IRAF (pkg/plot/t_contour.x, getdata.x, perim.x, vport.x) and its
NCAR utilities (sys/gio/ncarutil/conrec.f, support.f), Copyright (c) 1986
Association of Universities for Research in Astronomy Inc.; see
LICENSES/IRAF-COPYRIGHT.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from . import iraffmt, ncar

f32 = np.float32
NLA, NLM = 16, 40           # conbd: about 16 levels, at most 40
NULBLL = 3                  # unlabelled lines between labelled ones
EPSILON = 1.192e-7


class ContourError(RuntimeError):
    """What stops IRAF's contour: its message."""


# ------------------------------------------------------------- plt_getdata


def getdata(data: np.ndarray, sub: bool, pre: bool, xres: int, yres: int):
    """plt_getdata: the image as it is, or subsampled or block averaged down to the
    resolution asked for. Returns (grid[y, x] in real, message or "")."""
    nyin, nxin = data.shape
    data = data.astype(f32)
    if not ((nxin > xres and xres != 0) or (nyin > yres and yres != 0)):
        return data.copy(), ""
    if xres == 0:
        xres = nxin
    if yres == 0:
        yres = nyin
    if sub:
        xf, yf = (nxin + xres - 1) // xres, (nyin + yres - 1) // yres
        xrf = yrf = max(xf, yf) if pre else None
        if not pre:
            xrf, yrf = xf, yf
        nxout, nyout = nxin // xrf, nyin // yrf
        out = np.zeros((nyout, nxout), f32)
        yin = 0
        for jj in range(nyout):
            row = data[yin]
            out[jj] = row[::xrf][:nxout]
            yin += yrf
        return out, f"Image will be subsampled by {xrf} in x and {yrf} in y\n"
    bfx, bfy = nxin // xres, nyin // yres
    xfac = yfac = max(bfx, bfy) if pre else None
    if not pre:
        xfac, yfac = bfx, bfy
    message = f"Image will be block averaged by {xfac} in x and {yfac} in y\n"
    nxout, nyout = (nxin + xfac - 1) // xfac, (nyin + yfac - 1) // yfac
    nx_full, ny_full = nxin // xfac, nyin // yfac
    nxcols, nxrows = nxin - nx_full * xfac, nyin - ny_full * yfac
    out = np.zeros((nyout, nxout), f32)
    yin = 0

    def block_row(vec, jrow):
        for k in range(nx_full):
            s = f32(0)
            for v in vec[k * xfac:(k + 1) * xfac]:
                s = f32(s + v)
            out[jrow, k] = f32(s / f32(xfac))

    total = f32(0)
    for jj in range(ny_full):
        vec = np.zeros(nxin, f32)
        for _ in range(yfac):
            vec = (vec + data[yin]).astype(f32)
            yin += 1
        vec = (vec / f32(yfac)).astype(f32)
        block_row(vec, jj)
        if nxcols != 0:
            total = f32(0)
            for v in vec[nx_full * xfac:]:
                total = f32(total + v)
            out[jj, nx_full] = f32(total / f32(nxcols))
    if nxrows != 0:
        vec = np.zeros(nxin, f32)
        for ii in range(yin, nyin):
            vec = (vec + data[ii]).astype(f32)
        vec = (vec / f32(nxrows)).astype(f32)
        block_row(vec, nyout - 1)
        if nxcols != 0:
            # plt_blkaverage does not clear the sum here: it goes on from the last
            # full row's trailing columns
            for v in vec[nx_full * xfac:]:
                total = f32(total + v)
            out[nyout - 1, nx_full] = f32(total / f32(nxcols))
    return out, message


# ------------------------------------------------------------- CLGEN, REORD, ENCD


def _log10r(x) -> f32:
    return f32(0.43429448190325182765 * math.log(float(x)))


def _ifix(x) -> int:
    return int(x)


def clgen(z: np.ndarray, flo: float, hi: float, finc: float):
    """CLGEN: the contour levels and the floor, ceiling and step it settled on, or a
    constant field (None for the levels)."""
    clo, chi = f32(flo), f32(hi)
    glo, ha, fanc = clo, chi, f32(finc)
    crat = f32(NLA)
    if ha < glo:
        glo, ha = ha, clo
    elif ha == glo:
        if glo != 0:
            return [float(glo)], float(glo), float(glo), float(fanc)
        glo = ha = f32(z.min())
        glo, ha = f32(z.min()), f32(z.max())
        if glo >= ha:
            return None, float(glo), float(ha), float(fanc)
    if fanc <= 0:
        if fanc < 0:
            crat = f32(max(f32(1), -fanc))
        fanc = f32(f32(ha - glo) / crat)
        p = f32(iraffmt.pow_di(10.0, _ifix(f32(_log10r(fanc) + f32(5000.0))) - 5000))
        fanc = f32(f32(math.trunc(float(f32(fanc / p)))) * p)
    if chi == clo:
        glo = f32(f32(math.trunc(float(f32(glo / fanc)))) * fanc)
        ha = f32(f32(f32(math.trunc(float(f32(ha / fanc)))) * fanc)
                 * f32(1.0 + math.copysign(1e-6, float(ha))))
    levels = []
    for k in range(1, NLM + 1):
        cc = f32(glo + f32(f32(k - 1) * fanc))
        if cc > ha:
            break
        levels.append(float(cc))
    return levels, levels[0], levels[-1], float(fanc)


def reord(cl: list[float], nmg: int = NULBLL + 1):
    """REORD: the levels with the major ones (labelled) first, and how many those are."""
    nl = len(cl)
    if nl <= 4 or nmg <= 1:
        return list(cl), nl
    nml = 1 if nl <= 10 else nmg - 1
    nmlp1 = nml + 1
    isave = None
    for i in range(1, nl + 1):
        if cl[i - 1] == 0.0:
            isave = i
            break
    if isave is None:
        lev = int(f32(_log10r(abs(f32(cl[nl // 2 - 1]))) + f32(1.0)))
        q = f32(iraffmt.pow_di(10.0, lev))
        for _ in range(3):
            q = f32(q / f32(10.0))
            for i in range(1, nl + 1):
                c = f32(cl[i - 1])
                v = f32(abs(f32(c + f32(f32(1e-9) * c))) / q)
                if f32(math.fmod(v, float(nmlp1))) <= f32(0.0001):
                    isave = i
                    break
            if isave is not None:
                break
        if isave is None:
            isave = nl // 2
    istart = isave % nmlp1
    if istart == 0:
        istart = nmlp1
    c1 = [cl[i - 1] for i in range(istart, nl + 1, nmlp1)]
    mark = len(c1)
    c1 += [cl[i - 2] for i in range(2, istart + 1)]
    for i in range(2, mark + 1):
        for j in range(1, nml + 1):
            c1.append(cl[istart + (i - 2) * nmlp1 + j - 1])
    c1 += cl[len(c1):]
    return c1, mark


def encd(value: float, ash: float, ioffd: int) -> str:
    """ENCD: a level as its label: F(ns.nd) with the digits its size wants, or scaled
    by ash to an integer."""
    val = f32(value)
    if ioffd == 0:
        if val == 0:
            return "0.0"
        log = _ifix(f32(_log10r(abs(val)) + f32(0.00001)) + f32(5000.0)) - 5000
        ns = max(4, min(6, log + 2))
        nd = min(3, max(0, 2 - log))
        v = float(val)
    else:
        ns = 5 if val < 0 else 2 if val == 0 else 4
        nd = 0
        v = float(f32(val * f32(ash)))
    return fencd(v, ns, nd)


def fencd(v: float, ns: int, nd: int) -> str:
    """IRAF's fencd for an F(ns.nd) ENCODE: sprintf("%ns.ndf") of -|v|, and the minus
    sign blanked for a positive value -- so a field too narrow grows, and nothing
    is ever asterisks."""
    x = -v if v > 0 else v
    for d in range(nd, -1, -1):          # fmtio: fewer decimals until it fits,
        text = iraffmt.dtoc3(x, d, "f")  # keeping the point it had ("96.")
        if d == 0 and nd > 0:
            text += "."
        if len(text) <= ns:
            break
    text = text.rjust(ns)
    if v > 0 and "-" in text:
        text = text.replace("-", " ", 1)
    return text


def label_scale(gl: float, ha: float, gp: float) -> tuple[int, float]:
    """conrec's IOFFDT and ASH: labels scaled to integers when the levels are under
    0.1 or from 1e5 up."""
    ioffdt = 0
    for v in (gl, ha):
        if v != 0.0 and (abs(v) < 0.1 or abs(v) >= 1e5):
            ioffdt = 1
    big = max(abs(f32(gl)), abs(f32(ha)), abs(f32(gp)))
    ash = f32(iraffmt.pow_di(10.0, 3 - _ifix(f32(_log10r(big) - f32(5000.0))) - 5000))
    return ioffdt, (float(ash) if ioffdt else 1.0)


# ------------------------------------------------------------- MINMAX


def minmax(z: np.ndarray, ash: float, ioffdt: int):
    """MINMAX with nhi 0: the local highs and lows of the grid (at least MN columns
    and NM lines apart), as (x, y, 'H' or 'L', value label); grid coordinates."""
    n, m = z.shape                      # z[j-1, i-1] is Fortran's Z(I,J)
    Z = lambda i, j: z[j - 1, i - 1]    # noqa: E731
    out = []
    mn = min(15, max(2, int(m / 8.)))
    nm = min(15, max(2, int(n / 8.)))
    mm1 = m - 1

    def high(ip, jp, aa):
        js, jt = max(1, jp - nm), min(n, jp + nm)
        is_, it = max(1, ip - mn), min(m, ip + mn)
        for jk in range(js, jt + 1):
            if jk == jp:
                continue
            for ik in range(is_, it + 1):
                if Z(ik, jk) >= aa:
                    return False
        return True

    def low(ip, jp, aa):
        js, jt = max(1, jp - nm), min(n, jp + nm)
        is_, it = max(1, ip - mn), min(m, ip + mn)
        for jk in range(js, jt + 1):
            if jk == jp:
                continue
            for ik in range(is_, it + 1):
                if Z(ik, jk) <= aa:
                    return False
        return True

    for jp in range(2, n):
        im, ip = mn - 1, -1
        state = 126
        aa = an = None
        while True:
            if state == 126:
                ip += 1
                if ip >= mm1:
                    break
                im += 1
                if im <= mn:
                    continue
                im = 1
                an = Z(ip + 1, jp)
                d = Z(ip, jp) - an
                state = 101 if d < 0 else 103 if d == 0 else 113
                if state == 103:
                    im = 0
                    state = 101
                    continue
                continue
            if state == 101:                    # climbing: looking for a high
                ip += 1
                aa = an
                if ip == mm1:
                    state = 104
                else:
                    an = Z(ip + 1, jp)
                    if aa < an:
                        im += 1
                        continue
                    if aa == an:
                        im = 0
                        continue
                    state = 104
            if state == 104:
                ok = True
                if im < mn:
                    is_, it = max(1, ip - mn), ip - im - 1
                    if is_ <= it:
                        ok = all(not (aa <= Z(ii, jp)) for ii in range(is_, it + 1))
                if ok:
                    is_, it = ip + 2, min(m, ip + mn)
                    ok = all(not (aa <= Z(ii, jp)) for ii in range(is_, it + 1))
                if ok and high(ip, jp, aa):
                    out.append((float(ip), float(jp), "H", encd(float(aa), ash, ioffdt)))
                im = 1
                if ip - mm1 >= 0:
                    break
                state = 113
                continue
            if state == 113:                    # falling: looking for a low
                ip += 1
                aa = an
                if ip == mm1:
                    state = 116
                else:
                    an = Z(ip + 1, jp)
                    if aa > an:
                        im += 1
                        continue
                    if aa == an:
                        im = 0
                        continue
                    state = 116
            if state == 116:
                ok = True
                if im < mn:
                    is_, it = max(1, ip - mn), ip - im - 1
                    if is_ <= it:
                        ok = all(not (aa >= Z(ii, jp)) for ii in range(is_, it + 1))
                if ok:
                    is_, it = ip + 2, min(m, ip + mn)
                    ok = all(not (aa >= Z(ii, jp)) for ii in range(is_, it + 1))
                if ok and low(ip, jp, aa):
                    out.append((float(ip), float(jp), "L", encd(float(aa), ash, ioffdt)))
                im = 1
                if ip - mm1 >= 0:
                    break
                state = 101
                continue
    return out


def point_values(z: np.ndarray, ash: float, ioffdt: int):
    """MINMAX with nhi 1: the value at every few points, about 24 across and 48 up."""
    n, m = z.shape
    ii, jj = (m - 1 + 24) // 24, (n - 1 + 48) // 48
    return [(float(i), float(j), encd(float(z[j - 1, i - 1]), ash, ioffdt))
            for j in range(1, n + 1, jj) for i in range(1, m + 1, ii)]


# ------------------------------------------------------------- the curves


# ------------------------------------------------------------- the map


@dataclass
class Level:
    value: float
    major: bool
    label: str
    dashed: bool
    #: the polylines NCAR draws for the level, in the image's pixels: broken where a
    #: dash pattern has a gap and where the label is written
    curves: list[np.ndarray] = field(default_factory=list)


@dataclass
class ContourMap:
    """What contour draws: the levels and their curves (in the image's pixels), the
    marks, the texts and the message IRAF prints. `drawing` is NCAR's, in GKI units,
    as the metacode has it; `texts`, its texts in the image's pixels: (x, y, text,
    angle in degrees, height in the image's lines)."""
    title: str
    ncols: int
    nlines: int
    nx: int
    ny: int
    levels: list[Level]
    floor: float
    ceiling: float
    interval: float
    ash: float
    marks: list[tuple] = field(default_factory=list)
    message: str = ""
    perimeter: bool = True
    drawing: ncar.Drawing | None = None
    texts: list[tuple] = field(default_factory=list)

    @property
    def caption(self) -> str:
        """The line under the perimeter, as t_contour writes it with hold()."""
        g = iraffmt.g
        text = (f"contoured from {g(self.floor)} to {g(self.ceiling)}, "
                f"interval = {g(self.interval)}")
        if not iraffmt._equal(self.ash, 1.0):
            text += f", labels scaled by {g(self.ash)}"
        return text

    def to_pixels(self, gx, gy):
        """Grid coordinates to the image's: CONREC's SET maps 1..nx onto the same
        viewport as the perimeter's 1..ncols."""
        sx = (self.ncols - 1) / (self.nx - 1) if self.nx > 1 else 1.0
        sy = (self.nlines - 1) / (self.ny - 1) if self.ny > 1 else 1.0
        return 1 + (np.asarray(gx) - 1) * sx, 1 + (np.asarray(gy) - 1) * sy

    def from_gki(self, mx, my):
        """GKI units on the viewport to the image's pixels."""
        x1, x2, y1, y2 = self.drawing.viewport
        gx = 1 + (np.asarray(mx, float) / ncar.GKI_MAXNDC - x1) / (x2 - x1) * (self.nx - 1)
        gy = 1 + (np.asarray(my, float) / ncar.GKI_MAXNDC - y1) / (y2 - y1) * (self.ny - 1)
        return self.to_pixels(gx, gy)


def _text_height(size: int, viewport, nlines: int) -> float:
    """WTSTR's character height for NCAR's sizes 0..3 (NDC), in the image's lines."""
    cs = (8 + 4 * size + 4 * (size // 3)) / 2.0 ** ncar.YF * 25.5 / 27 * 2
    return cs / (viewport[3] - viewport[2]) * max(nlines - 1, 1)


def contour(data: np.ndarray, image: str, title_param: str = "imtitle",
            imtitle: str = "", floor=None, ceiling=None, zero: float = 0.0,
            ncontours: int = 0, interval: float = 0.0, nhi: int = -1,
            dashpat: int = 528, label: bool = True, xres: int = 64, yres: int = 64,
            subsample: bool = False, preserve: bool = True,
            perimeter: bool = True, fill: bool = False) -> ContourMap:
    """t_contour: the grid, the levels, their labels, and NCAR's drawing of them."""
    finc = f32(-abs(ncontours)) if ncontours > 0 else f32(interval if interval > 0 else 0)
    data = np.asarray(data, f32)
    if data.ndim == 1:
        data = data[None, :]
    nlines, ncols = data.shape
    title = f"{image}: {imtitle}" if title_param == "imtitle" else title_param
    z, message = getdata(data, subsample, preserve, xres, yres)
    dmin, dmax = f32(z.min()), f32(z.max())
    if iraffmt._equal(float(dmin), float(dmax)):
        raise ContourError("constant valued array, no plot drawn")
    fl = dmin if floor is None else f32(floor)
    ce = dmax if ceiling is None else f32(ceiling)
    fl, ce = f32(fl - f32(zero)), f32(ce - f32(zero))
    if abs(zero) > EPSILON:
        z = (z - f32(zero)).astype(f32)
    levels, gl, ha, gp = clgen(z, float(fl), float(ce), float(finc))
    if levels is None:
        raise ContourError(f"CONSTANT FIELD {gl:g}")
    ioffdt, ash = label_scale(gl, ha, gp)
    nml = 0
    if label:
        levels, nml = reord(levels)
    ny, nx = z.shape
    out = ContourMap(title, ncols, nlines, nx, ny, [], gl, ha, gp, ash,
                     message=message, perimeter=perimeter)
    if nhi == 0:
        grid_marks = minmax(z, ash, ioffdt)
    elif nhi > 0:
        grid_marks = point_values(z, ash, ioffdt)
    else:
        grid_marks = []
    drawn = [(value, encd(value, ash, ioffdt) if k < nml else "", k < nml)
             for k, value in enumerate(levels)]
    viewport = ncar.map_viewport(nx, ny, fill, perimeter)
    out.drawing = d = ncar.conrec(z, drawn, viewport, dashpat, nhi, grid_marks)
    bounds = [*d.starts, len(d.lines)]
    for k, (value, text, major) in enumerate(drawn):
        lev = Level(value, major, text, dashed=value < 0 and dashpat not in (0, 1, 1023))
        for line in d.lines[bounds[k]:bounds[k + 1]]:
            pts = np.asarray(line, float)
            px, py = out.from_gki(pts[:, 0], pts[:, 1])
            lev.curves.append(np.column_stack([px, py]))
        out.levels.append(lev)
    for mx, my, text, angle, size in d.texts:
        px, py = out.from_gki(mx, my)
        out.texts.append((float(px), float(py), text, float(angle),
                          _text_height(size, d.viewport, nlines)))
    if nhi == 0:
        for gx, gy, kind, text in grid_marks:
            px, py = out.to_pixels(gx, gy)
            out.marks.append((float(px), float(py), kind, text))
    elif nhi > 0:
        for gx, gy, text in grid_marks:
            px, py = out.to_pixels(gx, gy)
            out.marks.append((float(px), float(py), "", text))
    return out
