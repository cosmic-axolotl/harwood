"""NCAR's contour drawing as IRAF 2.18 runs it, point for point: the curves CONREC's
STLINE and DRLINE trace on the grid, DASHSMTH's smoothing of them (FDVDLD: a spline
under tension, KURV1S and KURV2S, through the points), its dash patterns with the
level's label written into a gap (DASHDC, CFVLD), and the way down to the metacode --
SPPS's FL2INT, PLOTIT and WTSTR, IRAF's GKS, and gio's transformation to GKI and its
clipping at the viewport (gadraw, gpl_clip*).

From sys/gio/ncarutil (conrec.f, conbd.f, dashsmth.f, dashbd.f, kurv.f, sysint/spps.f),
sys/gio/gks and sys/gio (gadraw.x, gamove.x, gplcache.x, gplflush.x, wcstogki.x, gtext.x)
of IRAF 2.18, in real as f2c compiles them: each operation of REALs rounded to single
precision, the intrinsics (SQRT, EXP, ATAN2, COS, SIN) in double.

What it gives is what IRAF writes into the metacode: the polylines, in GKI units
(0..32767), and the texts (the labels, with where and at what angle).

Translated from IRAF, Copyright (c) 1986 Association of Universities for Research in
Astronomy Inc.; the NCAR graphics routines IRAF carries are distributed under the same
notice, in LICENSES/IRAF-COPYRIGHT.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

f32 = np.float32
GKI_MAXNDC = 32767
IMPOS = -9999

# conbd: the directions DRLINE looks in, counterclockwise from -x
INX = (None, -1, -1, 0, 1, 1, 1, 0, -1)
INY = (None, 0, 1, 1, 1, 0, -1, -1, -1)
NR = 80000                     # conbd: STLINE's list of starts already traced
ISOLID, NREP, NCRT, ISIZEL = 1023, 4, 2, 0
ISIZEM, ISIZEP = 2, 0          # conbd: the H and L marks', nhi 1's values

# dashbd
IGP, MNCSTR = 9, 15
TENSN, NP, SMALL, L1, ICLOSE = f32(2.5), 150, 128, 70, 6
XF = YF = 10                   # utilbd: IU(3), IU(4); the metacode's resolution, bits
PB = 50                        # utilbd: IU(5), the points PLOTIT keeps before drawing


def _ifix(x) -> int:
    return int(math.trunc(float(x)))


def _tdiv(a: int, b: int) -> int:
    """C's long division: toward zero."""
    q = abs(a) // abs(b)
    return q if (a >= 0) == (b >= 0) else -q


# ------------------------------------------------------------- gio, to the metacode


class Gio:
    """The device end: gio's polylines (gamove, gadraw and the clipping at the
    viewport) and texts, for one WCS whose window and viewport the caller sets."""

    def __init__(self, viewport, clip: bool = True):
        self.vp = tuple(f32(v) for v in viewport)
        self.window = self.vp
        self.clip = clip
        self.lines: list[list[tuple[int, int]]] = []
        self.texts: list[tuple[int, int, str, int]] = []
        self.pl: list[int] = []
        self.cx = self.cy = None
        self.last_in = False
        self.xs = [0, 0, 0, 0]
        self.ys = [0, 0, 0, 0]
        self._cache()

    def _cache(self):
        """gpl_cache: the clipping box and world -> GKI."""
        v = self.vp
        self.mx1 = _ifix(f32(v[0] * f32(GKI_MAXNDC)))
        self.mx2 = _ifix(f32(v[1] * f32(GKI_MAXNDC)))
        self.my1 = _ifix(f32(v[2] * f32(GKI_MAXNDC)))
        self.my2 = _ifix(f32(v[3] * f32(GKI_MAXNDC)))
        w = self.window
        self.wxo, self.wyo = w[0], w[2]
        self.xscale = f32(f32(self.mx2 - self.mx1) / f32(w[1] - w[0]))
        self.yscale = f32(f32(self.my2 - self.my1) / f32(w[3] - w[2]))
        self.xo, self.yo = self.mx1, self.my1
        if not self.clip:
            # G_CLIP no: the clipping box is the device's, the transformation stays
            self.mx1, self.mx2, self.my1, self.my2 = 0, GKI_MAXNDC, 0, GKI_MAXNDC

    def set_window(self, window):
        """gswind: flush, the new window, the cache invalid (gpl_reset)."""
        self.flush()
        self.window = tuple(f32(v) for v in window)
        self.last_in = False
        self._cache()

    def to_gki(self, wx, wy):
        x = f32(f32(f32(f32(wx) - self.wxo) * self.xscale) + f32(self.xo))
        y = f32(f32(f32(f32(wy) - self.wyo) * self.yscale) + f32(self.yo))
        return (max(f32(0), min(f32(GKI_MAXNDC), x)), max(f32(0), min(f32(GKI_MAXNDC), y)))

    def flush(self):
        if len(self.pl) > 1:
            if len(self.pl) > 3:
                self.lines.append([(self.pl[i], self.pl[i + 1])
                                   for i in range(0, len(self.pl), 2)])
            self.pl = []

    def move(self, wx, wy):
        if self.pl:
            self.flush()
        self.cx, self.cy = self.to_gki(wx, wy)

    def draw(self, wx, wy):
        x, y = self.to_gki(wx, wy)
        if not self.pl:
            mx, my = _ifix(self.cx), _ifix(self.cy)
            if self._inside(mx, my):
                self.last_in = True
                self.pl += [mx, my]
            else:
                self.last_in = False
                self.xs = [mx] * 4
                self.ys = [my] * 4
        self.cx, self.cy = x, y
        mx, my = _ifix(x), _ifix(y)
        inside = self._inside(mx, my)
        if inside and self.last_in:
            self.pl += [mx, my]
        else:
            if self.last_in:
                self.xs = [self.pl[-2]] * 4
                self.ys = [self.pl[-1]] * 4
            self._clipl(1, mx, my)
        self.last_in = inside

    def _inside(self, mx, my):
        return self.my1 <= my <= self.my2 and self.mx1 <= mx <= self.mx2

    # gpl_clipl, _clipr, _clipb, _clipt: Sutherland and Hodgman, edge by edge
    def _clipl(self, pen, mx, my):
        xs, ys, m = self.xs, self.ys, self.mx1
        if (mx >= m and xs[0] < m) or (mx <= m and xs[0] > m):
            self._clipr(0 if mx >= m else pen, m,
                        _tdiv((my - ys[0]) * (m - mx), mx - xs[0]) + my)
        xs[0], ys[0] = mx, my
        if mx >= m:
            self._clipr(pen, mx, my)

    def _clipr(self, pen, mx, my):
        xs, ys, m = self.xs, self.ys, self.mx2
        if (mx <= m and xs[1] > m) or (mx >= m and xs[1] < m):
            self._clipb(0 if mx <= m else pen, m,
                        _tdiv((my - ys[1]) * (m - mx), mx - xs[1]) + my)
        xs[1], ys[1] = mx, my
        if mx <= m:
            self._clipb(pen, mx, my)

    def _clipb(self, pen, mx, my):
        xs, ys, m = self.xs, self.ys, self.my1
        if (my >= m and ys[2] < m) or (my <= m and ys[2] > m):
            self._clipt(0 if my >= m else pen,
                        _tdiv((mx - xs[2]) * (m - my), my - ys[2]) + mx, m)
        xs[2], ys[2] = mx, my
        if my >= m:
            self._clipt(pen, mx, my)

    def _clipt(self, pen, mx, my):
        xs, ys, m = self.xs, self.ys, self.my2
        if (my <= m and ys[3] > m) or (my >= m and ys[3] < m):
            if my <= m or pen == 0:
                self.flush()
            self.pl += [_tdiv((mx - xs[3]) * (m - my), my - ys[3]) + mx, m]
        xs[3], ys[3] = mx, my
        if my <= m:
            if pen == 0:
                self.flush()
            self.pl += [mx, my]

    def polyline(self, xs, ys):
        """GKS's GPL: gpline, a move and draws."""
        self.move(xs[0], ys[0])
        for x, y in zip(xs[1:], ys[1:], strict=True):
            self.draw(x, y)

    def text(self, wx, wy, chars: str, angle: int = 0, size: int = 0):
        """gtext: the polyline flushed, the position rounded."""
        self.flush()
        mx, my = self.to_gki(wx, wy)
        self.texts.append((math.floor(mx + 0.5), math.floor(my + 0.5), chars, angle, size))


# ------------------------------------------------------------- SPPS


class Spps:
    """NCAR's system plot package over IRAF's GKS: the user's window (SET) on a
    viewport, FL2INT to the metacode's integers, PLOTIT's buffer of points."""

    def __init__(self, viewport, window, gio: Gio | None = None):
        self.vp = tuple(f32(v) for v in viewport)
        self.wd = tuple(f32(v) for v in window)
        self.gio = gio if gio is not None else Gio(self.vp)
        self.gio.set_window(self.wd)
        self.nq, self.nf = 1, 1
        self.qx = [f32(0)] * (PB + 2)
        self.qy = [f32(0)] * (PB + 2)
        self.iff = [0, 1] + [0] * 60

    def fl2int(self, px, py) -> tuple[int, int]:
        wd, vp = self.wd, self.vp
        ix = _ifix(f32(f32(f32(f32(f32(f32(px) - wd[0]) / f32(wd[1] - wd[0]))
                                   * f32(vp[1] - vp[0])) + vp[0]) * f32(32767)))
        iy = _ifix(f32(f32(f32(f32(f32(f32(py) - wd[2]) / f32(wd[3] - wd[2]))
                                   * f32(vp[3] - vp[2])) + vp[2]) * f32(32767)))
        return ix, iy

    def cufx(self, rx):
        wd, vp = self.wd, self.vp
        return f32(f32(f32(f32(f32(rx) - wd[0]) / f32(wd[1] - wd[0])) * f32(vp[1] - vp[0]))
                   + vp[0])

    def cufy(self, ry):
        wd, vp = self.wd, self.vp
        return f32(f32(f32(f32(f32(ry) - wd[2]) / f32(wd[3] - wd[2])) * f32(vp[3] - vp[2]))
                   + vp[2])

    def cpux(self, ix):
        wd, vp = self.wd, self.vp
        v = float(wd[0]) + ((ix - 1) / (2.0 ** XF - 1.0) - float(vp[0])) / float(
            f32(vp[1] - vp[0])) * float(f32(wd[1] - wd[0]))
        return f32(v)

    def cpuy(self, iy):
        wd, vp = self.wd, self.vp
        v = float(wd[2]) + ((iy - 1) / (2.0 ** YF - 1.0) - float(vp[2])) / float(
            f32(vp[3] - vp[2])) * float(f32(wd[3] - wd[2]))
        return f32(v)

    def _point(self, gx, gy, ip):
        if ip != 0 or self.iff[self.nf] != self.nq:
            self.nq += 1
            if ip == 0:
                self.nf += 1
                self.iff[self.nf] = self.nq
        self.qx[self.nq] = gx
        self.qy[self.nq] = gy

    def plotit(self, ix, iy, ip):
        if ip != 2:
            jx = max(0, min(32767, ix))
            jy = max(0, min(32767, iy))
            self._point(f32(f32(jx) / f32(32767)), f32(f32(jy) / f32(32767)), ip)
            if not (ix == 0 and iy == 0 and ip == 0) and self.nq < PB:
                return
        self._flush()

    def plotif(self, fx, fy, ip):
        if ip != 2:
            gx = max(f32(0), min(f32(1), f32(fx)))
            gy = max(f32(0), min(f32(1), f32(fy)))
            self._point(gx, gy, ip)
            if self.nq < PB:
                return
        self._flush()

    def _flush(self):
        if self.nq <= 1:
            return
        g, iff, qx, qy = self.gio, self.iff, self.qx, self.qy
        g.set_window(self.vp)
        i = 1
        for i in range(1, self.nf):
            a, b = iff[i], iff[i + 1]
            g.polyline(qx[a:b], qy[a:b])
        i = self.nf
        if iff[self.nf] != self.nq:
            a = iff[i]
            g.polyline(qx[a:self.nq + 1], qy[a:self.nq + 1])
        g.set_window(self.wd)
        qx[1], qy[1] = qx[self.nq], qy[self.nq]
        self.nq, self.nf = 1, 1
        iff[1] = 1

    def frstpt(self, px, py):
        self.plotif(self.cufx(px), self.cufy(py), 0)

    def wtstr(self, px, py, chars: str, angle: int, size: int = 0):
        """WTSTR: the buffer drawn, the text at (px, py) in the window's units, the
        pen left there."""
        self.plotif(0, 0, 2)
        xn, yn = self.cufx(px), self.cufy(py)
        self.gio.set_window(self.vp)
        self.gio.text(xn, yn, chars, angle, size)
        self.gio.set_window(self.wd)
        self.frstpt(px, py)


# ------------------------------------------------------------- the spline: kurv.f

PI_K = f32(3.1415926535897932)
DEGRAD = f32(PI_K / f32(180))


def _sqrt(v) -> np.float32:
    return f32(math.sqrt(float(v)))


def _exp(v) -> np.float32:
    return f32(math.exp(float(v)))


def kurv1s(n, x, y, slp1, slpn, islpsw):
    """KURV1S: the tension spline's second derivatives through x(1..n), y(1..n) (lists
    indexed from 1); with islpsw the end slopes estimated from the points."""
    x, y = list(x), list(y)
    xp = [f32(0)] * (n + 1)
    yp = [f32(0)] * (n + 1)
    temp = [f32(0)] * (n + 1)
    slp1, slpn = f32(slp1), f32(slpn)
    nm1 = n - 1
    delx1 = f32(x[2] - x[1])
    dely1 = f32(y[2] - y[1])
    dels1 = _sqrt(f32(f32(delx1 * delx1) + f32(dely1 * dely1)))
    dx1 = f32(delx1 / dels1)
    dy1 = f32(dely1 / dels1)
    if islpsw != 0:
        if n == 2:
            if islpsw == 3:
                return xp, yp, None, f32(math.atan2(float(f32(y[2] - y[1])),
                                                    float(f32(x[2] - x[1]))) / float(DEGRAD))
            dx, dy = f32(x[2] - x[1]), f32(y[2] - y[1])
            if islpsw == 2:
                slp1 = f32(math.atan2(float(f32(dy - f32(slpn * dx))),
                                      float(f32(dx - f32(slpn * dy)))) / float(DEGRAD))
            if islpsw == 1:
                slpn = f32(math.atan2(float(f32(dy - f32(slp1 * dx))),
                                      float(f32(dx - f32(slp1 * dy)))) / float(DEGRAD))
        elif islpsw == 4:
            r1, r2 = f32(x[nm1] - x[n]), f32(y[nm1] - y[n])
            deln = _sqrt(f32(f32(r1 * r1) + f32(r2 * r2)))
            delnn = f32(dels1 + deln)
            c1 = f32(f32(-dels1 / deln) / delnn)
            c2 = f32(f32(f32(dels1 - deln) / dels1) / deln)
            c3 = f32(f32(deln / delnn) / dels1)
            sx = f32(f32(f32(c1 * x[nm1]) + f32(c2 * x[1])) + f32(c3 * x[2]))
            sy = f32(f32(f32(c1 * y[nm1]) + f32(c2 * y[1])) + f32(c3 * y[2]))
            if sx == 0 and sy == 0:
                sx = f32(1)
            slp1 = f32(math.atan2(float(sy), float(sx)) / float(DEGRAD))
            slpn = slp1
        else:
            if islpsw != 2:
                r1, r2 = f32(x[n - 2] - x[nm1]), f32(y[n - 2] - y[nm1])
                delnm1 = _sqrt(f32(f32(r1 * r1) + f32(r2 * r2)))
                r1, r2 = f32(x[nm1] - x[n]), f32(y[nm1] - y[n])
                deln = _sqrt(f32(f32(r1 * r1) + f32(r2 * r2)))
                delnn = f32(delnm1 + deln)
                c1 = f32(f32(f32(delnn + deln) / delnn) / deln)
                c2 = f32(f32(-delnn / deln) / delnm1)
                c3 = f32(f32(deln / delnn) / delnm1)
                sx = f32(f32(f32(c3 * x[n - 2]) + f32(c2 * x[nm1])) + f32(c1 * x[n]))
                sy = f32(f32(f32(c3 * y[n - 2]) + f32(c2 * y[nm1])) + f32(c1 * y[n]))
                slpn = f32(math.atan2(float(sy), float(sx)) / float(DEGRAD))
            if islpsw != 1:
                r1, r2 = f32(x[3] - x[2]), f32(y[3] - y[2])
                dels2 = _sqrt(f32(f32(r1 * r1) + f32(r2 * r2)))
                dels12 = f32(dels1 + dels2)
                c1 = f32(f32(-f32(dels12 + dels1) / dels12) / dels1)
                c2 = f32(f32(dels12 / dels1) / dels2)
                c3 = f32(f32(-dels1 / dels12) / dels2)
                sx = f32(f32(f32(c1 * x[1]) + f32(c2 * x[2])) + f32(c3 * x[3]))
                sy = f32(f32(f32(c1 * y[1]) + f32(c2 * y[2])) + f32(c3 * y[3]))
                slp1 = f32(math.atan2(float(sy), float(sx)) / float(DEGRAD))
    # label 10
    slpp1 = f32(slp1 * DEGRAD)
    slppn = f32(slpn * DEGRAD)
    xp[1] = f32(float(dx1) - math.cos(float(slpp1)))
    yp[1] = f32(float(dy1) - math.sin(float(slpp1)))
    temp[1] = dels1
    ss = dels1
    for i in range(2, nm1 + 1):
        delx2 = f32(x[i + 1] - x[i])
        dely2 = f32(y[i + 1] - y[i])
        dels2 = _sqrt(f32(f32(delx2 * delx2) + f32(dely2 * dely2)))
        dx2 = f32(delx2 / dels2)
        dy2 = f32(dely2 / dels2)
        xp[i] = f32(dx2 - dx1)
        yp[i] = f32(dy2 - dy1)
        temp[i] = dels2
        dels1 = dels2
        dx1, dy1 = dx2, dy2
        ss = f32(ss + dels1)
    xp[n] = f32(math.cos(float(slppn)) - float(dx1))
    yp[n] = f32(math.sin(float(slppn)) - float(dy1))
    sigmap = f32(f32(abs(TENSN) * f32(n - 1)) / ss)
    s = ss
    dels = f32(sigmap * temp[1])
    exps = _exp(dels)
    sinhs = f32(f32(0.5) * f32(exps - f32(f32(1) / exps)))
    sinhin = f32(f32(1) / f32(temp[1] * sinhs))
    diag1 = f32(sinhin * f32(f32(f32(dels * f32(0.5)) * f32(exps + f32(f32(1) / exps)))
                             - sinhs))
    diagin = f32(f32(1) / diag1)
    xp[1] = f32(diagin * xp[1])
    yp[1] = f32(diagin * yp[1])
    spdiag = f32(sinhin * f32(sinhs - dels))
    temp[1] = f32(diagin * spdiag)
    for i in range(2, nm1 + 1):
        dels = f32(sigmap * temp[i])
        exps = _exp(dels)
        sinhs = f32(f32(0.5) * f32(exps - f32(f32(1) / exps)))
        sinhin = f32(f32(1) / f32(temp[i] * sinhs))
        diag2 = f32(sinhin * f32(f32(dels * f32(f32(0.5) * f32(exps + f32(f32(1) / exps))))
                                 - sinhs))
        diagin = f32(f32(1) / f32(f32(diag1 + diag2) - f32(spdiag * temp[i - 1])))
        xp[i] = f32(diagin * f32(xp[i] - f32(spdiag * xp[i - 1])))
        yp[i] = f32(diagin * f32(yp[i] - f32(spdiag * yp[i - 1])))
        spdiag = f32(sinhin * f32(sinhs - dels))
        temp[i] = f32(diagin * spdiag)
        diag1 = diag2
    diagin = f32(f32(1) / f32(diag1 - f32(spdiag * temp[nm1])))
    xp[n] = f32(diagin * f32(xp[n] - f32(spdiag * xp[nm1])))
    yp[n] = f32(diagin * f32(yp[n] - f32(spdiag * yp[nm1])))
    for i in range(2, n + 1):
        ibak = n + 1 - i
        xp[ibak] = f32(xp[ibak] - f32(temp[ibak] * xp[ibak + 1]))
        yp[ibak] = f32(yp[ibak] - f32(temp[ibak] * yp[ibak + 1]))
    return xp, yp, s, None


class Kurv2s:
    """KURV2S: points along the spline, by arc length; its SAVEd place along the
    curve carries from call to call (t = 0 starts it again)."""

    def __init__(self):
        self.i1 = 2
        self.sum = f32(0)
        self.slp = f32(0)

    def __call__(self, t, n, x, y, xp, yp, s, slope: bool):
        t = f32(t)
        sigmap = f32(f32(abs(TENSN) * f32(n - 1)) / s)
        tn = f32(abs(f32(t * s)))
        if t >= 0:
            self.i1 = 2
            xs, ys = x[1], y[1]
            self.sum = f32(0)
        for i in range(self.i1, n + 1):
            delx = f32(x[i] - x[i - 1])
            dely = f32(y[i] - y[i - 1])
            dels = _sqrt(f32(f32(delx * delx) + f32(dely * dely)))
            if f32(f32(self.sum + dels) - tn) >= 0:
                break
            self.sum = f32(self.sum + dels)
        else:
            return x[n], y[n]
        del1 = f32(tn - self.sum)
        del2 = f32(dels - del1)
        exps1 = _exp(f32(sigmap * del1))
        sinhd1 = f32(f32(0.5) * f32(exps1 - f32(f32(1) / exps1)))
        exps2 = _exp(f32(sigmap * del2))
        sinhd2 = f32(f32(0.5) * f32(exps2 - f32(f32(1) / exps2)))
        exps = f32(exps1 * exps2)
        sinhs = f32(f32(0.5) * f32(exps - f32(f32(1) / exps)))
        xs = f32(f32(f32(f32(xp[i] * sinhd1) + f32(xp[i - 1] * sinhd2)) / sinhs)
                 + f32(f32(f32(f32(x[i] - xp[i]) * del1)
                           + f32(f32(x[i - 1] - xp[i - 1]) * del2)) / dels))
        ys = f32(f32(f32(f32(yp[i] * sinhd1) + f32(yp[i - 1] * sinhd2)) / sinhs)
                 + f32(f32(f32(f32(y[i] - yp[i]) * del1)
                           + f32(f32(y[i - 1] - yp[i - 1]) * del2)) / dels))
        self.i1 = i
        if slope:
            coshd1 = f32(f32(f32(0.5) * f32(exps1 + f32(f32(1) / exps1))) * sigmap)
            coshd2 = f32(f32(f32(0.5) * f32(exps2 + f32(f32(1) / exps2))) * sigmap)
            xt = f32(f32(f32(f32(xp[i] * coshd1) - f32(xp[i - 1] * coshd2)) / sinhs)
                     + f32(f32(f32(x[i] - xp[i]) - f32(x[i - 1] - xp[i - 1])) / dels))
            yt = f32(f32(f32(f32(yp[i] * coshd1) - f32(yp[i - 1] * coshd2)) / sinhs)
                     + f32(f32(f32(y[i] - yp[i]) - f32(y[i - 1] - yp[i - 1])) / dels))
            self.slp = f32(math.atan2(float(yt), float(xt)) / float(DEGRAD))
        return xs, ys


# ------------------------------------------------------------- DASHSMTH


class Dash:
    """DASHSMTH: FRSTD, VECTD and LASTD -- the line smoothed (FDVDLD) and drawn in its
    dash pattern, the label in its gap (CFVLD) -- onto an Spps."""

    PI_F = f32(3.14159265358)

    def __init__(self, spps: Spps, fpart: float = 1.0):
        self.p = spps
        self.fpart = f32(fpart)
        # DASHD1, DASHD2
        self.isl = 1
        self.l = 0
        self.isize = 0
        self.ip = [0] * 101
        self.ipflag = [0] * 101
        self.ipc = [[" "] * MNCSTR for _ in range(101)]
        self.lgbstr = 0
        # flags
        self.ifstfl = self.ifstf2 = self.iflag = 1
        self.ivctfg = 1
        self.ixstor = self.iystor = IMPOS
        # FDVDLD
        self.xsave = [f32(0)] * (L1 + 1)
        self.ysave = [f32(0)] * (L1 + 1)
        self.xsvn = self.ysvn = self.xsv1 = self.ysv1 = f32(0)
        self.slp1 = self.slpn = f32(-9999)
        self.sslp1 = self.sslpn = f32(0)
        self.n = self.nseg = 0
        self.npl = 0
        self.s = f32(0)
        self.deg = f32(0)
        self.kurv2s = Kurv2s()
        # CFVLD
        self.x = self.y = self.x1 = self.y1 = self.x2 = self.y2 = self.x3 = self.y3 = f32(0)
        self.dx = self.dy = self.d = self.bpbx = self.bpby = f32(0)
        self.m = 1
        self.bti = f32(0)
        self.ib = 0
        self.ix = self.iy = 0

    # ---- DASHDC
    def dashdc(self, ipat: str, jcrt: int = NCRT, jsize: int = ISIZEL):
        nchrts = 0
        self.l = 0
        self.isl = 0
        self.ifstfl = self.ifstf2 = 1
        iadjus = 1 << (15 - XF)
        icrt = jcrt * iadjus
        self.isize = jsize
        charw = f32(jsize * iadjus)
        if jsize <= 3:
            charw = f32(f32(256) + f32(f32(jsize) * f32(128)))
            if jsize == 3:
                charw = f32(768)
        if icrt < 1:
            self.isl = 1
            return
        ip, flag, ipc = self.ip, self.ipflag, self.ipc
        mode, ngo, nmode = 2, 0, 0
        for ch in ipat:
            if ch == " ":
                continue
            nmode = -1 if ch == "'" else 1 if ch == "$" else 0
            if self.l == 0 and nmode == -1:
                mode = 8
            ngo = nmode + mode
            mode = nmode * 3 + 5
            if ngo == 5:                                        # 60
                if nchrts == MNCSTR:
                    continue
                nchrts += 1
                ip[self.l] = nchrts
                ipc[self.l][nchrts - 1] = ch
                continue
            if ngo in (2, 8):                                   # 80
                self.lgbstr = min(self.l + 1, 100)
                self.l = min(self.lgbstr + 1, 100)
                flag[self.l] = 0
                nchrts = 1
                ip[self.l] = 1
                ipc[self.l][0] = ch
                continue
            if ngo in (4, 6):                                   # 90
                ip[self.lgbstr] = _ifix(f32(charw * f32(f32(nchrts) + f32(0.5))))
                flag[self.lgbstr] = -1 if IGP != 0 else 1
            if ngo in (4, 6, 3, 7):                             # 140
                self.l = min(self.l + 1, 100)
                ip[self.l] = 0
            ip[self.l] += icrt                                  # 150
            flag[self.l] = nmode
        if ngo in (2, 5, 8):
            ip[self.lgbstr] = _ifix(f32(charw * f32(f32(nchrts) + f32(0.5))))
            flag[self.lgbstr] = -1 if IGP != 0 else 1
        if self.l > 1:
            return
        ibig = 1 << max(XF, YF)
        if ip[self.l] >= ibig or flag[self.l] > 0:
            self.isl = 1
        else:
            self.isl = -1

    # ---- the entries
    def frstd(self, x, y):
        ix, iy = self.p.fl2int(x, y)
        self.fdvdld(1, ix, iy)

    def vectd(self, x, y):
        ix, iy = self.p.fl2int(x, y)
        self.fdvdld(2, ix, iy)

    def lastd(self):
        self.fdvdld(3, 0, 0)
        self.p.plotit(0, 0, 0)

    # ---- DRAWPV
    def drawpv(self, ix, iy, ind):
        if ind == 1:
            self.ixstor, self.iystor = ix, iy
            self.p.plotit(ix, iy, 1)
        elif ind == 0:
            if abs(self.ixstor - ix) + abs(self.iystor - iy) <= ICLOSE:
                return
            self.ixstor, self.iystor = ix, iy
            self.p.plotit(ix, iy, 0)
        else:
            self.ixstor, self.iystor = ix, iy

    # ---- FDVDLD
    def fdvdld(self, entry, iix, iiy):
        if entry == 1 or (entry == 2 and self.ifstf2 != 0):
            self.deg = f32(f32(180) / self.PI_F)
            self.ifstf2 = 0
            self.sslp1 = self.sslpn = f32(0)
            self.xsvn = self.ysvn = f32(0)
            self.n = self.nseg = 0
            self.xsv1, self.ysv1 = f32(iix), f32(iiy)
            self._save(iix, iiy)
            return
        if entry == 2:
            if abs(_ifix(self.xsvn) - iix) + abs(_ifix(self.ysvn) - iiy) < SMALL:
                return
            self.iflag = 0
            self._save(iix, iiy)
            return
        if self.ifstf2 != 0:
            return
        self.ifstf2 = 1
        if self.iflag == 1:
            return
        self._smooth()

    def _save(self, mx, my):
        self.n += 1
        self.xsave[self.n] = f32(mx)
        self.ysave[self.n] = f32(my)
        self.xsvn, self.ysvn = self.xsave[self.n], self.ysave[self.n]
        if self.n >= L1 - 1:
            self._smooth()

    def _smooth(self):                                          # label 40
        self.iflag = 1
        n, xsave, ysave = self.n, self.xsave, self.ysave
        iprd = 1
        if abs(f32(self.xsv1 - self.xsvn)) + abs(f32(self.ysv1 - self.ysvn)) < SMALL:
            iprd = 0
        islpsw = None
        if self.nseg < 1:
            if n < 2:
                return
            if n == 2:                                          # 140
                self.cfvld(1, _ifix(xsave[1]), _ifix(ysave[1]))
                self.cfvld(2, _ifix(xsave[n]), _ifix(ysave[n]))
                if self.ifstf2 == 1:
                    self.cfvld(3, 0, 0)
                return
            if n < 4 and iprd == 0:                             # 50
                dx = f32(xsave[2] - xsave[1])
                dy = f32(ysave[2] - ysave[1])
                slope = f32(math.atan2(float(dy), float(dx)) * float(self.deg) + 90.0)
                if slope >= 360:
                    slope = f32(slope - f32(360))
                if slope <= 0:
                    slope = f32(slope + f32(360))
                self.slp1 = self.slpn = slope
                islpsw = 0
        if islpsw is None:                                      # 60
            if iprd < 1:
                if self.nseg < 1:
                    islpsw = 4
                    xsave[n], ysave[n] = self.xsv1, self.ysv1
                else:                                           # 70
                    self.slp1, self.slpn = self.sslpn, self.sslp1
                    islpsw = 0
            elif self.nseg < 1:                                 # 80
                islpsw = 3
            else:                                               # 90
                self.slp1 = self.sslpn
                islpsw = 1
        xp, yp, s, _ = kurv1s(n, xsave, ysave, self.slp1, self.slpn, islpsw)
        if s is not None:
            self.s = s
        s = self.s
        if not (self.nseg >= 1 and n < L1 - 1):
            nprime = _ifix(f32(f32(NP) - f32(f32(f32(s * f32(NP)) * f32(0.5)) / f32(32767))))
            if s >= 32767:
                nprime = _ifix(f32(f32(0.5) * f32(NP)))
            self.npl = _ifix(max(f32(f32(f32(nprime) * s) / f32(32767)), f32(2.5)))
        dt = f32(f32(1) / f32(self.npl))
        ix, iy = _ifix(xsave[1]), _ifix(ysave[1])
        if self.nseg > 0:
            self.drawpv(ix, iy, 0)
        else:
            self.cfvld(1, ix, iy)
        t = f32(0)
        first = self.nseg < 1
        self.nseg = 1
        xs, ys = self.kurv2s(t, n, xsave, ysave, xp, yp, s, first)
        if first:
            self.sslp1 = self.kurv2s.slp
        for i in range(1, self.npl + 1):
            t = f32(t + dt)
            last = i == self.npl
            xs, ys = self.kurv2s(-t, n, xsave, ysave, xp, yp, s, last)
            if last:
                self.sslpn = self.kurv2s.slp
            self.cfvld(2, _ifix(xs), _ifix(ys))
        if iprd == 0:
            self.cfvld(2, _ifix(self.xsv1), _ifix(self.ysv1))
        xsave[1], ysave[1] = f32(xs), f32(ys)
        self.n = 1
        if self.ifstf2 == 1:
            self.cfvld(3, 0, 0)

    # ---- CFVLD
    def cfvld(self, entry, iix, iiy):
        if entry == 1:                                          # 330
            self.ix, self.iy = iix, iiy
            self.ifstfl = 2
            self.drawpv(IMPOS, IMPOS, 2)
            if self.isl < 0:
                return
            if self.isl > 0:
                self.drawpv(self.ix, self.iy, 0)
                return
            self.x = self.x2 = self.x3 = f32(self.ix)        # 30
            self.y = self.y2 = self.y3 = f32(self.iy)
            self.m = 1
            self.ib = self.ipflag[1]
            if self.ipflag[1] == 0:
                self.ib = 0
            self.bti = f32(f32(self.ip[1]) * self.fpart)
            return
        if entry == 2:                                          # 305
            if self.ifstfl != 2:
                self.cfvld(1, iix, iiy)
                return
            self.ivctfg = 2
            if self.isl < 0:
                return
            if self.isl > 0:
                self.ix, self.iy = iix, iiy
                self.drawpv(self.ix, self.iy, 1)
                return
            self._vector(iix, iiy)
            return
        if self.ifstfl != 2:                                    # 350
            return
        self.ifstfl = 1
        if self.isl != 0:
            return
        self._end()

    def _vector(self, iix, iiy):                                # 50
        self.x1, self.y1 = self.x2, self.y2
        self.x2, self.y2 = f32(iix), f32(iiy)
        self.dx = f32(self.x2 - self.x1)
        self.dy = f32(self.y2 - self.y1)
        self.d = _sqrt(f32(f32(self.dx * self.dx) + f32(self.dy * self.dy)))
        if self.d < f32(1.5):                                   # 190
            self.x2, self.y2 = self.x1, self.y1
        else:
            self._dashes()
        self._done()

    def _done(self):                                            # 200
        if self.ivctfg != 2:
            self._end()
            return
        self.ivctfg = 1

    def _dashes(self):
        """Labels 60-190: along the vector from (x1, y1) to (x2, y2), the pattern's
        elements one after another."""
        ip, flag = self.ip, self.ipflag
        self.bpbx = f32(self.dx / self.d)                       # 60
        self.bpby = f32(self.dy / self.d)
        self.drawpv(self.ix, self.iy, 0)
        while True:
            self.bti = f32(self.bti - self.d)                   # 70
            if self.bti > 0:                                    # 80
                self.x, self.y = self.x2, self.y2
                self.ix, self.iy = _ifix(self.x2), _ifix(self.y2)
                if self.ib < 0:
                    return
                if self.ib > 0:
                    self.drawpv(self.ix, self.iy, 1)
                    return
                self._label()
            else:
                self.bti = f32(self.bti + self.d)               # 100
                self.d = f32(self.d - self.bti)
                self.x = f32(self.x + f32(self.bpbx * self.bti))
                self.y = f32(self.y + f32(self.bpby * self.bti))
                self.ix = _ifix(f32(self.x + f32(0.5)))
                self.iy = _ifix(f32(self.y + f32(0.5)))
                if self.ib < 0:
                    self.drawpv(self.ix, self.iy, 0)
                elif self.ib > 0:
                    self.drawpv(self.ix, self.iy, 1)
                else:
                    self._label()
            # 130: the next element; a label is written at once
            while True:
                self.m = self.m % self.l + 1
                self.ib = flag[self.m]
                if self.ib < 0:                                 # 140
                    self.x3, self.y3 = self.x, self.y
                    self.bti = f32(ip[self.m])
                    break
                if self.ib > 0:                                 # 150
                    self.x3 = f32(-1)
                    self.bti = f32(ip[self.m])
                    break
                self._label()

    def _label(self):                                           # 160-184
        s = f32(0)
        if IGP == 9:
            dx = f32(self.x - self.x3)
            dy = f32(self.y - self.y3)
        else:
            dx = f32(self.x - self.x1)
            dy = f32(self.y - self.y1)
        if dy != 0:
            s = f32(math.atan2(float(dy), float(dx)))
            if abs(f32(s - f32(0.00005))) > f32(1.5708):
                s = f32(float(s) - math.copysign(float(f32(3.14159)), float(s)))
        if IGP == 9:
            mx = _ifix(f32(self.x3 + f32(dx * f32(0.5))))
            my = _ifix(f32(self.y3 + f32(dy * f32(0.5))))
        else:
            mx, my = _ifix(self.x), _ifix(self.y)
        angle = _ifix(f32(f32(f32(s * f32(180)) / f32(3.14)) + f32(0.5)))
        if angle < 0:
            angle += 360
        mx >>= 15 - XF
        my >>= 15 - YF
        chars = "".join(self.ipc[self.m][:self.ip[self.m]])
        self.p.wtstr(self.p.cpux(mx), self.p.cpuy(my), chars, angle, self.isize)
        self.drawpv(IMPOS, IMPOS, 2)
        self.drawpv(self.ix, self.iy, 0)

    def _end(self):                                             # 210
        ip, flag, ll = self.ip, self.ipflag, self.l
        if self.ib >= 0:
            return
        mo = self.m
        self.m = self.m % ll + 1
        if flag[self.m] != 0:
            return
        mprev = self.m - 2
        if mprev <= 0:
            mprev += ll
        self.ib = flag[mprev]
        if self.ib < 0:
            while True:                                         # 230, 240
                self.m = self.m % ll + 1
                if self.m == mo:
                    return
                self.ib = flag[self.m]
                if self.ib != 0:
                    self.bti = f32(ip[self.m])
                if self.ib > 0:
                    break
            self._resume()
            return
        while True:                                             # 250, 260
            self.m = self.m % ll + 1
            if self.m == mo:                                    # 290
                self.ix = _ifix(f32(self.x3 + f32(0.5)))
                self.iy = _ifix(f32(self.y3 + f32(0.5)))
                self.drawpv(self.ix, self.iy, 0)
                self.ix, self.iy = _ifix(self.x2), _ifix(self.y2)
                self.drawpv(self.ix, self.iy, 1)
                return
            self.ib = flag[self.m]
            if self.ib != 0:
                self.bti = f32(ip[self.m])
            if self.ib < 0:                                     # 270
                mt = self.m
                self.m = self.m % ll + 1
                if flag[self.m] == 0:
                    continue
                self.m = mt
                self._resume()
                return

    def _resume(self):                                          # 280
        self.x1 = self.x = self.x3
        self.y1 = self.y = self.y3
        self.ix = _ifix(f32(self.x + f32(0.5)))
        self.iy = _ifix(f32(self.y + f32(0.5)))
        self.dx = f32(self.x2 - self.x1)
        self.dy = f32(self.y2 - self.y1)
        self.d = _sqrt(f32(f32(self.dx * self.dx) + f32(self.dy * self.dy)))
        if self.d >= f32(1.5):
            self._dashes()
            self._done()


# ------------------------------------------------------------- CONREC's tracing


def _c(p1, p2, cv):
    return f32(f32(p1 - cv) / f32(p1 - p2))


def stline(z: np.ndarray, cv, dash: Dash) -> None:
    """STLINE: the curves of level cv on z (lines x columns, z[j-1, i-1] = Z(I,J)):
    first those that start at the grid's edge, then the closed ones inside."""
    zz = np.asarray(z, f32)
    n, m = zz.shape
    cv = f32(cv)
    st = {"np": 0, "ir": set()}

    def Z(i, j):
        return zz[j - 1, i - 1]

    def drline(ix, iy, idx, idy, is_, iss):
        ix0, iy0, is0 = ix, iy, is_
        if idx != 0:
            y = f32(iy)
            x = f32(f32(_c(Z(ix, iy), Z(ix + idx, iy), cv) * f32(idx)) + f32(ix))
        else:
            x = f32(ix)
            y = f32(f32(_c(Z(ix, iy), Z(ix, iy + idy), cv) * f32(idy)) + f32(iy))
        dash.frstd(x, y)
        while True:
            is_ += 1
            if is_ > 8:
                is_ -= 8
            idx, idy = INX[is_], INY[is_]
            ix2, iy2 = ix + idx, iy + idy
            if iss == 0 and (ix2 > m or iy2 > n or ix2 < 1 or iy2 < 1):
                break
            if cv <= Z(ix2, iy2):
                is_ += 4
                ix, iy = ix2, iy2
                continue
            if is_ % 2 == 0:
                continue
            if idx != 0:
                y = f32(iy)
                x = f32(f32(_c(Z(ix, iy), Z(ix + idx, iy), cv) * f32(idx)) + f32(ix))
            else:
                x = f32(ix)
                y = f32(f32(_c(Z(ix, iy), Z(ix, iy + idy), cv) * f32(idy)) + f32(iy))
            dash.vectd(x, y)
            if is_ == 1:
                st["np"] += 1
                if st["np"] > NR:
                    break
                st["ir"].add((ix, iy))
            if iss == 0:
                continue
            if ix != ix0 or iy != iy0 or is_ != is0:
                continue
            break
        dash.lastd()

    for ip1 in range(2, m + 1):
        i = ip1 - 1
        if not (Z(i, 1) >= cv or Z(ip1, 1) < cv):
            drline(ip1, 1, -1, 0, 1, 0)
        if not (Z(ip1, n) >= cv or Z(i, n) < cv):
            drline(i, n, 1, 0, 5, 0)
    for jp1 in range(2, n + 1):
        j = jp1 - 1
        if not (Z(m, j) >= cv or Z(m, jp1) < cv):
            drline(m, jp1, 0, -1, 7, 0)
        if not (Z(1, jp1) >= cv or Z(1, j) < cv):
            drline(1, j, 0, 1, 3, 0)
    for jp1 in range(3, n + 1):
        j = jp1 - 1
        for ip1 in range(2, m + 1):
            i = ip1 - 1
            if Z(i, j) >= cv or Z(ip1, j) < cv:
                continue
            if (ip1, j) in st["ir"]:
                continue
            st["np"] += 1
            if st["np"] > NR:
                return
            st["ir"].add((ip1, j))
            drline(ip1, j, -1, 0, 1, 1)


def pattern(value: float, label: str, major: bool, dashpat: int) -> str:
    """CONREC's dash pattern for a level: the ten bits of the pattern (solid for the
    levels not below zero when dashpat is given negative, as t_contour gives it), four
    times and the label after them on a major level."""
    ndot = -dashpat
    negpos = 1 if ndot >= 0 else -1
    idash = abs(ndot)
    if idash in (0, 1):
        idash = ISOLID
    ndash = ISOLID if negpos < 0 and value >= 0 else idash
    bits = "".join("$" if (ndash >> (10 - j)) & 1 else "'" for j in range(1, 11))
    return bits * NREP + label if major else bits


@dataclass
class Drawing:
    """What CONREC puts into the metacode: the polylines and the texts (x, y, text,
    angle in degrees, NCAR's size 0..3), in GKI units; for each level, the index of its
    first polyline; and the viewport, NDC."""
    lines: list
    texts: list
    starts: list
    viewport: tuple


def conrec(z: np.ndarray, levels, viewport, dashpat: int = 528, nhi: int = -1,
           marks=()) -> Drawing:
    """CONREC's drawing (nset 1: the caller's SET, the grid 1..nx, 1..ny on the
    viewport): each level in the order given, as (value, label or "", major), traced,
    smoothed and dashed; then MINMAX's texts, `marks` -- with nhi 0 (x, y, "H" or
    "L", value), the value under the letter; with nhi > 0 (x, y, value)."""
    ny, nx = z.shape
    spps = Spps(viewport, (1.0, float(nx), 1.0, float(ny)))
    # CONREC sets FPART to .5 before its first DASHDC, whose dashbd (NOAO made the
    # block data a first-call initialisation) sets it back to 1: in a fresh process,
    # as each IRAF task begins, the first dash is whole
    dash = Dash(spps, fpart=1.0)
    starts = []
    for value, label, major in levels:
        starts.append(len(spps.gio.lines))
        dash.dashdc(pattern(value, label, major, dashpat))
        stline(z, value, dash)
    scly = f32(f32(1) / f32(1 << (15 - YF)))
    if nhi == 0:
        sizem = f32(f32((ISIZEM + 1) * 256) * scly)
        for x, y, kind, text in marks:
            x, y = f32(x), f32(y)
            spps.wtstr(x, y, kind, 0, ISIZEM)
            _, ify = spps.fl2int(x, y)
            ify = _ifix(f32(f32(ify) * scly))
            my = _ifix(f32(f32(ify) - sizem))
            spps.wtstr(x, spps.cpuy(my), text, 0, ISIZEM)
    elif nhi > 0:
        for x, y, text in marks:
            spps.wtstr(f32(x), f32(y), text, 0, ISIZEP)
    spps.plotif(0, 0, 2)
    spps.gio.flush()
    return Drawing(spps.gio.lines, spps.gio.texts, starts, tuple(float(v) for v in viewport))


def map_viewport(ncols: int, nlines: int, fill: bool = False, perimeter: bool = True,
                 aspect: float = 0.77, requested=(0.0, 0.0, 0.0, 0.0)):
    """pl_map_viewport (pkg/plot/vport.x): the NDC viewport for an ncols x nlines grid,
    as CONREC and hafton would choose it; aspect is the device's "ar" (stdvdm's .77)."""
    ux1, ux2, uy1, uy2 = (f32(v) for v in requested)
    if any(abs(float(v)) > 1e-7 for v in (ux1, ux2, uy1, uy2)):
        return ux1, ux2, uy1, uy2
    ncolsr, nlinesr = f32(ncols), f32(nlines)
    if fill and not perimeter:
        x1, x2, y1, y2, xcen, ycen = (f32(0), f32(1), f32(0), f32(1), f32(0.5), f32(0.5))
    else:
        x1, x2, y1, y2 = f32(0.1), f32(0.93), f32(0.1), f32(0.93)
        xcen, ycen = f32(0.5), f32(0.52)
    if not fill:
        ratio = f32(min(ncolsr, nlinesr) / max(ncolsr, nlinesr))
        if ratio >= f32(0.0625):
            if ncols > nlines:
                y2 = f32(f32(f32(f32(y2 - y1) * nlinesr) / ncolsr) + y1)
            else:
                x2 = f32(f32(f32(f32(x2 - x1) * ncolsr) / nlinesr) + x1)
    xdis, ydis = f32(x2 - x1), f32(y2 - y1)
    if not fill:
        ar = f32(aspect) if abs(aspect) > 1e-7 else f32(1)
        if ar < 1:
            xdis = f32(xdis * ar)
        elif ar > 1:
            ydis = f32(ydis / ar)
    return (f32(xcen - f32(xdis / f32(2))), f32(xcen + f32(xdis / f32(2))),
            f32(ycen - f32(ydis / f32(2))), f32(ycen + f32(ydis / f32(2))))


# ------------------------------------------------------------- SRFACE


STEEP = f32(5.0)


class Horizon:
    """SRFACE's DRAWS: a line of the surface drawn only where it is above the upper
    horizon or below the lower one -- LIMU and LIML, 1024 columns of NCAR's frame --
    and, if marked, pushing them out."""

    def __init__(self, gio: Gio, nupper: int = 0, spval: int = -999,
                 frame=(0.0, 0.0, 0.0, 0.0)):
        self.gio = gio
        #: UMIN, UMAX, VMIN, VMAX, as SRFGK leaves them in the common block
        self.frame = tuple(float(f32(v)) for v in frame)
        self.nupper = nupper
        self.spval = spval
        self.limu = [0] * 1025
        self.liml = [1024] * 1025
        self.mx = 0
        self.my = 0

    def _line(self, nx1, ny1, nx2, ny2):
        self.gio.polyline([f32(nx1), f32(nx2)], [f32(ny1), f32(ny2)])

    def draws(self, mx1, my1, mx2, my2, idraw: int, imark: int):
        if mx1 == self.spval or mx2 == self.spval:
            return
        if self.nupper >= 0:
            self._side(mx1, my1, mx2, my2, idraw, imark, upper=True)
        if self.nupper <= 0:
            self._side(mx1, my1, mx2, my2, idraw, imark, upper=False)

    def _common(self, k: int) -> int:
        """The k-th INTEGER of COMMON /SRFBLK/ from LIMU's start, as DRAWS reads LIMU(k)
        and LIML(k) past their ends for a label outside the frame: LIMU, LIML, and
        then the rest of the block as f2c lays it out on IRAF's 64-bit systems --
        INTEGER 8 bytes, REAL 4, each aligned -- read 8 bytes at a time:
        CL(41) (no contours on the surface: zeros), NCL, LL, FACT, IROT, NDRZ, NUPPER,
        NRSWT, BIGD, UMIN, UMAX, VMIN, VMAX, RZERO, IOFFP, NSPVAL, SPVAL, BIGEST.
        Before LIMU and past the block, memory harwood cannot know: 0."""
        if 1 <= k <= 1024:
            return self.limu[k]
        if 1024 < k <= 2048:
            return self.liml[k - 1024]
        tail = self._tail()
        at = (k - 2049) * 8
        if at >= 0 and at + 8 <= len(tail):
            return int.from_bytes(tail[at:at + 8], "little", signed=True)
        return 0

    def _tail(self) -> bytes:
        import struct

        u = self.frame
        return (bytes(41 * 4) + bytes(4)
                + struct.pack("<qqf4xqqqq", 0, 1, 1.0, 0, 0, self.nupper, 0)
                + struct.pack("<6f", 0.0, *u, 0.0)
                + struct.pack("<qqff", 0, self.spval, 0.0, float(np.finfo(np.float32).max)))

    def _side(self, mx1, my1, mx2, my2, idraw, imark, upper: bool):
        if mx1 > mx2:
            mx1, my1, mx2, my2 = mx2, my2, mx1, my1
        nx1, ny1, nx2, ny2 = mx1, my1, mx2, my2
        mem = self._common

        class _Lim:
            def __getitem__(_, k):
                return mem(k) if upper else mem(k + 1024)

        lim = _Lim()

        def above(y, k):
            return y > lim[k] if upper else y < lim[k]

        if upper:
            vis1, vis2 = ny1 >= lim[nx1] - 1, ny2 >= lim[nx2] - 1
        else:
            vis1, vis2 = ny1 <= lim[nx1] + 1, ny2 <= lim[nx2] + 1
        if not (vis1 or vis2):
            return
        if not (vis1 and vis2):
            if nx1 == nx2:
                edge = (min if upper else max)(lim[nx1], lim[nx2])
                if vis1:
                    ny2 = edge
                if vis2:
                    ny1 = edge
            else:
                dy = f32(f32(ny2 - ny1) / f32(nx2 - nx1))
                fny1 = f32(ny1)
                if not vis1:
                    for k in range(nx1 + 1, nx2 + 1):
                        self.mx = k
                        self.my = _ifix(f32(fny1 + f32(f32(k - nx1) * dy)))
                        if above(self.my, k):
                            break
                    empty = 0 if upper else 1024
                    if abs(dy) >= STEEP and lim[self.mx] != empty:
                        nx1, ny1 = self.mx, lim[self.mx]
                    else:
                        nx1, ny1 = self.mx, self.my
                else:
                    for k in range(nx1 + 1, nx2 + 1):
                        self.mx = k
                        self.my = _ifix(f32(fny1 + f32(f32(k - nx1) * dy)))
                        if (self.my < lim[k]) if upper else (self.my > lim[k]):
                            break
                    if abs(dy) >= STEEP:
                        nx2, ny2 = self.mx - 1, lim[self.mx - 1]
                    else:
                        nx2, ny2 = self.mx - 1, self.my
        lim = self.limu if upper else self.liml
        if idraw != 0:
            self._line(nx1, ny1, nx2, ny2)
        if imark == 0:
            return
        if nx1 == nx2:
            if upper:
                lim[nx1] = max(lim[nx1], max(ny1, ny2))
            else:
                lim[nx1] = min(lim[nx1], min(ny1, ny2))
            return
        dy = f32(f32(ny2 - ny1) / f32(nx2 - nx1))
        fny1 = f32(ny1)
        for k in range(nx1, nx2 + 1):
            ltemp = _ifix(f32(fny1 + f32(f32(k - nx1) * dy)))
            if (ltemp > lim[k]) if upper else (ltemp < lim[k]):
                lim[k] = ltemp


def srfgk(m1: np.ndarray, m2: np.ndarray, x, y, eye, viewport=(0.0, 1.0, 0.0, 1.0),
          look=(0.0, 0.0, 0.0), iupper: int = 0, idrx: int = 1,
          idry: int = 1, frame=(0.0, 0.0, 0.0, 0.0)) -> tuple[Gio, Horizon]:
    """SRFGK's drawing of the mesh, for one picture (no stereo, no skirt, no contours
    on it): the grid's lines in the order that lets the floating horizon hide what
    is behind, from the side nearest the eye. m1, m2: M(1,I,J), M(2,I,J), the grid's
    points in NCAR's 1..1024 frame (m[j-1, i-1]); x, y the grid's coordinates; eye,
    S(1..3); frame, UMIN, UMAX, VMIN, VMAX. What SRFACE's SET puts into the metacode,
    on the viewport, and the horizon as the mesh leaves it."""
    gio = Gio(viewport)
    gio.set_window((1.0, 1024.0, 1.0, 1024.0))
    hz = Horizon(gio, iupper, frame=frame)
    nx, ny = len(x), len(y)
    s1, s2, s3 = (f32(v) for v in eye)
    s4, s5, s6 = (f32(v) for v in look)
    rx, ry = f32(s1 - s4), f32(s2 - s5)

    def M(i, j):
        return int(m1[j - 1, i - 1]), int(m2[j - 1, i - 1])

    def draws(a, b, idraw, imark):
        hz.draws(*a, *b, idraw, imark)

    def passes(eyec, grid, n):
        """The ranges (start, step, end) from the eye's side, one or two."""
        if eyec >= grid[n - 1]:
            return [(n, -1, 1)]
        if eyec <= grid[0]:
            return [(1, 1, n)]
        lx = next(i for i in range(2, n + 1) if eyec <= grid[i - 1])
        return [(lx - 1, -1, 1), (lx, 1, n)]

    xp = passes(s1, x, nx)
    yp = passes(s2, y, ny)
    if len(xp) == 2 and len(yp) == 2:                        # 490: from above or below
        if (hz.nupper > 0 and s3 < s6) or (hz.nupper < 0 and s3 > s6):
            return gio, hz
        hz.nupper = -1 if s3 < s6 else 1
        for i in range(1, nx + 1):
            for j in range(1, ny + 1):
                if idrx and j != ny:
                    draws(M(i, j), M(i, j + 1), 1, 0)
                if idry and i != nx:
                    draws(M(i, j), M(i + 1, j), 1, 0)
        gio.flush()
        return gio, hz
    li = xp[0][1]
    mi = xp[0][0] - li
    lj = yp[0][1]
    mj = yp[0][0] - lj
    nj = abs(mj - yp[0][2])
    if abs(rx) > abs(ry):
        if len(yp) == 1:
            i = xp[0][0]
            for j in range(2, ny + 1):
                draws(M(i, j - 1), M(i, j), 0, 1)
        for ii in range(1, nx + 1):
            i = mi + ii * li
            ipli = i + li
            if len(yp) != 1:
                k, ell = yp[0][0], yp[1][0]
                if idrx:
                    draws(M(i, k), M(i, ell), 1, 1)
            for start, step, end in yp:
                lj = step
                mj = start - lj
                nj = abs(mj - end)
                for jj in range(1, nj + 1):
                    j = mj + jj * lj
                    jplj = j + lj
                    if idrx and jj != nj:
                        draws(M(i, j), M(i, jplj), 1, 1)
                    if i != xp[0][2] and idry:
                        draws(M(ipli, j), M(i, j), 1, 1)
    else:
        if len(xp) == 1:
            j = yp[0][0]
            for i in range(2, nx + 1):
                draws(M(i - 1, j), M(i, j), 0, 1)
        for jj in range(1, ny + 1):
            j = mj + jj * lj
            jplj = j + lj
            if len(xp) != 1:
                k, ell = xp[0][0], xp[1][0]
                if idry:
                    draws(M(k, j), M(ell, j), 1, 1)
            for start, step, end in xp:
                li = step
                mi = start - li
                ni = abs(mi - end)
                for ii in range(1, ni + 1):
                    i = mi + ii * li
                    ipli = i + li
                    if idry and ii != ni:
                        draws(M(i, j), M(ipli, j), 1, 1)
                    if j != yp[0][2] and idrx:
                        draws(M(i, jplj), M(i, j), 1, 1)
    gio.flush()
    return gio, hz
