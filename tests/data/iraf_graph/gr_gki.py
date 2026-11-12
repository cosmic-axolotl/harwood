"""What GIO would write to the metacode for a list of strokes, on a frame whose
window, viewport and transforms IRAF's metacode gives: gamove, gadraw (the world to
GKI transform cached by gpl_cache, the clipping of gpl_clipl/r/b/t in long
integers, the 2048-short buffer), gpl_settype's polymarkers, gmark and gumark. For
tests only: harwood draws its own window. From sys/gio (iraf-community 2.18); see
LICENSES/IRAF-COPYRIGHT."""

import math

import numpy as np

f32 = np.float32
MAXNDC = 32767
LEN_PLBUF = 2048
LOG10E = 0.43429448190325182765

#: The unit-square polylines of the standard marks (markers.inc).
UNIT = {
    "box": ([0, 1, 1, 0, 0], [0, 0, 1, 1, 0]),
    "plus": ([.5, .5, .5, 0, 1], [0, 1, .5, .5, .5]),
    "cross": ([0, 1, .5, 0, 1], [0, 1, .5, 1, 0]),
    "diamond": ([.5, 1, .5, 0, .5], [0, .5, 1, .5, 0]),
    "hline": ([0, 1], [.5, .5]),
    "vline": ([.5, .5], [0, 1]),
    "hebar": ([0, 0, 0, 1, 1, 1], [0, 1, .5, .5, 1, 0]),
    "vebar": ([0, 1, .5, .5, 1, 0], [0, 0, 0, 1, 1, 1]),
    "circle": (
        [1.000, 0.996, 0.984, 0.963, 0.936, 0.901, 0.859, 0.812, 0.759, 0.702, 0.642,
         0.580, 0.516, 0.452, 0.389, 0.327, 0.269, 0.214, 0.164, 0.119, 0.081, 0.050,
         0.025, 0.009, 0.001, 0.001, 0.009, 0.025, 0.050, 0.081, 0.119, 0.164, 0.214,
         0.269, 0.327, 0.389, 0.452, 0.516, 0.580, 0.642, 0.702, 0.759, 0.812, 0.859,
         0.901, 0.936, 0.963, 0.984, 0.996, 1.000],
        [0.500, 0.564, 0.627, 0.688, 0.745, 0.799, 0.848, 0.891, 0.928, 0.957, 0.979,
         0.994, 1.000, 0.998, 0.987, 0.969, 0.943, 0.910, 0.870, 0.824, 0.773, 0.717,
         0.658, 0.596, 0.532, 0.468, 0.404, 0.342, 0.283, 0.227, 0.176, 0.130, 0.090,
         0.057, 0.031, 0.013, 0.002, 0.000, 0.006, 0.021, 0.043, 0.072, 0.109, 0.152,
         0.201, 0.255, 0.312, 0.373, 0.436, 0.500]),
}


def lg10(x):
    return f32(LOG10E * math.log(float(x)))


def elogr(x):
    x = f32(x)
    if x > 10:
        return lg10(x)
    if x >= -10:
        return f32(x / f32(10))
    return f32(-lg10(-x))


def _div(a: int, b: int) -> int:
    """Fortran's integer division, toward zero."""
    q = abs(a) // abs(b)
    return q if (a >= 0) == (b >= 0) else -q


class Gio:
    def __init__(self, frame, marker_size=0.0, aspect=1.0):
        w = [f32(v) for v in frame["window"]]
        self.w = w
        self.tran = frame["transforms"]
        self.marker_size, self.aspect = f32(marker_size), f32(aspect)
        self.mx1, self.mx2 = int(f32(w[4] * f32(MAXNDC))), int(f32(w[5] * f32(MAXNDC)))
        self.my1, self.my2 = int(f32(w[6] * f32(MAXNDC))), int(f32(w[7] * f32(MAXNDC)))
        self.xo, self.xs = self._cache(w[0], w[1], self.tran[0], self.mx1, self.mx2)
        self.yo, self.ys = self._cache(w[2], w[3], self.tran[1], self.my1, self.my2)
        self.out = []
        self.pl = []
        self.type = "line"
        self.pointmode = False
        self.cx = self.cy = None
        self.inb = False
        self.style = (1, 1)
        self.xsv = [0] * 4
        self.ysv = [0] * 4

    @staticmethod
    def _cache(w1, w2, tran, m1, m2):
        n = 0
        while _equal(w1, w2):
            w1 = f32(w1 - max(f32(abs(w1) * f32(0.01)), f32(0.01)))
            w2 = f32(w2 + max(f32(abs(w2) * f32(0.01)), f32(0.01)))
            n += 1
        if tran == 0:
            return w1, f32(f32(m2 - m1) / f32(w2 - w1))
        if tran == 1 and w1 > 0 and w2 > 0:
            o = lg10(w1)
            return o, f32(f32(m2 - m1) / f32(lg10(w2) - o))
        o = elogr(w1)
        return o, f32(f32(m2 - m1) / f32(elogr(w2) - o))

    def _gki(self, wx, wy):
        def one(v, tran):
            if tran == 0:
                return f32(v)
            if tran == 1:
                return lg10(v) if v > 0 else None
            return elogr(v)
        x, y = one(wx, self.tran[0]), one(wy, self.tran[1])
        if x is None or y is None:
            return None
        mx = max(f32(0), min(f32(MAXNDC), f32(f32(f32(x - self.xo) * self.xs)
                                               + f32(self.mx1))))
        my = max(f32(0), min(f32(MAXNDC), f32(f32(f32(y - self.yo) * self.ys)
                                               + f32(self.my1))))
        return mx, my

    # ---------------------------------------------------------------- gpl

    def flush(self):
        if len(self.pl) >= 1:
            if self.type == "marker":
                self.out.append(["marker", *self.style, [list(p) for p in self.pl]])
            elif len(self.pl) >= 2:
                self.out.append(["line", *self.style, [list(p) for p in self.pl]])
        self.pl = []

    def settype(self, kind, pointmode=False):
        if self.pl and self.type != kind:
            self.flush()
        self.type, self.pointmode = kind, pointmode

    def move(self, wx, wy):
        if self.pl:
            self.flush()
        if np.isnan(wx) or np.isnan(wy):
            self.cx = self.cy = None
            return
        got = self._gki(wx, wy)
        if got is not None:
            self.cx, self.cy = got

    def draw(self, wx, wy):
        if np.isnan(wx) or np.isnan(wy):
            self.move(wx, wy)
            return
        got = self._gki(wx, wy)
        if got is None:
            self.flush()
            return
        x, y = got
        if not self.pl:
            if self.cx is None:
                self.cx, self.cy = x, y
                return
            mx, my = int(self.cx), int(self.cy)
            if self.my1 <= my <= self.my2 and self.mx1 <= mx <= self.mx2:
                self.inb = True
                self.pl.append((mx, my))
            else:
                self.inb = False
                self.xsv = [int(self.cx)] * 4
                self.ysv = [int(self.cy)] * 4
        self.cx, self.cy = x, y
        mx, my = int(x), int(y)
        inside = self.my1 <= my <= self.my2 and self.mx1 <= mx <= self.mx2
        if inside and (self.inb or self.pointmode):
            self.pl.append((mx, my))
        elif not self.pointmode:
            if self.inb:
                self.xsv = [self.pl[-1][0]] * 4
                self.ysv = [self.pl[-1][1]] * 4
            self._clipl(True, mx, my)
        self.inb = inside
        if len(self.pl) * 2 + 1 > LEN_PLBUF - 2:
            self.flush()

    def _clipl(self, pen, mx, my):
        xs, ys, m1 = self.xsv, self.ysv, self.mx1
        if (mx >= m1 and xs[0] < m1) or (mx <= m1 and xs[0] > m1):
            self._clipr(False if mx >= m1 else pen, m1,
                        _div((my - ys[0]) * (m1 - mx), mx - xs[0]) + my)
        xs[0], ys[0] = mx, my
        if mx >= m1:
            self._clipr(pen, mx, my)

    def _clipr(self, pen, mx, my):
        xs, ys, m2 = self.xsv, self.ysv, self.mx2
        if (mx <= m2 and xs[1] > m2) or (mx >= m2 and xs[1] < m2):
            self._clipb(False if mx <= m2 else pen, m2,
                        _div((my - ys[1]) * (m2 - mx), mx - xs[1]) + my)
        xs[1], ys[1] = mx, my
        if mx <= m2:
            self._clipb(pen, mx, my)

    def _clipb(self, pen, mx, my):
        xs, ys, m1 = self.xsv, self.ysv, self.my1
        if (my >= m1 and ys[2] < m1) or (my <= m1 and ys[2] > m1):
            self._clipt(False if my >= m1 else pen,
                        _div((mx - xs[2]) * (m1 - my), my - ys[2]) + mx, m1)
        xs[2], ys[2] = mx, my
        if my >= m1:
            self._clipt(pen, mx, my)

    def _clipt(self, pen, mx, my):
        xs, ys, m2 = self.xsv, self.ysv, self.my2
        if (my <= m2 and ys[3] > m2) or (my >= m2 and ys[3] < m2):
            if my <= m2 or not pen:
                self.flush()
            self.pl.append((_div((mx - xs[3]) * (m2 - my), my - ys[3]) + mx, m2))
        xs[3], ys[3] = mx, my
        if my <= m2:
            if not pen:
                self.flush()
            self.pl.append((mx, my))

    # ---------------------------------------------------------------- g*

    def pline(self, x, y):
        self.move(x[0], y[0])
        for a, b in zip(x[1:], y[1:], strict=True):
            self.draw(a, b)

    def scale(self, x, y):
        w, xt, yt = self.w, self.tran[0], self.tran[1]
        xs, ys = f32(w[5] - w[4]), f32(w[7] - w[6])

        def one(v, a, b, tran, s):
            if tran == 1:
                return f32(f32(f32(v / f32(LOG10E)) * lg10(f32(b / a))) / s)
            if tran == 2:
                e = f32(f32(elogr(b) - elogr(a)) / s)
                if v < 10:
                    return f32(f32(-v / f32(LOG10E)) * e)
                if v > 10:
                    return f32(f32(v / f32(LOG10E)) * e)
                return f32(f32(f32(10) / f32(LOG10E)) * e)
            return f32(f32(b - a) / s)
        return one(x, w[0], w[1], xt, xs), one(y, w[2], w[3], yt, ys)

    def mark(self, x, y, kind, xsize, ysize):
        if kind == "point" or (xsize == 0 and ysize == 0):
            self.settype("marker")
            self.move(x, y)
            self.draw(x, y)
            self.settype("line")
            return
        sizes, wcs = [f32(xsize), f32(ysize)], [f32(0), f32(0)]
        d = None
        for i in range(2):
            if sizes[i] > 0:
                if f32(sizes[i] - f32(1)) > -f32(1.192e-7):
                    ndc = self.marker_size * (self.aspect if i == 0 else f32(1))
                else:
                    ndc = sizes[i]
                if d is None:
                    d = self.scale(x, y)
                wcs[i] = f32(ndc * abs(d[i]))
            else:
                wcs[i] = -sizes[i]
        saved, self.style = self.style, (1, self.style[1])
        if saved != self.style:
            self.flush()
        ux, uy = UNIT[kind]
        x1 = f32(x - f32(wcs[0] / f32(2)))
        y1 = f32(y - f32(wcs[1] / f32(2)))
        self.move(f32(f32(f32(ux[0]) * wcs[0]) + x1), f32(f32(f32(uy[0]) * wcs[1]) + y1))
        for a, b in zip(ux[1:], uy[1:], strict=True):
            self.draw(f32(f32(f32(a) * wcs[0]) + x1), f32(f32(f32(b) * wcs[1]) + y1))
        self.move(x, y)
        if saved != self.style:
            self.flush()
            self.style = saved

    def strokes(self, strokes):
        for s in strokes:
            if (s.ltype, s.color) != self.style:
                self.flush()            # gsetr flushes what was drawn in the old style
            self.style = (s.ltype, s.color)
            if s.kind == "line":
                self.pline(s.x, s.y)
            elif s.kind == "points":
                self.settype("marker", pointmode=True)
                self.pline(s.x, s.y)
                self.settype("line")
            else:
                self.mark(s.x[0], s.y[0], s.kind, s.xsize, s.ysize)
        self.flush()
        return self.out


def _equal(x, y) -> bool:
    from harwood.core.plotgraph import fp_equalr

    return fp_equalr(x, y)
