# ruff: noqa: E741
# A line-by-line port keeps IRAF's names (ix, iy, l).
"""The region functions of mskexpr, on one line of a mask at a time.

circle, box, polygon and the rest answer, for each pixel of a line, whether it
is inside, in real arithmetic. The polygons go through IRAF's clipper, which
finds where a line crosses the sides; an edge or a vertex on the line is where
it earns its keep.

From iraf-community 2.18: pkg/proto/maskexpr/meregfuncs.x. See
LICENSES/IRAF-COPYRIGHT.
"""

from __future__ import annotations

import math

import numpy as np

real = np.float32
RADIAN = real(57.295779513082320877)
PI = real(3.1415926535897932385)
SMALL_NUMBER = real(1.0e-24)
INDEFR = real(1.6e38)


def _rad(theta):
    return real(real(theta) / RADIAN)


def _cos(x):
    return real(math.cos(float(x)))


def _sin(x):
    return real(math.sin(float(x)))


def _nint(x: float) -> int:
    v = float(x)
    return int(math.floor(v + 0.5)) if v >= 0 else -int(math.floor(0.5 - v))


def point(ix, iy, x1, y1):
    return (ix == _nint(x1)) & (iy == _nint(y1))


def _r2(ix, iy, xc, yc):
    dx = (ix.astype(real) - real(xc)).astype(real)
    dy = (iy.astype(real) - real(yc)).astype(real)
    return ((dx * dx).astype(real) + (dy * dy).astype(real)).astype(real)


def circle(ix, iy, xc, yc, r):
    return _r2(ix, iy, xc, yc) <= real(real(r) * real(r))


def cannulus(ix, iy, xc, yc, r1, r2):
    rdist = _r2(ix, iy, xc, yc)
    return (rdist >= real(real(r1) * real(r1))) & (rdist <= real(real(r2) * real(r2)))


def _quadratic(a, ratio, theta):
    """The coefficients of an ellipse: aa dx dx + bb dx dy + cc dy dy <= rr."""
    a = real(a)
    b = real(real(ratio) * a)
    asq, bsq = real(a * a), real(b * b)
    cost, sint = _cos(_rad(theta)), _sin(_rad(theta))
    costsq, sintsq = real(cost * cost), real(sint * sint)
    aa = real(real(bsq * costsq) + real(asq * sintsq))
    bb = real(real(real(real(2.0) * real(bsq - asq)) * cost) * sint)
    cc = real(real(asq * costsq) + real(bsq * sintsq))
    return aa, bb, cc, real(asq * bsq)


def _distance(ix, iy, xc, yc, aa, bb, cc):
    dx = (ix.astype(real) - real(xc)).astype(real)
    dy = (iy.astype(real) - real(yc)).astype(real)
    return (((aa * dx).astype(real) * dx).astype(real)
            + ((bb * dx).astype(real) * dy).astype(real)).astype(real) \
        + ((cc * dy).astype(real) * dy).astype(real)


def ellipse(ix, iy, xc, yc, a, ratio, theta):
    aa, bb, cc, rr = _quadratic(a, ratio, theta)
    return _distance(ix, iy, xc, yc, aa, bb, cc).astype(real) <= rr


def eannulus(ix, iy, xc, yc, a1, a2, ratio, theta):
    aa1, bb1, cc1, rr1 = _quadratic(a1, ratio, theta)
    aa2, bb2, cc2, rr2 = _quadratic(a2, ratio, theta)
    d1 = _distance(ix, iy, xc, yc, aa1, bb1, cc1).astype(real)
    d2 = _distance(ix, iy, xc, yc, aa2, bb2, cc2).astype(real)
    return (d1 >= rr1) & (d2 <= rr2)


def box(ix, iy, x1, y1, x2, y2):
    xs, ys = ix.astype(real), iy.astype(real)
    xmin, xmax = min(real(x1), real(x2)), max(real(x1), real(x2))
    ymin, ymax = min(real(y1), real(y2)), max(real(y1), real(y2))
    return (xs >= xmin) & (xs <= xmax) & (ys >= ymin) & (ys <= ymax)


def _corners(xc, yc, x, y, cost, sint):
    """The four corners, in IRAF's order, of a rectangle about xc, yc."""
    xc, yc = real(xc), real(yc)
    xver, yver = [], []
    for sx, sy in ((x, y), (-x, y), (-x, -y), (x, -y)):
        xver.append(real(real(xc + real(sx * cost)) - real(sy * sint)))
        yver.append(real(real(yc + real(sx * sint)) + real(sy * cost)))
    return xver, yver


def rectangle(ix, iy, xc, yc, a, ratio, theta):
    cost, sint = _cos(_rad(theta)), _sin(_rad(theta))
    x, y = real(a), real(real(ratio) * real(a))
    return polygon(ix, iy, *_corners(xc, yc, x, y, cost, sint))


def rannulus(ix, iy, xc, yc, r1, r2, ratio, theta):
    cost, sint = _cos(_rad(theta)), _sin(_rad(theta))
    inner = _corners(xc, yc, real(r1), real(real(ratio) * real(r1)), cost, sint)
    outer = _corners(xc, yc, real(r2), real(real(ratio) * real(r2)), cost, sint)
    return apolygon(ix, iy, *inner, *outer)


def vector(ix, iy, x1, y1, x2, y2, width):
    x1, y1, x2, y2 = real(x1), real(y1), real(x2), real(y2)
    xc, yc = real(real(x2 + x1) / real(2.0)), real(real(y2 + y1) / real(2.0))
    dx, dy = real(x2 - x1), real(y2 - y1)
    x = real(real(math.sqrt(float(real(real(dx * dx) + real(dy * dy))))) / real(2.0))
    y = real(real(width) / real(2.0))
    theta = real(math.atan2(float(dy), float(dx)))
    return polygon(ix, iy, *_corners(xc, yc, x, y, _cos(theta), _sin(theta)))


def _intercept(width, height, xc, yc, angle):
    """me_pie_intercept: where a ray leaves the field, and by which side."""
    angl = real(angle)
    while angl < 0.0:
        angl = real(angl + real(360.0))
    while angl >= 360.0:
        angl = real(angl - real(360.0))
    if abs(angl) < SMALL_NUMBER:
        return real(width + 1), real(yc), 4
    if abs(real(angl - real(180.0))) < SMALL_NUMBER:
        return real(0.0), real(yc), 2
    if angl < 180.0:
        ycept = real(height + 1)
        if abs(real(angl - real(90.0))) < SMALL_NUMBER:
            return real(xc), ycept, 1
    else:
        ycept = real(0.0)
        if abs(real(angl - real(270.0))) < SMALL_NUMBER:
            return real(xc), ycept, 3
    slope = real(math.tan(float(real(real(angl / real(180.0)) * PI))))
    xcept = real(real(xc) + real(real(ycept - real(yc)) / slope))
    if xcept < 0:
        return real(0.0), real(real(yc) - real(real(xc) * slope)), 2
    if xcept > width + 1:
        return (real(width + 1),
                real(real(yc) + real(real(real(width + 1) - real(xc)) * slope)), 4)
    return xcept, ycept, (3 if ycept < height else 1)


_CORNERS = {1: (0.0, 1), 2: (0.0, 0), 3: (1, 0), 4: (1, 1)}


def pie(ix, iy, xc, yc, angle1, angle2, width, height):
    angle1, angle2 = real(angle1), real(angle2)
    sweep = real(angle2 - angle1)
    if abs(sweep) < SMALL_NUMBER:
        return np.zeros(ix.size, bool)
    if sweep < 0.0:
        sweep = real(sweep + real(360.0))
    vx, vy = [real(xc)], [real(yc)]
    x, y, side1 = _intercept(width, height, xc, yc, angle1)
    vx.append(x)
    vy.append(y)
    x2, y2, side2 = _intercept(width, height, xc, yc, angle2)
    if side1 != side2 or sweep > 180.0:
        while True:
            cx, cy = _CORNERS[side1]
            vx.append(real(cx * (width + 1)))
            vy.append(real(cy * (height + 1)))
            side1 = side1 + 1 if side1 < 4 else 1
            if side1 == side2:
                break
    vx.append(x2)
    vy.append(y2)
    return polygon(ix, iy, vx, vy)


def _pyclip(xver, yver, nver, lx, ld, xranges):
    """me_pyclip: where the line ld crosses the sides of the closed polygon,
    written in xranges. Returns how many."""
    slope = [real(0.0)] * (nver + 4)
    xs = [real(0.0)] * (nver + 4)
    for k in range(len(xs)):
        xs[k] = xranges[k] if k < len(xranges) else real(0.0)
    collinear = False
    nplus = nzero = nneg = nintr = 0

    def count(value):
        nonlocal nplus, nzero, nneg
        if value < 0.0:
            nneg += 1
        elif value > 0.0:
            nplus += 1
        else:
            nzero += 1

    u1 = real(lx * real(-yver[0] + ld))
    for i in range(2, nver + 1):
        u2 = real(lx * real(-yver[i - 1] + ld))
        if real(u1 * u2) <= 0.0:
            if u1 != 0.0 and u2 != 0.0:
                dy = real(yver[i - 2] - yver[i - 1])
                dx = real(xver[i - 2] - xver[i - 1])
                dd = real(real(xver[i - 2] * yver[i - 1]) - real(yver[i - 2] * xver[i - 1]))
                xa = real(lx * real(real(dx * ld) - dd))
                wa = real(dy * lx)
                nintr += 1
                xs[nintr - 1] = real(xa / wa)
                slope[nintr - 1] = real(-dy)
                count(slope[nintr - 1])
                collinear = False
            elif u1 == 0.0 and u2 == 0.0:
                if not collinear:
                    nintr += 1
                    xs[nintr - 1] = xver[i - 2]
                    slope[nintr - 1] = (real(yver[0] - yver[nver - 2]) if i == 2
                                        else real(yver[i - 2] - yver[i - 3]))
                    count(slope[nintr - 1])
                    nintr += 1
                    xs[nintr - 1] = xver[i - 1]
                    slope[nintr - 1] = real(0.0)
                    nzero += 1
                else:
                    xs[nintr - 1] = xver[i - 1]
                    slope[nintr - 1] = real(0.0)
                    nzero += 1
                collinear = True
            elif u1 != 0.0:
                if i == nver:
                    dx = real(xver[1] - xver[nver - 1])
                    dy = real(yver[1] - yver[nver - 1])
                    dd = real(dy * real(yver[nver - 2] - yver[nver - 1]))
                else:
                    dx = real(xver[i] - xver[i - 1])
                    dy = real(yver[i] - yver[i - 1])
                    dd = real(dy * real(yver[i - 2] - yver[i - 1]))
                if dy != 0.0:
                    nintr += 1
                    xs[nintr - 1] = xver[i - 1]
                    slope[nintr - 1] = real(yver[i - 1] - yver[i - 2])
                    count(slope[nintr - 1])
                if dd > 0.0:
                    nintr += 1
                    xs[nintr - 1] = xver[i - 1]
                    slope[nintr - 1] = dy
                    count(slope[nintr - 1])
                collinear = False
            else:
                collinear = False
        else:
            collinear = False
        u1 = u2

    if collinear and slope[0] == 0.0:
        xs[0] = xs[nintr - 2]
        slope[0] = slope[nintr - 2]
        nintr -= 2
        nzero -= 2

    if nzero == 0 or nplus == 0 or nneg == 0:
        xranges[:] = xs
        return nintr

    imin, imax = 1, 1
    low, high = xs[0], xs[0]
    for i in range(2, nintr + 1):
        if xs[i - 1] > high:
            imax, high = i, xs[i - 1]
        if xs[i - 1] < low:
            imin, low = i, xs[i - 1]
    u1 = real(xs[min(imin, imax) - 1] - xs[0])
    u2 = real(xs[nintr - 1] - xs[max(imin, imax) - 1])
    if (u1 >= 0.0 and u2 > 0.0) or (u1 > 0.0 and u2 >= 0.0) or (u1 == u2 and imax > imin):
        for i in range(imax + 1, nintr + 1):
            if xs[i - 1] != xs[i - 2]:
                break
            imax = i
        for i in range(imin - 1, 0, -1):
            if xs[i - 1] != xs[i]:
                break
            imin = i
    if (u1 <= 0.0 and u2 < 0.0) or (u1 < 0.0 and u2 <= 0.0) or (u1 == u2 and imax < imin):
        for i in range(imin + 1, nintr + 1):
            if xs[i - 1] != xs[i - 2]:
                break
            imin = i
        for i in range(imax - 1, 0, -1):
            if xs[i - 1] != xs[i]:
                break
            imax = i
    if imax < imin and not (imin == nintr and imax == 1):
        first = imax
    elif imin < imax and not (imin == 1 and imax == nintr):
        first = imin
    else:
        first = 0
    if first:
        last = imin if imax < imin else imax
        for values in (xs, slope):
            old = list(values)
            for i in range(1, first + 1):
                values[nintr - first + i - 1] = old[i - 1]
            for i in range(last, nintr + 1):
                values[i - first - 1] = old[i - 1]

    nadd = 0
    i = 1
    while i <= nintr - 2:
        if real(slope[i - 1] * slope[i + 1]) > 0.0:
            i += 2
        else:
            nadd += 1
            xs.append(real(0.0))
            xs[nintr + nadd - 1] = xs[i]
            j = i + 3
            while j <= nintr:
                if real(slope[i - 1] * slope[j - 1]) > 0:
                    break
                nadd += 1
                xs.append(real(0.0))
                xs[nintr + nadd - 1] = xs[j - 2]
                j += 1
            i = j
    xranges[:] = xs
    return nintr + nadd


def _segments(iy):
    """The runs of a line over which y stays the same: nearly always one."""
    edges = np.flatnonzero(iy[1:] != iy[:-1]) + 1
    return np.concatenate(([0], edges)).tolist(), np.concatenate((edges, [iy.size])).tolist()


def polygon(ix, iy, xver, yver):
    """me_polygon: a pair of crossings at a time; with an odd number the last pair
    reads a word past them, as IRAF does."""
    stat = np.zeros(ix.size, bool)
    nver = len(xver)
    xclosed, yclosed = list(xver) + [xver[0]], list(yver) + [yver[0]]
    lx = real(int(ix.max()) - int(ix.min()) + 1)
    xs = ix.astype(real)
    work = [real(0.0)] * (nver + 4)
    for first, stop in zip(*_segments(iy), strict=True):
        nintr = _pyclip(xclosed, yclosed, nver + 1, lx, real(iy[first]), work)
        crossings = sorted(work[:nintr])
        tail = work[nintr:] + [real(0.0)]
        for j in range(0, nintr, 2):
            right = crossings[j + 1] if j + 1 < nintr else tail[0]
            stat[first:stop] |= (xs[first:stop] >= crossings[j]) & (xs[first:stop] <= right)
    return stat


def apolygon(ix, iy, ixver, iyver, oxver, oyver):
    """me_apolygon: inside the outer polygon and not inside the inner."""
    stat = np.zeros(ix.size, bool)
    nver = len(ixver)
    ixc, iyc = list(ixver) + [ixver[0]], list(iyver) + [iyver[0]]
    oxc, oyc = list(oxver) + [oxver[0]], list(oyver) + [oyver[0]]
    lx = real(int(ix.max()) - int(ix.min()) + 1)
    xs = ix.astype(real)
    work = [real(0.0)] * (nver + 4)
    ibegin, iend = 1, 1
    for first, stop in zip(*_segments(iy), strict=True):
        part = xs[first:stop]
        ld = real(iy[first])
        onintr = _pyclip(oxc, oyc, nver + 1, lx, ld, work)
        outer = sorted(work[:onintr])
        if onintr <= 0:
            continue
        inintr = _pyclip(ixc, iyc, nver + 1, lx, ld, work)
        inner = sorted(work[:inintr])
        out = np.zeros(part.size, bool)
        for j in range(0, onintr, 2):
            right = outer[j + 1] if j + 1 < onintr else real(0.0)
            if inintr <= 0:
                out |= (part >= outer[j]) & (part <= right)
                continue
            for jj in range(0, inintr, 2):
                if outer[j] < inner[jj] < right:
                    ibegin = jj + 1
                    break
            for jj in range(inintr - 1, -1, -2):
                if outer[j] < inner[jj] < right:
                    iend = jj + 1
                    break
            done = (part >= outer[j]) & (part <= inner[ibegin - 1])
            done |= (part >= inner[iend - 1]) & (part <= right)
            for jj in range(ibegin, iend - 1, 2):
                done |= (part >= inner[jj]) & (part <= inner[jj + 1])
            out |= done
        stat[first:stop] = out
    return stat


def expand(xin, yin, width):
    """me_pyexpand: the polygon moved out by width, side by side."""
    nver = len(xin)
    xcen = real(_asum(xin) / real(nver))
    ycen = real(_asum(yin) / real(nver))
    width = real(width)
    xout, yout = [], []
    for i in range(nver):
        before = i - 1 if i > 0 else nver - 1
        after = i + 1 if i < nver - 1 else 0
        m1, b1, xp1, yp1 = _segment(xcen, ycen, xin[before], yin[before], xin[i], yin[i],
                                    width)
        m2, b2, xp2, yp2 = _segment(xcen, ycen, xin[i], yin[i], xin[after], yin[after], width)
        if m1 == m2:
            xout.append(xp2)
            yout.append(yp2)
        elif m1 == INDEFR:
            xout.append(xp1)
            yout.append(real(real(m2 * xp1) + b2))
        elif m2 == INDEFR:
            xout.append(xp2)
            yout.append(real(real(m1 * xp2) + b1))
        else:
            xout.append(real(real(b2 - b1) / real(m1 - m2)))
            yout.append(real(real(real(m2 * b1) - real(m1 * b2)) / real(m2 - m1)))
    return xout, yout


def _asum(values):
    total = real(0.0)
    for v in values:
        total = real(total + v)
    return total


def _dist2(dx, dy):
    return real(real(dx * dx) + real(dy * dy))


def _segment(xcen, ycen, xb, yb, xe, ye, width):
    """me_psegment: a side, moved width away from the centre."""
    m = real(xe - xb)
    m = INDEFR if m == 0.0 else real(real(ye - yb) / m)
    if m == INDEFR:
        x1, y1, x2, y2 = real(xb - width), yb, real(xb + width), yb
    elif m == 0.0:
        x1, y1, x2, y2 = xb, real(yb - width), xb, real(yb + width)
    else:
        mw = real(m * width)
        off = real(math.sqrt(float(real(real(mw * mw) / real(real(m * m) + real(1.0))))))
        x1 = real(xb - off)
        y1 = real(yb - real(real(x1 - xb) / m))
        x2 = real(xb + off)
        y2 = real(yb - real(real(x2 - xb) / m))
    d1 = _dist2(real(x1 - xcen), real(y1 - ycen))
    d2 = _dist2(real(x2 - xcen), real(y2 - ycen))
    xp, yp = (x2, y2) if d1 <= d2 else (x1, y1)
    b = INDEFR if m == INDEFR else real(yp - real(m * xp))
    return m, b, xp, yp
