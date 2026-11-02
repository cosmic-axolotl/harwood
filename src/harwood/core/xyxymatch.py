"""IRAF's xyxymatch, ported: match two coordinate lists by tolerance or by
triangles.

From iraf-community 2.18 pkg/images/immatch/src/listmatch/t_xyxymatch.x and
pkg/images/lib (xymatch.x, rgsort.x, rgtransform.x, rgmerge.x). The
triangles algorithm is Groth's (1986, AJ 91, 1244) as IRAF implements it:
every triangle of up to `nmatch` stars of each list, described by the log
of its perimeter, the ratio of its longest to shortest side and the cosine
at its first vertex; triangles matched within tolerances that propagate
the position tolerance; false matches rejected by k-sigma about the mode of
the log-perimeter differences and by the sense (clockwise or not) of the
pair; and a vote: a pair of stars that turns up in many matched triangles
is a match.

IRAF's arithmetic is single precision (REAL), and its sorts are its own
unstable quicksort -- which pair of equal values comes first decides,
here and there, which star is kept -- so both are reproduced.

Lists are 1-based: element 0 of every array is padding, and indices are
IRAF's.

Translated from IRAF, Copyright (c) 1986 Association of Universities for
Research in Astronomy Inc.; the notice it is distributed under is in
LICENSES/IRAF-COPYRIGHT.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

f32 = np.float32
RADIAN = f32(57.295779513082320877)
PI = f32(3.1415926535897932385)
MAX_REAL = f32(9.9e36)
EPSILONR = 1.192e-7

# Triangle table columns (xyxymatch.h).
INDEX, X1, X2, X3, CC, MATCH = 1, 2, 3, 4, 5, 6
LOGP, RATIO, COS1, TOLR, TOLC = 1, 2, 3, 4, 5


def _sqrt(v) -> np.float32:
    return f32(math.sqrt(float(v))) if v >= 0 else f32("nan")


def _fp_equal(a, b) -> bool:
    """fp_equalr: equal to within the precision of a REAL."""
    a, b = float(a), float(b)
    if a == b:
        return True
    scale = max(abs(a), abs(b))
    return abs(a - b) <= 2 * EPSILONR * scale if scale > 0 else True


# ----------------------------------------------------------- reading lists


def read_list(path, xcolumn: int = 1, ycolumn: int = 2):
    """rg_rdxyi: the coordinates in columns xcolumn and ycolumn of every line
    that has them as numbers, and each one's line number (counting every
    line, comments and blank ones included)."""
    from pathlib import Path

    xs, ys, lines = [f32(0)], [f32(0)], [0]
    for number, text in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        words = text.split()
        if len(words) < max(xcolumn, ycolumn):
            continue
        try:
            x, y = _number(words[xcolumn - 1]), _number(words[ycolumn - 1])
        except ValueError:
            continue
        xs.append(f32(x))
        ys.append(f32(y))
        lines.append(number)
    return np.array(xs, dtype=f32), np.array(ys, dtype=f32), np.array(lines)


def _number(text: str) -> float:
    """ctor: a number, sexagesimal included; INDEF is not one."""
    if text.upper() == "INDEF":
        raise ValueError(text)
    if ":" in text:
        sign = -1.0 if text.startswith("-") else 1.0
        parts = [float(p) for p in text.lstrip("+-").split(":")]
        return sign * sum(p / 60.0 ** k for k, p in enumerate(parts))
    return float(text)


# ------------------------------------------------------------------- sorts


def qsort(data: np.ndarray, index: list[int], first: int, n: int) -> None:
    """rg_qsortr / rg_qsorti: IRAF's quicksort of the index (1-based,
    index[first .. first + n - 1]) by data[index[k]], in place."""
    b = index
    lv, uv = [0] * 22, [0] * 22
    p = 1
    lv[1], uv[1] = first, first + n - 1
    while p > 0:
        if lv[p] >= uv[p]:
            p -= 1
            continue
        i, j = lv[p] - 1, uv[p]
        pivot = data[b[j]]
        while i < j:
            i += 1
            while data[b[i]] < pivot:
                i += 1
            j -= 1
            while j > i:
                if data[b[j]] <= pivot:
                    break
                j -= 1
            if i < j:
                b[i], b[j] = b[j], b[i]
        j = uv[p]
        b[i], b[j] = b[j], b[i]
        if i - lv[p] < uv[p] - i:
            lv[p + 1], uv[p + 1] = lv[p], i - 1
            lv[p] = i + 1
        else:
            lv[p + 1], uv[p + 1] = i + 1, uv[p]
            uv[p] = i - 1
        p += 1


def sort_unique(x: np.ndarray, y: np.ndarray, npts: int, separation) -> tuple[list[int], int]:
    """rg_sort: an index sorted in y and then in x, without the points closer
    than separation to an earlier one. Returns the index and its length."""
    index = list(range(npts + 1))
    if npts > 0:
        qsort(y, index, 1, npts)
        # rg_sqsort: runs of equal y sorted on x -- except the last run.
        ndup = 0
        for i in range(2, npts + 1):
            if y[index[i]] <= y[index[i - 1]]:
                ndup += 1
            elif ndup > 0:
                qsort(x, index, i - 1 - ndup, ndup + 1)
                ndup = 0
    return index, _coincide(x, y, index, npts, f32(separation))


def _coincide(x, y, a: list[int], npts: int, tolerance) -> int:
    """rg_xycoincide: drop points within tolerance of an earlier one."""
    tol2 = f32(tolerance * tolerance)
    nunique = npts
    iprev = 1
    while True:
        for i in range(iprev + 1, npts + 1):
            if a[iprev] == 0:
                break
            if a[i] == 0:
                continue
            dy = f32(y[a[i]] - y[a[iprev]])
            r2 = f32(dy * dy)
            if r2 > tol2:
                break
            dx = f32(x[a[i]] - x[a[iprev]])
            r2 = f32(r2 + f32(dx * dx))
            if r2 <= tol2:
                a[i] = 0
                nunique -= 1
        iprev += 1
        if iprev >= npts:
            break
    if nunique < npts:
        kept = [v for v in a[1:npts + 1] if v != 0]
        a[1:len(kept) + 1] = kept
    return nunique


# --------------------------------------------------------- transformations


def make_coeff(xin=0.0, yin=0.0, xmag=1.0, ymag=1.0, xrot=0.0, yrot=0.0,
               xout=0.0, yout=0.0) -> np.ndarray:
    """rg_lmkcoeff: input -> reference from origins, scales and rotations."""
    c = np.zeros(7, dtype=f32)
    xmag, ymag, xin, yin = f32(xmag), f32(ymag), f32(xin), f32(yin)
    c[1] = f32(float(xmag) * math.cos(float(f32(f32(xrot) / RADIAN))))
    c[2] = f32(-float(ymag) * math.sin(float(f32(f32(yrot) / RADIAN))))
    c[3] = f32(f32(f32(xout) - f32(c[1] * xin)) - f32(c[2] * yin))
    c[4] = f32(float(xmag) * math.sin(float(f32(f32(xrot) / RADIAN))))
    c[5] = f32(float(ymag) * math.cos(float(f32(f32(yrot) / RADIAN))))
    c[6] = f32(f32(f32(yout) - f32(c[4] * xin)) - f32(c[5] * yin))
    return c


def apply(c: np.ndarray, x: np.ndarray, y: np.ndarray):
    """rg_compute."""
    xt = ((c[1] * x).astype(f32) + (c[2] * y).astype(f32)).astype(f32) + c[3]
    yt = ((c[4] * x).astype(f32) + (c[5] * y).astype(f32)).astype(f32) + c[6]
    return xt.astype(f32), yt.astype(f32)


def _posangle(x, y) -> np.float32:
    if _fp_equal(y, 0.0):
        return f32(PI) if x < 0 else f32(0)
    if _fp_equal(x, 0.0):
        return f32(PI / f32(2)) if y > 0 else (f32(f32(3) * PI / f32(2)) if y < 0
                                                else f32(0))
    t = f32(math.atan(float(f32(y / x))))
    if x > 0 and y > 0:
        return t
    if x > 0:
        return f32(float(f32(2) * PI) + float(t))
    return f32(float(PI) + float(t))


def tie_coeff(xref, yref, xin, yin) -> np.ndarray | None:
    """rg_lincoeff: the transformation from one (a shift), two (shift,
    scale and rotation), three (a full linear map) or more (least squares)
    pairs of points; None when it cannot be found."""
    n = len(xref)
    c = np.zeros(7, dtype=f32)
    xr, yr = np.asarray(xref, dtype=f32), np.asarray(yref, dtype=f32)
    xl, yl = np.asarray(xin, dtype=f32), np.asarray(yin, dtype=f32)
    if n == 0:
        return None
    if n == 1:
        c[1:] = [1, 0, xr[0] - xl[0], 0, 1, yr[0] - yl[0]]
        return c
    if n == 2:
        return _twostar(xr, yr, xl, yl)
    if n == 3:
        det = f32(f32(f32(xl[0] * f32(yl[1] - yl[2])) - f32(xl[1] * f32(yl[0] - yl[2])))
                  + f32(xl[2] * f32(yl[0] - yl[1])))
        if _fp_equal(det, 0.0):
            return _twostar(xr, yr, xl, yl)
        dx23, dx13, dx12 = f32(xl[1] - xl[2]), f32(xl[0] - xl[2]), f32(xl[0] - xl[1])
        dy23, dy13, dy12 = f32(yl[1] - yl[2]), f32(yl[0] - yl[2]), f32(yl[0] - yl[1])
        cross = (f32(f32(xl[1] * yl[2]) - f32(xl[2] * yl[1])),
                 f32(f32(yl[0] * xl[2]) - f32(xl[0] * yl[2])),
                 f32(f32(xl[0] * yl[1]) - f32(yl[0] * xl[1])))
        for base, v in ((1, xr), (4, yr)):
            c[base] = f32(f32(f32(f32(v[0] * dy23) - f32(v[1] * dy13)) + f32(v[2] * dy12))
                          / det)
            c[base + 1] = f32(f32(f32(f32(-v[0] * dx23) + f32(v[1] * dx13))
                                  - f32(v[2] * dx12)) / det)
            c[base + 2] = f32(f32(f32(f32(v[0] * cross[0]) + f32(v[1] * cross[1]))
                                  + f32(v[2] * cross[2])) / det)
        return c
    fx, fy = _plane(xl, yl, xr), _plane(xl, yl, yr)
    if fx is None:                     # IRAF tests xier twice and never yier
        return None
    c[1:] = [fx[1], fx[2], fx[0], fy[1], fy[2], fy[0]]
    return c


def _plane(x: np.ndarray, y: np.ndarray, z: np.ndarray):
    """gsfit of a GS_POLYNOMIAL, xorder = yorder = 2, GS_XNONE, uniform
    weights: z = a + b x + c y, accumulated and solved in single precision
    as gsacpts and gssolve do. None when the matrix is singular."""
    from . import nlfit

    def ssum(v):
        return np.cumsum(np.asarray(v, dtype=f32), dtype=f32)[-1]

    one = np.ones(x.size, dtype=f32)
    vector = np.array([ssum(one * z), ssum((x * z).astype(f32)),
                       ssum((y * z).astype(f32))], dtype=f32)
    matrix = np.zeros((3, 3), dtype=f32)
    matrix[0, 0], matrix[1, 0], matrix[2, 0] = ssum(one), ssum(x), ssum(y)
    matrix[0, 1], matrix[1, 1] = ssum((x * x).astype(f32)), ssum((x * y).astype(f32))
    matrix[0, 2] = ssum((y * y).astype(f32))
    factor = np.zeros((3, 3), dtype=f32)
    if nlfit.chfac(matrix, 3, 3, factor, singular=f32(1000) / MAX_REAL) != nlfit.OK:
        return None
    return nlfit.chslv(factor, 3, 3, vector)


def _twostar(xr, yr, xl, yl) -> np.ndarray:
    c = np.zeros(7, dtype=f32)
    dxl, dyl = f32(xl[1] - xl[0]), f32(yl[1] - yl[0])
    dxr, dyr = f32(xr[1] - xr[0]), f32(yr[1] - yr[0])
    rot = f32(_posangle(dxr, dyr) - _posangle(dxl, dyl))
    cosr, sinr = f32(math.cos(float(rot))), f32(math.sin(float(rot)))
    mag = f32(f32(dxl * dxl) + f32(dyl * dyl))
    mag = f32(0) if mag <= 0 else _sqrt(f32(f32(f32(dxr * dxr) + f32(dyr * dyr)) / mag))
    c[1], c[2] = f32(mag * cosr), f32(-mag * sinr)
    c[3] = f32(f32(xr[0] - f32(f32(mag * cosr) * xl[0])) + f32(f32(mag * sinr) * yl[0]))
    c[4], c[5] = f32(mag * sinr), f32(mag * cosr)
    c[6] = f32(f32(yr[0] - f32(f32(mag * sinr) * xl[0])) - f32(f32(mag * cosr) * yl[0]))
    return c


def geometry(c: np.ndarray) -> tuple[float, float, float, float]:
    """rg_ctogeo(c1, -c2, -c4, c5): x and y scales and rotations (degrees)."""
    a, b, cc, d = c[1], -c[2], -c[4], c[5]
    xscale = _sqrt(f32(f32(a * a) + f32(cc * cc)))
    yscale = _sqrt(f32(f32(b * b) + f32(d * d)))
    xrot = (f32(0) if _fp_equal(a, 0) and _fp_equal(cc, 0)
            else f32(math.atan2(float(-cc), float(a)) * float(RADIAN)))
    if xrot < 0:
        xrot = f32(xrot + f32(360))
    yrot = (f32(0) if _fp_equal(b, 0) and _fp_equal(d, 0)
            else f32(math.atan2(float(b), float(d)) * float(RADIAN)))
    if yrot < 0:
        yrot = f32(yrot + f32(360))
    return float(xscale), float(yscale), float(xrot), float(yrot)


# ---------------------------------------------------------- tolerance match


def intersect(xref, yref, refindex, nref, xtrans, ytrans, listindex, nlist,
              tolerance) -> list[tuple[int, int]]:
    """rg_intersection: for each reference point in sorted order, the
    nearest transformed input point within tolerance (the last of equals)."""
    tol = f32(tolerance)
    tol2 = f32(tol * tol)
    pairs = []
    blp = 1
    for rp in range(1, nref + 1):
        r = refindex[rp]
        dy = f32(0)
        while blp <= nlist:
            dy = f32(yref[r] - ytrans[listindex[blp]])
            if dy < tol:
                break
            blp += 1
        if blp > nlist:
            break
        if dy < -tol:
            continue
        best, match = tol2, 0
        for lp in range(blp, nlist + 1):
            k = listindex[lp]
            dy = f32(yref[r] - ytrans[k])
            if dy < -tol:
                break
            dx = f32(xref[r] - xtrans[k])
            r2 = f32(f32(dx * dx) + f32(dy * dy))
            if r2 <= best:
                best, match = r2, k
        if match > 0:
            pairs.append((r, match))
    return pairs


# ----------------------------------------------------------------- triangles


@dataclass
class Triangles:
    """One list's triangles: tri[k, INDEX..MATCH] and par[k, LOGP..TOLC]."""

    tri: np.ndarray = field(repr=False)
    par: np.ndarray = field(repr=False)
    n: int = 0


def ncombinations(n: int) -> int:
    """rg_factorial(n, 3)."""
    if n <= 0:
        return 1
    return n * (n - 1) * (n - 2) // 6


def triangles(x, y, index, npoints: int, nmax: int, maxnpts: int, tolerance,
              maxratio, tri: Triangles | None = None) -> Triangles:
    """rg_triangle: the triangles of up to maxnpts of the points (every
    nsample-th when there are more), vertices ordered shortest side 1-2 and
    longest 1-3, sorted on the ratio of the longest to the shortest side."""
    if tri is None:
        tri = Triangles(np.zeros((nmax + 1, 7), dtype=np.int64),
                        np.zeros((nmax + 1, 6), dtype=f32))
    t, par = tri.tri, tri.par
    tol2 = f32(f32(tolerance) * f32(tolerance))
    maxratio = f32(maxratio)
    nsample = max(1, npoints // maxnpts)
    npts = min(npoints, nsample * maxnpts)
    ntri = 1

    def d2(p, q):
        dx, dy = f32(x[p] - x[q]), f32(y[p] - y[q])
        return f32(f32(dx * dx) + f32(dy * dy))

    for i in range(1, npts - 2 * nsample + 1, nsample):
        for j in range(i + nsample, npts - nsample + 1, nsample):
            for k in range(j + nsample, npts + 1, nsample):
                pi, pj, pk = index[i], index[j], index[k]
                rij = d2(pi, pj)
                if rij <= tol2:
                    continue
                rjk = d2(pj, pk)
                if rjk <= tol2:
                    continue
                rki = d2(pk, pi)
                if rki <= tol2:
                    continue
                if rij <= rjk:
                    if rki <= rij:
                        v1, v2, v3 = pk, pi, pj
                    elif rki >= rjk:
                        v1, v2, v3 = pi, pj, pk
                    else:
                        v1, v2, v3 = pj, pi, pk
                elif rki <= rjk:
                    v1, v2, v3 = pi, pk, pj
                elif rki >= rij:
                    v1, v2, v3 = pk, pj, pi
                else:
                    v1, v2, v3 = pj, pk, pi
                t[ntri, INDEX], t[ntri, X1], t[ntri, X2], t[ntri, X3] = ntri, v1, v2, v3
                dx1, dy1 = f32(x[v3] - x[v2]), f32(y[v3] - y[v2])
                dx2, dy2 = f32(x[v2] - x[v1]), f32(y[v2] - y[v1])
                dx3, dy3 = f32(x[v3] - x[v1]), f32(y[v3] - y[v1])
                r1 = _sqrt(f32(f32(dx1 * dx1) + f32(dy1 * dy1)))
                r2sq = f32(f32(dx2 * dx2) + f32(dy2 * dy2))
                r2 = _sqrt(r2sq)
                r3sq = f32(f32(dx3 * dx3) + f32(dy3 * dy3))
                r3 = _sqrt(r3sq)
                if r2 <= 0:
                    continue
                ratio = f32(r3 / r2)
                if ratio > maxratio:
                    continue
                cosc = f32(f32(f32(dx3 * dx2) + f32(dy3 * dy2)) / f32(r3 * r2))
                cosc2 = max(f32(0), min(f32(1), f32(cosc * cosc)))
                sinc2 = max(f32(0), min(f32(1), f32(f32(1) - cosc2)))
                t[ntri, CC] = 1 if f32(f32(dx2 * dy1) - f32(dy2 * dx1)) > 0 else 0
                tol = f32(f32(f32(f32(1) / r3sq) - f32(cosc / f32(r3 * r2)))
                          + f32(f32(1) / r2sq))
                par[ntri, TOLR] = f32(f32(f32(f32(2) * f32(ratio * ratio)) * tol2) * tol)
                par[ntri, TOLC] = f32(
                    f32(f32(f32(f32(2) * sinc2) * tol2) * tol)
                    + f32(f32(f32(f32(f32(3) * cosc2) * f32(tol2 * tol2)) * tol) * tol))
                par[ntri, LOGP] = f32(math.log(float(f32(f32(r1 + r2) + r3))))
                par[ntri, RATIO] = ratio
                par[ntri, COS1] = cosc
                ntri += 1
    ntri -= 1
    order = list(t[:, INDEX])
    qsort(par[:, RATIO], order, 1, ntri)
    t[1:ntri + 1, INDEX] = order[1:ntri + 1]
    tri.n = ntri
    return tri


def _merge(a: Triangles, b: Triangles) -> int:
    """rg_tmerge: for each triangle of `a` in ratio order, the closest of
    `b` within both tolerances; the pairs go into the MATCH columns.

    The inner search is IRAF's loop done a candidate range at a time with
    numpy: the same single-precision operations element by element, and the
    first of equal best sums, as the loop's strict comparison keeps."""
    if a.n <= 0 or b.n <= 0:
        return 0
    maxtol = _sqrt(f32(a.par[1:a.n + 1, TOLR].max() + b.par[1:b.n + 1, TOLR].max()))
    border = b.tri[1:b.n + 1, INDEX].astype(np.int64)
    bratio = b.par[border, RATIO]
    bcos, btolr, btolc = b.par[border, COS1], b.par[border, TOLR], b.par[border, TOLC]
    blp, ninter = 0, 0
    for rp in range(1, a.n + 1):
        r = a.tri[rp, INDEX]
        ar = a.par[r, RATIO]
        while blp < b.n and f32(ar - bratio[blp]) > maxtol:
            blp += 1
        if blp >= b.n:
            break
        dr = (ar - bratio[blp:]).astype(f32)
        if dr[0] < -maxtol:
            continue
        beyond = dr < -maxtol
        stop = int(np.argmax(beyond)) if beyond.any() else dr.size
        dr = dr[:stop]
        dr2 = (dr * dr).astype(f32)
        dc = (a.par[r, COS1] - bcos[blp:blp + stop]).astype(f32)
        dcos2 = (dc * dc).astype(f32)
        ok = ((dr2 <= (a.par[r, TOLR] + btolr[blp:blp + stop]).astype(f32))
              & (dcos2 <= (a.par[r, TOLC] + btolc[blp:blp + stop]).astype(f32)))
        if not ok.any():
            continue
        total = np.where(ok, (dr2 + dcos2).astype(f32), np.inf)
        best = int(np.argmin(total))
        ninter += 1
        a.tri[ninter, MATCH] = r
        b.tri[ninter, MATCH] = border[blp + best]
    return ninter


def _mode(a: list[float], nmin: int, zrange: float, zbin: float, zstep: float) -> float:
    """rg_moded, on a sorted list (1-based through a[0] padding)."""
    n = len(a) - 1
    if n < nmin:
        return a[1 + n // 2] if n % 2 == 1 else (a[n // 2] + a[1 + n // 2]) / 2.0
    x1 = max(1, int(1.0 + n * (1.0 - zrange) / 2.0))
    x3 = min(n, int(1.0 + n * (1.0 + zrange) / 2.0))
    if a[x1] == a[x3] or abs(a[x1] - a[x3]) <= 2.22e-16 * max(abs(a[x1]), abs(a[x3])):
        return a[x1]
    nmax, x2, y1, mode = 0, x1, a[x1], a[x1]
    while x2 < x3:
        while a[x1] < y1:
            x1 += 1
        y2 = y1 + zbin
        while x2 < x3 and a[x2] < y2:
            x2 += 1
        if x2 - x1 > nmax:
            nmax = x2 - x1
            mode = (a[(x2 + x1) // 2] if (x2 + x1) % 2 == 0
                    else (a[(x2 + x1) // 2] + a[(x2 + x1) // 2 + 1]) / 2.0)
        y1 += zstep
    return mode


def _reject(ref: Triangles, inp: Triangles, nmatch: int, maxiter: int) -> int:
    """rg_treject: k-sigma rejection about the mode of the log-perimeter
    differences, k set by how many pairs keep their sense, then only the
    pairs of the majority sense."""
    rt, lt = ref.tri, inp.tri

    def dif(i):
        r, k = rt[i, MATCH], lt[i, MATCH]
        return float(f32(ref.par[r, LOGP] - inp.par[k, LOGP])), rt[r, CC] == lt[k, CC]

    total = squares = 0.0
    nplus = 0
    values = []
    for i in range(1, nmatch + 1):
        d, same = dif(i)
        values.append(d)
        total += d
        squares += d * d
        nplus += same
    nminus = nmatch - nplus

    def factor_for(nplus, nminus):
        ntrue = abs(nplus - nminus)
        nfalse = nplus + nminus - ntrue
        if nfalse > ntrue:
            return f32(1)
        return f32(3) if f32(0.1) * ntrue > nfalse else f32(2)

    sigma = f32(0) if nmatch <= 1 else f32((squares - (total / nmatch) * total) / (nmatch - 1))
    if sigma <= 0:
        return nmatch
    sigma = _sqrt(sigma)
    mode = _mode([0.0, *sorted(values)], 10, 1.0, 0.1 * float(sigma), 0.01 * float(sigma))
    factor = factor_for(nplus, nminus)
    npts, niter = nmatch, 0
    while True:
        locut = float(f32(mode - float(f32(factor * sigma))))
        hicut = float(f32(mode + float(f32(factor * sigma))))
        count = rejected = 0
        values = []
        for i in range(1, npts + 1):
            r, k = rt[i, MATCH], lt[i, MATCH]
            d, same = dif(i)
            if d < locut or d > hicut:
                total -= d
                squares -= d * d
                if same:
                    nplus -= 1
                else:
                    nminus -= 1
                rejected += 1
            else:
                values.append(d)
                count += 1
                rt[count, MATCH], lt[count, MATCH] = r, k
        npts = count
        if rejected <= 0 or npts <= 0:
            break
        niter += 1
        if niter >= maxiter:
            break
        sigma = f32(0) if npts <= 1 else f32((squares - (total / npts) * total) / (npts - 1))
        if sigma <= 0:
            break
        sigma = _sqrt(sigma)
        mode = _mode([0.0, *sorted(values)], 10, 1.0, 0.1 * float(sigma),
                     0.01 * float(sigma))
        factor = factor_for(nplus, nminus)
    if npts <= 0:
        return 0
    keep_same = nplus > nminus
    count = 0
    for i in range(1, npts + 1):
        r, k = rt[i, MATCH], lt[i, MATCH]
        if (rt[r, CC] == lt[k, CC]) == keep_same:
            count += 1
            rt[count, MATCH], lt[count, MATCH] = r, k
    return count


def _vote(ref: Triangles, inp: Triangles, nrefstars: int, nliststars: int,
          nmatch: int) -> int:
    """rg_tvote: every matched triangle votes for its three vertex pairs; a
    reference star is matched to the input star with the most votes, when
    that is unique, more than half the most any pair got, more than one
    (unless there was a single triangle), and not taken already."""
    rt, lt = ref.tri, inp.tri
    votes: dict[int, dict[int, int]] = {}
    for i in range(1, nmatch + 1):
        r, k = rt[i, MATCH], lt[i, MATCH]
        for col in (X1, X2, X3):
            line = votes.setdefault(int(rt[r, col]), {})
            star = int(lt[k, col])
            line[star] = (line.get(star, 0) + 1) & 0xFFFF
    maxvote = max((v for line in votes.values() for v in line.values()), default=0)
    if maxvote <= 0:
        return 0
    half = maxvote // 2
    taken: set[int] = set()
    ninter = 0
    for j in range(1, nrefstars + 1):
        found = votes.get(j)
        if not found:
            continue
        line = found
        ranked = sorted(line.values(), reverse=True)
        top = ranked[0]
        second = ranked[1] if len(ranked) > 1 else 0
        if top <= 0 or top == second or top <= half:
            continue
        vp = max(line, key=lambda star: line[star])
        if vp in taken:
            continue
        if top == 1 and (maxvote > 1 or nmatch > 1):
            continue
        ninter += 1
        rt[ninter, MATCH], lt[ninter, MATCH] = j, vp
        taken.add(vp)
    return ninter


def match_triangles(xref, yref, refindex, nref, ntref, xin, yin, listindex, nin,
                    ntin, reftri: Triangles, intri: Triangles, nrmax, ninmax,
                    tolerance, ratio, nreject) -> int:
    """rg_match: merge, reject, vote -- and, when not every star is matched,
    once more on the matched stars alone, keeping the result only if it
    does not shrink."""
    def merge():
        return (_merge(reftri, intri) if reftri.n < intri.n else _merge(intri, reftri))

    nmerge = merge()
    if nmerge <= 0:
        return 0
    nkeep = _reject(reftri, intri, nmerge, nreject)
    if nkeep <= 0:
        return 0
    nmatch = _vote(reftri, intri, ntref, ntin, nkeep)
    if nmatch <= 0 or (nmatch <= 3 and nkeep < nmerge):
        return 0
    if 2 < nmatch < min(nref, nin):
        rindex = [0, *[int(reftri.tri[i, MATCH]) for i in range(1, nmatch + 1)]]
        lindex = [0, *[int(intri.tri[i, MATCH]) for i in range(1, nmatch + 1)]]
        triangles(xref, yref, rindex, nmatch, nrmax, ntref, tolerance, ratio, reftri)
        triangles(xin, yin, lindex, nmatch, ninmax, ntin, tolerance, ratio, intri)
        nmerge = merge()
        if nmerge > 0:
            nkeep = _reject(reftri, intri, nmerge, nreject)
        if nkeep > 0:
            ncheck = _vote(reftri, intri, ntref, ntin, nkeep)
            if ncheck <= 3 and nkeep < nmerge:
                ncheck = 0
        else:
            ncheck = 0
        nmatch = 0 if ncheck < nmatch else ncheck
    return nmatch


# --------------------------------------------------------------- the task


@dataclass
class Result:
    pairs: list[tuple[int, int]]          #: (reference, input) 1-based indices
    initial: np.ndarray                    #: the initial transformation
    matched: np.ndarray | None = None      #: from the matched triangles
    ntie: int = 0
    ties: list = field(default_factory=list)
    message: str = ""
    columns: bool = False                  #: rg_pxycolumns was called


def xyxymatch(xref, yref, xin, yin, tolerance=3.0, matching="triangles",
              separation=9.0, nmatch=30, ratio=10.0, nreject=10,
              coeff: np.ndarray | None = None, ties=None) -> Result:
    """t_xyxymatch for one input list against one reference list (both
    1-based with padding). `coeff` is the initial transformation
    (make_coeff); `ties` the reference and input tie points, as read from
    a refpoints file, which override it when they give one."""
    ntref, ntin = len(xref) - 1, len(xin) - 1
    if coeff is None:
        coeff = make_coeff()
    base = coeff                           # what rg_lmkcoeff gives, ties aside
    ntie = 0
    tie_list = []
    if ties:
        (rx, ry), (lx, ly) = ties
        ntie = min(len(rx), len(lx))
        tie_list = list(zip(rx[:ntie], ry[:ntie], lx[:ntie], ly[:ntie], strict=True))
        solved = tie_coeff(rx[:ntie], ry[:ntie], lx[:ntie], ly[:ntie]) if ntie else None
        if solved is not None:
            coeff = solved
    result = Result([], coeff, ntie=ntie, ties=tie_list)

    refindex, nref = sort_unique(xref, yref, ntref, separation)
    if matching != "triangles":
        nreftri = nref
        reftri = None
    elif nref > 2:
        nrmax = ncombinations(min(nref, nmatch))
        reftri = triangles(xref, yref, refindex, nref, nrmax, nmatch, tolerance, ratio)
        nreftri = reftri.n
    else:
        nreftri, reftri = 0, None

    if ntref <= 0:
        result.message = "The reference coordinate list is empty"
        return result
    if ntin <= 0:
        result.message = "The input coordinate list is empty"
        return result
    if nreftri <= 0:
        result.message = "No valid reference triangles can be defined"
        return result

    xt, yt = apply(coeff, xin, yin)
    listindex, nlist = sort_unique(xt, yt, ntin, separation)
    if matching != "triangles":
        result.columns = True
        result.pairs = intersect(xref, yref, refindex, nref, xt, yt, listindex, nlist,
                                 tolerance)
        return result
    if nlist <= 2:
        result.message = "The input coordinate list has < 3 stars"
        return result
    ninmax = ncombinations(min(max(nlist, nref), nmatch))
    intri = triangles(xt, yt, listindex, nlist, ninmax, nmatch, tolerance, ratio)
    ninter = 0
    if intri.n <= 0:
        result.message = "No valid input triangles can be defined"
    else:
        assert reftri is not None
        ninter = match_triangles(xref, yref, refindex, nref, ntref, xt, yt, listindex,
                                 nlist, ntin, reftri, intri, reftri.tri.shape[0] - 1,
                                 ninmax, tolerance, ratio, nreject)
    result.columns = True
    if nref <= nmatch and nlist <= nmatch:
        assert reftri is not None
        result.pairs = [(int(reftri.tri[i, MATCH]), int(intri.tri[i, MATCH]))
                        for i in range(1, ninter + 1)]
        return result
    matched = None
    if ninter > 0:
        assert reftri is not None
        r = [int(reftri.tri[i, MATCH]) for i in range(1, ninter + 1)]
        k = [int(intri.tri[i, MATCH]) for i in range(1, ninter + 1)]
        matched = tie_coeff(xref[r], yref[r], xin[k], yin[k])
    result.matched = matched if matched is not None else base
    xt, yt = apply(result.matched, xin, yin)
    listindex, nlist = sort_unique(xt, yt, ntin, separation)
    result.pairs = intersect(xref, yref, refindex, nref, xt, yt, listindex, nlist,
                             tolerance)
    return result


def format_output(result: Result, xref, yref, rlines, xin, yin, ilines, input_name,
                  reference_name, xformat="%13.3f", yformat="%13.3f") -> str:
    """The output file, as t_xyxymatch writes it."""
    out = [f"\n# Input: {input_name}  Reference: {reference_name}  "
           f"Number of tie points: {result.ntie}"]
    for i, (rx, ry, lx, ly) in enumerate(result.ties, 1):
        out.append(f"#     tie point: {i:3d}  ref: {rx:9.3f} {ry:9.3f}  "
                   f"input: {lx:9.3f} {ly:9.3f}")
    if result.ties:
        out.append("#")
    out += _coeff_lines("Initial linear transformation", result.initial)
    if result.matched is not None:
        out += _coeff_lines("Matched triangles transformation", result.matched)
    if result.columns:
        out += ["# Column definitions", "#    Column 1: X reference coordinate",
                "#    Column 2: Y reference coordinate",
                "#    Column 3: X input coordinate", "#    Column 4: Y input coordinate",
                "#    Column 5: Reference line number", "#    Column 6: Input line number",
                ""]
    xf, yf = xformat or "%13.7g", yformat or "%13.7g"
    for r, k in result.pairs:
        out.append(f"{xf % xref[r]} {yf % yref[r]}  {xf % xin[k]} {yf % yin[k]}  "
                   f"{rlines[r]:5d} {ilines[k]:5d}")
    return "\n".join(out) + "\n"


def _g(value, width: int = 10) -> str:
    """SPP's %10g: seven significant figures, fewer when they do not fit the
    width, a whole number with its point ('1.'), and no negative zero."""
    v = float(value) + 0.0
    text = ""
    for digits in range(7, 0, -1):
        text = f"{v:.{digits}g}"
        if "e" not in text and "." not in text and text not in ("inf", "-inf", "nan"):
            text += "."
        if len(text) <= width:
            break
    return f"{text:>{width}s}"


def _coeff_lines(title: str, c: np.ndarray) -> list[str]:
    xmag, ymag, xrot, yrot = geometry(c)
    return [f"# {title}",
            f"#     xref[tie] = {_g(c[3])} + {_g(c[1])} * x[tie] + {_g(c[2])} * y[tie]",
            f"#     yref[tie] = {_g(c[6])} + {_g(c[4])} * x[tie] + {_g(c[5])} * y[tie]",
            f"# dx: {float(c[3]):0.2f} dy: {float(c[6]):0.2f} xmag: {xmag:0.3f} "
            f"ymag: {ymag:0.3f} xrot: {xrot:0.1f} yrot: {yrot:0.1f}",
            "#"]
