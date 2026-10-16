"""IRAF's math/llsq: Lawson and Hanson's HFTI and H12, as IRAF builds them.

HFTI solves a least-squares system by Householder transformations with
column pivoting; imexamine's j and k keys (mr_invert in iejimexam.x) solve
their Levenberg-Marquardt steps with it. The Fortran is single precision
with double-precision accumulators; after f2c -R, a REAL expression turns
double from a call to sqrt on and is rounded when stored. That is the
arithmetic here.

Arrays are stored as IRAF passes them: flat, column-major, 1-based through
the offsets below.

From "Solving Least Squares Problems", C. L. Lawson and R. J. Hanson
(1974), as distributed in iraf-community 2.18 math/llsq; the notice IRAF
is distributed under is in LICENSES/IRAF-COPYRIGHT.
"""

from __future__ import annotations

import math

import numpy as np

f32 = np.float32


def h12(mode: int, lpivot: int, l1: int, m: int, u: np.ndarray, uoff: int, iue: int,
        up: list, c: np.ndarray, coff: int, ice: int, icv: int, ncv: int) -> None:
    """H12: construct (mode 1) and/or apply a Householder transformation.

    u(1, j) is u[uoff + (j - 1) iue]; c(i) is c[coff + i - 1]; `up` is a
    one-element list (the Fortran argument is written to)."""
    if lpivot <= 0 or lpivot >= l1 or l1 > m:
        return

    def U(j):
        return uoff + (j - 1) * iue

    cl = abs(u[U(lpivot)])
    if mode != 2:
        for j in range(l1, m + 1):
            cl = max(abs(u[U(j)]), cl)
        if cl <= 0:
            return
        clinv = f32(f32(1.0) / cl)
        sm = (float(u[U(lpivot)]) * float(clinv)) ** 2
        for j in range(l1, m + 1):
            sm += (float(u[U(j)]) * float(clinv)) ** 2
        sm1 = f32(sm)
        cl = f32(float(cl) * math.sqrt(float(sm1)))
        if u[U(lpivot)] > 0:
            cl = -cl
        up[0] = f32(u[U(lpivot)] - cl)
        u[U(lpivot)] = cl
    elif cl <= 0:
        return
    if ncv <= 0:
        return
    b = float(up[0]) * float(u[U(lpivot)])
    if b >= 0:
        return
    b = 1.0 / b
    i2 = 1 - icv + ice * (lpivot - 1)
    incr = ice * (l1 - lpivot)
    for _ in range(ncv):
        i2 += icv
        i3 = i2 + incr
        i4 = i3
        sm = float(c[coff + i2 - 1]) * float(up[0])
        for i in range(l1, m + 1):
            sm += float(c[coff + i3 - 1]) * float(u[U(i)])
            i3 += ice
        if sm == 0:
            continue
        sm *= b
        c[coff + i2 - 1] = f32(float(c[coff + i2 - 1]) + sm * float(up[0]))
        for i in range(l1, m + 1):
            c[coff + i4 - 1] = f32(float(c[coff + i4 - 1]) + sm * float(u[U(i)]))
            i4 += ice


def hfti(a: np.ndarray, mda: int, m: int, n: int, b: np.ndarray, mdb: int, nb: int,
         tau: float) -> int:
    """HFTI: a (mda x n, column-major, flat) and b (mdb x nb) are
    overwritten; b's first n rows become the solution. Returns the rank."""
    a = a  # modified in place
    h = np.zeros(n + 1, dtype=f32)
    g = np.zeros(n + 1, dtype=f32)
    ip = [0] * (n + 1)
    factor = f32(0.001)
    hmax = f32(0.0)

    def A(i, j):
        return (i - 1) + (j - 1) * mda

    def B(i, j):
        return (i - 1) + (j - 1) * mdb

    k = 0
    ldiag = min(m, n)
    if ldiag <= 0:
        return 0
    for j in range(1, ldiag + 1):
        recompute = j == 1
        if not recompute:
            lmax = j
            for col in range(j, n + 1):
                h[col] = f32(h[col] - f32(a[A(j - 1, col)] * a[A(j - 1, col)]))
                if h[col] > h[lmax]:
                    lmax = col
            recompute = bool(f32(f32(hmax + f32(factor * h[lmax])) - hmax) <= 0)
        if recompute:
            lmax = j
            for col in range(j, n + 1):
                h[col] = 0
                for i in range(j, m + 1):
                    h[col] = f32(h[col] + f32(a[A(i, col)] * a[A(i, col)]))
                if h[col] > h[lmax]:
                    lmax = col
            hmax = h[lmax]
        ip[j] = lmax
        if ip[j] != j:
            for i in range(1, m + 1):
                a[A(i, j)], a[A(i, lmax)] = a[A(i, lmax)], a[A(i, j)]
            h[lmax] = h[j]
        hj = [h[j]]
        h12(1, j, j + 1, m, a, A(1, j), 1, hj, a, A(1, j + 1), 1, mda, n - j)
        h12(2, j, j + 1, m, a, A(1, j), 1, hj, b, 0, 1, mdb, nb)
        h[j] = hj[0]
    k = ldiag
    for j in range(1, ldiag + 1):
        if abs(a[A(j, j)]) <= tau:
            k = j - 1
            break
    kp1 = k + 1
    if k <= 0:
        for jb in range(1, nb + 1):
            for i in range(1, n + 1):
                b[B(i, jb)] = 0
        return k
    if k != n:
        for ii in range(1, k + 1):
            i = kp1 - ii
            gi = [g[i]]
            h12(1, i, kp1, n, a, A(i, 1), mda, gi, a, 0, mda, 1, i - 1)
            g[i] = gi[0]
    for jb in range(1, nb + 1):
        for ll in range(1, k + 1):
            sm = 0.0
            i = kp1 - ll
            for j in range(i + 1, k + 1):
                sm += float(a[A(i, j)]) * float(b[B(j, jb)])
            sm1 = f32(sm)
            b[B(i, jb)] = f32(f32(b[B(i, jb)] - sm1) / a[A(i, i)])
        if k != n:
            for j in range(kp1, n + 1):
                b[B(j, jb)] = 0
            for i in range(1, k + 1):
                gi = [g[i]]
                h12(2, i, kp1, n, a, A(i, 1), mda, gi, b, B(1, jb), 1, mdb, 1)
        for jj in range(1, ldiag + 1):
            j = ldiag + 1 - jj
            if ip[j] != j:
                lp = ip[j]
                b[B(lp, jb)], b[B(j, jb)] = b[B(j, jb)], b[B(lp, jb)]
    return k
