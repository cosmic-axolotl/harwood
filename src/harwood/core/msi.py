"""IRAF's 2-D image interpolator, math/iminterp's msi: msifit and msivector for the
types illumination offers -- nearest, linear, poly3, poly5 and spline3.

The coefficient array is IRAF's: one flat vector, 1-based, the data placed at
MSI_FSTPNT with room around them, and extended by boundary projection (a row's
extra points 2 a[edge] - a[edge -+ k], then the columns the same way), or for
spline3 the B-spline coefficients with natural end conditions. Everything is
single precision, as IRAF computes it, and the evaluators are ii_bieval.x's
loops: the central differences of poly3 and poly5, cached while the point stays
in the same pixel, and the cubic B-splines of spline3.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .jit import jit

f32 = np.float32
TYPES = ("nearest", "linear", "poly3", "poly5", "spline3")
_MIN = {"nearest": 1, "linear": 2, "poly3": 4, "poly5": 6, "spline3": 4}
_PAD = {"nearest": 0, "linear": 1, "poly3": 3, "poly5": 5, "spline3": 3}


class MsiError(ValueError):
    """What msifit stops on, in its words."""


@dataclass
class Msi:
    kind: str
    nxcoeff: int
    nycoeff: int
    fstpnt: int
    coeff: np.ndarray            #: 1-based: coeff[0] is unused


def msifit(kind: str, data: np.ndarray) -> Msi:
    """msifit of data[ny, nx] (IRAF's datain[nx, ny])."""
    data = np.asarray(data, dtype=f32)
    nypix, nxpix = data.shape
    if nxpix < _MIN[kind] or nypix < _MIN[kind]:
        raise MsiError("MSIFIT: Too few data points.")
    pad = _PAD[kind]
    nxc, nyc = nxpix + pad, nypix + pad
    fstpnt = {"nearest": 0, "linear": 0, "poly3": nxc + 1, "poly5": 2 * nxc + 2,
              "spline3": nxc + 1}[kind]
    c = np.zeros(nxc * nyc + 1, dtype=f32)
    fptr = fstpnt                    # COEFF(fptr + 1) is the first data point
    rptr = fptr
    for j in range(nypix):
        c[rptr + 1:rptr + 1 + nxpix] = data[j]
        rptr += nxc

    def row(first: int) -> slice:
        return slice(first, first + nxc)

    def awsur(a: int, b: int, out: int) -> None:
        """c = a * 2 + b * -1, a row at a time: rows given by their first element."""
        c[row(out)] = (c[row(a)] * f32(2.0) + c[row(b)] * f32(-1.0)).astype(f32)

    two = f32(2.0)
    if kind == "linear":
        rptr = fptr + nxpix
        for _ in range(nypix):
            c[rptr + 1] = f32(two * c[rptr] - c[rptr - 1])
            rptr += nxc
        last = 1 + (nyc - 1) * nxc
        awsur(last - nxc, last - 2 * nxc, last)
    elif kind == "poly3":
        rptr, nptr = fptr, fptr + nxpix
        for _ in range(nypix):
            c[rptr] = f32(two * c[rptr + 1] - c[rptr + 2])
            c[nptr + 1] = f32(two * c[nptr] - c[nptr - 1])
            c[nptr + 2] = f32(two * c[nptr] - c[nptr - 2])
            rptr += nxc
            nptr += nxc
        first = [1 + i * nxc for i in range(3)]
        awsur(first[1], first[2], first[0])
        last = [1 + (nyc - 1) * nxc - i * nxc for i in range(5)]
        awsur(last[2], last[3], last[1])
        awsur(last[2], last[4], last[0])
    elif kind == "poly5":
        rptr, nptr = fptr, fptr + nxpix
        for _ in range(nypix):
            c[rptr - 1] = f32(two * c[rptr + 1] - c[rptr + 3])
            c[rptr] = f32(two * c[rptr + 1] - c[rptr + 2])
            c[nptr + 1] = f32(two * c[nptr] - c[nptr - 1])
            c[nptr + 2] = f32(two * c[nptr] - c[nptr - 2])
            c[nptr + 3] = f32(two * c[nptr] - c[nptr - 3])
            rptr += nxc
            nptr += nxc
        first = [1 + i * nxc for i in range(5)]
        awsur(first[2], first[4], first[0])
        awsur(first[2], first[3], first[1])
        last = [1 + (nyc - 1) * nxc - i * nxc for i in range(7)]
        awsur(last[3], last[6], last[0])
        awsur(last[3], last[5], last[1])
        awsur(last[3], last[4], last[2])
    elif kind == "spline3":
        tmp = np.zeros(nxc * nyc + 1, dtype=f32)
        _spline2d(c, tmp, nxpix, nyc, nxc, nyc)
        _spline2d(tmp, c, nypix, nxc, nyc, nxc)
    return Msi(kind, nxc, nyc, fstpnt, c)


@jit
def _spline2d(data, coeff, nxpix, nvectors, len_data, len_coeff):
    """ii_spline2d: univariate B-spline coefficients of each row of data, natural
    end conditions, written transposed into coeff. Both 1-based and flat:
    data[i, j] is data[i + (j - 1) * len_data]."""
    diag = np.zeros(nxpix + 2, dtype=np.float32)
    diag[0] = np.float32(-2.0)
    diag[1] = np.float32(0.0)
    for i in range(3, nxpix + 2):
        diag[i - 1] = np.float32(1.0) / (np.float32(4.0) - diag[i - 2])
    six = np.float32(6.0)
    for j in range(1, nvectors + 1):
        for i in range(1, nxpix + 3):
            coeff[j + (i - 1) * len_coeff] = data[i + (j - 1) * len_data]
        c1 = j
        c2 = j + len_coeff
        coeff[c1] = coeff[c1] / six
        coeff[c2] = (coeff[c2] - coeff[c1]) / six
        for i in range(3, nxpix + 2):
            at = j + (i - 1) * len_coeff
            coeff[at] = diag[i - 1] * (coeff[at] - coeff[at - len_coeff])
        top = j + (nxpix + 1) * len_coeff
        n1 = j + nxpix * len_coeff
        n0 = j + (nxpix - 1) * len_coeff
        coeff[top] = ((diag[nxpix - 1] + np.float32(2.0)) * coeff[n1] - coeff[n0]
                      + coeff[top] / six) / (np.float32(1.0) + diag[nxpix]
                                             * (diag[nxpix - 1] + np.float32(2.0)))
        for i in range(nxpix + 1, 2, -1):
            at = j + (i - 1) * len_coeff
            coeff[at] = coeff[at] - diag[i - 1] * coeff[at + len_coeff]
        coeff[c1] = coeff[c1] + np.float32(2.0) * coeff[c2] - coeff[j + 2 * len_coeff]


def msivector(msi: Msi, x, y) -> np.ndarray:
    """msivector at the points (x, y), in the data's own 1-based coordinates."""
    x = np.asarray(x, dtype=f32)
    y = np.asarray(y, dtype=f32)
    fn = {"nearest": _nearest, "linear": _linear, "poly3": _poly3, "poly5": _poly5,
          "spline3": _bspline3}[msi.kind]
    return fn(msi.coeff, msi.fstpnt, msi.nxcoeff, x, y)


@jit
def _nearest(coeff, first, lenc, x, y):
    out = np.empty(x.size, dtype=np.float32)
    for i in range(x.size):
        nx = int(x[i] + np.float32(0.5))
        ny = int(y[i] + np.float32(0.5))
        out[i] = coeff[first + (ny - 1) * lenc + nx]
    return out


@jit
def _linear(coeff, first, lenc, x, y):
    out = np.empty(x.size, dtype=np.float32)
    one = np.float32(1.0)
    for i in range(x.size):
        nx = int(x[i])
        ny = int(y[i])
        sx = x[i] - np.float32(nx)
        tx = one - sx
        sy = y[i] - np.float32(ny)
        ty = one - sy
        k = first + (ny - 1) * lenc + nx
        out[i] = (tx * ty * coeff[k] + sx * ty * coeff[k + 1]
                  + sy * tx * coeff[k + lenc] + sx * sy * coeff[k + lenc + 1])
    return out


@jit
def _poly3(coeff, first, lenc, x, y):
    out = np.empty(x.size, dtype=np.float32)
    one, two = np.float32(1.0), np.float32(2.0)
    sixth = np.float32(1.0) / np.float32(6.0)
    cd20 = np.zeros(4, dtype=np.float32)
    cd21 = np.zeros(4, dtype=np.float32)
    zt = np.zeros(4, dtype=np.float32)
    nxold = -1
    nyold = -1
    for i in range(x.size):
        nx = int(x[i])
        sx = x[i] - np.float32(nx)
        tx = one - sx
        sx2m1 = sx * sx - one
        tx2m1 = tx * tx - one
        ny = int(y[i])
        sy = y[i] - np.float32(ny)
        ty = one - sy
        first_row = first + (ny - 2) * lenc + nx
        index = first_row
        if nx != nxold or ny != nyold:
            for j in range(4):
                cd20[j] = sixth * (coeff[index + 1] - two * coeff[index] + coeff[index - 1])
                cd21[j] = sixth * (coeff[index + 2] - two * coeff[index + 1] + coeff[index])
                index += lenc
        index = first_row
        for j in range(4):
            zt[j] = (sx * (coeff[index + 1] + sx2m1 * cd21[j])
                     + tx * (coeff[index] + tx2m1 * cd20[j]))
            index += lenc
        cd20y = sixth * (zt[2] - two * zt[1] + zt[0])
        cd21y = sixth * (zt[3] - two * zt[2] + zt[1])
        out[i] = (sy * (zt[2] + (sy * sy - one) * cd21y)
                  + ty * (zt[1] + (ty * ty - one) * cd20y))
        nxold = nx
        nyold = ny
    return out


@jit
def _poly5(coeff, first, lenc, x, y):
    out = np.empty(x.size, dtype=np.float32)
    one, two, four, six = (np.float32(1.0), np.float32(2.0), np.float32(4.0),
                           np.float32(6.0))
    sixth = np.float32(1.0) / np.float32(6.0)
    v120 = np.float32(1.0) / np.float32(120.0)
    cd20 = np.zeros(6, dtype=np.float32)
    cd21 = np.zeros(6, dtype=np.float32)
    cd40 = np.zeros(6, dtype=np.float32)
    cd41 = np.zeros(6, dtype=np.float32)
    zt = np.zeros(6, dtype=np.float32)
    nxold = -1
    nyold = -1
    for i in range(x.size):
        nx = int(x[i])
        sx = x[i] - np.float32(nx)
        sx2 = sx * sx
        sx2m1 = sx2 - one
        sx2m4 = sx2 - four
        tx = one - sx
        tx2 = tx * tx
        tx2m1 = tx2 - one
        tx2m4 = tx2 - four
        ny = int(y[i])
        sy = y[i] - np.float32(ny)
        sy2 = sy * sy
        ty = one - sy
        ty2 = ty * ty
        first_row = first + (ny - 3) * lenc + nx
        index = first_row
        if nx != nxold or ny != nyold:
            for j in range(6):
                cd20[j] = sixth * (coeff[index + 1] - two * coeff[index] + coeff[index - 1])
                cd21[j] = sixth * (coeff[index + 2] - two * coeff[index + 1] + coeff[index])
                cd40[j] = v120 * (coeff[index - 2] - four * coeff[index - 1]
                                  + six * coeff[index] - four * coeff[index + 1]
                                  + coeff[index + 2])
                cd41[j] = v120 * (coeff[index - 1] - four * coeff[index]
                                  + six * coeff[index + 1] - four * coeff[index + 2]
                                  + coeff[index + 3])
                index += lenc
        index = first_row
        for j in range(6):
            zt[j] = (sx * (coeff[index + 1] + sx2m1 * (cd21[j] + sx2m4 * cd41[j]))
                     + tx * (coeff[index] + tx2m1 * (cd20[j] + tx2m4 * cd40[j])))
            index += lenc
        cd20y = sixth * (zt[3] - two * zt[2] + zt[1])
        cd21y = sixth * (zt[4] - two * zt[3] + zt[2])
        cd40y = v120 * (zt[0] - four * zt[1] + six * zt[2] - four * zt[3] + zt[4])
        cd41y = v120 * (zt[1] - four * zt[2] + six * zt[3] - four * zt[4] + zt[5])
        out[i] = (sy * (zt[3] + (sy2 - one) * (cd21y + (sy2 - four) * cd41y))
                  + ty * (zt[2] + (ty2 - one) * (cd20y + (ty2 - four) * cd40y)))
        nxold = nx
        nyold = ny
    return out


@jit
def _bspline3(coeff, first, lenc, x, y):
    out = np.empty(x.size, dtype=np.float32)
    one, three = np.float32(1.0), np.float32(3.0)
    bx = np.zeros(4, dtype=np.float32)
    by = np.zeros(4, dtype=np.float32)
    for i in range(x.size):
        nx = int(x[i])
        sx = x[i] - np.float32(nx)
        tx = one - sx
        ny = int(y[i])
        sy = y[i] - np.float32(ny)
        ty = one - sy
        bx[0] = tx * tx * tx
        bx[1] = one + tx * (three + tx * (three - three * tx))
        bx[2] = one + sx * (three + sx * (three - three * sx))
        bx[3] = sx * sx * sx
        by[0] = ty * ty * ty
        by[1] = one + ty * (three + ty * (three - three * ty))
        by[2] = one + sy * (three + sy * (three - three * sy))
        by[3] = sy * sy * sy
        first_row = first + (ny - 2) * lenc + nx
        accum = np.float32(0.0)
        index = first_row
        for j in range(4):
            s = (coeff[index - 1] * bx[0] + coeff[index] * bx[1]
                 + coeff[index + 1] * bx[2] + coeff[index + 2] * bx[3])
            accum = accum + s * by[j]
            index += lenc
        out[i] = accum
    return out
