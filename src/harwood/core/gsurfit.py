"""IRAF's 2-D surface fitting, math/gsurfit, in double precision (dgsurfit): what
fitcoords fits and transform evaluates.

A surface is a sum of products of basis functions in x and y -- Chebyshev or
Legendre polynomials of x and y normalized to -1..1 over xmin..xmax and
ymin..ymax, or plain powers -- with no cross terms (xorder + yorder - 1
coefficients), half of them or all. The fit accumulates the normal equations
point by point as dgsacpts does and solves them with gsurfit's Cholesky
factorization; the evaluators are gs_feval's, gs_f1deval's (a surface of order
1 in x or y is evaluated as a curve), gseval's for a single point (its Legendre
recurrence is written differently from the vector one, and gives other last
digits) and gsder's for derivatives. Every loop runs in IRAF's order, so that
the numbers are IRAF's to the last bit.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .jit import jit

CHEBYSHEV, LEGENDRE, POLYNOMIAL = 1, 2, 3
XNONE, XFULL, XHALF = 0, 1, 2
NAMES = {"chebyshev": CHEBYSHEV, "legendre": LEGENDRE, "polynomial": POLYNOMIAL}
OK, SINGULAR, NO_DEG_FREEDOM = 0, 1, 2
MAX_DOUBLE = float(np.finfo(np.float64).max)


class GsurfitError(ValueError):
    """What gsurfit stops on, in its words."""


def _ncoeff(xorder: int, yorder: int, xterms: int) -> int:
    if xterms == XNONE:
        return xorder + yorder - 1
    if xterms == XHALF:
        order = min(xorder, yorder)
        return xorder * yorder - order * (order - 1) // 2
    return xorder * yorder


@dataclass
class Surface:
    type: int
    xorder: int
    yorder: int
    xterms: int
    xmin: float
    xmax: float
    ymin: float
    ymax: float
    coeff: np.ndarray = field(default=None)
    matrix: np.ndarray = field(default=None, repr=False)
    vector: np.ndarray = field(default=None, repr=False)
    npts: int = 0

    @property
    def nxcoeff(self) -> int:
        return self.xorder

    @property
    def nycoeff(self) -> int:
        return self.yorder

    @property
    def ncoeff(self) -> int:
        return _ncoeff(self.xorder, self.yorder, self.xterms)

    @property
    def xrange(self) -> float:
        return 1.0 if self.type == POLYNOMIAL else 2.0 / (self.xmax - self.xmin)

    @property
    def xmaxmin(self) -> float:
        return 0.0 if self.type == POLYNOMIAL else -(self.xmax + self.xmin) / 2.0

    @property
    def yrange(self) -> float:
        return 1.0 if self.type == POLYNOMIAL else 2.0 / (self.ymax - self.ymin)

    @property
    def ymaxmin(self) -> float:
        return 0.0 if self.type == POLYNOMIAL else -(self.ymax + self.ymin) / 2.0


def init(kind, xorder: int, yorder: int, xterms: int, xmin: float, xmax: float,
         ymin: float, ymax: float) -> Surface:
    """dgsinit."""
    kind = NAMES.get(kind, kind) if isinstance(kind, str) else int(kind)
    if xorder < 1 or yorder < 1:
        raise GsurfitError("GSINIT: Illegal order.")
    if xmax <= xmin:
        raise GsurfitError("GSINIT: xmax <= xmin.")
    if ymax <= ymin:
        raise GsurfitError("GSINIT: ymax <= ymin.")
    if kind not in (CHEBYSHEV, LEGENDRE, POLYNOMIAL):
        raise GsurfitError("GSINIT: Unknown surface type.")
    sf = Surface(kind, int(xorder), int(yorder), int(xterms), float(xmin), float(xmax),
                 float(ymin), float(ymax))
    n = sf.ncoeff
    sf.matrix = np.zeros((n, n), dtype=np.float64)      # [row n, band j]
    sf.vector = np.zeros(n, dtype=np.float64)
    sf.coeff = np.zeros(n, dtype=np.float64)
    return sf


# ------------------------------------------------------------------ bases

def _bases(kind: int, x: np.ndarray, order: int, k1: float, k2: float) -> np.ndarray:
    """gs_bpol, gs_bcheb, gs_bleg: the basis functions of x, one row each."""
    x = np.asarray(x, dtype=np.float64)
    b = np.zeros((order, x.size), dtype=np.float64)
    for k in range(1, order + 1):
        if k == 1:
            b[0] = 1.0
        elif k == 2:
            b[1] = x if kind == POLYNOMIAL else (x + k1) * k2
        elif kind == POLYNOMIAL:
            b[k - 1] = b[k - 2] * x
        elif kind == CHEBYSHEV:
            b[k - 1] = (b[1] * b[k - 2]) * 2.0 - b[k - 3]
        else:
            ri = float(k)
            ri1 = (2.0 * ri - 3.0) / (ri - 1.0)
            ri2 = -(ri - 2.0) / (ri - 1.0)
            b[k - 1] = (b[1] * b[k - 2]) * ri1 + b[k - 3] * ri2
    return b


def _bases_one(kind: int, x: float, order: int, k1: float, k2: float) -> list[float]:
    """gs_b1pol, gs_b1cheb, gs_b1leg: the basis functions of one x."""
    b = [1.0]
    if order == 1:
        return b
    xn = x if kind == POLYNOMIAL else (x + k1) * k2
    b.append(xn)
    for i in range(3, order + 1):
        if kind == POLYNOMIAL:
            b.append(xn * b[i - 2])
        elif kind == CHEBYSHEV:
            b.append(2.0 * xn * b[i - 2] - b[i - 3])
        else:
            ri = float(i)
            b.append(((2.0 * ri - 3.0) * xn * b[i - 2] - (ri - 2.0) * b[i - 3]) / (ri - 1.0))
    return b


# -------------------------------------------------------------------- fit

def fit(sf: Surface, x, y, z, w=None) -> int:
    """dgsfit with user weights (all ones when none are given): gszero, gsacpts,
    gssolve. The status, OK, SINGULAR or NO_DEG_FREEDOM."""
    zero(sf)
    accumulate(sf, x, y, z, w)
    return solve(sf)


def zero(sf: Surface) -> None:
    sf.matrix[:] = 0.0
    sf.vector[:] = 0.0
    sf.coeff[:] = 0.0
    sf.npts = 0


def accumulate(sf: Surface, x, y, z, w=None) -> None:
    """dgsacpts: the normal equations, point by point."""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    z = np.asarray(z, dtype=np.float64)
    w = np.ones(x.size) if w is None else np.asarray(w, dtype=np.float64)
    sf.npts += x.size
    xb = _bases(sf.type, x, sf.xorder, sf.xmaxmin, sf.xrange)
    yb = _bases(sf.type, y, sf.yorder, sf.ymaxmin, sf.yrange)
    _acpts(sf.matrix, sf.vector, xb, yb, z, w, sf.xorder, sf.yorder, sf.xterms,
           sf.ncoeff)


@jit
def _acpts(matrix, vector, xb, yb, z, w, xorder0, yorder, xterms, ncoeff):
    npts = z.size
    maxorder = max(xorder0 + 1, yorder + 1)
    xorder = xorder0
    ntimes = 0
    vz = 0                       # vzptr + 1, 0-based
    row = 0                      # mzptr, the matrix row
    byw = np.empty(npts)
    bw = np.empty(npts)
    for ly in range(1, yorder + 1):
        for i in range(npts):
            byw[i] = w[i] * yb[ly - 1, i]
        for k in range(1, xorder + 1):
            for i in range(npts):
                bw[i] = byw[i] * xb[k - 1, i]
            dot = 0.0
            for i in range(npts):
                dot = dot + bw[i] * z[i]
            vector[vz + k - 1] = vector[vz + k - 1] + dot
            bby = ly - 1
            bbx = k - 1
            xxorder = xorder
            jj = k
            ll = ly
            ii = 0
            for _j in range(k + ntimes, ncoeff + 1):
                for i in range(npts):
                    matrix[row, ii] = matrix[row, ii] + bw[i] * xb[bbx, i] * yb[bby, i]
                if jj % xxorder == 0:
                    jj = 1
                    ll = ll + 1
                    bbx = 0
                    bby = bby + 1
                    if xterms == 0:
                        xxorder = 1
                    elif xterms == 2 and ll + xorder0 > maxorder:
                        xxorder = xxorder - 1
                else:
                    jj = jj + 1
                    bbx = bbx + 1
                ii = ii + 1
            row = row + 1
        vz = vz + xorder
        ntimes = ntimes + xorder
        if xterms == 0:
            xorder = 1
        elif xterms == 2 and ly + xorder0 + 1 > maxorder:
            xorder = xorder - 1


def solve(sf: Surface) -> int:
    """dgssolve: dgschofac and dgschoslv over the full matrix."""
    n = sf.ncoeff
    if sf.npts - n < 0:
        return NO_DEG_FREEDOM
    fac, ier = _chofac(sf.matrix, n, n)
    sf.coeff = _choslv(fac, n, n, sf.vector)
    return ier


def _chofac(matrix, nbands, nrows):
    """dgschofac; the matrix is [row n, band j] (IRAF's matrix[j, n])."""
    fac = np.zeros((nrows, nbands), dtype=np.float64)
    ier = OK
    if nrows == 1:
        if matrix[0, 0] > 0.0:
            fac[0, 0] = 1.0 / matrix[0, 0]
        return fac, ier
    fac[:, :] = matrix[:nrows, :nbands]
    for n in range(1, nrows + 1):
        if (fac[n - 1, 0] + matrix[n - 1, 0]) - matrix[n - 1, 0] <= 1000.0 / MAX_DOUBLE:
            fac[n - 1, :] = 0.0
            ier = SINGULAR
            continue
        fac[n - 1, 0] = 1.0 / fac[n - 1, 0]
        imax = min(nbands - 1, nrows - n)
        if imax < 1:
            continue
        jmax = imax
        for i in range(1, imax + 1):
            ratio = fac[n - 1, i] * fac[n - 1, 0]
            for j in range(1, jmax + 1):
                fac[n + i - 1, j - 1] = fac[n + i - 1, j - 1] - fac[n - 1, j + i - 1] * ratio
            jmax -= 1
            fac[n - 1, i] = ratio
    return fac, ier


def _choslv(fac, nbands, nrows, vector):
    coeff = np.array(vector, dtype=np.float64)
    if nrows == 1:
        coeff[0] = vector[0] * fac[0, 0]
        return coeff
    nbndm1 = nbands - 1
    for n in range(1, nrows + 1):
        jmax = min(nbndm1, nrows - n)
        for j in range(1, jmax + 1):
            coeff[j + n - 1] = coeff[j + n - 1] - fac[n - 1, j] * coeff[n - 1]
    for n in range(nrows, 0, -1):
        coeff[n - 1] = coeff[n - 1] * fac[n - 1, 0]
        jmax = min(nbndm1, nrows - n)
        for j in range(1, jmax + 1):
            coeff[n - 1] = coeff[n - 1] - fac[n - 1, j] * coeff[j + n - 1]
    return coeff


# ------------------------------------------------------------- evaluation

def _sum_terms(coeff, xb, yb, xterms, xorder, yorder, npts, derivative=False):
    """gs_evpoly/gs_evcheb/gs_evleg (and gs_der*'s): the coefficients times the
    products of the bases, a y term at a time."""
    zfit = np.zeros(npts, dtype=np.float64)
    if xterms != XNONE:
        maxorder = max(xorder + 1, yorder + 1)
        xincr = xorder
        cptr = 0
        for i in range(1, yorder + 1):
            accum = np.zeros(npts, dtype=np.float64)
            for k in range(1, xincr + 1):
                accum = accum * 1.0 + xb[k - 1] * coeff[cptr + k - 1]
            zfit = zfit + accum * yb[i - 1]
            cptr += xincr
            if xterms == XHALF and i + xorder + 1 > maxorder:
                xincr -= 1
        return zfit
    if derivative:
        zfit = (xb[0] * yb[0]) * coeff[0]
        for k in range(1, xorder):
            zfit = zfit * 1.0 + xb[k] * coeff[k]
    else:
        for k in range(1, xorder + 1):
            zfit = zfit * 1.0 + xb[k - 1] * coeff[k - 1]
    for k in range(1, yorder):
        zfit = zfit * 1.0 + yb[k] * coeff[xorder + k - 1]
    return zfit


def _curve(kind, coeff, x, order, k1, k2):
    """gs_1devpoly, gs_1devcheb, gs_1devleg."""
    x = np.asarray(x, dtype=np.float64)
    yfit = np.full(x.size, coeff[0], dtype=np.float64)
    if order == 1:
        return yfit
    if kind == POLYNOMIAL:
        yfit = x * coeff[1] + coeff[0]
        if order == 2:
            return yfit
        temp = x.copy()
        for i in range(3, order + 1):
            temp = temp * x
            yfit = yfit * 1.0 + temp * coeff[i - 1]
        return yfit
    c1 = k2 * coeff[1]
    c2 = c1 * k1 + coeff[0]
    yfit = x * c1 + c2
    if order == 2:
        return yfit
    pnm2 = np.ones(x.size, dtype=np.float64)
    sx = (x + k1) * k2
    pnm1 = sx.copy()
    if kind == CHEBYSHEV:
        sx = sx * 2.0
    for i in range(3, order + 1):
        if kind == CHEBYSHEV:
            pn = sx * pnm1 - pnm2
        else:
            ri = float(i)
            ri1 = (2.0 * ri - 3.0) / (ri - 1.0)
            ri2 = -(ri - 2.0) / (ri - 1.0)
            pn = (sx * pnm1) * ri1 + pnm2 * ri2
        if i < order:
            pnm2 = pnm1
            pnm1 = pn
        yfit = yfit + pn * coeff[i - 1]
    return yfit


def vector(sf: Surface, x, y) -> np.ndarray:
    """dgsvector."""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    c = sf.coeff
    if sf.xorder == 1:
        return _curve(sf.type, c, y, sf.yorder, sf.ymaxmin, sf.yrange)
    if sf.yorder == 1:
        return _curve(sf.type, c, x, sf.xorder, sf.xmaxmin, sf.xrange)
    if sf.type == POLYNOMIAL and sf.xorder == 2 and sf.yorder == 2 and sf.xterms == XNONE:
        return c[0] + x * c[1] + y * c[2]
    xb = _bases(sf.type, x, sf.xorder, sf.xmaxmin, sf.xrange)
    yb = _bases(sf.type, y, sf.yorder, sf.ymaxmin, sf.yrange)
    return _sum_terms(c, xb, yb, sf.xterms, sf.xorder, sf.yorder, x.size)


def evaluate(sf: Surface, x: float, y: float) -> float:
    """dgseval: one point, with the single-point bases."""
    xb = _bases_one(sf.type, float(x), sf.nxcoeff, sf.xmaxmin, sf.xrange)
    yb = _bases_one(sf.type, float(y), sf.nycoeff, sf.ymaxmin, sf.yrange)
    total = 0.0
    maxorder = max(sf.xorder + 1, sf.yorder + 1)
    xorder = sf.xorder
    ii = 0
    for i in range(1, sf.yorder + 1):
        accum = 0.0
        for k in range(1, xorder + 1):
            accum = accum + sf.coeff[ii] * xb[k - 1]
            ii += 1
        accum = accum * yb[i - 1]
        total = total + accum
        if sf.xterms == XNONE:
            xorder = 1
        elif sf.xterms == XHALF and i + sf.xorder + 1 > maxorder:
            xorder -= 1
    return total


def _derivative_bases(kind, x, order, nder, k1, k2):
    """gs_dpol, gs_dcheb, gs_dleg."""
    x = np.asarray(x, dtype=np.float64)
    n = x.size
    if nder == 0:
        return _bases(kind, x, order, k1, k2)
    if kind == POLYNOMIAL:
        b = _bases(kind, x, order, k1, k2)
        if nder == 1:
            for k in range(1, order + 1):
                b[k - 1] = b[k - 1] * float(k)
            return b
        for k in range(1, order + 1):
            fac = 1.0
            if k == 1:
                for kk in range(2, nder + 1):
                    fac = fac * float(kk)
            else:
                for kk in range(k + nder - 1, k - 1, -1):
                    fac = fac * float(kk)
            b[k - 1] = b[k - 1] * fac
        return b
    total = order + nder
    fn = np.zeros((total, n), dtype=np.float64)
    dfn = np.zeros((total, n), dtype=np.float64)
    xnorm = (x + k1) * k2
    for k in range(1, total + 1):
        if k == 1:
            fn[0] = 1.0
        elif k == 2:
            fn[1] = xnorm
        elif kind == CHEBYSHEV:
            fn[k - 1] = (xnorm * fn[k - 2]) * 2.0 - fn[k - 3]
        else:
            ri = float(k)
            ri1 = (2.0 * ri - 3.0) / (ri - 1.0)
            ri2 = -(ri - 2.0) / (ri - 1.0)
            fn[k - 1] = (xnorm * fn[k - 2]) * ri1 + fn[k - 3] * ri2
    for i in range(1, nder + 1):
        for k in range(1, total + 1):
            if k == 1:
                dfn[0] = 0.0
            elif k == 2:
                dfn[1] = 1.0 if i == 1 else 0.0
            elif kind == CHEBYSHEV:
                d = (xnorm * dfn[k - 2]) * 2.0 - dfn[k - 3]
                dfn[k - 1] = fn[k - 2] * (2.0 * float(i)) + d * 1.0
            else:
                ri = float(k)
                ri1 = (2.0 * ri - 3.0) / (ri - 1.0)
                ri2 = -(ri - 2.0) / (ri - 1.0)
                d = (xnorm * dfn[k - 2]) * ri1 + dfn[k - 3] * ri2
                dfn[k - 1] = fn[k - 2] * (ri1 * float(i)) + d * 1.0
        if i < nder:
            fn = dfn.copy()
    return dfn[nder:nder + order].copy()


def derivative(sf1: Surface, x, y, nxd: int, nyd: int) -> np.ndarray:
    """dgsder: the nxd-th derivative in x and nyd-th in y of the surface."""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if nxd < 0 or nyd < 0:
        raise GsurfitError("GSDER: Order of derivatives cannot be < 0")
    if nxd == 0 and nyd == 0:
        return vector(sf1, x, y)
    nxder = min(nxd, sf1.nxcoeff)
    nyder = min(nyd, sf1.nycoeff)
    nx1, ny1 = sf1.nxcoeff, sf1.nycoeff
    xt = sf1.xterms
    if xt == XNONE:
        if nxder > 0 and nyder > 0:
            nx2 = ny2 = 1
            n2 = 1
        elif nxder > 0:
            nx2, ny2 = max(1, nx1 - nxder), 1
            n2 = nx2
        else:
            nx2, ny2 = 1, max(1, ny1 - nyder)
            n2 = ny2
        xo2, yo2 = nx2, ny2
    elif xt == XHALF:
        if nxder >= nx1 or nyder >= ny1 or nxder + nyder >= max(nx1, ny1):
            nx2 = ny2 = xo2 = yo2 = n2 = 1
        else:
            maxorder1 = max(sf1.xorder + 1, sf1.yorder + 1)
            nx2 = xo2 = max(1, min(maxorder1 - 1 - nyder - nxder, nx1 - nxder))
            ny2 = yo2 = max(1, min(maxorder1 - 1 - nyder - nxder, ny1 - nyder))
            order = min(xo2, yo2)
            n2 = nx2 * ny2 - order * (order - 1) // 2
    else:
        if nxder >= nx1 or nyder >= ny1:
            nx2 = ny2 = xo2 = yo2 = n2 = 1
        else:
            nx2 = max(1, nx1 - nxder)
            xo2 = max(1, sf1.xorder - nxder)
            ny2 = max(1, ny1 - nyder)
            yo2 = max(1, sf1.yorder - nyder)
            n2 = nx2 * ny2
    c1 = sf1.coeff
    c2 = np.zeros(n2, dtype=np.float64)
    if xt == XFULL:
        if not (nxder >= nx1 or nyder >= ny1):
            p2 = (ny2 - 1) * nx2
            p1 = (ny1 - 1) * nx1
            for _ in range(ny1, nyder, -1):
                c2[p2:p2 + nx2] = c1[p1 + nxder:p1 + nxder + nx2]
                p2 -= nx2
                p1 -= nx1
    elif xt == XHALF:
        if not (nxder >= nx1 or nyder >= ny1 or nxder + nyder >= max(nx1, ny1)):
            maxorder1 = max(sf1.xorder + 1, sf1.yorder + 1)
            maxorder2 = max(xo2 + 1, yo2 + 1)
            p2, p1 = n2, sf1.ncoeff
            for i in range(ny1, nyder, -1):
                nmove1 = max(0, min(maxorder1 - i, nx1))
                nmove2 = max(0, min(maxorder2 - i + nyder, nx2))
                p1 -= nmove1
                p2 -= nmove2
                c2[p2:p2 + nmove2] = c1[p1 + nxder:p1 + nxder + nmove2]
    else:
        if nxder > 0 and nyder > 0:
            pass
        elif nxder > 0:
            if nxder < nx1:
                p2 = n2 - 1
                for j in range(nx1, nxder, -1):
                    c2[p2] = c1[j - 1]
                    p2 -= 1
        elif nyder > 0 and nyder < ny1:
            p1 = sf1.ncoeff - 1
            for _ in range(ny1, nyder, -1):
                p1 -= 1
            c2[:n2] = c1[p1 + 1:p1 + 1 + n2]
    k1x, k2x, k1y, k2y = sf1.xmaxmin, sf1.xrange, sf1.ymaxmin, sf1.yrange
    xb = _derivative_bases(sf1.type, x, xo2, nxder, k1x, k2x)
    yb = _derivative_bases(sf1.type, y, yo2, nyder, k1y, k2y)
    zfit = _sum_terms(c2, xb, yb, xt, xo2, yo2, x.size, derivative=True)
    if sf1.type != POLYNOMIAL:
        norm = (2.0 / (sf1.xmax - sf1.xmin)) ** nxder * (2.0 / (sf1.ymax - sf1.ymin)) ** nyder
        zfit = zfit * norm
    return zfit


# ------------------------------------------------------------ save, restore

def save(sf: Surface) -> list[float]:
    """dgssave: type, xorder, yorder, xterms, xmin, xmax, ymin, ymax, coefficients."""
    return [float(sf.type), float(sf.xorder), float(sf.yorder), float(sf.xterms),
            sf.xmin, sf.xmax, sf.ymin, sf.ymax, *map(float, sf.coeff)]


def restore(fit: list[float]) -> Surface:
    """dgsrestore."""
    xorder = int(round(fit[1]))
    yorder = int(round(fit[2]))
    if xorder < 1:
        raise GsurfitError("GSRESTORE: Illegal x order.")
    if yorder < 1:
        raise GsurfitError("GSRESTORE: Illegal y order.")
    xmin, xmax, ymin, ymax = fit[4], fit[5], fit[6], fit[7]
    if xmax <= xmin:
        raise GsurfitError("GSRESTORE: Illegal x range.")
    if ymax <= ymin:
        raise GsurfitError("GSRESTORE: Illegal y range.")
    kind = int(round(fit[0]))
    if kind not in (CHEBYSHEV, LEGENDRE, POLYNOMIAL):
        raise GsurfitError("GSRESTORE: Unknown surface type.")
    sf = Surface(kind, xorder, yorder, int(fit[3]), float(xmin), float(xmax),
                 float(ymin), float(ymax))
    sf.coeff = np.array(fit[8:8 + sf.ncoeff], dtype=np.float64)
    return sf


# ------------------------------------------------------ adding, coefficients

def _rows(sf: Surface):
    """How many coefficients each y term holds, in order (gsadd's nmove)."""
    out, nmove = [], sf.nxcoeff
    maxorder = max(sf.xorder + 1, sf.yorder + 1)
    for i in range(1, sf.nycoeff + 1):
        out.append(nmove)
        if sf.xterms == XNONE:
            nmove = 1
        elif sf.xterms == XHALF and i + sf.xorder + 1 > maxorder:
            nmove -= 1
    return out


def add(sf1: Surface, sf2: Surface) -> Surface:
    """dgsadd: the two surfaces' coefficients added on the larger one's grid."""
    if sf1.type != sf2.type:
        raise GsurfitError("GSADD: Incompatable surface types.")
    for a, b, what in ((sf1.xmin, sf2.xmin, "X"), (sf1.xmax, sf2.xmax, "X"),
                       (sf1.ymin, sf2.ymin, "Y"), (sf1.ymax, sf2.ymax, "Y")):
        if not _fp_equald(a, b):
            raise GsurfitError(f"GSADD: {what} ranges not identical.")
    if sf1.xterms == sf2.xterms:
        xterms = sf1.xterms
    elif XFULL in (sf1.xterms, sf2.xterms):
        xterms = XFULL
    else:
        xterms = XHALF
    sf3 = Surface(sf1.type, max(sf1.xorder, sf2.xorder), max(sf1.yorder, sf2.yorder),
                  xterms, sf1.xmin, sf1.xmax, sf1.ymin, sf1.ymax)
    sf3.coeff = np.zeros(sf3.ncoeff, dtype=np.float64)
    rows3 = _rows(sf3)
    for source, copy in ((sf1, True), (sf2, False)):
        p_in = p_out = 0
        for i, nmove in enumerate(_rows(source)):
            if copy:
                sf3.coeff[p_out:p_out + nmove] = source.coeff[p_in:p_in + nmove]
            else:
                sf3.coeff[p_out:p_out + nmove] = (sf3.coeff[p_out:p_out + nmove]
                                                  + source.coeff[p_in:p_in + nmove])
            p_in += nmove
            p_out += rows3[i]
    return sf3


def _fp_equald(x: float, y: float) -> bool:
    """fpequald: equal to the last few bits of a double."""
    if x == y:
        return True
    if x == 0.0 or y == 0.0:
        return False
    import math

    nx, ex = math.frexp(x)
    ny, ey = math.frexp(y)
    return ex == ey and abs(nx - ny) <= 32 * np.finfo(np.float64).eps


def _index(sf: Surface, xorder: int, yorder: int) -> int | None:
    """dgsgcoeff's position of a coefficient, 1-based; None where it has none."""
    if xorder > sf.xorder or yorder > sf.yorder:
        return None
    if sf.xterms == XNONE:
        if yorder == 1:
            return xorder
        if xorder == 1:
            return sf.nxcoeff + yorder - 1
        return None
    if sf.xterms == XHALF:
        maxorder = max(sf.xorder + 1, sf.yorder + 1)
        if xorder + yorder > maxorder:
            return None
        n, xincr = xorder, sf.xorder
        for i in range(2, yorder + 1):
            n += xincr
            if i + sf.xorder + 1 > maxorder:
                xincr -= 1
        return n
    return xorder + (yorder - 1) * sf.nxcoeff


def gcoeff(sf: Surface, xorder: int, yorder: int) -> float:
    n = _index(sf, xorder, yorder)
    return 1.6e308 if n is None else float(sf.coeff[n - 1])


def scoeff(sf: Surface, xorder: int, yorder: int, value: float) -> None:
    n = _index(sf, xorder, yorder)
    if n is not None:
        sf.coeff[n - 1] = value
