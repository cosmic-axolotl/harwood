"""xcsao -- RVSAO's radial velocity by cross-correlation with templates, as RVSAO
2.8.5 computes it (Tonry & Davis 1979).

For each spectrum and each template: both are rebinned onto one grid, even in log
wavelength (ncols points, a power of two) over the range they share (or st_lambda,
end_lambda); each has its continuum fit and taken off (contpars: icfit, spline3
of 10 pieces, by default) -- and, if asked, its emission or absorption lines
clipped -- its ends tapered by a cosine bell (bell_window), and is transformed.
The transforms are filtered (a cosine taper from low_bin to top_low, flat to
top_nrun, down to nrun), multiplied, transformed back: the correlation, normalised
by the two spectra's rms. Its highest peak inside minvel..maxvel is fit with a
parabola (pkmode 1; the fit above pkfrac of the peak, interpolated between the
fits on the points just above and just below it), which gives the shift and so cz;
R is the peak's height over twice the rms of the correlation's antisymmetric part,
and the error 3w/8(1+R) of its width w. The template with the highest R is best.

From Xcsao/{t_xcsao,xcfit,xcorfit,correlate,aspart,buffers,pkfitp,pkwidth,
earlin,xcrslts}.x and Util/{rebin,apodize,fourm,icsubcon,gettemp,vhead}.x, with
IRAF's arieval (math/iminterp); what the RVSAO tasks share is core/rvsao.py. In
real as f2c compiles it: each operation of REALs rounded to single precision, the
intrinsics and the REAL constants as f2c leaves them (the transforms are RVSAO's
own, in single precision with double twiddle factors).

Translated from RVSAO, Copyright (c) 1991-2022 Smithsonian Astrophysical
Observatory, whose notice is in LICENSES/RVSAO-COPYRIGHT; and from IRAF's iminterp
and icfit, Copyright (c) 1986 AURA (LICENSES/IRAF-COPYRIGHT).
"""

from __future__ import annotations

import contextlib
import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .emsao import vcombine
from .rvsao import C0, INDEFD, VERSION, RvsaoError, dlog10

f32 = np.float32
PI = 3.1415926535897932
LN2 = float(f32(0.69314718))           # rvsao.h's, a REAL constant
SPLPTS = 16                             # iminterp's points for a local spline

DEFAULTS = {
    "specnum": "0", "specband": 0, "specdir": "", "correlate": "velocity",
    "templates": "", "tempnum": "0", "tempband": 1, "tempdir": "", "echelle": False,
    "st_lambda": None, "end_lambda": None, "fixbad": False, "badlines": "badlines.dat",
    "s_emchop": "no", "t_emchop": "no", "s_abs_reject": 100.0, "s_em_reject": 2.0,
    "t_abs_reject": 0.0, "t_em_reject": 0.0, "bell_window": 0.05,
    "renormalize": False, "ncols": 2048, "interp_mode": "spline3", "zeropad": "no",
    "low_bin": 5, "top_low": 20, "top_nrun": 125, "nrun": 250, "vel_init": "zero",
    "czguess": 0.0, "nzpass": 0, "tshift": 0.0, "svel_corr": "barycentric",
    "tvel_corr": "file", "pkmode": 1, "pkfrac": 0.5, "pksrch": 25,
    "minvel": -5000.0, "maxvel": 100000.0, "report_mode": 1,
    "logfiles": "STDOUT,xcsao.log", "nsmooth": 0, "save_vel": False,
    # contpars
    "c_function": "spline3", "c_sample": "*", "naverage": 1, "order": 10,
    "s_low_reject": 2.0, "s_high_reject": 2.0, "t_low_reject": 2.0,
    "t_high_reject": 2.0, "niterate": 10, "grow": 1.0,
}
VINIT = ("zero", "guess", "zguess", "correlation", "emission", "combination")
CHOPS = ("yes", "no", "tempfile", "specfile")
NOCONT, SUBCONT, DIVCONT, ZEROCONT, CONTFIT = 0, 1, 2, 3, 4
COR_VEL, COR_PIX, COR_WAV = 2, 3, 4
#: correlate: in velocity (log wavelength), in pixels or in wavelength (Angstroms)
COR_NO = 1                       #: correlate=no: the results read from the header
CORRELATE = {"no": COR_NO, "velocity": COR_VEL, "yes": COR_VEL, "pixel": COR_PIX,
             "wavelength": COR_WAV}


# ------------------------------------------------------------- arithmetic


def _yes(value) -> bool:
    """A boolean parameter as given: True, or yes."""
    return value is True or str(value).strip().lower() in ("yes", "y", "true", "1")


def _ifix(x) -> int:
    return int(math.trunc(float(x)))


def _nint(x) -> int:
    """SPP's nint: half away from zero."""
    x = float(x)
    return int(math.floor(x + 0.5)) if x >= 0 else -int(math.floor(-x + 0.5))


def arieval(x: float, data: np.ndarray, kind: str = "spline3") -> np.float32:
    """IRAF's arieval at one point of data (1-based): spline3 fits a cubic spline
    to the SPLPTS points around x, there and then; linear, nearest, poly3, poly5
    as iminterp has them."""
    npts = data.size
    x = f32(x)
    if kind == "nearest":
        return f32(data[int(x + f32(0.5)) - 1])
    if kind == "linear":
        nearx = int(x)
        hold = (f32(f32(2) * data[nearx - 1]) - data[nearx - 2]) if nearx >= npts \
            else data[nearx]
        return f32(f32(f32(x - f32(nearx)) * hold)
                   + f32(f32(f32(nearx + 1) - x) * data[nearx - 1]))
    if kind == "spline3":
        from .asi import _spline

        nearx = int(x)
        deltax = f32(x - f32(nearx))
        bcoeff = np.zeros(SPLPTS + 4, f32)
        k = 0
        pindex = 0
        for i in range(nearx - SPLPTS // 2 + 1, nearx + SPLPTS // 2 + 1):
            if 1 <= i <= npts:
                k += 1
                if k == 1:
                    pindex = nearx - i + 1
                bcoeff[k + 1] = data[i - 1]
        bcoeff[1] = 0
        bcoeff[k + 2] = 0
        bcoeff[1:k + 3] = _spline(bcoeff[1:k + 3], k)
        pindex += 1
        bcoeff[k + 3] = 0
        b = bcoeff
        p1 = f32(f32(b[pindex - 1] + f32(f32(4) * b[pindex])) + b[pindex + 1])
        p2 = f32(f32(3) * f32(b[pindex + 1] - b[pindex - 1]))
        p3 = f32(f32(3) * f32(f32(b[pindex - 1] - f32(f32(2) * b[pindex]))
                              + b[pindex + 1]))
        p4 = f32(f32(f32(-b[pindex - 1] + f32(f32(3) * b[pindex]))
                     - f32(f32(3) * b[pindex + 1])) + b[pindex + 2])
        return f32(p1 + f32(deltax * f32(p2 + f32(deltax * f32(p3 + f32(deltax * p4))))))
    raise RvsaoError(f"interp_mode {kind} is not ported")


def rebinl(data: np.ndarray, wcs, indlog: float, npts: int, w0: float, dw: float,
           kind: str = "spline3") -> np.ndarray:
    """rebinl: the spectrum at npts points even in log wavelength from w0 by dw
    (shifted by indlog), 0 outside it."""
    out = np.zeros(npts, f32)
    n = data.size
    for i in range(1, npts + 1):
        lw = w0 + dw * float(i - 1)
        if indlog != 0.0:
            lw -= indlog
        xpos = f32(wcs.l2p(lw))
        if xpos < f32(1) or xpos > f32(n):
            out[i - 1] = 0
        else:
            out[i - 1] = arieval(xpos, data, kind)
    return out


def rebin(data: np.ndarray, wcs, indlog: float, npts: int, w0: float, dw: float,
          ispix: bool, kind: str = "spline3") -> np.ndarray:
    """rebin: the spectrum at npts points even in wavelength (or, ispix, in pixels)
    from w0 by dw -- each moved by indlog in log wavelength first -- 0 outside it."""
    out = np.zeros(npts, f32)
    n = data.size
    for i in range(1, npts + 1):
        if indlog != 0.0:
            wl0 = w0 + (dw * float(i - 1))
            wl = 10.0 ** (dlog10(wl0) - indlog)
        else:
            wl = w0 + (dw * float(i - 1))
        xpos = f32(wl if ispix else wcs.w2p(wl))
        if xpos < f32(1) or xpos > f32(n):
            out[i - 1] = 0
        else:
            out[i - 1] = arieval(xpos, data, kind)
    return out


def apodize(fdata: np.ndarray, fraction) -> None:
    """apodize: a cosine bell over `fraction` of the points at each end."""
    n = fdata.size
    nsquash = min(_nint(f32(f32(n) * f32(fraction))), n // 2)
    if nsquash <= 1:
        return
    fact = PI / float(nsquash - 1)
    for i in range(nsquash):
        factor = 0.5 * (1.0 - math.cos(fact * float(i)))
        fdata[i] = f32(factor * float(fdata[i]))
        fdata[n - 1 - i] = f32(factor * float(fdata[n - 1 - i]))


def fourm(re: np.ndarray, im: np.ndarray, log2: int, isign: int) -> None:
    """fourm: RVSAO's in-place FFT of 2**log2 complex points (Numerical Recipes'
    four1), in single precision, the twiddle factors recurring in double; the
    inverse divided by the number of points."""
    nn = 2 ** log2
    # bit reversal, as four1 swaps
    j = 0
    for i in range(nn):
        if j > i:
            re[i], re[j] = re[j], re[i]
            im[i], im[j] = im[j], im[i]
        m = nn >> 1
        while m >= 1 and j >= m:
            j -= m
            m >>= 1
        j += m
    mmax = 1
    while nn > mmax:
        istep = 2 * mmax
        theta = 6.28318530717959 / (isign * 2 * mmax)
        wpr = -2.0 * math.sin(0.5 * theta) ** 2
        wpi = math.sin(theta)
        wr, wi = 1.0, 0.0
        for m in range(mmax):
            xr, xi = f32(wr), f32(wi)
            i = np.arange(m, nn, istep)
            k = i + mmax
            tempr = (xr * re[k] - xi * im[k]).astype(f32)
            tempi = (xr * im[k] + xi * re[k]).astype(f32)
            re[k] = re[i] - tempr
            im[k] = im[i] - tempi
            re[i] = re[i] + tempr
            im[i] = im[i] + tempi
            wtemp = wr
            wr = (wr * wpr) - (wi * wpi) + wr
            wi = (wi * wpr) + (wtemp * wpi) + wi
        mmax = istep
    if isign < 0:
        re /= f32(nn)
        im /= f32(nn)


def flter2(re: np.ndarray, im: np.ndarray, k1: int, k2: int, k3: int, k4: int,
           tfilt: int = 0) -> None:
    """flter2: the transform's amplitudes multiplied by the square root of a cosine
    taper -- up from k1 to k2, flat, down from k3 to k4, nothing outside."""
    n = re.size
    n1, n2, n3, n4 = (k1, k2, n, n) if tfilt > 1 else (k1, k2, k3, k4)
    for j in range(1, n + 1):
        number = j - 1 if j < n // 2 + 1 else j - n - 1
        numa = abs(number)
        if numa < n1 or numa > n4:
            re[j - 1] = im[j - 1] = 0
        elif numa < n2:
            factor = 0.5 * (1.0 - math.cos(PI * float(numa - n1) / float(n2 - n1)))
            s = math.sqrt(factor)
            re[j - 1] = f32(float(re[j - 1]) * s)
            im[j - 1] = f32(float(im[j - 1]) * s)
        elif numa > n3:
            factor = 0.5 * (1.0 + math.cos(PI * float(numa - n3) / float(n4 - n3)))
            s = math.sqrt(factor)
            re[j - 1] = f32(float(re[j - 1]) * s)
            im[j - 1] = f32(float(im[j - 1]) * s)


def rmsnorm(re1, im1, re2, im2) -> tuple[float, float, float]:
    """rmsnorm: each transform's rms, and the correlation's normalisation."""
    n = re1.size
    f = np.full(n, 2.0)
    f[0] = 1.0
    f[n // 2] = 1.0
    rms1 = 0.0
    for k in range(n):
        xr, xi = float(re1[k]), float(im1[k])
        rms1 += f[k] * (xr * xr + xi * xi)
    rms1n = math.sqrt(rms1)
    rms1 = rms1n / n
    rms2 = 0.0
    for k in range(n):
        xr, xi = float(re2[k]), float(im2[k])
        rms2 += f[k] * (xr * xr + xi * xi)
    rms2n = math.sqrt(rms2)
    rms2 = rms2n / n
    if rms1 != 0 and rms2 != 0:
        fnorm = 2.0 / (rms1 * rms2 * n)
    elif rms1 != 0:
        fnorm = 2.0 / rms1n
    elif rms2 != 0:
        fnorm = 2.0 / rms2n
    else:
        fnorm = 2.0 / n
    return rms1, rms2, fnorm


def aspartf(re, im, k1: int, k4: int, shift: float, tfilt: int = 0):
    """aspartf: the rms of the correlation's antisymmetric and symmetric parts about
    the peak at `shift`, from its transform, over the filter's band."""
    n = re.size
    n4 = n if tfilt > 1 else k4
    arms = srms = 0.0
    for k in range(k1, n4 + 1):
        angle = -2.0 * PI * float(k) * shift / float(n)
        pr, pi = f32(math.cos(angle)), f32(math.sin(angle))
        f = 1.0 if k == 0 or k == n // 2 else 2.0
        # past the end (nrun at n): the rest of RVSAO's buffer, twice as long, zeroed
        xr, xi = (re[k], im[k]) if k < n else (f32(0), f32(0))
        aterm = float(f32(f32(pr * xi) + f32(pi * xr)))
        sterm = float(f32(f32(pr * xr) - f32(pi * xi)))
        arms += f * aterm * aterm
        srms += f * sterm * sterm
    return math.sqrt(arms) / n, math.sqrt(srms) / n


# ------------------------------------------------------------- the peak


def dsimul(n: int, a: np.ndarray) -> np.ndarray:
    """dsimul (indic 0): the solution of a's n x n system with right side column n+1,
    by Gauss-Jordan with full pivoting (1-based a[1..n, 1..n+1])."""
    mx = n + 1
    irow = [0] * (n + 1)
    jcol = [0] * (n + 1)
    x = np.zeros(n + 1)
    for k in range(1, n + 1):
        pivot = 0.0
        for i in range(1, n + 1):
            for j in range(1, n + 1):
                taken = k > 1 and (i in irow[1:k] or j in jcol[1:k])
                if abs(a[i, j]) > abs(pivot) and not taken:
                    pivot = a[i, j]
                    irow[k], jcol[k] = i, j
        if abs(pivot) <= 0.0:
            return x
        irowk, jcolk = irow[k], jcol[k]
        for j in range(1, mx + 1):
            a[irowk, j] = a[irowk, j] / pivot
        a[irowk, jcolk] = 1.0 / pivot
        for i in range(1, n + 1):
            aijck = a[i, jcolk]
            if i != irowk:
                a[i, jcolk] = -aijck / pivot
                for j in range(1, mx + 1):
                    if j != jcolk:
                        a[i, j] = a[i, j] - aijck * a[irowk, j]
    for i in range(1, n + 1):
        x[jcol[i]] = a[irow[i], mx]
    return x


def earlin(x, y, m: int = 3) -> np.ndarray:
    """earlin: the least-squares polynomial of m coefficients through (x, y)."""
    c = np.zeros((m + 2, m + 2))
    xx = np.zeros(m + 1)
    for xi, yi in zip(x, y, strict=True):
        xxx = 1.0
        for j in range(1, m + 1):
            if j != 1:
                xxx = xxx * float(xi)
            xx[j] = xxx
            c[j, m + 1] += xx[j] * float(yi)
            for k in range(1, j + 1):
                c[j, k] += xx[j] * xx[k]
    for j in range(2, m + 1):
        for k in range(1, j):
            c[k, j] = c[j, k]
    return dsimul(m, c)


def parabolapeak(a) -> tuple[float, float, float]:
    if a[3] == 0:
        return 0.0, 1.0, 1.0
    x0 = -a[2] / (2.0 * a[3])
    h = a[1] - a[2] * a[2] / (4.0 * a[3])
    w = -h / (2.0 * a[3])
    w = math.sqrt(w) if w >= 0 else -math.sqrt(-w)
    return x0, h, w


def pkwidth(xcor, xvel, fracpeak: float, imax: int, minvel, maxvel):
    """pkwidth: the highest local peak (inside minvel..maxvel) when imax is 0, its
    height, and where it crosses fracpeak of that height on each side. 1-based."""
    n = xcor.size
    pmin = pmax = 0
    if imax > 0:
        height = float(xcor[imax - 1])
        pmin = pmax = imax
    else:
        height = 0.0
        velmin = f32(-100000.0 if minvel is None else minvel)
        velmax = f32(200000.0 if maxvel is None else maxvel)
        for i in range(1, n + 1):
            if (minvel is None or xvel[i - 1] >= velmin) and \
                    (maxvel is None or xvel[i - 1] <= velmax):
                if pmin == 0:
                    pmin = i
                pmax = i
                if xcor[i - 1] >= height and i + 1 <= n and i - 1 > 0 and \
                        xcor[i - 1] >= xcor[i - 2] and xcor[i - 1] >= xcor[i]:
                    height = float(xcor[i - 1])
                    imax = i
    il = ir = 0
    if pmax == 0 and pmin == 0:
        rr = rl = height = 0.0
    elif fracpeak > 1:
        iwidth = int(fracpeak * 0.5)
        il, ir = imax - iwidth, imax + iwidth
        rl, rr = float(il), float(ir)
    else:
        fracp = height * (0.5 if fracpeak == 0 else fracpeak)

        def at(i: int):
            # xcor[i], 1-based; before the first, RVSAO reads the buffer's slot of
            # the template before -- its zero padding (with no peak in minvel..maxvel,
            # imax is 0 and both ends are read there)
            return xcor[i - 1] if 1 <= i <= n else f32(0)
        il = imax
        while il > 0 and at(il) > fracp:
            il -= 1
        a, b = at(il), at(il + 1)
        rl = il + ((fracp - float(a)) / float(f32(b - a))) if b != a else float(il)
        ir = imax
        while at(ir) > fracp:
            ir += 1
        a, b = at(ir), at(ir - 1)
        rr = ir - ((fracp - float(a)) / float(f32(b - a))) if b != a else float(ir)
    return height, rr - rl, il, ir, imax


def pkfitp(xind, xcor, xvel, fracpeak: float, pkindx: int, minvel, maxvel):
    """pkfitp: the peak's centre, height and width from parabolas fit above
    fracpeak of it -- on the points inside and on those one wider, weighted by
    where the fraction falls between them. (npfit, centre, width, height, imax)."""
    height, pwidth, il, ir, imax = pkwidth(xcor, xvel, fracpeak, pkindx, minvel, maxvel)
    if il == ir:
        if imax > 0:
            return 0, float(xind[imax - 1]), 0.0, float(xcor[imax - 1]), imax
        return 0, 0.0, 0.0, 0.0, imax
    if fracpeak == 0:
        fracpeak = 0.5
    npfit = 0
    if fracpeak < 1.0:
        ilu, iru = il + 1, ir - 1
        co = earlin(xind[ilu - 1:iru], xcor[ilu - 1:iru])
        cu, hu, wu = parabolapeak(co)
        npfit = ir - il + 1
        co = earlin(xind[il - 1:ir], xcor[il - 1:ir])
        cl, hl, wl = parabolapeak(co)
        pwl = float(ir - il)
        pweight = (pwidth - pwl) / 2.0
        center = cl + ((cl - cu) * pweight)
        height = hl + ((hl - hu) * pweight)
        width = wl + ((wl - wu) * pweight)
    else:
        npfit = int(fracpeak)
        ir = il + npfit - 1
        co = earlin(xind[il - 1:ir], xcor[il - 1:ir])
        center, height, width = parabolapeak(co)
    if center > xind[ir - 1] or center < xind[il - 1] or height <= 0:
        if imax > 0:
            center, height = float(xind[imax - 1]), float(xcor[imax - 1])
        else:
            center = height = 0.0
        width = 0.0
    return npfit, center, width, height, imax


_QUARTIC = [0.0] * 6     #: pkfitq's coefficients, SAVEd from one fit to the next


def quarticpeak(a, xl: float, xr: float, x0: float, out: list[str] | None = None):
    """quarticpeak: the quartic's maximum near x0 by Newton's method, and the points
    where it falls to half on each side, from xl and xr. (x0, h, w, xl, xr)."""
    def poly(z):
        return a[1] + z * (a[2] + z * (a[3] + z * (a[4] + z * a[5])))

    acc = 1.0e-3
    diff = acc + 1.0
    x = x0
    i = 1
    while i <= 50 and abs(diff) > acc:
        xd = a[2] + x0 * (2 * a[3] + x0 * (3 * a[4] + x0 * 4 * a[5]))
        xc = 2 * a[3] + x0 * (6 * a[4] + x0 * 12 * a[5])
        diff = xd / xc
        x = x0 - diff
        if abs(diff) > acc:
            x0 = x
        i += 1
    x0 = 0.5 * (x + x0)
    h = poly(x0)
    a[1] = a[1] - 0.5 * h
    step = float(f32(0.1))
    for iroot in (1, 2):
        x1 = xl if iroot == 1 else xr
        s = float(f32(0.1))
        while s < 3.0 and poly(x0 + s * (x1 - x0)) >= 0:
            s = s + step
        if s >= 3:
            if out is not None:
                out.append("QUARTICPEAK: poly never reaches half maximum\n")
            x = x1
        else:
            i = 1
            x = (s - step / 2) * (x1 - x0) + x0
            # diff is the last pass's: a root found closely leaves the next unsought
            while i <= 50 and abs(diff) >= acc:
                xd = a[2] + x * (2 * a[3] + x * (3 * a[4] + x * 4 * a[5]))
                diff = poly(x) / xd
                i += 1
                x = x - diff
            x = 0.5 * (x1 + x)
        if iroot == 1:
            xl = x
        else:
            xr = x
    a[1] = a[1] + 0.5 * h
    return x0, h, xr - xl


def _pow_di(base: float, n: int) -> float:
    """f2c's pow_di: a double to an integer power by repeated squaring (1/x first
    for a negative power), which is what x**n compiles to in SPP, not C's pow."""
    value = 1.0
    if n == 0:
        return value
    if n < 0:
        n = -n
        base = 1.0 / base
    while True:
        if n & 1:
            value *= base
        n >>= 1
        if n == 0:
            return value
        base *= base


def _matinv(array: list[list[float]], norder: int) -> float:
    """Util/matinv.x (Bevington): the symmetric matrix inverted in place, with full
    pivoting; its determinant, 0 when singular. 0-based."""
    det = 1.0
    ik = [0] * norder
    jk = [0] * norder
    for k in range(norder):
        while True:
            amax = 0.0
            for i in range(k, norder):
                for j in range(k, norder):
                    if abs(amax) <= abs(array[i][j]):
                        amax = array[i][j]
                        ik[k] = i
                        jk[k] = j
            if amax == 0:
                return 0.0
            i = ik[k]
            if i < k:
                continue
            if i > k:
                for j in range(norder):
                    save = array[k][j]
                    array[k][j] = array[i][j]
                    array[i][j] = -save
            j = jk[k]
            if j < k:
                continue
            if j > k:
                for i in range(norder):
                    save = array[i][k]
                    array[i][k] = array[i][j]
                    array[i][j] = -save
            break
        for i in range(norder):
            if i != k:
                array[i][k] = -array[i][k] / amax
        for i in range(norder):
            for j in range(norder):
                if i != k and j != k:
                    array[i][j] = array[i][j] + array[i][k] * array[k][j]
        for j in range(norder):
            if j != k:
                array[k][j] = array[k][j] / amax
        array[k][k] = 1.0 / amax
        det = det * amax
    for back in range(1, norder + 1):
        k = norder - back
        j = jk[k]
        if j > k:
            for i in range(norder):
                save = array[i][k]
                array[i][k] = -array[i][j]
                array[i][j] = save
        i = ik[k]
        if i > k:
            for j in range(norder):
                save = array[k][j]
                array[k][j] = -array[i][j]
                array[i][j] = save
    return det


def _mfunk(x: list[float], y: list[float], a: list[float]) -> float:
    """pkfitc's function: chi² of A cos(B z) / (1 + D z²), z = x - C, over n - 4."""
    chi = 0.0
    for xi, yi in zip(x, y, strict=True):
        psi = xi - a[2]
        f = a[0] * math.cos(a[1] * psi) / (1.0 + a[3] * (psi * psi))
        d = yi - f
        chi = chi + d * d
    return chi / (len(x) - 4)


def xcmin(x, y, a: list[float], out: list[str]) -> int:
    """Xcsao/xcmin.x, after Tonry's MINI: a minimum of _mfunk by numerical derivatives
    and a Levenberg-like mix of gradient and Newton steps. `a` is the start and the
    result (changed in place). 0, 1 (iterations exceeded) or 2 (singular)."""
    n = len(a)
    lam = -3
    base = 10.0
    itermax = 40
    qfrac, dfrac, places, vary = 1.0e-6, 0.02, 1.0e-7, 1.0e-5
    a0 = list(a)
    d2f = [[0.0] * n for _ in range(n)]
    cov = [[0.0] * n for _ in range(n)]
    df = [0.0] * n
    da = [0.0] * n
    fnow = _mfunk(x, y, a)
    ai = [abs(vary * v) or 1.0e-6 for v in a0]
    it = 0
    for it in range(1, itermax + 2):
        if it > itermax:
            break
        fthen = fnow
        a[:] = a0
        for j in range(n):
            a[j] = a0[j] + ai[j]
            df[j] = _mfunk(x, y, a)
            a[j] = a0[j]
        for j in range(1, n):
            for k in range(j):
                a[k] = a0[k] + ai[k]
                a[j] = a0[j] + ai[j]
                chi = _mfunk(x, y, a)
                if ai[j] == 0.0:
                    out.append(f"XCMIN: ai[{j + 1}] is zero\n")
                    d2f[j][k] = 0.0
                elif ai[k] == 0.0:
                    out.append(f"XCMIN: ai[{k + 1}] is zero\n")
                    d2f[j][k] = 0.0
                else:
                    d2f[j][k] = (chi - df[j] - df[k] + fnow) / (ai[j] * ai[k])
                d2f[k][j] = d2f[j][k]
                a[j] = a0[j]
                a[k] = a0[k]
        for j in range(n):
            a[j] = a0[j] - ai[j]
            fminus = _mfunk(x, y, a)
            if ai[j] == 0.0:
                out.append(f"XCMIN: ai[{j + 1}] is zero\n")
                d2f[j][j] = 0.0
            else:
                d2f[j][j] = (fminus + df[j] - 2 * fnow) / (ai[j] * ai[j])
            df[j] = (df[j] - fminus) / (2 * ai[j])
            a[j] = a0[j]
        for j in range(n):
            curve = d2f[j][j]
            if curve == 0:
                curve = 1.0e-5
            g = df[j] * dfrac / curve
            ai[j] = math.sqrt(g * g + abs(fnow * places / curve))
        for _ii in range(15):
            for j in range(n):
                for k in range(j):
                    cov[j][k] = d2f[j][k]
                    cov[k][j] = d2f[j][k]
                cov[j][j] = abs(d2f[j][j] * (1.0 + _pow_di(base, lam)))
            if _matinv(cov, n) == 0:
                return 2
            for j in range(n):
                da[j] = 0.0
                for k in range(n):
                    da[j] = da[j] - (cov[j][k] * df[k])
            for j in range(n):
                a[j] = a0[j] + da[j]
            fnow = _mfunk(x, y, a)
            if fnow < fthen:
                break
            lam = lam + 1
        a0[:] = a
        lam = lam - 1
        if abs((fthen - fnow) / fnow) < qfrac:
            break
    return 1 if it > itermax else 0


def pkfitc(xind, xcor, xvel, fracpeak: float, pkindx: int, minvel, maxvel,
           out: list[str] | None = None):
    """pkfitc (pkmode 3, after G. Torres): A cos(B z) / (1 + D z²) fit to the peak
    above fracpeak of it by xcmin, D tried as computed and then twice it; the FWHM by
    Newton-Raphson on the fitted curve. (npfit, centre, width, height, imax)."""
    out = [] if out is None else out
    height, width, il, ir, imax = pkwidth(xcor, xvel, fracpeak, pkindx, minvel, maxvel)
    npfit = ir - il + 1
    x = [float(v) for v in xind[il - 1:il - 1 + npfit]]
    y = [float(v) for v in xcor[il - 1:il - 1 + npfit]]
    pi = 3.1415926535897932
    a = [0.0] * 4
    ierr, itry = 1, 0
    dv = 0.0
    while ierr > 0 and itry < 2:
        dv = float(f32(xind[ir - 1]) - f32(xind[il - 1]))
        a = [height, pi / dv, float(xind[imax - 1]),
             16.0 * (math.sqrt(2.0) - 1.0) / (dv * dv)]
        a[3] = a[3] / 2.0
        g = float(f32(f32(f32(-1.25) * f32(itry * itry)) + f32(f32(2.25) * f32(itry)))
                  + f32(1.0))
        a[3] = g * a[3]
        ierr = xcmin(x, y, a, out)
        if ierr > 0:
            out.append("PKFITC Error:  singular matrix\n")
            itry = itry + 1
    psi0 = dv / 4.0
    dpsi = 1.0
    stop = float(f32(1.0e-4))
    while dpsi > stop:
        denom = 1.0 + a[3] * psi0 * psi0
        ff = (math.cos(a[1] * psi0) / denom) - 0.5
        dff = -a[1] * math.sin(a[1] * psi0) / denom
        dff = dff - (2.0 * a[3] * psi0 * math.cos(a[1] * psi0) / (denom * denom))
        dpsi = ff / dff
        psi0 = abs(psi0 - dpsi)
    return npfit, a[2], 2.0 * psi0, a[0], imax


def pkfitq(xind, xcor, xvel, fracpeak: float, pkindx: int, minvel, maxvel,
           out: list[str] | None = None):
    """pkfitq: the peak fit by quartics, above fracpeak of it, on the points inside and
    one wider, the half-height points from pkwidth's at 0.5."""
    _h, _w, ilh, irh, imax = pkwidth(xcor, xvel, 0.5, pkindx, minvel, maxvel)
    height, pwidth, il, ir, imax = pkwidth(xcor, xvel, fracpeak, imax, minvel, maxvel)
    if fracpeak == 0:
        fracpeak = 0.5
    a = _QUARTIC
    if fracpeak < 1.0:
        ilu, iru = il + 1, ir - 1
        co = earlin(xind[ilu - 1:iru], xcor[ilu - 1:iru], 5)
        a[1:6] = list(co[1:6])
        cu, hu, wu = quarticpeak(a, float(xind[ilh - 1]), float(xind[irh - 1]),
                                 float(xind[imax - 1]), out)
        npfit = ir - il + 1
        co = earlin(xind[il - 1:ir], xcor[il - 1:ir], 5)
        a[1:6] = list(co[1:6])
        cl, hl, wl = quarticpeak(a, float(xind[ilh - 1]), float(xind[irh - 1]),
                                 float(xind[imax - 1]), out)
        pwl = float(ir - il)
        pweight = (pwidth - pwl) / 2.0
        center = cl + ((cl - cu) * pweight)
        height = hl + ((hl - hu) * pweight)
        width = wl + ((wl - wu) * pweight)
    else:
        npfit = int(fracpeak)
        co = earlin(xind[il - 1:il - 1 + npfit], xcor[il - 1:il - 1 + npfit], 3)
        a[1:4] = list(co[1:4])
        center, height, width = quarticpeak(a, float(xind[ilh - 1]),
                                            float(xind[irh - 1]), float(xind[imax - 1]),
                                            out)
    return npfit, center, width, height, imax


# ------------------------------------------------------------- the continuum


def icsubcon(fdata: np.ndarray, p: dict, spectype: int, conproc: int, chop: bool,
             abrej: float, emrej: float, out: list[str] | None = None) -> np.ndarray:
    """icsubcon: the continuum fit (contpars; the spectrum's or the template's
    rejection) to the points from the first to the last not zero; with `chop`, the
    points beyond abrej (below) or emrej (above) sigmas of it set to it; and the
    continuum taken off (or divided out). In place; the fit returned."""
    from . import icfitbits

    n = fdata.size
    order = int(p["order"])
    if order < 1 or not np.any(fdata != 0):
        return fdata.copy()
    w = np.ones(n, f32)
    for i in range(n):
        if fdata[i] == 0:
            w[i] = 0
        else:
            break
    for i in range(n - 1, -1, -1):
        if fdata[i] == 0:
            w[i] = 0
        else:
            break
    x = np.arange(1, n + 1, dtype=f32)
    low, high = ("s_low_reject", "s_high_reject") if spectype == 1 else \
        ("t_low_reject", "t_high_reject")
    ic = icfitbits.Ic(function=str(p["c_function"]), order=order,
                      sample=str(p["c_sample"]), naverage=int(p["naverage"]),
                      low_reject=float(f32(p[low])), high_reject=float(f32(p[high])),
                      niterate=int(p["niterate"]), grow=float(f32(p["grow"])))
    curve = icfitbits.fit(ic, x, fdata.astype(f32), w)
    if out is not None:            # ic_fit's errors, as it prints them
        out.extend(f"{m}\n" for m in ic.messages)
    if chop and (abrej > 0 or emrej > 0):
        clip = icfitbits.Ic(low_reject=float(f32(abrej)), high_reject=float(f32(emrej)),
                            grow=float(f32(p["grow"])))
        clip.rejpts = np.zeros(n, int)
        icfitbits._deviant(curve, x, fdata.astype(f32), w, clip, 0, refit=False)
        values = np.asarray(curve.vector_at(x), f32)
        fdata[clip.rejpts == 1] = values[clip.rejpts == 1]
    fit = np.asarray(curve.vector_at(x), f32)
    if conproc in (DIVCONT, ZEROCONT):
        fitmean = f32(float(np.sum(fit.astype(np.float64))) / n)
        for i in range(n):
            if w[i] > 0:
                fdata[i] = f32(1.0) if fit[i] <= 0 else f32(fdata[i] / fit[i])
        if conproc == ZEROCONT:
            fdata -= f32(1.0)
        fdata *= fitmean
    elif conproc == SUBCONT:
        keep = w > 0
        fdata[keep] = (fdata[keep] - fit[keep]).astype(f32)
    elif conproc == CONTFIT:
        fdata[:] = fit
    return fit


# ------------------------------------------------------------- templates and results


@dataclass
class Template:
    """gettemp's template and what its header asks (vthead, TSHIFT, FI-FLAG,
    SUBCONT, DIVCONT, OVERLAP, ZEROPAD, PEAKFRAC, CHOPEM, EMCHOP)."""
    path: str
    id: str                       #: tempid: the name as listed (with :w1-w2, [ap])
    spec: object                  #: rvsao.Spectrum, getimage's
    vel: float = 0.0              #: tempvel
    shift: float = 0.0            #: tempshift (TSHIFT)
    hcv: float = 0.0              #: temphcv
    filt: int = 0                 #: tempfilt (FI-FLAG)
    start_log: float = 0.0        #: testrt
    end_log: float = 0.0          #: tefnsh
    wl1: float = INDEFD           #: tempwl1, from name:w1-w2
    wl2: float = INDEFD


@dataclass
class TempResult:
    """One template's correlation: what xcrslts reports and xcplot draws."""
    template: Template
    cz: float = 0.0               #: the peak's velocity (the shift alone)
    czerr: float = 0.0
    czr: float = 0.0              #: R
    zvel: float = 0.0             #: cz with the template's and spectrum's corrections
    height: float = 0.0           #: thght
    width: float = 0.0            #: twdth (bins)
    center: float = 0.0           #: tcent (bins)
    arms: float = 0.0
    srms: float = 0.0
    npfit: int = 0
    vwidth: float = 0.0           #: tvw, km/s
    wl1: float = 0.0              #: twl1, twl2: the range correlated
    wl2: float = 0.0
    schop: bool = False           #: tschop: the spectrum's emission lines clipped
    achop: bool = False           #: tachop: its absorption lines clipped
    scont: bool = True            #: tscont: the template's continuum taken off
    conproc: int = SUBCONT
    dlogw: float = 0.0            #: the grid's step in log wavelength (taa's inverse)
    vshift: float = 0.0           #: tvshift: the initial velocity the template was moved by
    pcent: float = 0.0            #: tpcent, tphght, tpwdth: the peak as measured, not fit
    phght: float = 0.0
    pwdth: float = 0.0
    xvel: np.ndarray | None = None
    xcor: np.ndarray | None = None
    overlap: bool = False         #: toverlap: the whole overlap correlated


@dataclass
class XcResult:
    spectrum: object
    temps: list[TempResult] = field(default_factory=list)
    best: int = 0                 #: itmax, 1-based
    ncor: int = 0
    npts: int = 0
    filter: tuple = (0, 0, 0, 0)  #: lo, toplo, topnrn, nrun
    zpad: bool = False
    han: float = 0.05
    fracpeak: float = 0.5
    tshift: float = 0.0
    xcvel: float = INDEFD
    xcerr: float = INDEFD
    xcr: float = INDEFD
    emvel: float = INDEFD         #: spevel, speerr, spnlf: emsao's, from the header
    emerr: float = INDEFD
    emnlf: int = 0
    emnl: int = 0
    vel: float = INDEFD           #: spvel, sperr: vcombine's of the two
    err: float = INDEFD
    saved: str = ""               #: the file save_vel wrote into
    specfile: str = ""            #: the name as listed
    span: tuple = (1, 1)          #: np1, np2: the pixels with data
    quality: int = 0              #: spvqual: the graph's y, n, j
    qplot: bool = False
    npmax: int = 0                #: the points fit for the best template's peak
    old: tuple = (INDEFD, INDEFD, 0.0, INDEFD, "_")  #: vrhead's: vel, err, R, BCV, quality


def emrhead(head, mspec: int) -> tuple[float, float, int]:
    """emrhead's velocity: emsao's cz, its error and the lines it fit, from CZEM,
    CZEMERR and CZEMNLF (a multispec's APVEMn); INDEFD when emsao wrote none."""
    if mspec == 0:
        if not head.has("CZEM"):
            return INDEFD, INDEFD, 0
        return head.d("CZEM", INDEFD), head.d("CZEMERR", INDEFD), head.i("CZEMNLF", 0)
    if mspec >= 1000 or not head.has(f"APVEM{mspec}"):
        return INDEFD, INDEFD, 0
    from .rvsao import Scan

    scan = Scan(head.string(f"APVEM{mspec}") or "")
    vel, err = scan.d(), scan.d()
    scan.i()
    nlf = scan.i()
    return (INDEFD if vel is None else vel), (INDEFD if err is None else err), nlf or 0


def _apvem_nl(head, mspec: int) -> int:
    from .rvsao import Scan

    if mspec >= 1000 or not head.has(f"APVEM{mspec}"):
        return 0
    scan = Scan(head.string(f"APVEM{mspec}") or "")
    scan.d()
    scan.d()
    return scan.i() or 0


def vrhead(head, mspec: int) -> tuple:
    """vrhead: the velocity a run before left in the header -- VELOCITY (with
    VELERR or CZERR, CZXCR and VELQUAL), or a multispec's APVELn -- and the BCV:
    (vel, err, R, bcv, quality), INDEFD where there is none."""
    from .rvsao import Scan

    bcv = head.d("BCV", INDEFD)
    vfound = afound = False
    key = "VELOCITY"
    if mspec == 0:
        vfound = head.has("VELOCITY")
        if not vfound:
            afound = head.has("APVEL1")
            key = "APVEL0"
    elif mspec < 1000:
        key = f"APVEL{mspec}"
        afound = head.has(key)
        if not afound:
            vfound = head.has("VELOCITY")
    else:
        vfound = head.has("VELOCITY")
    vel = err = INDEFD
    r0 = 0.0
    qflag = "\\000"            # a char never set: IRAF's %c prints NUL so
    if vfound:
        vel = head.d("VELOCITY", INDEFD)
        err = head.d("VELERR", INDEFD)
        err = head.d("CZERR", err)
        r0 = head.d("CZXCR", 0.0)
        qflag = (head.string("VELQUAL") or "_")[:1] or "_"
    elif afound:
        scan = Scan(head.string(key) or "")
        vel = scan.d()
        err = scan.d()
        word = scan.wrd()
        qflag = word[:1] if word else "_"
        extra = scan.d()
        vel = INDEFD if vel is None else vel
        err = INDEFD if err is None else err
        if extra is not None:
            bcv = extra
    return vel, err, r0, bcv, qflag


def report_head(rmode: int, ntemp: int, tab: bool = True) -> str:
    """xcrshead: the column names and dashes a tab-separated report (a negative
    report_mode) starts with -- for modes 8, 10, 11, 12 and 14."""
    sp = "\t" if tab else " "
    d = "-" * 32

    def temps(n1: int, n2: int, n3: int, n4: int, left: bool = True,
              plain: bool = False) -> str:
        out = ""
        for i in range(1, ntemp + 1):
            if plain:
                out += (f"{sp}{f'temp{i}':<{n1}.{n1}}{sp}{f'vel{i}':>7}{sp}"
                        f"{f'err{i}':<5}{sp}{f'xcr{i}':>4}")
            else:
                out += (f"{sp}{f'temp{i}':<{n1}.{n1}}{sp}{f'vel{i}':<{n2}.{n2}}{sp}"
                        f"{f'err{i}':<{n3}.{n3}}{sp}{f'xcr{i}':<{n4}.{n4}}")
        return out

    def dashes(n1: int, n2: int, n3: int, n4: int, left: bool = False) -> str:
        cell = (lambda n: f"{d:<{n}.{n}}") if left else (lambda n: f"{d:>{n}.{n}}")
        return "".join(f"{sp}{cell(n1)}{sp}{cell(n2)}{sp}{cell(n3)}{sp}{cell(n4)}"
                       for _ in range(ntemp))

    if rmode == 8:
        return (f"{'specid':<32.32}{sp}{'instrument':<8.8}{sp}{'title':<16.16}{sp}"
                f"{'julian_date':<15.15}{sp}{'emvel':<5.5}{sp}{'emerr':>5.5}{sp}"
                f"{'xcvel':<5.5}{sp}{'xcerr':>5.5}{sp}{'xcr':<5.5}{sp}{'vel':>5.5}{sp}"
                f"{'err':>3.3}{sp}C{sp}{'nl':>2.2}{sp}{'nf':>2.2}{sp}{'best':<8.8}\n"
                f"{d:>32.32}{sp}{d:>8.8}{sp}{d:>16.16}{sp}{d:>15.15}{sp}{d:>5.5}{sp}"
                f"{d:>5.5}{sp}{d:>5.5}{sp}{d:>5.5}{sp}{d:>5.5}{sp}{d:>5.5}{sp}{d:>3.3}"
                f"{sp}{d:>1.1}{sp}{d:>2.2}{sp}{d:>2.2}{sp}{d:>8.8}\n")
    if rmode == 10:
        best = (f"{sp}{'besttemp':<12.12}{sp}{'bestvel':<7.7}{sp}{'besterr':<7.7}{sp}"
                f"{'bestr':<5.5}") if ntemp > 1 else ""
        bestd = f"{sp}{d:>12.12}{sp}{d:>7.7}{sp}{d:>5.5}{sp}{d:>4.4}" if ntemp > 1 else ""
        return (f"{'specid':<14.14}{sp}{'julian_date':<14.14}{sp}{'best':<4.4}{best}"
                f"{temps(12, 7, 5, 4)}\n{d:>14.14}{sp}{d:>14.14}{sp}{d:>2.2}{bestd}"
                f"{dashes(12, 7, 5, 4)}\n")
    if rmode == 11:
        best = (f"{sp}{'besttemp':<14.14}{sp}{'xcvel':<7.7}{sp}{'xcerr':<5.5}{sp}"
                f"{'xcr':>4.4}") if ntemp > 1 else ""
        bestd = f"{sp}{d:>14.14}{sp}{d:>7.7}{sp}{d:>5.5}{sp}{d:>4.4}" if ntemp > 1 else ""
        return (f"{'specid':<32.32}{sp}{'julian_date':<14.14}{sp}best{best}"
                f"{temps(14, 7, 5, 4)}\n{d:>32.32}{sp}{d:>14.14}{sp}{d:>2.2}{bestd}"
                f"{dashes(14, 7, 5, 4)}\n")
    if rmode == 12:
        return (f"{'specid':<32.32}{sp}{'julian_date':<14.14}{sp}{'best':<4.4}"
                f"{temps(14, 7, 5, 4, plain=True)}\n{d:>32.32}{sp}{d:>14.14}{sp}"
                f"{d:>4.4}{dashes(14, 7, 5, 4, left=True)}\n")
    if rmode == 14:
        return (f"{'specid':<32.32}{sp}{'julian_date':<14.14}{sp}{'emvel':<7.7}{sp}"
                f"{'emerr':<5}{sp}nl{sp}nf{sp}best{sp}{'besttemp':<14.14}"
                f"{temps(14, 7, 5, 4, plain=True)}\n{d:>32.32}{sp}{d:>14.14}{sp}"
                f"{d:>7.7}{sp}{d:>5.5}{sp}{d:>2.2}{sp}{d:>2.2}{sp}{d:>2.2}{sp}{d:>14.14}"
                f"{dashes(14, 7, 5, 4)}\n")
    return ""


def peak_width(t: TempResult) -> float:
    """The correlation peak's width in km/s, from its width in bins (taa: the bins
    per unit of ln wavelength), or tvw when there is none."""
    taa = 1.0 / t.dlogw / math.log(10.0) if t.dlogw else 0.0
    if taa != 0:
        dlogw = 1.0 / (math.log(10.0) * taa)
        return C0 * ((10.0 ** (t.width * dlogw)) - 1.0)
    return t.vwidth


def _vthead(head) -> tuple[float, float, float]:
    """vthead for a one-dimensional template: VELOCITY (or APVEL1's first word) and
    the BCV and HCV it carries; (vel, bcv, hcv), INDEFD for those it does not."""
    tbcv = head.d("BCV", INDEFD)
    thcv = head.d("HCV", INDEFD)
    tvel = INDEFD
    if head.has("VELOCITY"):
        tvel = head.d("VELOCITY", INDEFD)
    elif head.has("APVEL1"):
        words = (head.string("APVEL1") or "").split()
        try:
            tvel = float(words[0])
        except (IndexError, ValueError):
            tvel = INDEFD
        if len(words) > 3:
            with contextlib.suppress(ValueError):
                tbcv = thcv = float(words[3])
    return (0.0 if tvel == INDEFD else tvel), tbcv, thcv


# ------------------------------------------------------------- xcsao


class Xcsao:
    """t_xcsao for a list of spectra: xcfit on each, and the report."""

    def __init__(self, params: dict):
        from .rvsao import VEL_CORR

        self.p = {**DEFAULTS, **params}
        self.svcor = VEL_CORR.index(str(self.p["svel_corr"])) + 1
        self.tvcor = VEL_CORR.index(str(self.p["tvel_corr"])) + 1
        self.vinit = VINIT.index(str(self.p["vel_init"])) + 1
        self.results: list[XcResult] = []
        self.state: dict = {}
        self.out = ""
        self.err = ""                 # what RVSAO writes with eprintf
        self.log: list[str] = []
        self.tempfilt: dict[int, int] = {}
        self._counted = False         # countemp runs once, when the buffers are made
        self._tempinfo = [""] * 36    # xcrslts' tempinfo, as the stack keeps it
        # xcplot's: the commons and statics its keys change, the graphs drawn
        self.nsmooth = int(self.p.get("nsmooth", 0))
        self._spec_subcon = (SUBCONT, (False, 0.0, 0.0))
        self.emview = None            # emsao's commons for its graph, made when first drawn
        self.show_lines = None        # the window's emplot from xcsao: (Plot) -> None
        self.show_plotspec = None     # the window's plotspec: (Plotspec) -> None
        self.pltcor = _yes(self.p.get("xcor_plot", False))
        self.drawn: list = []         # every graph drawn, in the order drawn
        self._specname = ""
        self.jtemp = 0
        self.pkindx = 0
        self.plots: list = []
        self.read_key = None
        self.read_line = None
        self.show = None

    def logfiles(self) -> list[str]:
        return [w.strip() for w in str(self.p.get("logfiles", "STDOUT")).split(",")
                if w.strip()]

    def _log(self, text: str) -> None:
        self.log.append(text)
        if self.to_stdout:
            self.out += text

    def _say(self, text: str) -> None:
        self.out += text

    @staticmethod
    def xlabel(spec) -> str:
        """LABEL and UNITS of the spectrum's WCS: "Wavelength in angstroms"."""
        label = getattr(spec.sh, "label", "") or ""
        units = getattr(spec.sh, "units", "") or ""
        return f"{label} in {units}" if units else label

    def plot(self, result) -> bool:
        """xcplot after the report: the graph and its keys. True when they ask to
        correlate again."""
        from .xcplot import Plot

        p = self.p
        if not (_yes(p.get("displot")) or _yes(p.get("hardcopy"))):
            return False
        plot = Plot(self, result, self.read_key or (lambda: None),
                    self.read_line or (lambda: None), self._say,
                    curmode=_yes(p.get("curmode", False)))
        self.plots.append(plot)
        if self.show is not None and _yes(p.get("displot")):
            self.show(plot)
        else:
            plot.run()
        result.qplot = plot.qplot
        return plot.newfit

    def run(self, names: list[str]) -> None:
        from . import rvsao
        from .emsao import decode_ranges

        self.to_stdout = "STDOUT" in self.logfiles()
        if int(self.p["report_mode"]) == 1:
            self._log(rvsao.loghead("rvsao.xcsao") + "\n")
        if int(self.p["report_mode"]) < 0:
            self._log(report_head(-int(self.p["report_mode"]), 0))
        for name in names:
            path = str(self.p["specdir"]) + name
            text = str(self.p["specnum"]).strip() or "0"
            mspecs = [0] if text == "0" else ([n for n in decode_ranges(text) if n > 0]
                                              or [0])
            if not Path(path).exists() and not Path(path + ".fits").exists():
                self.err += f"XCSAO: cannot read spectrum file {path} \n"
                continue
            for k, mspec in enumerate(mspecs):
                try:
                    spec = rvsao.getspec(path, mspec, int(self.p["specband"]),
                                         self.svcor, self.state)
                except (RvsaoError, OSError, ValueError):
                    continue
                while True:
                    result = self.xcfit(spec)
                    if result is None:
                        break
                    result.specfile = name
                    result.old = vrhead(spec.head, mspec)
                    result.emvel, result.emerr, result.emnlf = emrhead(spec.head, mspec)
                    result.emnl = spec.head.i("CZEMNL", 0) if mspec == 0 else \
                        _apvem_nl(spec.head, mspec)
                    # xcfit's vcombine: the correlation's velocity joined with emsao's
                    # (not for correlate=no: VELOCITY stays what the header has)
                    if self.cor == COR_NO:
                        result.vel, result.err = result.old[0], result.old[1]
                    else:
                        result.vel, result.err = vcombine(result.xcvel, result.xcerr,
                                                          result.xcr, result.emvel,
                                                          result.emerr, result.emnlf)
                    self._report(result)
                    if not self.plot(result):
                        break
                if result is not None:
                    self.results.append(result)
                    if self.cor == COR_NO:
                        if _yes(self.p.get("save_vel")) and result.qplot:
                            self.qsave(path, mspec, result, len(mspecs))
                    elif _yes(self.p.get("save_vel")):
                        self.save(path, mspec, result, last=k == len(mspecs) - 1)

    # ---------------------------------------------------------------- save_vel

    def save(self, path: str, mspec: int, r: XcResult, last: bool) -> None:
        """vwhead and xcwhead: the velocity -- the correlation's, joined by vcombine
        with emsao's when the header has it -- and each template's correlation into
        the spectrum's header, the cards RVSAO writes; the XCSAO card before it kept
        as HISTORY. Apertures of a multispec each have their own cards (APVELn,
        APVXCn, XTn_m), apertures past 999 none.

        RVSAO's FITS kernel writes back the header it holds in memory, so the
        RA, DEC, HA and ST of 00:00:00 that getimage gives a header without them go
        into the file too, with IRAF's DATE: the next task (xcsao on the next
        aperture, among them) then corrects the velocity for an object at 0h 0deg
        observed today. harwood writes RVSAO's own cards only."""
        import os

        from .iraffmt import Format
        from .rvsao import HeaderUpdate, history_card, logtime, number_card, string_card

        if not Path(path).exists():
            path += ".fits"
        if not os.access(path, os.W_OK):
            self.err += f"XCSAO: cannot write to {path}; not saving results\n"
            return
        if mspec >= 1000:
            return
        best = r.temps[r.best - 1]
        hcv = r.spectrum.hcv
        q = {1: "?", 2: "?", 3: "X", 4: "Q"}.get(r.quality, "_")   # the graph's y n j
        lo, toplo, topnrn, nrun = r.filter
        g7, f3 = Format("%7g"), Format("%.3f")
        params = (f"{len(r.temps)} {r.npts} {2 if r.zpad else 1} {lo} {toplo} {topnrn} "
                  f"{nrun} {Format('%.2f')(float(f32(r.han)))}")
        cards: list[str] = []
        if mspec <= 0:
            cards += [number_card("VELOCITY", r.vel), number_card("Z", r.vel / C0),
                      number_card("CZERR", r.err), number_card("ZERR", r.err / C0),
                      string_card("VELQUAL", q), number_card("BCV", hcv),
                      number_card("CZXC", r.xcvel), number_card("ZXC", r.xcvel / C0),
                      number_card("CZXCERR", r.xcerr), number_card("CZXCR", r.xcr),
                      string_card("BESTTEMP", best.template.id),
                      string_card("CZPARAMS", params)]
        else:
            cards += [string_card(f"APVEL{mspec}",
                                  f"{f3(r.vel)} {f3(r.err)} {q} {f3(hcv)}"),
                      string_card(f"APVXC{mspec}",
                                  f"{f3(r.xcvel)} {f3(r.xcerr)} {g7(r.xcr)} {r.best} "
                                  f"{best.template.id} {params}")]
        cards += [history_card("COMMENT", text) for text in (
            "Template information from rvsao.xcsao:",
            "file name, cross-correlation velocity, error, R-value",
            "correlation peak height (0-1) and width (km/sec)",
            "velocity relative to template, template BCV",
            "template em. chopped, template filtered")]
        for i, t in enumerate(r.temps, 1):
            key = f"XCTEMP{i}" if mspec <= 0 else f"XT{mspec}_{i}"
            tmpvel = t.template.vel + t.template.shift + t.vshift
            cards.append(string_card(key, (
                f"{t.template.id} {f3(t.zvel)} {f3(t.czerr)} {g7(t.czr)} "
                f"{f3(t.height)} {f3(peak_width(t))} {f3(tmpvel)} {f3(t.template.hcv)} "
                f"{'yes' if t.schop else 'no'} {t.template.filt}")))
        if mspec > 0 and last:
            stamp = f"rvsao.xcsao {VERSION} {logtime()} CZXC [{self.p['specnum']}]"
        else:
            stamp = (f"rvsao.xcsao {VERSION} {logtime()} CZXC = "
                     f"{Format('%.2f')(r.xcvel)} R = {Format('%.2f')(r.xcr)}")
        head = HeaderUpdate(path)
        try:
            for image in cards:
                head.put(image)
            old = head.string("XCSAO")
            if old:
                head.put(history_card("HISTORY", old))
            head.put(string_card("XCSAO", stamp))
        finally:
            head.close()
        r.saved = path

    # ---------------------------------------------------------------- xcfit

    def _templates(self) -> list[tuple[str, str, float, float]]:
        """The template list: (path, tempid, w1, w2) for each name, name:w1-w2 read."""
        from . import rvsao

        tempdir = str(self.p["tempdir"])
        if tempdir and not tempdir.endswith(("/", "$")):
            tempdir += "/"
        out = []
        for word in str(self.p["templates"]).split(","):
            word = word.strip()
            if not word:
                continue
            w1 = w2 = INDEFD
            name = word
            if ":" in word:
                name, rest = word.split(":", 1)
                if "-" in rest:
                    a, b = rest.split("-", 1)
                    with contextlib.suppress(ValueError):
                        w1, w2 = float(a), float(b)
            path = name if name.startswith(("/", ".")) or "$" in name else tempdir + name
            path = rvsao.template_file(path)
            # tempid: the name; the range is cut off it (what follows w2 would stay)
            out.append((path, name, w1, w2))
        return out

    def _template(self, path: str, tempid: str, w1: float, w2: float, itemp: int):
        from .rvsao import BCV, FBCV, FHCV, HCV, compbcv, getimage

        spec = getimage(path, 0, int(self.p["tempband"]), self.state)
        head = spec.head
        t = Template(path, tempid, spec, wl1=w1, wl2=w2)
        t.filt = head.i("FI-FLAG", self.tempfilt.get(itemp, 0))
        self.tempfilt[itemp] = t.filt
        t.vel, bcvel, hcvel = _vthead(head)
        t.shift = head.d("TSHIFT", 0.0)
        if self.tvcor == HCV:
            t.hcv = compbcv(head, False)
        elif self.tvcor == BCV:
            t.hcv = compbcv(head, True)
        elif self.tvcor == FHCV:
            t.hcv = hcvel if hcvel != INDEFD else bcvel if bcvel != INDEFD else \
                compbcv(head, False)
        elif self.tvcor == FBCV:
            t.hcv = bcvel if bcvel != INDEFD else hcvel if hcvel != INDEFD else \
                compbcv(head, True)
        npix = spec.data.size
        if spec.wcs.p2w(1.0) <= 0 or spec.wcs.p2w(float(npix)) <= 0:
            self.out += f"GETTEMP: Illegal starting wavelength in {path}\n"
            return None
        t.start_log = spec.wcs.p2l(1.0)
        t.end_log = spec.wcs.p2l(float(npix))
        return t

    def _chops(self, spec, temp: Template) -> tuple[bool, float, float, bool, float, float]:
        """setschop and settchop: whether the spectrum's and the template's lines are
        clipped, and beyond how many sigmas (absorption, emission)."""
        p = self.p
        sab, sem = float(f32(p["s_abs_reject"])), float(f32(p["s_em_reject"]))
        mode = str(p["s_emchop"])
        schop = False
        if mode == "yes":
            schop = True
        elif mode == "tempfile":
            schop = temp.spec.head.b("CHOPEM", False)
            if schop:
                sem = temp.spec.head.d("EMREJ", sem)
                sab = temp.spec.head.d("ABSREJ", sab)
        elif mode == "specfile":
            schop = spec.head.b("CHOPEM", True)
        if spec.head.b("EMCHOP", False):
            schop = False
        tab, tem = float(f32(p["t_abs_reject"])), float(f32(p["t_em_reject"]))
        mode = str(p["t_emchop"])
        tchop = False
        if mode == "yes":
            tchop = True
        elif mode == "tempfile":
            tchop = temp.spec.head.b("CHOPEM", True)
            if tchop:
                tem = temp.spec.head.d("EMREJ", tem)
                tab = temp.spec.head.d("ABSREJ", tab)
        elif mode == "specfile":
            tchop = spec.head.b("CHOPEM", True)
        if temp.spec.head.b("EMCHOP", False):
            tchop = False
        return schop, sab, sem, tchop, tab, tem

    def xcfit(self, spec) -> XcResult | None:
        p = self.p
        self._specname = spec.object
        corr = str(p["correlate"])
        if corr not in CORRELATE:
            raise RvsaoError(f"correlate={corr} is not ported yet")
        self.cor = CORRELATE[corr]
        if self.cor == COR_NO:
            return self._from_header(spec)
        linear = self.cor in (COR_WAV, COR_PIX)
        ispix = self.cor == COR_PIX
        data = spec.data
        sh = spec.sh
        np1, np2 = sh.np1, sh.np2
        while data[np1 - 1] == 0.0:
            np1 += 1
        while data[np2 - 1] == 0.0:
            np2 -= 1
        tcont = spec.head.b("SUBCONT", True)
        divcon = spec.head.b("DIVCONT", False)
        conproc = (ZEROCONT if divcon else SUBCONT) if tcont else NOCONT
        contfit = tcont
        part = data[np1 - 1:np2]
        spmax, spmin = float(part.max()), float(part.min())
        renorm = bool(p["renormalize"]) or (spmax < 1.0 and spmin > -1.0)
        if renorm:
            nz = data[part.nonzero()[0] + np1 - 1] if False else part[part != 0]
            if nz.size == 0:
                self.out += "XCFIT: Spectrum is all zeroes\n"
                return None
            spmean = float(np.sum(nz.astype(np.float64))) / nz.size or 1.0
            factor = f32(1000.0)
            for i in range(data.size):
                if data[i] != 0.0:
                    data[i] = f32(float(factor) * (float(data[i]) / spmean))
        minwav0 = p["st_lambda"] if p["st_lambda"] not in (None, 0.0) else None
        maxwav0 = p["end_lambda"] if p["end_lambda"] not in (None, 0.0) else None
        zero_pad = str(p["zeropad"])
        han = f32(p["bell_window"])
        lo0, toplo0 = int(p["low_bin"]), int(p["top_low"])
        nrun0, tpnrn0 = int(p["nrun"]), int(p["top_nrun"])
        tshift = float(p["tshift"])
        ncols = int(p["ncols"])
        npts = 2 ** int(math.log(float(f32(ncols))) / LN2 + 1.0)
        lo0 = lo0 if 1 <= lo0 <= npts else 1
        toplo0 = toplo0 if 1 <= toplo0 <= npts else 1
        nrun0 = nrun0 if 1 <= nrun0 <= npts else npts
        tpnrn0 = tpnrn0 if 1 <= tpnrn0 <= npts else npts
        czguess, zinit = 0.0, False
        if self.vinit == 2:
            czguess, zinit = float(p["czguess"]), True
        elif self.vinit == 3:
            czguess, zinit = C0 * float(p["czguess"]), True
        nzpass = int(p["nzpass"]) + 1
        pkfrac0 = float(p["pkfrac"])
        minvel, maxvel = p["minvel"], p["maxvel"]
        if minvel is not None and maxvel is not None and maxvel < minvel:
            minvel, maxvel = maxvel, minvel
        result = XcResult(spec, npts=npts, han=float(han), tshift=tshift, span=(np1, np2))
        rmax = 0.0
        itmax = 0
        templates = self._templates()
        if not self._counted:
            self._counted = True
            for path, *_ in templates:
                if not Path(path).exists() and not Path(path + ".fits").exists():
                    self.err += f"COUNTEMP: cannot read template file {path} \n"
        itemp = 0
        for path, tempid, w1, w2 in templates:
            if not Path(path).exists() and not Path(path + ".fits").exists():
                self.err += f"XCFIT: cannot read template file {path} \n"
                continue
            itemp += 1
            temp = self._template(path, tempid, w1, w2, itemp)
            if temp is None:
                continue
            tr = TempResult(temp)
            schop, sab, sem, tchop, tab, tem = self._chops(spec, temp)
            if sab < sem:
                tr.achop, tr.schop = schop, False
            else:
                tr.schop, tr.achop = schop, False
            ttcont = temp.spec.head.b("SUBCONT", True)
            tr.scont = ttcont
            if not divcon:
                conproc = ZEROCONT if temp.spec.head.b("DIVCONT", False) else SUBCONT
            tr.conproc = conproc
            overlap = temp.spec.head.b("OVERLAP", False)
            if minwav0 is None and maxwav0 is None:
                overlap = False
            tr.overlap = overlap
            zpad = zero_pad == "yes" or (zero_pad == "tempfile"
                                         and temp.spec.head.b("ZEROPAD", False))
            if zpad:
                ncor = 2 * npts
                lo = 2 * lo0 if lo0 > 1 else lo0
                toplo = 2 * toplo0 if toplo0 > 1 else toplo0
                topnrn = min(2 * tpnrn0, ncor)
                nrun = min(2 * nrun0, ncor)
            else:
                ncor, lo, toplo, topnrn, nrun = npts, lo0, toplo0, tpnrn0, nrun0
            pkfrac = temp.spec.head.d("PEAKFRAC", pkfrac0)
            if ispix:
                tz = 1.0
            else:
                tz = ((1.0 + spec.hcv / C0) * (1.0 + temp.vel / C0)
                      * (1.0 + temp.shift / C0) * (1.0 + tshift / C0)
                      / (1.0 + temp.hcv / C0))
            zshift = (1.0 + czguess / C0) / tz if zinit else 1.0
            tcz = 0.0
            for izpass in range(1, nzpass + 1):
                tmstrt, tmfnsh = temp.start_log, temp.end_log
                if zinit or izpass > 1:
                    tmshft = dlog10(zshift)
                    tmstrt, tmfnsh = tmstrt + tmshft, tmfnsh + tmshft
                else:
                    tmshft = 0.0
                if ispix:
                    spwl1, spwl2 = 1.0, float(np2 - np1 + 1)
                    tmwl1, tmwl2 = 1.0, float(temp.spec.sh.np2 - temp.spec.sh.np1 + 1)
                else:
                    spwl1, spwl2 = sorted((10.0 ** spec.start_log, 10.0 ** spec.end_log))
                    tmwl1, tmwl2 = sorted((10.0 ** tmstrt, 10.0 ** tmfnsh))
                if spwl1 > tmwl2:
                    self.out += ("XCFIT: Spectrum starts after template "
                                 f"{temp.id} ends: {spwl1:9.2f} > {tmwl2:9.2f}\n")
                    break
                if spwl2 < tmwl1:
                    self.out += ("XCFIT: Spectrum ends before template "
                                 f"{temp.id} starts {spwl2:9.2f} < {tmwl1:9.2f}\n")
                    break
                minwav1, maxwav1 = max(spwl1, tmwl1), min(spwl2, tmwl2)
                if temp.wl1 != INDEFD:
                    minwav = temp.wl1
                elif overlap or minwav0 is None or minwav0 < minwav1:
                    minwav = minwav1
                else:
                    minwav = minwav0
                if temp.wl2 != INDEFD:
                    maxwav = temp.wl2
                elif overlap or maxwav0 is None or maxwav0 > maxwav1:
                    maxwav = maxwav1
                else:
                    maxwav = maxwav0
                if ispix:
                    minwav, maxwav = float(np1), float(np2)
                tr.wl1, tr.wl2 = minwav, maxwav
                logmin, logmax = dlog10(minwav), dlog10(maxwav)
                dlogw = (logmax - logmin) / float(npts - 1)
                self.delwav = (maxwav - minwav) / float(npts - 1)
                kind = str(p["interp_mode"])
                if linear:
                    shspec = rebin(data, spec.wcs, 0.0, npts, minwav, self.delwav, ispix,
                                   kind)
                    shtemp = rebin(temp.spec.data, temp.spec.wcs, tmshft, npts, minwav,
                                   self.delwav, ispix, kind)
                else:
                    shspec = rebinl(data, spec.wcs, 0.0, npts, logmin, dlogw, kind)
                    shtemp = rebinl(temp.spec.data, temp.spec.wcs, tmshft, npts, logmin,
                                    dlogw, kind)
                tvel = ((tz * zshift) - 1.0) * C0 if (zinit or izpass > 1) else \
                    (tz - 1.0) * C0
                tcz = self.xcorfit(shspec, shtemp, tr, ncor, (lo, toplo, topnrn, nrun),
                                   dlogw, tvel, contfit, conproc, ttcont, han, pkfrac,
                                   minvel, maxvel, (schop, sab, sem), (tchop, tab, tem))
                if linear:
                    tr.zvel = tr.cz
                    if zinit or izpass > 1:
                        tr.zvel += zshift
                        tr.vshift = zshift
                    else:
                        tr.vshift = 0.0
                else:
                    tr.zvel = (1.0 + (tr.cz / C0)) * tz
                    if zinit or izpass > 1:
                        tr.zvel *= zshift
                        tr.vshift = (zshift - 1.0) * C0
                    else:
                        tr.vshift = 0.0
                    tr.zvel = (tr.zvel - 1.0) * C0
                if nzpass > 1 and izpass < nzpass:
                    if linear:
                        zshift += tcz
                    else:
                        zshift *= (1.0 + tcz / C0)
            else:
                result.filter = (lo, toplo, topnrn, nrun)
                result.zpad, result.ncor, result.fracpeak = zpad, ncor, abs(pkfrac)
                result.temps.append(tr)
                if tr.czr >= rmax:
                    rmax = tr.czr
                    itmax = len(result.temps)
                continue
            result.temps.append(tr)
        if not result.temps:
            self.out += "XCFIT: No templates; no cross-correlation\n"
            return None
        result.best = max(itmax, 1)
        best = result.temps[result.best - 1]
        result.npmax = best.npfit
        result.xcvel, result.xcerr, result.xcr = best.zvel, best.czerr, best.czr
        return result

    def _from_header(self, spec) -> XcResult:
        """correlate=no: what an xcsao before wrote to the header (xcrhead), shown
        and reported as if just correlated -- with what xcrhead does not read left
        as RVSAO leaves it: no correlation (0 points), the peak at 0, the peak
        fraction 0, the templates' continuum not subtracted (+tc), the filter 0 0 0 0
        when there is no CZPARAMS. One empty template when there is nothing."""
        from types import SimpleNamespace

        from .emsao import xcrhead

        p = self.p
        head = spec.head
        xcvel, xcerr, xcr, temps, best = xcrhead(head, spec.mspec)
        ncols = int(p["ncols"])
        npts = 2 ** int(math.log(float(f32(ncols))) / LN2 + 1.0)
        filt, zpad, han = (0, 0, 0, 0), False, f32(p["bell_window"])
        words = []
        if spec.mspec <= 0 and head.has("CZPARAMS"):
            words = (head.string("CZPARAMS") or "").split()
        elif spec.mspec > 0 and head.has(f"APVXC{spec.mspec}"):
            words = (head.string(f"APVXC{spec.mspec}") or "").split()[5:]
        if len(words) >= 8 and int(words[0]) > 0:
            npts, zpad = int(words[1]), words[2] == "2"
            filt = (int(words[3]), int(words[4]), int(words[5]), int(words[6]))
            han = f32(float(words[7]))
        minwav, maxwav = spec.sh.w0, spec.sh.w1
        st, end = p.get("st_lambda"), p.get("end_lambda")
        if st not in (None, 0.0) and st > minwav:
            minwav = st
        if end not in (None, 0.0) and end < maxwav:
            maxwav = end
        blank = SimpleNamespace(object="")        # tempname: gettemp never ran
        result = XcResult(spec, npts=npts, han=float(han), tshift=float(p["tshift"]),
                          span=(spec.sh.np1, spec.sh.np2), filter=filt, zpad=zpad,
                          ncor=0, fracpeak=0.0)
        for t in temps or [None]:
            template = Template("", t.id if t else "", blank, shift=t.shift if t else 0.0,
                                hcv=t.hcv if t else 0.0, filt=t.filt if t else 0)
            tr = TempResult(template, czerr=t.czerr if t else 0.0,
                            czr=t.czr if t else 0.0, zvel=t.zvel if t else 0.0,
                            height=t.height if t else 0.0, vwidth=t.width if t else 0.0,
                            wl1=minwav, wl2=maxwav, schop=t.schop if t else False,
                            scont=False, conproc=0)
            result.temps.append(tr)
        result.best = max(best, 1)
        top = result.temps[result.best - 1]
        result.xcvel, result.xcerr, result.xcr = top.zvel, top.czerr, top.czr
        return result

    def qsave(self, path: str, mspec: int, r: XcResult, nmspec: int) -> None:
        """qwhead after correlate=no with save_vel (which then only lets the file be
        written): VELOCITY, Z, CZERR, VELQUAL and BCV (when absent), or APVELn, and
        the QPLOT card, the one before kept as HISTORY."""
        import os

        from .emsao import _dcard, _f
        from .rvsao import HeaderUpdate, history_card, logtime, string_card

        if not Path(path).exists():
            path += ".fits"
        if not os.access(path, os.W_OK):
            self.err += f"XCSAO: cannot write to {path}; not saving results\n"
            return
        q = {1: "?", 2: "?", 3: "X", 4: "Q"}.get(r.quality, "_")
        head = HeaderUpdate(path)
        try:
            if mspec <= 0:
                cards = [_dcard("VELOCITY", r.vel), _dcard("Z", r.vel / C0),
                         _dcard("CZERR", r.err), string_card("VELQUAL", q)]
                if head.h.get("BCV") is None:
                    cards.append(_dcard("BCV", r.spectrum.hcv))
            else:
                cards = [string_card(f"APVEL{mspec}",
                                     f"{_f(r.vel, 0, 3)} {_f(r.err, 0, 3)} {q} "
                                     f"{_f(r.spectrum.hcv, 0, 3)}")]
            for image in cards:
                head.put(image)
            stamp = f"rvsao.xcsao {VERSION} {logtime()} "
            stamp += (f"Quality[{self.p['specnum']}] set" if mspec > 0 and nmspec == 1
                      else f"VELOCITY = {_f(r.vel, 0, 2)} Q = {q}")
            old = head.string("QPLOT")
            if old:
                head.put(history_card("HISTORY", old))
            head.put(string_card("QPLOT", stamp))
        finally:
            head.close()
        r.saved = path

    def _choose_peak(self, xcor, xvel, tr: TempResult) -> int:
        """xcorfit's plotspec of the filtered correlation, for its peak to be chosen
        (p there): the point chosen, 0 for none."""
        from .plotspec import Plotspec

        label = {COR_PIX: "Shift in Pixels", COR_WAV: "Shift in Wavelength (Angstroms)"}\
            .get(self.cor, "Shift in Velocity (km/sec)")
        title = (f"Filtered cross-correlation: {self._specname} X "
                 f"{tr.template.spec.object}")
        ps = Plotspec(xcor, xvel, title, label, self.nsmooth, "rvsao.xcsao", self.p,
                      self.read_key or (lambda: None), self.read_line or (lambda: None),
                      self._say)
        if self.show_plotspec is not None:
            self.show_plotspec(ps)
        else:
            ps.run()
        self.drawn.extend(ps.frames)
        return ps.peak

    def xcorfit(self, spectrum, template, tr: TempResult, ncor: int, filt, dlogw: float,
                tvel: float, contfit: bool, conproc: int, tcont: bool, han, pkfrac,
                minvel, maxvel, schop, tchop) -> float:
        lo, toplo, topnrn, nrun = filt
        ncor2 = ncor // 2
        log2 = int(math.log(float(f32(ncor))) / LN2 + 0.001)
        index = np.arange(1, ncor + 1) - ncor2 - 1
        xind = index.astype(f32)
        linear = self.cor in (COR_WAV, COR_PIX)
        delwav = self.delwav
        if linear:
            xvel = np.array([f32(float(k) * delwav) for k in index], f32)
        else:
            xvel = np.array([f32(C0 * (10.0 ** (k * dlogw) - 1.0) + tvel) for k in index],
                            f32)
        npts = spectrum.size
        tfilt = tr.template.filt
        self._spec_subcon = (conproc, schop)   # the commons xcplot's icsubcon reads
        if contfit:
            said: list[str] = []
            icsubcon(spectrum, self.p, 1, conproc, schop[0], schop[1], schop[2], said)
            self.out += "".join(said)
        apodize(spectrum, han)
        re1 = np.zeros(ncor, f32)
        im1 = np.zeros(ncor, f32)
        re1[:npts] = spectrum
        fourm(re1, im1, log2, 1)
        if tcont:
            said = []
            icsubcon(template, self.p, 2, conproc, tchop[0], tchop[1], tchop[2], said)
            self.out += "".join(said)
            apodize(template, han)
        re2 = np.zeros(ncor, f32)
        im2 = np.zeros(ncor, f32)
        re2[:npts] = template
        fourm(re2, im2, log2, 1)
        if lo != nrun:
            flter2(re1, im1, lo, toplo, topnrn, nrun, tfilt)
        if lo != nrun and tfilt not in (1, 3):
            flter2(re2, im2, lo, toplo, topnrn, nrun, tfilt)
        # correl: ft1 times the conjugate of ft2
        cre = ((re1 * re2).astype(f32) + (im1 * im2).astype(f32)).astype(f32)
        cim = ((-(re1 * im2)).astype(f32) + (im1 * re2).astype(f32)).astype(f32)
        fourm(cre, cim, log2, -1)
        xcor = cre.copy()
        half = ncor // 2
        xcor = np.concatenate([xcor[half:], xcor[:half]])
        _rms1, _rms2, fnorm = rmsnorm(re1, im1, re2, im2)
        xcor = (xcor.astype(np.float64) * fnorm).astype(f32)
        fracpeak = abs(pkfrac)
        # pltcor (xcplot's p): the correlation shown to choose its peak (z[1])
        z = self._choose_peak(xcor, xvel, tr) if self.pltcor else 0
        start = z if z > 0 else 0
        pkmode = int(self.p["pkmode"])
        if pkmode == 2:
            said = []
            npfit, pc, pw, ph, pkindx = pkfitq(xind, xcor, xvel, fracpeak, start, minvel,
                                               maxvel, said)
            self.out += "".join(said)
        elif pkmode == 3:
            said: list[str] = []
            npfit, pc, pw, ph, pkindx = pkfitc(xind, xcor, xvel, fracpeak, start, minvel,
                                               maxvel, said)
            for line in said:
                self.out += line
        else:
            # xcorfit: any other mode is 1, and is reported as 1
            self.p["pkmode"] = 1
            npfit, pc, pw, ph, pkindx = pkfitp(xind, xcor, xvel, fracpeak, start, minvel,
                                               maxvel)
        tcz = pc * delwav if linear else C0 * (10.0 ** (pc * dlogw) - 1.0)
        fourm(cre, cim, log2, 1)
        arms, srms = aspartf(cre, cim, lo, nrun, pc, tfilt)
        arms, srms = arms * fnorm, srms * fnorm
        tr.center, tr.height, tr.width = pc, ph, pw
        tr.arms, tr.srms, tr.npfit = arms, srms, npfit
        if arms > 0:
            tczr = ph / (2.0 * arms) if pw > 0 else 0.0
            pheight, pwidth, _il, _ir, pkindx = pkwidth(xcor, xvel, 0.5, start, minvel,
                                                        maxvel)
            pwidth = abs(pwidth)
            epsilon = 3.0 * pwidth / 8.0 / (1.0 + tczr)
            tczerr = epsilon * delwav if linear else \
                C0 * (10.0 ** (epsilon * dlogw) - 1.0)
        else:
            tczr = tczerr = pwidth = pheight = 0.0
        # the peak as measured: its bin, its height, and its half width in bins
        # (Memr[xind+pkindx+pwidth/2-1]: the pointer's sum truncated)
        self.pkindx = pkindx                  # the common xcorplot reads it from
        tr.pcent, tr.phght = float(xind[pkindx - 1]), pheight
        hi = min(max(math.floor(pkindx - 1 + pwidth * 0.5), 0), ncor - 1)
        lo_ = min(max(math.floor(pkindx - 1 - pwidth * 0.5), 0), ncor - 1)
        tr.pwdth = 0.5 * (float(xind[hi]) - float(xind[lo_]))
        if (minvel is not None and tcz + tvel < minvel) or \
                (maxvel is not None and tcz + tvel > maxvel):
            tcz = float(xvel[pkindx - 1])
            tczr = tczerr = 0.0
            tr.center, tr.height, tr.width = float(xind[pkindx - 1]), \
                float(xcor[pkindx - 1]), 0.0
        tcz1 = (pc - (pwidth * 0.5)) * delwav if linear else \
            C0 * (10.0 ** ((pc - (pwidth * 0.5)) * dlogw) - 1.0)
        tr.vwidth = tcz - tcz1
        tr.cz, tr.czerr, tr.czr = tcz, tczerr, tczr
        tr.xvel, tr.xcor, tr.dlogw = xvel, xcor, dlogw
        return tcz

    # ---------------------------------------------------------------- report

    def _report_line(self, rmode: int, r: XcResult, sp: str) -> str:
        """xcrslts' report modes 3 to 18: a line per spectrum or per template, in
        the formats of SAO's archives and instruments (sp: a tab between the fields
        of modes 8, 10, 11, 12 and 14 when report_mode is negative)."""
        from .emsao import round_half, vcombine
        from .iraffmt import Format
        from .rvsao import juldate

        def f(fmt: str, value) -> str:
            return Format(fmt)(value)

        spec, head = r.spectrum, r.spectrum.head
        specid = spec.name
        _ut, dj, djh, _date = juldate(head)
        instrument = (head.string("INSTRUME") or "")[:8]
        incode = "T" if instrument.startswith("echelle") else \
            "M" if instrument.startswith("mmtech") else \
            "W" if instrument.startswith("oroech") else "_"
        temps = r.temps
        ntemp = len(temps)
        best = temps[r.best - 1]
        hcv = spec.hcv
        vel0, err0, r0, bcv0, qual0 = r.old
        out = ""

        def stem(text: str) -> str:   # strcpy (specid, filename, strldx ('.') - 1)
            nc = text.rfind(".")
            return specid[:nc] if nc > 0 else ""

        if rmode == 3:
            out += (f"{specid[:16]:<16.16} {f('%6.2f', r0)} {f('%6.2f', best.czr)} "
                    f"{f('%9.3f', vel0)} {f('%9.3f', best.zvel)} {f('%8.3f', err0)} "
                    f"{f('%8.3f', best.czerr)} {f('%6.3f', bcv0)} {f('%6.3f', hcv)} "
                    f"{f('%12.2f', djh)} {qual0} {best.template.id}\n")
        elif rmode == 4:
            name = stem(r.specfile)
            for t in temps:
                out += (f"{name:<16.16} {t.template.id} {f('%6.2f', t.czr)} "
                        f"{f('%9.3f', t.zvel)} {f('%8.3f', t.czerr)}\n")
        elif rmode == 5:
            lo, toplo, topnrn, nrun = r.filter
            for t in temps:
                out += (f"{lo:2d} {toplo:2d} {topnrn:3d} {nrun:3d} "
                        f"{t.template.id:<16.16} {f('%5.2f', t.template.shift)} "
                        f"{specid:<14.14} {f('%6.2f', t.zvel)} {f('%4.1f', t.czr)} "
                        f"{f('%4.2f', t.height)} {f('%5.1f', peak_width(t))} "
                        f"{f('%5.2f', hcv)}\n")
        elif rmode in (6, 9, 15):
            for t in temps:
                width = peak_width(t)
                if rmode != 9:
                    width = max(width, 0.0)
                if rmode == 6:
                    out += (f"{incode} {specid:<14.14} {t.template.id:<16.16} "
                            f"{f('%14.5f', djh)} {f('%7.2f', t.zvel)} {f('%6.2f', t.czerr)} "
                            f"{f('%5.3g', t.czr)} {f('%4.2f', t.height)} {f('%5.1f', width)} "
                            f"{f('%6.2f', hcv)}\n")
                elif rmode == 9:
                    out += (f"{incode} {specid:<14.14} {t.template.id:<16.16} "
                            f"{f('%14.5f', djh)} {f('%8.3f', t.zvel)} {f('%7.3f', t.czerr)} "
                            f"{f('%4.1f', t.czr)} {f('%4.2f', t.height)} {f('%5.1f', width)} "
                            f"{f('%6.2f', hcv)}\n")
                else:
                    out += (f"{incode} {specid:<14.14} {t.template.id:<18.18} "
                            f"{f('%14.5f', djh)} {f('%9.3f', t.zvel)} {f('%7.3f', t.czerr)} "
                            f"{f('%5.3g', t.czr)} {f('%5.3f', t.height)} {f('%7.3f', width)} "
                            f"{f('%7.3f', hcv)}\n")
        elif rmode == 7:
            nnova = head.i("NTEMP", 0)
            qnova = head.i("QCSTAT", 0)
            nova = {}
            for i in range(1, nnova + 1):
                name = (head.string(f"TEMPL{i}") or "")
                name = name.split(" ")[0].lower() if name else ("ztemp" if i == 1 else "")
                nova[name] = (head.d(f"TCZXCR{i}", 0.0), head.d(f"TCZXC{i}", 0.0),
                              head.d(f"TCZXCER{i}", 0.0), head.d(f"TXCHGHT{i}", 0.0),
                              head.d(f"TXCARMS{i}", 0.0))
            q = {1: "?", 2: "?", 3: "X", 4: "Q"}.get(qnova, "_")
            for t in temps:
                name = t.template.id if t.template.id in nova else "none"
                nr, nv, ne, nh, na = nova.get(t.template.id, (0.0, 0.0, 0.0, 0.0, 0.0))
                out += (f"{specid[:16]:>16.16} {f('%6.2f', nr)} {f('%6.2f', t.czr)} "
                        f"{f('%9.3f', nv)} {f('%9.3f', t.zvel)} {f('%8.3f', ne)} "
                        f"{f('%8.3f', t.czerr)} {f('%7.5f', float(f32(nh)))} "
                        f"{f('%7.5f', t.height)} {f('%7.5f', float(f32(na)))} "
                        f"{f('%7.5f', t.arms)} {f('%12.2f', dj)} {t.template.id} {name} "
                        f"{q}\n")
        elif rmode == 8:
            vel, err = vcombine(r.xcvel, r.xcerr, r.xcr, r.emvel, r.emerr, r.emnlf)

            def pair(v, e):
                return (int(round_half(v)), int(round_half(e))) if v != INDEFD else (0, 0)
            icz, icze = pair(vel, err)
            iczxc, iczxce = pair(r.xcvel, r.xcerr)
            iczem, iczeme = pair(r.emvel, r.emerr)
            title = (head.string("OBJECT") or "").strip() or "_" * 16
            out += (f"{specid:>6}{sp}{instrument or '_' * 8:<8}{sp}{title:<16}{sp}"
                    f"{f('%15.5f', dj)}{sp}{iczem:5d}{sp}{iczeme:3d}{sp}{iczxc:5d}{sp}"
                    f"{iczxce:3d}{sp}{f('%5.1f', r.xcr)}{sp}{icz:5d}{sp}{icze:3d}{sp}C"
                    f"{sp}{r.emnl:2d}{sp}{r.emnlf:2d}{sp}{best.template.id}\n")
        elif rmode in (10, 11, 12, 14):
            name = specid
            if rmode == 12:
                name = name.replace("[", " ", 1).replace("]", " ", 1)
            width = 14 if rmode == 10 else 32
            best_i = r.best
            if rmode == 11 and best.czr < 1.0:
                best_i = 0
            line = f"{name:<{width}.{width}}{sp}{f('%14.5f', dj)}"
            if rmode == 14:
                line += (f"{sp}{f('%7.1f', r.emvel)}{sp}{f('%5.1f', r.emerr)}{sp}"
                         f"{r.emnl}{sp}{r.emnlf}{sp}{r.best}{sp}{best.template.id:<14.14}")
            else:
                line += f"{sp}{best_i}"
            tw = 12 if rmode == 10 else 14
            if rmode in (10, 11) and ntemp > 1:
                if rmode == 11 and best.czr < 1.0:
                    line += (f"{sp}{'none':<14.14}{sp}{f('%7.1f', 0.0)}{sp}"
                             f"{f('%5.1f', 0.0)}{sp}{f('%4.2f', best.czr)}")
                else:
                    line += (f"{sp}{best.template.id:<{tw}.{tw}}{sp}{f('%7.1f', best.zvel)}"
                             f"{sp}{f('%5.1f', best.czerr)}{sp}{f('%4.2f', best.czr)}")
            for t in temps:
                if rmode in (11, 12):
                    line += (f"{sp}{t.template.id:<14.14}{sp}{f('%7.7g', t.zvel)}{sp}"
                             f"{f('%5.5g', t.czerr)}{sp}{f('%4.2f', t.czr)}")
                else:
                    line += (f"{sp}{t.template.id:<{tw}.{tw}}{sp}{f('%7.1f', t.zvel)}{sp}"
                             f"{f('%5.1f', t.czerr)}{sp}{f('%4.2f', t.czr)}")
            out += line + "\n"
        elif rmode == 13:
            for t in temps:
                taa = 1.0 / t.dlogw / math.log(10.0) if t.dlogw else 0.0
                if taa != 0:
                    dlogw = 1.0 / (math.log(10.0) * taa)
                    width = C0 * ((10.0 ** (t.pwdth * dlogw)) - 1.0)
                    z1 = (10.0 ** (t.pcent * dlogw)) * (1.0 + (hcv / C0))
                    vel = C0 * (z1 - 1.0)
                else:
                    width, vel = t.vwidth, t.cz
                out += (f"{incode} {specid:<14.14} {t.template.id:<16.16} "
                        f"{f('%14.5f', djh)} {f('%7.2f', vel)} {f('%7.5f', t.phght)} "
                        f"{f('%6.2f', width)} {f('%6.2f', hcv)}\n")
        elif rmode == 16:
            for t in temps:
                w1, w2 = t.template.wl1, t.template.wl2
                mid = (w1 + w2) * 0.5
                # INDEF + INDEF overflows: IRAF prints the infinity so
                tail = f('%7.1f', mid) if math.isfinite(mid) else "0.00E309"
                out += (f"{specid:<24.24} {t.template.id:<16.16} {f('%6.2f', t.czr)} "
                        f"{f('%9.3f', t.zvel)} {f('%8.3f', t.czerr)} {f('%5.3f', t.height)} "
                        f"{f('%7.1f', w1)} {f('%7.1f', w2)} {tail}\n")
        elif rmode in (17, 18):
            aperture = r.spectrum.mspec if r.spectrum.mspec > 0 else head.i("APERTURE", 1)
            fiber, beam = head.i("FIBER", 0), head.i("BEAM", 1)
            for t in temps:
                tid = t.template.id
                if rmode == 17:
                    width = max(peak_width(t), 0.0)
                    k = tid.rfind("/") + 1
                    start = k if k > 0 else max(len(tid) - 35, 0)
                    # tempinfo: copied in without its EOS, over what the template
                    # before left (an EOS at 28 only)
                    buf = self._tempinfo
                    buf[27] = ""
                    for j, ch in enumerate(tid[start:start + min(len(tid), 35)]):
                        buf[j] = ch
                    info = "".join(buf[:buf.index("")])
                    out += (f"{aperture:03d} {fiber:03d} {beam} {spec.object:<24.24} "
                            f"{info:<35.35} {f('%14.5f', djh)} {f('%9.3f', t.zvel)} "
                            f"{f('%8.3f', t.czerr)} {f('%6.3f', t.czr)} "
                            f"{f('%6.4f', t.height)} {f('%6.2f', width)} {f('%7.3f', hcv)}\n")
                else:
                    tempfile = tid.split("[")[0]
                    out += (f"{aperture:02d} {r.specfile} {tempfile} {f('%14.5f', djh)} "
                            f"{f('%9.5f', t.zvel)} {f('%8.5f', t.czerr)} {f('%6.3f', t.czr)} "
                            f"{f('%7.3f', hcv)}\n")
        return out

    def _report(self, r: XcResult) -> None:
        from . import iraffmt
        from .emsao import _f
        from .rvsao import BCV, FBCV, FHCV, HCV, NONE, juldate

        rmode0 = int(self.p["report_mode"])
        rmode = abs(rmode0)
        if rmode >= 3:
            self._log(self._report_line(rmode, r, "\t" if rmode0 < 0 else " "))
            return
        if rmode == 2:
            text = ""
            for t in r.temps:
                width = peak_width(t)
                text += (f"{r.spectrum.name[:16]:<16s} {t.template.id[:16]:<16s} "
                         f"{_f(t.czr, 6, 2)} {_f(t.zvel, 9, 3)} {_f(t.czerr, 8, 3)} "
                         f"{_f(t.height, 5, 3)} {_f(width, 7, 3)}\n")
            self._log(text)
            return
        if rmode != 1:
            return                      # 0: the archive's, which xcsao does not write
        spec = r.spectrum
        head = spec.head
        ra = head.d("RA", INDEFD)
        if ra != INDEFD and ":" not in (head.string("RA") or ""):
            ra = ra / 15.0
        dec = head.d("DEC", INDEFD)
        epoch = head.d("EPOCH", 1950.0)
        epoch = head.d("EQUINOX", epoch)
        text = f"\n{spec.name} Object: {spec.object} \n"
        if ra != INDEFD and dec != INDEFD:
            text += (f"RA: {iraffmt.Format('%011.2h')(ra)} Dec: "
                     f"{iraffmt.Format('%010.1h')(dec)} {_f(epoch, 0, 1)}\n")
        else:
            text += "(No position)\n"
        ut, dj, djh, (mm, dd, yyyy) = juldate(head)
        if None in (mm, dd, yyyy):
            text += "(No observation date) "
        else:
            month = "JanFebMarAprMayJunJulAugSepOctNovDec"[3 * (mm - 1):3 * mm] \
                if 1 <= mm <= 12 else ""
            jd = f"=HJD{_f(djh, 0, 4)} " if djh > 0 else f"=JD{_f(dj, 0, 4)} "
            text += f"{yyyy:4d}-{month:>3.3s}-{dd:02d} {iraffmt.Format('%011.2h')(ut)} {jd}"
        hcv = f"HCV: {_f(spec.hcv, 6, 2)}"
        bcv = f"BCV: {_f(spec.hcv, 6, 2)}"
        if self.svcor == HCV:
            text += hcv
        elif self.svcor == BCV:
            text += bcv
        elif self.svcor in (FHCV, FBCV):
            text += bcv if spec.vb else hcv
        elif self.svcor == NONE:
            pass
        text += "\n"
        best = r.temps[r.best - 1]
        text += f"{_f(best.wl1, 0, 1)}A- {_f(best.wl2, 0, 1)}A "
        text += f"{r.npts}x2 points, " if r.zpad else f"{r.ncor} points, "
        lo, toplo, topnrn, nrun = r.filter
        text += (f"filter: {lo} {toplo} {topnrn} {nrun} apodize {_f(r.han, 4, 2)} "
                 f"fit {_f(r.fracpeak, 3, 2)} best {r.best}\n")
        if r.tshift != 0.0:
            text += f"Tshift: {_f(r.tshift, 6, 2)} \n"
        for t in r.temps:
            line = (f"Temp: {t.template.id:>16s} vel: {_f(t.template.vel, 6, 2)} "
                    f" tsh: {_f(t.template.shift, 6, 2)} HCV: {_f(t.template.hcv, 6, 2)} "
                    f"Peak: {_f(t.cz, 7, 3)}"
                    f" h:{_f(t.height, 5, 3)} R: {_f(t.czr, 6, 2)} "
                    f"{'Shift' if self.cor in (COR_PIX, COR_WAV) else 'CZ'}: "
                    f"{_f(t.zvel, 7, 3)} "
                    f"+/- {_f(t.czerr, 7, 3)} {int(self.p['pkmode'])}")
            if t.schop:
                line += " -el"
            if t.achop:
                line += " -al"
            filt = t.template.filt
            line += {1: " -tf", 2: " +hi", 3: " +hi -tf"}.get(filt, "")
            if not t.scont:
                line += " +tc"
            elif t.conproc == ZEROCONT:
                line += " divc"
            text += line + "\n"
        self._log(text)
