"""mkapfile / apfile: aperture corrections from the curve of growth.

From photcal/mkobsfile (t_apfile.x, phaimtable.x, phagrow.x, dinvers.f),
non-interactive: Stetson's DAOGROW. Each star's magnitude differences
between successive apertures are fitted, all images together, by a model
profile -- a Moffat wing, a Gaussian core and an exponential, in proportions
that are the model's parameters -- with one seeing radius per image. The
adopted curve of growth of an image is then the weighted mean of its stars'
differences, pulled towards the model where they are few or noisy, and its
sum from the small aperture to the large one is the aperture correction.
Each star's best magnitude is the one, of those corrected to the large
aperture, with the smallest error.

The arithmetic is IRAF's, as its Fortran was compiled (f2c -R): single
precision, but the model and its parameters are double; a real to a power
that is not a constant is pow_ri's double, and the log10 is libf2c's
d_lg10, log10(e) * log(x), which differs from log10 in the last bit of
about one value in eight. Each of the three changes the last digit of
the fitted model in some of the Skinakas images.
Two quirks kept: the model's d1 = 1 + r1**2 is in pixels, not seeing radii;
and the first fit's convergence test starts from 0 (a static variable in
a fresh process -- run twice in one IRAF process, the second differs).

Translated from IRAF, Copyright (c) 1986 Association of Universities for
Research in Astronomy Inc.; the notice it is distributed under is in
LICENSES/IRAF-COPYRIGHT.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .. import iraffmt
from ..jit import jit
from .config import ctor
from .io import cnvtime

f32 = np.float32
f64 = np.float64       # in numba, float() of a float32 stays a float32
EPSILONR = f32(1.192e-7)
MAX_REAL = f32(3.4028235e38)
DEF_AIROFFSET = f32(1.25)
MAX_MTERMS = 5
AGR_ITMAX = 300
NPERLINE = 7
NAN = f32(np.nan)
LOG10E = 0.43429448190325182765


# ------------------------------------------------------------------ input


@dataclass
class Image:
    """IMT_*: an image's entry in the table."""
    name: str
    number: int
    ifilter: str = "INDEF"
    itime: float = float("nan")
    xairmass: np.float32 = NAN
    nxairmass: np.float32 = NAN
    otime: np.float32 = NAN
    ro: np.float32 = NAN
    stars: list = field(default_factory=list)


@dataclass
class Star:
    id: int
    x: np.float32
    y: np.float32
    nap: int
    rap: np.ndarray
    mag: np.ndarray       # the first, then differences
    merr: np.ndarray
    name: str = ""        # the star's name in the photometry, when it has one


def _real(text) -> np.float32:
    value = ctor(str(text).strip())
    return NAN if value is None else f32(value)


def star_data(x, y, rap, mag, merr, maglim) -> Star | None:
    """ph_astardata: the apertures with a magnitude, an error and an error
    under maglim, from the first; then the differences between successive
    ones, their errors with a floor of 0.001."""
    rap = np.array(rap, dtype=f32)
    mag = np.array(mag, dtype=f32)
    merr = np.array(merr, dtype=f32)
    nap = 0
    for i in range(len(mag)):
        if np.isnan(mag[i]) or np.isnan(merr[i]) or merr[i] > f32(maglim):
            break
        nap += 1
    if nap <= 0:
        return None
    for i in range(nap - 1, 0, -1):
        mag[i] = f32(mag[i] - mag[i - 1])
        merr[i] = np.sqrt(f32(f32(1.0e-6) + f32(merr[i] * merr[i])))
    return Star(0, f32(x), f32(y), nap, rap, mag, merr)


def read_photfiles(paths: list[str], naperts: int, maglim: float = 0.10,
                   notes: list[str] | None = None) -> list[Image]:
    """ph_amkimtable over txdump of apphot files: the images in the order
    they first appear, each with its stars in order."""
    from .photfiles import read

    images: dict[str, Image] = {}
    notes = notes if notes is not None else []
    for path in paths:
        table = read(path)
        for row in table:
            name = str(row["IMAGE"]).strip()
            image = images.get(name)
            if image is None:
                image = images[name] = Image(name, len(images) + 1)
            if image.ifilter == "INDEF":
                image.ifilter = str(row["IFILTER"]).strip()
            if np.isnan(image.itime):
                itime = _real(row["ITIME"])
                image.itime = 1.0 if np.isnan(itime) else float(itime)
            if np.isnan(image.xairmass):
                airmass = _real(row["XAIRMASS"])
                if not np.isnan(airmass):
                    image.xairmass = airmass
                    image.nxairmass = f32(airmass - DEF_AIROFFSET)
                else:
                    image.xairmass = DEF_AIROFFSET
                    image.nxairmass = f32(0.0)
                    notes.append(f"    Warning: Setting airmass for image {name} to "
                                 f"{iraffmt.g(float(DEF_AIROFFSET))}")
            if np.isnan(image.otime):
                image.otime = _real(row["OTIME"])
            star = star_data(
                _real(row["XCENTER"]), _real(row["YCENTER"]),
                [_real(row[f"RAPERT{k}"]) for k in range(1, naperts + 1)],
                [_real(row[f"MAG{k}"]) for k in range(1, naperts + 1)],
                [_real(row[f"MERR{k}"]) for k in range(1, naperts + 1)], maglim)
            if star is None:
                continue
            if "ID" in table.colnames:
                star.name = "".join(str(row["ID"]).split())
            image.stars.append(star)
            star.id = len(image.stars)
            if star.id == 1:
                image.ro = f32(f32(0.5) * star.rap[0])
    return list(images.values())


# ------------------------------------------------------------------ model
#
# The fit is the slow part -- the model is evaluated some thirty thousand
# times for twelve images -- so it is written as kernels over arrays, which
# numba compiles when it is installed (core.jit); without it they run as
# they are, to the same numbers.

D001 = float(f32(0.001))          # the parameter steps, a real in IRAF
D002 = float(f32(0.002))
CONVERGED = float(f32(3.0e-4))


@jit
def dmag(r1, r2, x, r0, p) -> float:
    """ph_dmag: the model's magnitude difference between two radii."""
    bpex = p[1] + p[4] * f64(x)
    pm1 = p[0] - 1.0
    x1 = f64(f32(f32(r1) / f32(r0)))
    x2 = f64(f32(f32(r2) / f32(r0)))
    x1sq = x1 * x1
    x2sq = x2 * x2
    x1 = x1 / p[3]
    x2 = x2 / p[3]
    d1 = 1.0 + f64(f32(f32(r1) * f32(r1)))
    d2 = 1.0 + f64(f32(f32(r2) * f32(r2)))
    i1 = (bpex * (1.0 - 1.0 / d1 ** pm1) + (1.0 - bpex)
          * (p[2] * (1.0 - math.exp(-0.5 * x1sq)) + (1.0 - p[2])
             * (1.0 - (1.0 + x1) * math.exp(-x1))))
    i2 = (bpex * (1.0 - 1.0 / d2 ** pm1) + (1.0 - bpex)
          * (p[2] * (1.0 - math.exp(-0.5 * x2sq)) + (1.0 - p[2])
             * (1.0 - (1.0 + x2) * math.exp(-x2))))
    return -2.5 * (LOG10E * math.log(i2 / i1))       # libf2c's d_lg10


@jit
def _pow_ri(x: float, n: int) -> float:
    """libf2c pow_ri: a real to an integer power, in double."""
    result = 1.0
    if n:
        u = n
        while True:
            if u & 1:
                result *= x
            u >>= 1
            if not u:
                break
            x *= x
    return result


@jit
def _weight(diff, merr, power, constant):
    """1 / (1 + |diff / 2 merr| ** power). For a constant exponent f2c
    writes the power inline and it stays in single precision; for a
    variable one it calls pow_ri, which returns a double, and the sum and
    the division are done in double and rounded once."""
    q = abs(f32(diff / f32(f32(2.0) * merr)))
    if constant:
        p = q
        for _ in range(power - 1):
            p = f32(p * q)
        return f32(f32(1.0) / f32(p + f32(1.0)))
    return f32(1.0 / (_pow_ri(f64(q), power) + 1.0))


@jit
def dinver(a: np.ndarray, n: int) -> int:
    """dinvers.f: invert the n by n leading part of a in place, no pivoting;
    1 when a diagonal element is zero."""
    for i in range(n):
        if a[i, i] == 0.0:
            return 1
        a[i, i] = 1.0 / a[i, i]
        for j in range(n):
            if j == i:
                continue
            a[j, i] = -a[j, i] * a[i, i]
            for k in range(n):
                if k == i:
                    continue
                a[j, k] = a[j, k] + a[j, i] * a[i, k]
        for k in range(n):
            if k != i:
                a[i, k] = a[i, k] * a[i, i]
    return 0


@dataclass
class Fit:
    params: list[float]
    errors: list[float]
    nterms: int
    ok: bool
    old: float = 0.0      # the static the next fit in this process starts from


def fit_model(images: list[Image], params0: list[float], lterms: int,
              old: float = 0.0) -> Fit:
    """ph_amfit: each image's seeing radius, and the model's parameters.
    The images and their stars go to the kernel as arrays. `old` is the static
    the convergence test starts from: zero in a fresh process, and for a
    refit the last fit's Fit.old."""
    nimg = len(images)
    naperts = max((len(i.stars[0].rap) for i in images if i.stars), default=1)
    allstars = [s for image in images for s in image.stars]
    rap = np.zeros((nimg, naperts), dtype=f32)
    start = np.zeros(nimg + 1, dtype=np.int64)
    ngood = np.zeros(nimg, dtype=np.int64)
    for k, image in enumerate(images):
        start[k + 1] = start[k] + len(image.stars)
        if image.stars:
            rap[k, :len(image.stars[0].rap)] = image.stars[0].rap
            # IRAF counts the good points of the first stars of all,
            # not this image's.
            ngood[k] = sum(max(0, allstars[i].nap - 1) for i in range(len(image.stars)))
    nstars = len(allstars)
    mag = np.zeros((nstars, naperts), dtype=f32)
    merr = np.ones((nstars, naperts), dtype=f32)
    nap = np.zeros(nstars, dtype=np.int64)
    for i, star in enumerate(allstars):
        mag[i, :star.nap] = star.mag[:star.nap]
        merr[i, :star.nap] = star.merr[:star.nap]
        nap[i] = star.nap
    nxair = np.array([image.nxairmass for image in images], dtype=f32)
    ro = np.array([image.ro for image in images], dtype=f32)
    first_rap = allstars[0].rap[0] if allstars else f32(0.0)
    params = np.array(params0, dtype=np.float64)
    perrors = np.zeros(MAX_MTERMS)
    ok, nterms, old = _amfit(rap, start, ngood, mag, merr, nap, nxair, ro, f32(first_rap),
                             params, perrors, lterms, float(old))
    for image, r in zip(images, ro, strict=True):
        image.ro = f32(r)
    return Fit([float(v) for v in params], [float(v) for v in perrors], int(nterms),
               bool(ok), float(old))


@jit
def _amfit(rap, start, ngood, mag, merr, nap, nxair, ro, first_rap, params, perrors,
           lterms, old):
    """ph_amfit's iterations, on arrays: ro and params are updated in place.
    Returns whether the fit converged, the number of terms fitted, and the
    static the next fit starts from."""
    nimg, naperts = rap.shape
    pclamps = np.full(MAX_MTERMS, 0.2)
    pclamps[3] = 0.5
    pold = np.zeros(lterms)
    ipow, nterms = 1, 0
    gain = f32(0.0)
    sumn = 0.0
    u = np.zeros((lterms, lterms))
    v = np.zeros(lterms)
    dp = np.zeros(lterms)
    dm = np.zeros(naperts)
    dddr = np.zeros(naperts)
    t = np.zeros((lterms, naperts))
    niter = 1
    while niter <= AGR_ITMAX:
        sumd = 0.0
        sumw = 0.0
        sumn = 0.0
        n = 1
        if nterms >= 1 and gain <= f32(0.04):
            n = 2
            if nterms >= 2 and gain <= f32(0.016):
                n = 3
                if nterms >= 3 and gain <= f32(0.006):
                    n = 4
                    if nterms >= 4 and gain <= f32(0.0025):
                        n = 5
        nterms = min(n, lterms)
        u[:, :] = 0.0
        v[:] = 0.0
        for k in range(nimg):
            if start[k + 1] == start[k] or ngood[k] <= 0:
                ro[k] = NAN
                continue
            if niter == 1:
                if k > 0 and not np.isnan(ro[k - 1]):
                    ro[k] = ro[k - 1]
                else:
                    ro[k] = f32(f32(0.5) * first_rap)
            x = nxair[k]
            rold = f32(0.0)
            rclamp = f32(ro[k] / f32(4.0))
            ntimes = 0
            dm[:] = 0.0
            while True:
                sumr = 0.0
                sumy = 0.0
                dddr[:] = 0.0
                for i in range(1, naperts):
                    dm[i] = dmag(rap[k, i - 1], rap[k, i], x, ro[k], params)
                    dddr[i] = 500.0 * (
                        dmag(rap[k, i - 1], rap[k, i], x, f32(ro[k] - f32(0.001)), params)
                        - dmag(rap[k, i - 1], rap[k, i], x, f32(ro[k] + f32(0.001)),
                               params))
                for s in range(start[k], start[k + 1]):
                    if nap[s] <= 1:
                        continue
                    wprev = f32(1.0)
                    for j in range(1, nap[s]):
                        diff = f32(f64(mag[s, j]) - dm[j])
                        w = min(_weight(diff, merr[s, j], ipow, False), wprev)
                        wprev = w
                        wt = f32(w / f32(merr[s, j] * merr[s, j]))
                        sumr = sumr + f64(f32(wt * diff)) * dddr[j]
                        sumy = sumy + f64(wt) * dddr[j] ** 2
                if sumy > 0.0:
                    dr = f32(sumr / sumy)
                    if f32(dr * rold) < 0:
                        rclamp = f32(f32(0.5) * rclamp)
                    ro[k] = f32(ro[k] - f32(dr / f32(abs(f32(dr / rclamp)) + f32(1.0))))
                    if ro[k] <= EPSILONR:
                        ro[k] = NAN
                        return False, nterms, old
                    rold = dr
                    if abs(f64(f32(dr / ro[k]))) <= CONVERGED:
                        break
                else:
                    ro[k] = NAN
                    break
                ntimes += 1
                if ntimes >= 100:
                    ro[k] = NAN
                    return False, nterms, old
            # ph_adp: the parameter derivatives at the new radius
            t[:, :] = 0.0
            for j in range(1, naperts):
                dm[j] = dmag(rap[k, j - 1], rap[k, j], x, ro[k], params)
                for i in range(nterms):
                    params[i] = params[i] - D001
                    lower = dmag(rap[k, j - 1], rap[k, j], x, ro[k], params)
                    params[i] = params[i] + D002
                    t[i, j] = 500.0 * (lower - dmag(rap[k, j - 1], rap[k, j], x, ro[k],
                                                    params))
                    params[i] = params[i] - D001
            # ph_aaccum
            for s in range(start[k], start[k + 1]):
                if nap[s] <= 1:
                    continue
                wprev = f32(1.0)
                for j in range(1, nap[s]):
                    diff = f32(f64(mag[s, j]) - dm[j])
                    w = min(_weight(diff, merr[s, j], ipow, False), wprev)
                    wprev = w
                    wt = f32(w / f32(merr[s, j] * merr[s, j]))
                    sumd = sumd + f64(f32(wt * f32(diff * diff)))
                    sumw = sumw + f64(wt)
                    sumn = sumn + f64(w)
                    wd = f64(f32(wt * diff))
                    for ll in range(nterms):
                        v[ll] = v[ll] + wd * t[ll, j]
                        for m in range(nterms):
                            u[ll, m] = u[ll, m] + f64(wt) * t[ll, j] * t[m, j]
        if dinver(u, nterms):
            return False, nterms, old
        for i in range(nterms):
            dp[i] = 0.0
            for j in range(nterms):
                dp[i] += u[j, i] * v[j]
        # ph_apsolve
        for i in range(nterms):
            if dp[i] * pold[i] < 0.0:
                pclamps[i] = 0.5 * pclamps[i]
            pold[i] = dp[i]
        params[0] = params[0] - dp[0] / (1.0 + abs(dp[0] / (pclamps[0] * (params[0] - 1.0))))
        if nterms >= 2:
            params[1] = params[1] - dp[1] / (1.0 + abs(
                dp[1] / (pclamps[1] * params[1] * (1.0 - params[1]))))
            if nterms >= 3:
                params[2] = params[2] - dp[2] / (1.0 + abs(
                    dp[2] / (pclamps[2] * params[2] * (1.0 - params[2]))))
                if ipow == 1:
                    ipow = 2
                    pclamps[0] = 0.1
                    pclamps[1] = 0.1
                if nterms >= 4:
                    params[3] = params[3] - dp[3] / (1.0 + abs(
                        dp[3] / (pclamps[3] * params[3])))
                    if nterms >= 5:
                        params[4] = params[4] - dp[4] / (1.0 + abs(
                            dp[4] / (pclamps[4] * params[1])))
        sumn = math.sqrt(sumd / (sumn - (nterms + 1.0))) if sumn > nterms + 1.0 else 0.0
        sumd = math.sqrt(sumd / sumw)
        gain = f32(2.0 * abs(old - sumd) / (old + sumd))
        old = sumd
        niter += 1
        if nterms < lterms or gain > f32(0.001):
            continue
        if ipow <= 2:
            ipow = 3
            pclamps[:4] = 0.1
            continue
        break
    if niter > AGR_ITMAX:
        return False, nterms, old
    for k in range(nterms):
        perrors[k] = sumn * math.sqrt(u[k, k]) if u[k, k] > 0.0 else 0.0
    return True, nterms, old


# --------------------------------------------------------------- evaluate


@dataclass
class Curve:
    """What ph_aeval works out for one image."""
    rbar: np.ndarray
    theo: np.ndarray
    obs: np.ndarray
    wobs: np.ndarray
    adopt: np.ndarray
    wado: np.ndarray
    cum: np.ndarray
    tcum: np.ndarray
    wcum: np.ndarray
    ave: np.ndarray
    resid: np.ndarray
    ressq: np.ndarray
    wr: np.ndarray


def evaluate(image: Image, params: list[float], naperts: int) -> Curve:
    """ph_rinit, ph_tinit, ph_taccum and ph_tcum."""
    params = np.asarray(params, dtype=np.float64)
    rap = image.stars[0].rap if image.stars else np.full(naperts, NAN, dtype=f32)
    z = lambda: np.zeros(naperts, dtype=f32)  # noqa: E731
    rbar, theo, obs = z(), z(), z()
    adopt, wobs, wado = z(), z(), z()
    ave, resid, ressq, wr = z(), z(), z(), z()
    for j in range(1, naperts):
        rbar[j] = f32(f32(0.5) * f32(rap[j - 1] + rap[j]))
        theo[j] = (NAN if np.isnan(image.ro)
                   else f32(dmag(rap[j - 1], rap[j], image.nxairmass, image.ro, params)))
    stars = image.stars
    for star in stars:
        if star.nap <= 1:
            continue
        wprev = f32(1.0)
        for j in range(1, star.nap):
            if np.isnan(theo[j]):
                wt = f32(f32(1.0) / f32(star.merr[j] * star.merr[j]))
                for arr in (ave, resid, ressq, wr):
                    arr[0] = arr[j] = NAN
            else:
                diff = f32(star.mag[j] - theo[j])
                w = min(_weight(diff, star.merr[j], 1, True), wprev)
                wprev = w
                wt = f32(w / f32(star.merr[j] * star.merr[j]))
                for jj in (0, j):
                    ave[jj] = f32(ave[jj] + f32(wt * theo[j]))
                    resid[jj] = f32(resid[jj] + f32(wt * diff))
                    ressq[jj] = f32(ressq[jj] + f32(wt * f32(diff * diff)))
                    wr[jj] = f32(wr[jj] + wt)
            adopt[j] = f32(adopt[j] + f32(wt * star.mag[j]))
            wobs[j] = f32(wobs[j] + wt)

    def mean() -> None:
        for j in range(1, naperts):
            if wobs[j] <= 0:
                continue
            obs[j] = f32(adopt[j] / wobs[j])
            adopt[j] = 0
            wobs[j] = 0

    mean()
    for power in (2, 3):
        for star in stars:
            if star.nap <= 1:
                continue
            wprev = f32(1.0)
            for j in range(1, star.nap):
                diff = f32(star.mag[j] - obs[j])
                w = min(_weight(diff, star.merr[j], power, True), wprev)
                wprev = w
                wt = f32(w / f32(star.merr[j] * star.merr[j]))
                adopt[j] = f32(adopt[j] + f32(wt * star.mag[j]))
                wobs[j] = f32(wobs[j] + wt)
        if power == 2:
            mean()
    scale = f32(0.1)
    for j in range(1, naperts):
        if wobs[j] > 0:
            obs[j] = f32(adopt[j] / wobs[j])
            if np.isnan(theo[j]):
                wt = f32(0.0)
            else:
                st = f32(scale * theo[j])
                wt = f32(f32(1.0) / f32(st * st))
                adopt[j] = f32(adopt[j] + f32(wt * theo[j]))
            wt = f32(f32(1.0) / f32(wobs[j] + wt))
            adopt[j] = f32(wt * adopt[j])
            sa = f32(scale * adopt[j])
            wado[j] = np.sqrt(f32(wt + f32(sa * sa)))
            wobs[j] = np.sqrt(f32(f32(1.0) / wobs[j]))
        else:
            adopt[j] = theo[j]
            if np.isnan(theo[j]):
                wado[j] = obs[j] = wobs[j] = NAN
            else:
                wado[j] = f32(float(f32(f32(2.0) * scale)) * abs(float(theo[j])))
    cum = np.zeros(naperts + 1, dtype=f32)
    tcum = np.zeros(naperts + 1, dtype=f32)
    wcum = np.zeros(naperts + 1, dtype=f32)
    if np.isnan(image.ro):
        cum[:] = tcum[:] = wcum[:] = NAN
    else:
        tcum[naperts] = f32(dmag(rap[naperts - 1], f32(f32(2.0) * rap[naperts - 1]),
                                 image.nxairmass, image.ro, params))
        for i in range(naperts - 1, 0, -1):
            cum[i] = f32(adopt[i] + cum[i + 1])
            tcum[i] = f32(adopt[i] + tcum[i + 1])
            wcum[i] = f32(f32(wado[i] * wado[i]) + wcum[i + 1])
    return Curve(rbar, theo, obs, wobs, adopt, wado, cum, tcum, wcum, ave, resid, ressq, wr)


def correction(image: Image, curve: Curve, smallap: int, largeap: int):
    """ph_wapcat: the sum of the adopted differences from the small aperture
    to the large one (1-based), and its error."""
    if np.isnan(image.ro):
        return NAN, NAN
    cum = wcum = f32(0.0)
    for i in range(largeap - 1, smallap - 1, -1):
        cum = f32(cum + curve.adopt[i])
        wcum = f32(wcum + f32(curve.wado[i] * curve.wado[i]))
    return cum, np.sqrt(wcum)


class Work:
    """AGR_MAGS, AGR_CMAGS, AGR_TMAGS, AGR_WMAGS: working arrays shared by the
    log and the best magnitudes, never cleared -- what a star does not
    reach keeps the last star's values, and IRAF prints them. Before any star
    has reached it IRAF prints the memory it was given: zeros in a plain run,
    and after the graphs of an interactive one 77272.055 or the like. Here it
    is always zero."""

    def __init__(self, naperts: int):
        self.mags = np.zeros(naperts, dtype=f32)
        self.cmags = np.zeros(naperts, dtype=f32)
        self.tmags = np.zeros(naperts, dtype=f32)
        self.wmags = np.zeros(naperts, dtype=f32)


def best_magnitudes(image: Image, curve: Curve, largeap: int, work: Work | None = None):
    """ph_wmags: each star's magnitude corrected to the large aperture at
    the aperture where its error is smallest -- (star, mag, merr, radius)."""
    out = []
    naperts = len(curve.adopt)
    work = work or Work(naperts)
    for star in image.stars:
        tmp, obs, wobs = work.mags, work.cmags, work.wmags
        tmp[0] = star.mag[0]
        sigmin = MAX_REAL
        jfinal = largeap - 1
        if largeap < star.nap:
            sigcor = f32(0.0)
            for value in curve.wado[largeap:star.nap]:
                sigcor = f32(sigcor + f32(value * value))
        else:
            sigcor = f32(0.0)
        wprev = f32(1.0)
        for j in range(min(star.nap, largeap)):
            if j == 0:
                w = f32(1.0)
            else:
                diff = f32(star.mag[j] - curve.adopt[j])
                tmp[j] = f32(tmp[j - 1] + star.mag[j])
                q = f32(diff / f32(f32(2.0) * star.merr[j]))
                w = min(f32(f32(1.0) / f32(f32(1.0) + f32(q * q))), wprev)
            wprev = w
            if np.isnan(curve.cum[j + 1]):
                obs[j] = wobs[j] = NAN
                continue
            obs[j] = f32(f32(tmp[j] + curve.cum[j + 1]) - curve.cum[largeap])
            wobs[j] = np.sqrt(f32(f32(curve.wcum[j + 1] - sigcor)
                                  + f32(f32(star.merr[j] * star.merr[j]) / w)))
            if wobs[j] < sigmin:
                jfinal = j
                sigmin = wobs[j]
        out.append((star, obs[jfinal], wobs[jfinal], star.rap[jfinal]))
    return out


# ------------------------------------------------------------------ output


def _f(value, width: int, decpl: int) -> str:
    if value is None or np.isnan(value):
        return "INDEF".rjust(width)
    return iraffmt.printf(float(value), "f", decpl).rjust(width)


def _hours(value, decpl: int = 1) -> str:
    """IRAF's %h: h:mm:ss.s."""
    if np.isnan(value):
        return "INDEF"
    sign = "-" if value < 0 else ""
    seconds = round(abs(float(value)) * 3600.0, decpl)
    h, rest = divmod(seconds, 3600.0)
    m, s = divmod(rest, 60.0)
    return f"{sign}{int(h)}:{int(m):02d}:{s:0{3 + decpl}.{decpl}f}"


def _rows(label: str, values) -> list[str]:
    """Seven %9.4f to a line, the label before the first."""
    lines = []
    values = list(values)
    for k in range(0, len(values), NPERLINE):
        lead = f"{label:>8}  " if k == 0 else " " * 10
        lines.append(lead + "".join(_f(v, 9, 4) for v in values[k:k + NPERLINE]))
    return lines


def _g15(value: float) -> str:
    return iraffmt.printf(value, "g", 7, double=True).rjust(15)


def magfile_header(largeap: int, names: bool = False) -> str:
    stops = [(19, "Image"), (30, "Filter"), (39, "Exptime"), (47, "Airmass"),
             (62, "Otime"), (70, "Xcenter"), (80, "Ycenter"), (93, "Mag"),
             (101, "Merr"), (107, "Radius")]
    line = "#"
    for column, text in stops:
        line = line.ljust(column - 1) + text
    if names:
        line += "  Obj"
    return f"# Magnitudes corrected to aperture {largeap}\n\n{line}\n\n"


PARAMETERS = ("swings", "pwings", "pgauss", "rgescale", "xwings")


@dataclass
class Result:
    images: list[Image]
    fit: Fit
    corrections: dict
    best: dict


def apfile(images: list[Image], naperts: int, apercors: str, smallap: int = 1,
           largeap: int = 0, magfile: str = "", logfile: str = "",
           params: tuple = (1.2, 0.1, 0.5, 0.9, 0.0), nparams: int = 3,
           append: bool = False, names: bool = False) -> Result:
    """t_apfile after the input: fit, evaluate, and write the corrections,
    the best magnitudes and the log. With `names`, each best magnitude also
    carries the star's name from the photometry as an eleventh column, Obj
    -- what obsfile needs to write the stars under the names the standards
    catalog knows them by; IRAF's file has ten columns."""
    smallap, largeap = limits(naperts, smallap, largeap)
    lterms = max(1, min(MAX_MTERMS, nparams))
    p0 = [float(f32(v)) for v in params]
    log = log_title(apercors) if logfile else []
    images = [image for image in images if image is not None]
    if not images:
        if logfile:
            log.append("There is no input data to fit")
        raise ValueError("Error: There is no input data to fit")
    fit = fit_model(images, p0, lterms)
    return write_results(images, fit, naperts, apercors, smallap, largeap, magfile, logfile,
                         p0, lterms, log, append, names)


def limits(naperts: int, smallap: int, largeap: int) -> tuple[int, int]:
    smallap = max(1, min(naperts, smallap))
    largeap = naperts if largeap <= 0 else max(1, min(naperts, largeap))
    if largeap <= smallap:
        raise ValueError("The large aperture is less than or equal to the large aperture")
    return smallap, largeap


def log_title(apercors: str) -> list[str]:
    return [f"NEW LOGFILE ENTRY AT: {cnvtime()}", f"APFILE: {apercors}", ""]


def show_initial(params, lterms: int) -> list[str]:
    """ph_ishow."""
    return ["", f"The initial cog model parameter values for nparams {lterms}",
            *(f"\t{name:>10}: {_g15(value)}" for name, value in zip(PARAMETERS, params,
                                                                    strict=True)), ""]


def show_model(fit: Fit, lterms: int) -> list[str]:
    """ph_pshow."""
    return ["", f"The computed cog model parameters and their errors for nparams {lterms}",
            *(f"\t{name:>10}: {_g15(value)} +/- {_g15(error)}"
              for name, value, error in zip(PARAMETERS, fit.params, fit.errors, strict=True)),
            ""]


def show_seeing(images: list[Image]) -> list[str]:
    """ph_rshow."""
    return ["", "The seeing radius and assumed airmass for each image",
            *(f"{image.name:>30}  {_f(image.ro, 8, 4)}  {_f(image.xairmass, 8, 3)}"
              for image in images), ""]


def write_results(images: list[Image], fit: Fit, naperts: int, apercors: str, smallap: int,
                  largeap: int, magfile: str, logfile: str, p0, lterms: int, log: list[str],
                  append: bool = False, names: bool = False, stale: bool = False) -> Result:
    """What is left once the fit is settled: ph_aeval and the log around it.
    `stale`: points were deleted since the fit, as an interactive run may leave
    it; IRAF says so in the log and goes on with the fit it has."""
    def ishow() -> None:
        log.extend(show_initial(p0, lterms) + show_model(fit, lterms) + show_seeing(images))

    corrections, best = {}, {}
    apcat = [f"# The aperture correction from apertures {smallap} to {largeap} "
             "in magnitudes", ""]
    mags = [magfile_header(largeap, names)] if magfile else []
    if not fit.ok:
        if logfile:
            ishow()
            log.append("Error: The cog model fit did not converge")
    else:
        if logfile:
            if stale:
                log.append("Warning: The cog model fit is out-of-date")
            ishow()
        tave = np.zeros(naperts, dtype=f32)
        tresid, tressq, twr = (np.zeros(naperts, dtype=f32) for _ in range(3))
        work = Work(naperts)
        for image in images:
            curve = evaluate(image, fit.params, naperts)
            cum, wcum = correction(image, curve, smallap, largeap)
            corrections[image.name] = (cum, wcum)
            apcat.append(f"{image.name} {iraffmt.g(float(cum))} {iraffmt.g(float(wcum))}")
            rap = image.stars[0].rap if image.stars else np.full(naperts, NAN, dtype=f32)
            if logfile:
                log += _twrite(image, curve, rap, naperts)
                log += _tmags(image, curve, naperts, work)
                log.append(f"Image: {image.name} rin={_f(rap[smallap - 1], 0, 2)} "
                           f"rout={_f(rap[largeap - 1], 0, 2)} "
                           f"apercor={_f(cum, 0, 3)} +/- {_f(wcum, 0, 4)}")
                log.append("")
            rows = best_magnitudes(image, curve, largeap, work)
            best[image.name] = rows
            for star, mag, merr, radius in rows:
                mags.append(
                    f"{image.name:>23.23}  {image.ifilter:>10.10}  "
                    f"{_f(image.itime, 8, 2)}  {_f(image.xairmass, 6, 3)}  "
                    f"{_hours(image.otime):>11}  {_f(star.x, 8, 2)}  {_f(star.y, 8, 2)}  "
                    f"{_f(mag, 7, 3)}  {_f(merr, 7, 3)}  {_f(radius, 6, 3)}"
                    + (f"  {star.name or star.id}" if names else "") + "\n")
            if not np.isnan(image.ro):
                tave = (tave + curve.ave).astype(f32)
                tresid = (tresid + curve.resid).astype(f32)
                tressq = (tressq + curve.ressq).astype(f32)
                twr = (twr + curve.wr).astype(f32)
        if logfile:
            log.append("")
            log.append("Average model cog, residual, and rms residual for each aperture "
                       "all images")
            for j in range(naperts):
                if twr[j] <= 0:
                    a = r = s = NAN
                else:
                    a = f32(tave[j] / twr[j])
                    r = f32(tresid[j] / twr[j])
                    rtmp = f32(f32(tressq[j] / twr[j]) - f32(r * r))
                    s = f32(0.0) if rtmp <= 0 else np.sqrt(rtmp)
                log.append(f"\t{j + 1:3d}  {_f(a, 9, 5)}  {_f(r, 9, 5)}  {_f(s, 9, 5)}")
            log.append("")
    Path(apercors).write_text("\n".join(apcat) + "\n", encoding="utf-8")
    if magfile:
        Path(magfile).write_text("".join(mags), encoding="utf-8")
    if logfile:
        with open(logfile, "a" if append else "w", encoding="utf-8") as handle:
            handle.write("\n".join(log) + "\n")
    return Result(images, fit, corrections, best)


def _twrite(image: Image, c: Curve, rap, naperts: int) -> list[str]:
    out = ["", f"The cog for image {image.name} from radius {_f(c.rbar[1], 0, 2)} to "
               f"{_f(c.rbar[naperts - 1], 0, 2)}"]
    for label, values in (("radius", c.rbar), ("model", c.theo), ("observed", c.obs),
                          ("sigma", c.wobs), ("adopted", c.adopt), ("sigma", c.wado)):
        out += _rows(label, values[1:])
    out += ["", f"The aperture correction for image {image.name} from radius "
                f"{_f(rap[0], 0, 2)} to {_f(rap[naperts - 1], 0, 2)}"]
    out += _rows("radius", rap)
    out += _rows("apercor", c.cum[1:naperts])
    out += _rows("sigma", c.wcum[1:naperts])
    out += ["", f"The aperture correction for image {image.name} from radius "
                f"{_f(rap[0], 0, 2)} to {_f(f32(f32(2.0) * rap[naperts - 1]), 0, 2)}"]
    out += _rows("tapercor", c.tcum[1:naperts + 1])
    out += ["", ""]
    return out


def _tmags(image: Image, c: Curve, naperts: int, work: Work) -> list[str]:
    stars = image.stars
    rlast = stars[0].rap[naperts - 1] if stars else NAN
    out = ["", f"The observed, and corrected to radii {_f(rlast, 0, 2)} and "
               f"{_f(f32(f32(2.0) * rlast), 0, 2)}, magnitudes"]
    for star in stars:
        tmp, obs, tobs, wobs = work.mags, work.cmags, work.tmags, work.wmags
        tmp[0] = star.mag[0]
        wprev = f32(1.0)
        for j in range(star.nap):
            if j == 0:
                w = f32(1.0)
            else:
                diff = f32(star.mag[j] - c.adopt[j])
                tmp[j] = f32(tmp[j - 1] + star.mag[j])
                q = f32(diff / f32(f32(2.0) * star.merr[j]))
                w = min(f32(f32(1.0) / f32(f32(1.0) + f32(q * q))), wprev)
            wprev = w
            if np.isnan(c.cum[j + 1]):
                obs[j] = tobs[j] = wobs[j] = NAN
                continue
            obs[j] = f32(tmp[j] + c.cum[j + 1])
            tobs[j] = f32(tmp[j] + c.tcum[j + 1])
            wobs[j] = np.sqrt(f32(c.wcum[j + 1] + f32(f32(star.merr[j] * star.merr[j]) / w)))
        obs[star.nap:] = tobs[star.nap:] = wobs[star.nap:] = NAN
        out.append(f"Star: {star.id}  x: {_f(star.x, 0, 3)} y: {_f(star.y, 0, 3)}")
        out += _rows("radius", star.rap)
        out += _rows("mags", tmp)
        out += _rows("cmags", obs)
        out += _rows("tcmags", tobs)
        out += _rows("sigma", wobs)
    out.append("")
    return out
