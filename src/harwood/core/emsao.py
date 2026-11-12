"""emsao -- RVSAO's redshift from emission lines, as RVSAO 2.8.5 computes it.

The spectrum is smoothed (nsmooth passes of 1-2-1) and its continuum fit and
subtracted (contpars: icfit, spline3 of 10 pieces, 2-sigma clipping by default).
An initial redshift comes from the tallest line found near each line of the search
list (emsearch.dat: [O II], Hbeta, [O III], Halpha), or from czguess. Each line of
emlines.dat redshifted by it is looked for within wspan Angstroms, and kept when
it stands linesig times the rms of its continuum region above it. The unsmoothed
spectrum, its continuum subtracted, is then fit around each line found with one,
two or three Gaussians and a straight background -- three for [N II]-Halpha-[N II]
(emcomb.dat), two for the [S II] and [O III] pairs -- and each line's redshift is
weighted by its centre's error; a line too near an end of the spectrum, or whose
equivalent width is under lsmin times its error, is left out of the mean.

From Emsao/{t_emsao,emfit,emguess,emsrch,emfind,emlfit,emmin,emvfit,eminit,
emrslts}.x and Util/{icsubcon,smooth,vcombine}.x; what is shared with the other
RVSAO tasks is core/rvsao.py. Defects kept, because they decide the numbers:

* isline's continuum is the spectrum itself (ycont sums fdata, not fcont).
* emlfit's half-height search to the left is a DO loop from lp-1 up to pixleft,
  which does not run: the initial width is the distance to the right half-height
  point, minus one.
* A line fit alone uses the fitting half-width of the reference line at the same
  place in emlines.dat as the line's place among those found (iscombo: wgfit[n]).
* wspan is read with clgeti: its fraction is dropped.
* emvfit's second pass uses the dispersion error of the last line of the first.
* The equivalent width's error divides by the square of the equivalent width.
* gxyline's continuum rms starts 16 pixels below the continuum region; IRAF reads
  whatever memory is there when that is before the spectrum, harwood reads 0.

Translated from RVSAO, Copyright (c) 1991-2022 Smithsonian Astrophysical
Observatory, whose notice is in LICENSES/RVSAO-COPYRIGHT; and from IRAF's icfit,
Copyright (c) 1986 AURA (LICENSES/IRAF-COPYRIGHT).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from . import iraffmt
from .rvsao import C0, INDEFD, RvsaoError, Scan, Spectrum, f32, line_file, smooth

MAXREF, MAXABS, MAXCOMB, MAXSEARCH = 500, 10, 20, 5
SZ_ELINE = 9
WEXTRA = 0.02
PIXMAX = 20
NOCONT, SUBCONT = 1, 2
VINIT = ("correlation", "emission", "combination", "search", "guess", "zguess", "cortemp")
VPLOT = ("correlation", "emission", "combination", "search", "guess")

DEFAULTS = {
    "specnum": "0", "specband": 0, "skynum": "0", "skyband": 0, "specdir": "",
    "linefit": True, "fixbad": False, "badlines": "badlines.dat", "renormalize": False,
    "st_lambda": None, "end_lambda": None, "nsmooth": 10, "vel_init": "search",
    "czguess": 0.0, "cortemp": "", "wspan": 10.0, "linesig": 1.5,
    "emsearch": "emsearch.dat", "emlines": "emlines.dat", "linedir": "rvsao$lib/",
    "npfit": 2, "nlcont": 1, "esmooth": 0, "emcombine": "emcomb.dat", "mincont": 0.0,
    "lwmin": 0.4, "lwmax": 1.7, "lsmin": 1.0, "sigline": 0.0, "disperr": 0.01,
    "vel_corr": "barycentric", "report_mode": 1, "verbose": True,
    "ablines": "ablines.dat",
    # contpars
    "c_function": "spline3", "c_sample": "*", "naverage": 1, "order": 10,
    "s_low_reject": 2.0, "s_high_reject": 2.0, "niterate": 10, "grow": 1.0,
}


# ------------------------------------------------------------- line lists


@dataclass
class Combo:
    edwl: float
    elines: list[float]
    eht: list[float]
    edrop: list[bool]


@dataclass
class Lines:
    """eminit: the absorption lines, the emission lines (wavelength, continuum
    region, fitting half-width, name), the blends and the search lines."""
    wlref: list[float] = field(default_factory=list)
    bcont: list[float] = field(default_factory=list)
    rcont: list[float] = field(default_factory=list)
    wgfit: list[float] = field(default_factory=list)
    nmref: list[str] = field(default_factory=list)
    combos: list[Combo] = field(default_factory=list)
    search: list[tuple[float, float, float, str]] = field(default_factory=list)
    absorption: list[tuple[float, str]] = field(default_factory=list)


def _records(path, name: str, linedir: str = "") -> list[str]:
    """open of a line list: RVSAO stops when it cannot (an errchk'd open), naming
    the path it built."""
    if "/" not in name and "$" not in name and linedir:
        name = (linedir if linedir.endswith("/") else linedir + "/") + name
    try:
        return path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError as error:
        raise RvsaoError(f"Cannot open file ({name})") from error


def read_lines(params: dict) -> Lines:
    out = Lines()
    linedir = str(params["linedir"])
    if params["ablines"]:
        name = str(params["ablines"])
        for line in _records(line_file(name, linedir), name, linedir)[:MAXABS]:
            s = Scan(line)
            out.absorption.append((s.d() or 0.0, (s.wrd(SZ_ELINE) or "")))
    name = str(params["emlines"])
    if name:
        name = name.lstrip("+")
        for line in _records(line_file(name, linedir), name, linedir)[:MAXREF]:
            s = Scan(line)
            wl = s.d()
            wl = 0.0 if wl is None else wl
            b = s.d()
            b = 0.0 if b is None else b
            out.wlref.append(wl)
            out.bcont.append(b)
            if b > 0.0:
                r = s.d()
                w = s.d()
                out.rcont.append(0.0 if r is None else r)
                out.wgfit.append(0.0 if w is None else w)
                out.nmref.append((s.wrd(SZ_ELINE) or "")[:SZ_ELINE])
            else:
                out.rcont.append(0.0)
                out.wgfit.append(1.0)
                out.nmref.append(f"{wl:9.4f}"[:SZ_ELINE])
    name = str(params["emcombine"])
    if name:
        for line in _records(line_file(name, linedir), name, linedir)[:MAXCOMB]:
            s = Scan(line)
            n = s.i() or 0
            edwl = s.d() or 0.0
            combo = Combo(edwl, [], [], [])
            for _ in range(n):
                wl = s.d() or 0.0
                ht = s.d() or 0.0
                combo.elines.append(wl)
                combo.edrop.append(ht < 0)
                combo.eht.append(-ht if ht < 0 else ht)
            out.combos.append(combo)
    name = str(params["emsearch"])
    if name:
        for line in _records(line_file(name, linedir), name, linedir)[:MAXSEARCH]:
            s = Scan(line)
            out.search.append((s.d() or 0.0, s.d() or 0.0, s.d() or 0.0,
                               (s.wrd(SZ_ELINE) or "")))
    return out


# ------------------------------------------------------------- finding lines


def isline(x: int, fdata, npts: int, npfit: int, lsigma: float) -> bool:
    """isline: a local maximum over npfit pixels each side, whose mean over them
    stands lsigma times the square root of twice itself above zero."""
    ymax = fdata[x - 1]
    for i in range(1, npfit + 1):
        if fdata[x - 1 - i] >= ymax:
            return False
    for i in range(1, npfit + 1):
        if fdata[x - 1 + i] >= ymax:
            return False
    i1, i2 = max(1, x - npfit), min(npts, x + npfit)
    ymax = ycont = dp = f32(0)
    for i in range(i1, i2 + 1):
        ymax = f32(ymax + fdata[i - 1])
        ycont = f32(ycont + fdata[i - 1])
        dp = f32(dp + f32(1))
    if dp <= 0:
        return False
    ymax = f32(ymax / dp)
    ycont = f32(ycont / dp)
    sigma = math.sqrt(abs(float(f32(ymax + ycont))))
    if float(ymax) >= sigma * lsigma:
        return True
    return bool(ymax > abs(f32(ycont * f32(1.e10))))


def linepar(xzero: int, fdata, npts: int, npfit: int):
    """linepar: a parabola through 2 npfit + 1 pixels -- its centre, height and
    width -- or None."""
    if npfit <= 0 or xzero < npfit:
        return None
    sumy = sumyi = sumyii = dn = 0.0
    i1, i2 = max(1, xzero - npfit), min(npts, xzero + npfit)
    for i in range(i1, i2 + 1):
        y = float(fdata[i - 1])
        di = float(i - xzero)
        sumy += y
        sumyi += y * di
        sumyii += y * di * di
        dn += 1
    if dn <= 0:
        return None
    u = 3.0 / (dn * dn + dn)
    v = 2.0 * dn + 1.0
    w = ((4.0 * dn) + 4.0) * dn - 3.0
    a = (-15.0 * u) / (v * w) * (sumy / u - sumyii)
    b = (-u * sumyi) / v
    c = sumy / v
    with np.errstate(all="ignore"):
        w = np.float64(b) / (2.0 * np.float64(a))
        center = xzero + w
        height = c - a * (1.0 / u + (w * w))
        width = math.sqrt(abs(float(-2.0 * height / np.float64(a)))) if a != 0 else math.inf
    return float(center), float(height), width


def emfind(fdata, npts: int, ipix1: int, ipix2: int, npfit: int, lsigma: float,
           maxlines: int = 30):
    """emfind: the lines between two pixels, their centres and heights."""
    centers, heights = [], []
    for i in range(ipix1 + npfit, ipix2 - npfit + 1):
        if isline(i, fdata, npts, npfit, lsigma):
            got = linepar(i, fdata, npts, npfit)
            if got is not None and len(centers) < maxlines:
                centers.append(got[0])
                heights.append(got[1])
    return centers, heights


# ------------------------------------------------------------- the fit (emmin)


def _lineq(nv: int, xmat, soln, nrhs: int, tol: float):
    """lineq: Gaussian elimination with partial pivoting on the rows' index list."""
    ir = list(range(nv))
    irank = nv
    determ = 1.0
    idiag = 0
    while True:
        hold, irp = 0.0, idiag
        for i in range(idiag, nv):
            if hold < abs(xmat[ir[i], idiag]):
                irp = i
                hold = abs(xmat[ir[irp], idiag])
        determ = determ * xmat[ir[irp], idiag]
        if hold <= tol:
            return idiag, ir
        if ir[irp] != ir[idiag]:
            determ = -determ
            ir[irp], ir[idiag] = ir[idiag], ir[irp]
        if idiag == nv - 1:
            if nrhs == 0:
                return irank, ir
            hold = xmat[ir[nv - 1], nv - 1]
            for i in range(nrhs):
                soln[ir[nv - 1], i] = soln[ir[nv - 1], i] / hold
            k = nv - 2
            for i in range(nv - 1, 0, -1):
                hold = xmat[ir[k], k]
                for j in range(nrhs):
                    s = 0.0
                    for ll in range(i, nv):
                        s = s + (xmat[ir[k], ll] * soln[ir[ll], j])
                    soln[ir[k], j] = (soln[ir[k], j] - s) / hold
                k -= 1
            return irank, ir
        k = idiag + 1
        for i in range(k, nv):
            hold = xmat[ir[i], idiag] / xmat[ir[idiag], idiag]
            for j in range(k, nv):
                xmat[ir[i], j] = xmat[ir[i], j] - (xmat[ir[idiag], j] * hold)
            for j in range(nrhs):
                soln[ir[i], j] = soln[ir[i], j] - (soln[ir[idiag], j] * hold)
        idiag += 1


def _solve2(a, nv: int):
    """solve2 with nrhs < 0: the inverse of a, or None when it cannot be solved."""
    zero = 0.0
    s = np.zeros((12, 12))
    for i in range(nv):
        for j in range(nv):
            zero = zero + (a[i, j] * a[i, j])
            s[i, j] = a[i, j]
    c = float(nv * nv)
    zero = math.sqrt(zero) * 1.0e-16 / c
    x = np.zeros((12, 12))
    for i in range(nv):
        x[i, i] = 1.0
    irank, ir = _lineq(nv, s, x, nv, zero)
    if nv > irank:
        return None
    out = x.copy()
    for j in range(nv):
        t = [x[ir[i], j] for i in range(nv)]
        for i in range(nv):
            out[i, j] = t[i]
    return out


def _tgauss(y, j: int, np_: int, gp, state: dict, compute: bool):
    """tgauss: the model (a straight background and up to three Gaussians) at pixel
    j, its derivatives, and data minus model."""
    deriv = [0.0] * 12
    if compute:
        state["sj0"] = float(state["j0"])
        state["back"], state["bslope"] = gp[0], gp[1]
        t = [0.0] * 9
        for g in range(3):
            if np_ > 2 + 3 * g:
                t[3 * g:3 * g + 3] = gp[2 + 3 * g:5 + 3 * g]
        state["t"] = t
        for g in range(3):
            s = t[3 * g + 2]
            state[f"s{g}s"] = s * s
            state[f"s{g}c"] = s * s * s
            state[f"ts{g}s"] = 2.0 * s * s
    t = state["t"]
    sj = float(j)
    db = state["back"] + (state["bslope"] * (sj - state["sj0"]))
    dg, dtt, dts = [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]
    for g in range(3):
        tc, a, s = t[3 * g], t[3 * g + 1], t[3 * g + 2]
        if a != 0 and s > 0:
            dtt[g] = sj - tc
            dts[g] = dtt[g] * dtt[g]
            dg[g] = a * math.exp(-dts[g] / state[f"ts{g}s"])
    dyj = db + dg[0] + dg[1] + dg[2]
    deriv[0] = 1.0
    deriv[1] = sj - state["sj0"]
    with np.errstate(all="ignore"):
        for g in range(3):
            if np_ > 2 + 3 * g:
                a = t[3 * g + 1]
                ss, sc = state[f"s{g}s"], state[f"s{g}c"]
                deriv[2 + 3 * g] = float(np.float64(dg[g]) * dtt[g] / np.float64(ss))
                deriv[3 + 3 * g] = float(np.float64(dg[g]) / np.float64(a))
                deriv[4 + 3 * g] = float(np.float64(dg[g]) * dts[g] / np.float64(sc))
    diff = float(y[j - 1]) - dyj
    return deriv, dyj, diff


def emmin(y, np_: int, gp1: list[float], pixleft: int, pixright: int):
    """emmin: Gauss-Newton on the background and Gaussians, until each parameter
    changes by less than 1e-5 of itself (300 iterations at most). The parameters,
    their errors, the iterations, the sum of squares and that per degree of freedom,
    and the error code (1 not converged, 2 singular)."""
    gp1 = list(gp1) + [0.0] * (12 - len(gp1))
    eps, nit = 1.0e-5, 300
    nfit = pixright - pixleft + 1
    vf1 = 0.0
    nerr = 0
    gp0 = [0.0] * 12
    c = np.zeros((12, 12))
    tb = [0.0] * 12
    state = {"j0": pixleft}
    vf2 = 1.0
    converged = False
    it = nit
    for n in range(1, nit + 1):
        a = np.zeros((12, 12))
        b = [0.0] * 12
        vf0 = vf1
        vf1 = 0.0
        compute = True
        for j in range(pixleft, pixright + 1):
            deriv, _fj, f1 = _tgauss(y, j, np_, gp1, state, compute)
            compute = False
            vf1 = vf1 + (f1 * f1)
            for p_ in range(np_):
                b[p_] = b[p_] + (deriv[p_] * f1)
                for m in range(p_, np_):
                    a[p_, m] = a[p_, m] + (deriv[p_] * deriv[m])
        vf2 = 1.0
        if n > 1 and vf1 > vf0:
            tb = [(gp1[j] + gp0[j]) * 0.5 for j in range(np_)] + tb[np_:]
        else:
            if np_ > 1:
                for m in range(1, np_):
                    for i in range(m):
                        a[i, m] = a[i, m] / vf2
                        a[m, i] = a[i, m]
                for m in range(np_):
                    a[m, m] = a[m, m] / vf2
                    b[m] = b[m] / vf2
                inv = _solve2(a, np_)
                if inv is None:
                    return gp1, [0.0] * 12, n, vf1, 0.0, 2
                c = inv
            else:
                if a[0, 0] <= 1.0e-70:
                    return gp1, [0.0] * 12, n, vf1, 0.0, 2
                c[0, 0] = 1.0 / a[0, 0]
            for i in range(np_):
                t = 0.0
                for j in range(np_):
                    t = t + (c[i, j] * b[j])
                tb[i] = gp1[i] + (1.0 * t)
        icon = 0
        with np.errstate(all="ignore"):
            for i in range(np_):
                if abs((np.float64(gp1[i]) / np.float64(tb[i])) - 1.0) > eps:
                    icon = 1
        for i in range(np_):
            gp0[i] = gp1[i]
            gp1[i] = tb[i]
        if icon == 0:
            converged, it = True, n
            break
    if not converged:
        nerr, it = 1, nit
    vf = vf1 / float(nfit - np_)
    stde = [0.0] * 12
    with np.errstate(all="ignore"):
        for i in range(np_):
            stde[i] = float(1.0 * np.sqrt(np.float64(c[i, i]) / vf2 * vf))
    return gp1, stde, it, vf1, vf, nerr


# ------------------------------------------------------------- the task


@dataclass
class Found:
    """A line emsrch found, and what emlfit and emvfit made of it."""
    name: str
    wlrest: float
    pxobs: float
    wlobs: float
    params: np.ndarray = field(default_factory=lambda: np.zeros((2, 10)))
    drop: int = 0
    weight: float = 0.0


@dataclass
class Result:
    spectrum: Spectrum
    lines: list[Found]
    czguess: float
    vel: float = INDEFD          #: spevel: emission velocity, km/s, corrected
    err: float = 0.0             #: speerr
    nfit: int = 0
    combined: float = INDEFD     #: spvel
    combined_err: float = INDEFD
    xcvel: float = INDEFD        #: spxvel (from a header written by xcsao)
    xcerr: float = INDEFD
    xcr: float = 0.0


class Emsao:
    """t_emsao for a list of spectra: emfit on each, and the report."""

    def __init__(self, params: dict):
        self.p = {**DEFAULTS, **params}
        self.lines = Lines()
        from .rvsao import VEL_CORR
        self.svcor = VEL_CORR.index(str(self.p["vel_corr"])) + 1
        self.vinit = VINIT.index(str(self.p["vel_init"])) + 1
        self.results: list[Result] = []
        self.state: dict = {}         # getimage's static buffer, across spectra
        self.out = ""                 # what goes to STDOUT
        self.log: list[str] = []      # what goes to each log file

    def run(self, names: list[str]) -> None:
        from . import rvsao

        rmode = int(self.p["report_mode"])
        self.to_stdout = "STDOUT" in self.logfiles()
        if rmode == 1:
            self._log(rvsao.loghead("rvsao.emsao") + "\n")
        self.lines = read_lines(self.p)
        for name in names:
            path = str(self.p["specdir"]) + name
            mspecs = self._apertures(name)
            for mspec in mspecs:
                try:
                    spec = rvsao.getspec(path, mspec, int(self.p["specband"]), self.svcor,
                                         self.state)
                except (RvsaoError, OSError, ValueError) as error:
                    self.out += f"EMSAO: {error}\n"
                    continue
                self.results.append(self.emfit(spec))

    def logfiles(self) -> list[str]:
        return [w.strip() for w in str(self.p.get("logfiles", "STDOUT")).split(",")
                if w.strip()]

    def _apertures(self, name: str) -> list[int]:
        text = str(self.p["specnum"]).strip() or "0"
        if text == "0":
            return [0]
        return [n for n in decode_ranges(text) if n > 0] or [0]

    def _log(self, text: str) -> None:
        self.log.append(text)
        if self.to_stdout:
            self.out += text

    # ---------------------------------------------------------------- emfit

    def emfit(self, spec: Spectrum) -> Result:
        p = self.p
        y = spec.data
        specpix = y.size
        wcs = spec.wcs
        tcont = spec.head.b("SUBCONT", True)
        conproc = SUBCONT if tcont else NOCONT
        pix1, pix2 = self._limits(spec)
        npix = pix2 - pix1 + 1
        wred, wblue = wcs.p2w(float(pix2)), wcs.p2w(float(pix1))
        if wred < wblue:
            wblue, wred = wcs.p2w(float(pix2)), wcs.p2w(float(pix1))
        renorm = bool(p["renormalize"])
        if not renorm:
            renorm = float(max(y[pix1 - 1:pix2])) < 1.0
        if renorm:
            raise RvsaoError("renormalized spectra are not ported yet")
        nsmooth = int(p["nsmooth"])
        smplot = y.astype(f32).copy()
        smooth(smplot, specpix, nsmooth)
        smspec = smplot.copy()
        smcont = np.zeros(specpix, f32)
        self._subcon(smspec, smcont, pix1, pix2, conproc)
        npfit = int(p["npfit"])
        zsig = float(p["linesig"])
        wspan = float(int(float(p["wspan"])))
        czvel = INDEFD
        if bool(p["linefit"]):
            if self.vinit == 4:
                czvel = self._guess(smspec, specpix, wblue, wred, wcs, npfit, zsig)
            elif self.vinit == 5:
                czvel = float(p["czguess"])
            elif self.vinit == 6:
                czvel = C0 * float(p["czguess"])
            if czvel == INDEFD:
                czvel = 0.0
        else:
            czvel = 0.0
        cvel = 1.0 + (czvel / C0)
        found = self._search(smspec, specpix, wblue, wred, cvel, wcs, npfit, zsig, wspan)
        sfspec = y.astype(f32).copy()
        part = sfspec[pix1 - 1:pix2]
        smooth(part, npix, int(p["esmooth"]))
        sfspec[pix1 - 1:pix2] = part
        sfcont = np.zeros(specpix, f32)
        self._subcon(sfspec, sfcont, pix1, pix2, conproc)
        self._fit_lines(found, sfspec, sfcont, specpix, wcs)
        result = Result(spec, found, czvel)
        self._velocity(result, wblue, wred, wcs)
        if bool(p["verbose"]):
            self._report(result)
        return result

    def _limits(self, spec: Spectrum) -> tuple[int, int]:
        wcs, sh = spec.wcs, spec.sh
        minwav, maxwav = spec.sh.w0, spec.sh.w1
        minwav, maxwav = wcs.p2w(float(sh.np1)), wcs.p2w(float(sh.np2))
        if maxwav < minwav:
            minwav, maxwav = maxwav, minwav
        st, end = self.p["st_lambda"], self.p["end_lambda"]
        if st is not None:
            if st > 0.0 and st > minwav:
                minwav = st
            elif st < 0.0:
                minwav = minwav - st
        pix1 = int(round_half(wcs.w2p(minwav)))
        minwav = wcs.p2w(float(pix1))
        if st is not None:
            if end is not None and end > 0.0 and end < maxwav:
                maxwav = end
            elif end is not None and end < 0.0:
                maxwav = maxwav + end
        pix2 = int(round_half(wcs.w2p(maxwav)))
        if pix1 > pix2:
            pix2 = int(round_half(wcs.w2p(minwav)))
            pix1 = int(round_half(wcs.w2p(maxwav)))
        return pix1, pix2

    def _subcon(self, fdata, fit, pix1: int, pix2: int, conproc: int) -> None:
        """icsubcon on pixels pix1..pix2: the continuum, and the spectrum minus it
        (where the weight is not zero)."""
        from . import icfitbits

        p = self.p
        n = pix2 - pix1 + 1
        data = fdata[pix1 - 1:pix2]
        order = int(p["order"])
        if order < 1 or not np.any(data != 0):
            fit[pix1 - 1:pix2] = data
            return
        w = np.ones(n, f32)
        for i in range(n):
            if data[i] == 0:
                w[i] = 0
            else:
                break
        for i in range(n - 1, -1, -1):
            if data[i] == 0:
                w[i] = 0
            else:
                break
        x = np.arange(1, n + 1, dtype=f32)
        ic = icfitbits.Ic(function=str(p["c_function"]), order=order,
                          sample=str(p["c_sample"]), naverage=int(p["naverage"]),
                          low_reject=float(f32(p["s_low_reject"])),
                          high_reject=float(f32(p["s_high_reject"])),
                          niterate=int(p["niterate"]), grow=float(f32(p["grow"])))
        curve = icfitbits.fit(ic, x, data.astype(f32), w)
        values = np.asarray(curve.vector_at(x), f32)
        fit[pix1 - 1:pix2] = values
        if conproc == SUBCONT:
            for i in range(n):
                if w[i] > 0:
                    fdata[pix1 - 1 + i] = f32(data[i] - values[i])

    def _guess(self, fdata, npts, blue, red, wcs, npfit, zsig) -> float:
        """emguess: the redshift of the tallest line near any search line."""
        blue_limit = blue - (red - blue) * WEXTRA
        red_limit = red + (red - blue) * WEXTRA
        zvel, hmax = 0.0, 0.0
        for restwave, bluelim, redlim, _name in reversed(self.lines.search):
            if bluelim > red or redlim < blue:
                continue
            ip1 = int(round_half(wcs.w2p(bluelim)))
            ip = int(round_half(wcs.w2p(blue_limit)))
            ip1 = max(ip1, ip, 1)
            ip2 = int(round_half(wcs.w2p(redlim)))
            ip = int(round_half(wcs.w2p(red_limit)))
            ip2 = min(ip2, ip, npts)
            centers, heights = emfind(fdata, npts, ip1, ip2, npfit, zsig)
            wavelength = height = 0.0
            for c, h in zip(centers, heights, strict=True):
                if h > height:
                    height = h
                    wavelength = wcs.p2w(c)
            if hmax < height:
                hmax = height
                zvel = ((wavelength / restwave) - 1.0) * C0
        return zvel

    def _search(self, smdata, npts, blue, red, zval, wcs, npfit, zsig, wspan):
        """emsrch: each line of the list where the redshift puts it, if it is there,
        sorted by rest wavelength."""
        L = self.lines
        blue_limit = blue - (red - blue) * WEXTRA
        red_limit = red + (red - blue) * WEXTRA
        found = []
        for k, wl in enumerate(L.wlref):
            zbcont = L.bcont[k] * zval
            if zbcont < blue_limit:
                continue
            zwave = wl * zval
            if zwave < blue_limit or zwave > red_limit:
                continue
            zrcont = L.rcont[k] * zval
            if zrcont > red_limit:
                continue
            center = self._gxyline(smdata, npts, blue_limit, red_limit, zwave, zbcont,
                                   zrcont, wcs, npfit, zsig, wspan)
            if center > 0.0:
                if found and abs(center - found[-1].pxobs) < 1.0:
                    continue
                found.append(Found(L.nmref[k], wl, center, wcs.p2w(center)))
            if len(found) == MAXREF:
                break
        out = []
        pool = list(found)
        rest = [f.wlrest for f in pool]
        for _ in range(len(pool)):
            wlmax, imax = 0.0, -1
            for j, r in enumerate(rest):
                if r > wlmax:
                    wlmax, imax = r, j
            if imax >= 0:
                f = pool[imax]
                out.append(Found(f.name, f.wlrest, f.pxobs, wcs.p2w(f.pxobs)))
                rest[imax] = 0.0
        return out[::-1]

    def _gxyline(self, fdata, npts, blue_limit, red_limit, lineval, blue_cont,
                 red_cont, wcs, npfit, zsig, wspan) -> float:
        blue_pix = int(round_half(wcs.w2p(blue_cont)))
        red_pix = int(round_half(wcs.w2p(red_cont)))
        if red_pix < blue_pix:
            blue_pix, red_pix = red_pix, blue_pix
        if red_pix < blue_pix + 4:
            return 0.0

        def at(i: int) -> float:
            return float(fdata[i - 1]) if 1 <= i <= npts else 0.0
        vave = 0.0
        for i in range(-16, 16):
            vave = vave + at(blue_pix + i)
        vrms = 0.0
        for i in range(blue_pix, red_pix + 1):
            val = at(i) - (vave / 32.0)
            vrms = vrms + (val * val)
            vave = vave + at(i + 16) - at(i - 16)
        vrms = zsig * math.sqrt(vrms / (red_pix - blue_pix + 1))
        center = wcs.w2p(lineval)
        ip1 = int(round_half(wcs.w2p(lineval - wspan)))
        ip2 = int(round_half(wcs.w2p(lineval + wspan)))
        if ip2 < ip1:
            ip1, ip2 = ip2, ip1
        ipl = int(round_half(wcs.w2p(blue_limit)))
        ipr = int(round_half(wcs.w2p(red_limit)))
        if ipl > ipr:
            ipl, ipr = ipr, ipl
        ip1 = max(ip1, ipl, 1)
        ip2 = min(ip2, ipr, npts)
        centers, heights = emfind(fdata, npts, ip1, ip2, npfit, zsig)
        height = 0.0
        for c, h in zip(centers, heights, strict=True):
            if h > height:
                height, center = h, c
        return center if height > vrms else 0.0

    # ---------------------------------------------------------------- emlfit

    def _iscombo(self, found: list[Found], n: int):
        """iscombo: the blend the n-th line found begins (index, half-width, how
        many Gaussians, which are found), or a single line."""
        restw = [f.wlrest for f in found]
        igauss = [False, False, False]
        ngauss, m = 0, n
        icombo, wlfit, ngfit = -1, 0.0, 1
        for i, combo in enumerate(self.lines.combos):
            if ngauss != 0:
                break
            for j in range(len(combo.elines)):
                if ngauss != 0:
                    break
                if abs(restw[m] - combo.elines[j]) < 0.1:
                    ngauss = 1
                    igauss[j] = True
                    wlfit = combo.edwl
                    ngfit = len(combo.elines)
                    icombo = i
                    for k in range(j + 1, ngfit):
                        m += 1
                        if m > len(found) - 1:
                            break
                        if abs(restw[m] - combo.elines[k]) < 0.1:
                            ngauss += 1
                            igauss[k] = True
                        else:
                            break
        if ngauss == 0:
            ngauss, ngfit = 1, 1
            wlfit = self.lines.wgfit[n] if n < len(self.lines.wgfit) else 0.0
            igauss[0] = True
        return ngauss, icombo, wlfit, ngfit, igauss

    def _fit_lines(self, found: list[Found], fdata, fcont, npix: int, wcs) -> None:
        nfound = len(found)
        mincont = float(self.p["mincont"])
        s2pi = math.sqrt(2.0 * 3.14159265358979323846)
        i, m = 0, 0
        while i < nfound:
            ngauss, ic, wlfit, ngfit, igauss = self._iscombo(found, i)
            npoly = 2
            pixleft = int(round_half(wcs.w2p(found[i].wlobs - wlfit)))
            pixright = int(round_half(wcs.w2p(found[i + ngauss - 1].wlobs + wlfit)))
            if pixright < pixleft:
                pixright = int(round_half(wcs.w2p(found[i].wlobs - wlfit)))
                pixleft = int(round_half(wcs.w2p(found[i + ngauss - 1].wlobs + wlfit)))
            pixleft = max(pixleft, 1)
            pixright = min(pixright, npix)
            gc, gh, gw = [0.0] * 3, [0.0] * 3, [0.0] * 3
            n = i
            for j in range(ngfit):
                if igauss[j]:
                    gc[j] = found[n].pxobs
                    gh[j] = float(fdata[int(gc[j]) - 1])
                    hheight = f32(0.5 * gh[j])
                    lp = int(round_half(gc[j]))
                    for _ in range(lp - 1, pixleft + 1):
                        lp -= 1
                        if fdata[lp - 1] < hheight:
                            break
                    rp = int(round_half(gc[j]))
                    for _ in range(rp + 1, pixright + 1):
                        rp += 1
                        if fdata[rp - 1] < hheight:
                            break
                    gw[j] = max(float(rp - lp - 1), 1.0)
                    n += 1
            combo = self.lines.combos[ic] if ic >= 0 else None
            for j in range(ngfit):
                if igauss[j]:
                    continue
                if combo.edrop[j]:
                    gh[j] = gw[j] = gc[j] = 0.0
                else:
                    for k in range(ngfit):
                        if igauss[k]:
                            gh[j] = gh[k] * combo.eht[j] / combo.eht[k]
                            gw[j] = gw[k]
                            wlj = combo.elines[j] * found[i].wlobs / found[i].wlrest
                            gc[j] = wcs.w2p(wlj)
                            break
            if combo is not None:
                if ngfit > 2 and combo.edrop[2] and gc[2] == 0.0:
                    ngfit = 2
                if ngfit > 1 and combo.edrop[1] and gc[1] == 0.0:
                    if ngfit == 2:
                        ngfit = 1
                    else:
                        ngfit = 2
                        gc[1], gh[1], gw[1], igauss[1] = gc[2], gh[2], gw[2], igauss[2]
                if ngfit > 1 and combo.edrop[0] and gc[0] == 0.0:
                    if ngfit == 2:
                        ngfit = 1
                        gc[0], gh[0], gw[0], igauss[0] = gc[1], gh[1], gw[1], igauss[1]
                    else:
                        ngfit = 2
                        gc[0], gh[0], gw[0], igauss[0] = gc[1], gh[1], gw[1], igauss[1]
                        gc[1], gh[1], gw[1], igauss[1] = gc[2], gh[2], gw[2], igauss[2]
            nparams = ngfit * 3 + npoly
            fitvec = [0.0] * 12
            diffback = float(f32(fdata[pixright - 1] - fdata[pixleft - 1]))
            fitvec[1] = diffback / float(pixright - pixleft)
            meanback = 0.5 * float(f32(fdata[pixleft - 1] + fdata[pixright - 1]))
            meanpix = 0.5 * float(pixright + pixleft)
            fitvec[0] = meanback - (meanpix * fitvec[1])
            for g in range(min(ngfit, 3)):
                fitvec[2 + 3 * g:5 + 3 * g] = [gc[g], gh[g], gw[g]]
            fitvec, covar, _nit, xi2, xi2dof, ierr = emmin(fdata, nparams, fitvec,
                                                           pixleft, pixright)
            for j in range(ngfit):
                if not igauss[j]:
                    continue
                f = found[m]
                f.params = np.zeros((2, 10))
                if ierr < 2:
                    jj = npoly + j * 3
                    for k in range(npoly):
                        f.params[0, k] = fitvec[k]
                        f.params[1, k] = covar[k]
                    for k in range(3):
                        f.params[0, k + 3] = fitvec[jj + k]
                        f.params[1, k + 3] = covar[jj + k]
                    px = f.params[0, 3]
                    wv = wcs.p2w(px)
                    f.wlobs = wv
                    dwdp = abs(wcs.p2w(px + 1.0) - wv)
                    sigma = fitvec[jj + 2]
                    height = abs(fitvec[jj + 1])
                    area = s2pi * height * sigma
                    icenter = int(round_half(fitvec[jj]))
                    cont = (float(fcont[icenter - 1]) + fitvec[0]
                            + (fitvec[1] * (fitvec[jj] - pixleft)))
                    if cont <= 0:
                        cont = 1.0
                    eqw = area / cont if cont > mincont else area
                    eqw = eqw * dwdp
                    f.params[0, 6] = eqw
                    if cont <= mincont:
                        relsig = covar[jj + 2] / sigma if sigma > 0.0 else 1.0e-4
                        relht = covar[jj + 1] / height if height > 0.0 else 1.0e-4
                        f.params[1, 6] = area * math.sqrt(relht * relht + relsig * relsig)
                    else:
                        skylevel = 0.0
                        with np.errstate(all="ignore"):
                            area = float(area + 8.0 * (cont + skylevel) * sigma
                                         + np.float64(cont + 2.0 * skylevel)
                                         / (dwdp * float(pixright - pixleft + 1)
                                            * (eqw * eqw)))
                        f.params[1, 6] = math.sqrt(abs(area)) / cont
                    f.params[0, 7] = xi2
                    f.params[1, 7] = xi2 / xi2dof if xi2dof else 0.0
                m += 1
            i += ngauss

    # ---------------------------------------------------------------- emvfit

    def _velocity(self, r: Result, blue, red, wcs) -> None:
        p = self.p
        drms = float(p["disperr"])
        lwmin = float(f32(p["lwmin"]))      # noqa: F841 -- meanwidth is 0: unused
        lsmin = float(f32(p["lsmin"]))
        found = r.lines
        nfound = len(found)
        weight = ave_vel = disp = 0.0
        nz = 0
        fiterr2 = verr2 = 0.0
        ok = [False] * nfound
        for k, f in enumerate(found):
            f.drop = 0
            if f.params[0, 3] > 0.0:
                ok[k] = True
            else:
                ok[k] = False
                f.drop = 1
                continue
            wv = wcs.p2w(f.params[0, 3])
            wv1 = wcs.p2w(f.params[0, 3] + 1.0)
            f.wlobs = wv
            dwdp = wv1 - wv
            if wv == wv1:
                wv1 = wcs.p2w(f.params[0, 3] - 1.0)
                dwdp = wv - wv1
            if f.params[1, 6] != 0.0 and lsmin > (f.params[0, 6] / f.params[1, 6]):
                ok[k], f.drop = False, 4
            if wv - blue < PIXMAX * dwdp:
                ok[k], f.drop = False, 5
            if red - wv < PIXMAX * dwdp:
                ok[k], f.drop = False, 6
            zvel = (f.wlobs / f.wlrest) - 1.0
            f.params[0, 8] = zvel
            verr = 2 * f.params[1, 3] * dwdp / wv
            if verr == 0.0:
                ok[k], f.drop = False, 7
            verr2 = verr * verr
            fiterr = drms / wv
            fiterr2 = fiterr * fiterr
            verr2 = verr2 + fiterr2
            f.params[1, 8] = math.sqrt(verr2)
            if ok[k]:
                weight = weight + (1.0 / verr2)
                ave_vel = ave_vel + (zvel / verr2)
                nz += 1
                f.params[0, 9] = math.sqrt(verr2)
            else:
                f.params[0, 9] = 0.0
            if f.params[1, 7] < 1.0:
                f.params[1, 7] = -1.0
        rms2 = 0.0
        if nz > 0:
            ave_vel = ave_vel / weight
            weight2 = 1.0 / weight
            for k, f in enumerate(found):
                if not ok[k]:
                    continue
                wv = wcs.p2w(f.params[0, 3])
                wv1 = wcs.p2w(f.params[0, 3] + 1.0)
                zvel = (wv / f.wlrest) - 1.0 - ave_vel
                verr = 2 * f.params[1, 3] * (wv1 - wv) / wv
                verr2 = verr * verr + fiterr2
                zvel2 = 0.0 if zvel == 0.0 else zvel * zvel
                if verr2 != weight2:
                    disp = disp + (zvel2 / (verr2 - weight2))
                rms2 = rms2 + zvel2
            disp = disp / float(nz)
            rms2 = rms2 / float(nz)
            fiterr2 = 0.0
        if nz == 1:
            rms2 = verr2
            fiterr = (drms * 2.0) / (blue + red)
            fiterr2 = fiterr * fiterr
        zav = ave_vel + 1.0 if nfound > 0 else 0.0
        zrms = math.sqrt(rms2 + fiterr2) if nfound > 0 else 0.0
        if nfound > 0 and zav != 0.0:
            r.vel = (zav - 1.0) * C0 + r.spectrum.hcv
        elif nfound > 0:
            r.vel = r.spectrum.hcv
        else:
            r.vel = INDEFD
        r.err = zrms * C0
        r.nfit = nz
        if nz == 1 and float(f32(p["sigline"])) > 0:
            r.err = float(f32(p["sigline"]))
        r.combined, r.combined_err = vcombine(r.xcvel, r.xcerr, r.xcr, r.vel, r.err, nz)
        total = 0.0
        for f in found:
            wt = f.params[0, 9]
            if wt != 0.0:
                total += 1.0 / (wt * wt)
        for f in found:
            wt = f.params[0, 9]
            f.weight = (1.0 / (wt * wt)) / total if wt != 0.0 else 0.0

    # ---------------------------------------------------------------- report

    def _report(self, r: Result) -> None:
        from .rvsao import juldate
        spec = r.spectrum
        head = spec.head
        self.out += "\n"     # emrslts' printf, to STDOUT whatever the log files
        ra = head.d("RA", INDEFD)
        dec = head.d("DEC", INDEFD)
        epoch = head.d("EPOCH", 1950.0)
        epoch = head.d("EQUINOX", epoch)
        names = {1: "BCV", 3: "HCV", 4: "BCV", 5: "File HCV",
                 2: "File BCV" if spec.vb else "File HCV"}
        ssvcor = names.get(self.svcor, "BCV")
        # every report mode above 0 turns the brackets of the name into spaces
        text = f"File: {spec.name.replace('[', ' ', 1).replace(']', ' ', 1)}\n"
        if ra != INDEFD and dec != INDEFD:
            text += (f"Object: {spec.object}  RA: {iraffmt.Format('%011.2h')(ra)} "
                     f"Dec: {iraffmt.Format('%010.1h')(dec)} {_f(epoch, 0, 1)}\n")
        else:
            text += f"Object: {spec.object}  (No position)\n"
        ut, dj, _hdj, (mm, dd, yyyy) = juldate(head)
        if None not in (mm, dd, yyyy):
            month = "JanFebMarAprMayJunJulAugSepOctNovDec"[3 * (mm - 1):3 * mm] \
                if 1 <= mm <= 12 else ""
            text += (f"Observed {yyyy:4d}-{month:>3.3s}-{dd:02d} "
                     f"{iraffmt.Format('%011.2h')(ut)} = JD {_f(dj, 0, 4)} {ssvcor}: "
                     f"{_f(spec.hcv, 6, 2)}\n")
        else:
            text += f" (No observation date) {ssvcor}: {_f(spec.hcv, 6, 2)}\n"
        if r.combined != INDEFD:
            text += (f"Combined vel = {_f(r.combined, 8, 2)} +- "
                     f"{_f(r.combined_err, 7, 2)} km/sec, z= {_f(r.combined / C0, 6, 4)} \n")
        if r.xcvel != INDEFD:
            text += (f"Correlation vel = {_f(r.xcvel, 8, 2)} +- {_f(r.xcerr, 7, 2)} km/sec, "
                     f"z= {_f(r.xcvel / C0, 6, 4)} R= {_f(r.xcr, 6, 1)} \n")
        if r.vel != INDEFD:
            text += (f"Emission vel = {_f(r.vel, 8, 2)} +- {_f(r.err, 7, 2)} km/sec, "
                     f"z= {_f(r.vel / C0, 6, 4)} for {r.nfit}/{len(r.lines)} lines\n")
        if r.lines:
            text += ("\nLine  Rest lam  Obs. lam    Pixel     z         vel    dvel"
                     "    eqw    wt\n")
            for f in r.lines:
                if f.params[0, 9] > 0:
                    wch = " "
                elif f.drop > 0:
                    wch = f"X{f.drop}"
                else:
                    wch = "X"
                text += (f"{f.name:>4s}   {_f(f.wlrest, 7, 2)}   {_f(f.wlobs, 7, 2)}  "
                         f"{_f(f.params[0, 3], 7, 2)}  {_f(f.params[0, 8], 6, 4)}  "
                         f"{_f((C0 * f.params[0, 8]) + spec.hcv, 8, 2)} "
                         f"{_f(C0 * f.params[1, 8], 7, 2)} {_f(f.params[0, 6], 6, 2)} "
                         f"{_f(f.weight, 6, 3)} {wch}\n")
        else:
            text += "No emission lines found\n"
        self._log(text)


def vcombine(velxc, vxerr, vr, velem, veerr, nlfit):
    """vcombine: the cross-correlation and emission velocities as one."""
    if velem == INDEFD and velxc == INDEFD:
        return INDEFD, INDEFD
    if velem == INDEFD or veerr <= 0.0 or nlfit == 0:
        if velxc != INDEFD:
            return velxc, math.sqrt(vxerr * vxerr)
        return 0.0, 0.0
    if velxc == INDEFD or vxerr <= 0.0:
        return velem, math.sqrt(veerr * veerr)
    emerr2, xcerr2 = veerr * veerr, vxerr * vxerr
    vdiff = velxc - velem
    fc = vdiff * vdiff / (xcerr2 + emerr2)
    if fc > 8:
        if vr > 15.0 or (nlfit < 3 and vr > 4.0):
            vel, err = velxc, vxerr
        else:
            vel, err = velem, veerr
    else:
        err2 = 1.0 / ((1.0 / emerr2) + (1.0 / xcerr2))
        vel = ((velxc / xcerr2) + (velem / emerr2)) * err2
        err = math.sqrt(err2)
    return vel, math.sqrt(err * err)


def decode_ranges(text: str) -> list[int]:
    """decode_ranges: "1-3,5 8" as 1 2 3 5 8."""
    import re

    out: list[int] = []
    for word in re.split(r"[\s,]+", text.strip()):
        if not word:
            continue
        m = re.fullmatch(r"(\d+)(?:-(\d+))?(?:x(\d+))?", word)
        if m is None:
            raise RvsaoError(f"T_EMSAO: Illegal multispec object list <{text}>")
        a = int(m.group(1))
        b = int(m.group(2)) if m.group(2) else a
        step = int(m.group(3)) if m.group(3) else 1
        out.extend(range(min(a, b), max(a, b) + 1, step))
    return out


def _f(value: float, width: int, decpl: int) -> str:
    """fmtio's %W.Df: fewer decimals until the number fits the width."""
    if width == 0:
        return iraffmt.dtoc3(value, decpl, "f")
    return iraffmt.printf(value, "f", decpl, width).rjust(width)


def round_half(x: float) -> float:
    """idnint: to the nearest integer, halves away from zero."""
    return math.floor(x + 0.5) if x >= 0 else -math.floor(-x + 0.5)
