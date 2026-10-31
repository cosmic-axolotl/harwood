"""IRAF's identify and autoidentify (noao.onedspec), ported.

From iraf-community 2.18, noao/onedspec/identify: idlinelist.x (the line
list, matching and locating lines -- the 'l' key), idnewfeature.x,
idfitdata.x (the fit at every pixel and its inverse), iddofit.x (the fit,
through icfit), idcenter.x and xtools' center1d, peaks.gx; and from
identify/autoid: aidautoid.x and autoid.x, the automatic identification --
spacing ratios of three lines voted on by patterns of up to npattern lines,
linear dispersions through the best-voted pairs, each evaluated by fitting,
locating more lines from the list and scoring the rms and the fractions of
lines matched.

Pixels are 1-based. Without a fit, the coordinate of a pixel is the pixel.

Translated from IRAF, Copyright (c) 1986 Association of Universities for
Research in Astronomy Inc.; the notice it is distributed under is in
LICENSES/IRAF-COPYRIGHT.
"""

from __future__ import annotations

import bisect
import contextlib
import math
from dataclasses import dataclass, field, replace
from pathlib import Path

import numpy as np

from . import icfit, jit
from .center1d import center1d

f32 = np.float32
MAX_REAL = 3.4028235e38
MAX_DOUBLE = 1.7976931348623157e308
DXMIN = 0.01


def nint(value) -> int:
    v = float(value)
    return int(math.floor(v + 0.5)) if v >= 0 else -int(math.floor(-v + 0.5))


# ------------------------------------------------------------------ line list

@dataclass
class LineList:
    values: np.ndarray                      # sorted, distinct
    labels: list[str]
    units: str = "Angstroms"

    def __len__(self) -> int:
        return self.values.size


def read_linelist(path: str | Path) -> LineList:
    """id_mapll: a value and an optional label per line; '# units' sets the
    units; sorted, with repeated values dropped."""
    values, labels = [], []
    units = "Angstroms"
    for line in Path(path).read_text(encoding="utf-8", errors="replace").splitlines():
        words = line.split()
        if not words:
            continue
        if words[0].startswith("#"):
            rest = (" ".join(words))[1:].split()
            if len(rest) >= 2 and rest[0].lower() == "units":
                units = " ".join(rest[1:]).strip()
            continue
        try:
            value = float(words[0])
        except ValueError:
            continue
        values.append(value)
        labels.append(" ".join(words[1:]))
    order = sorted(range(len(values)), key=lambda k: values[k])
    out_v: list[float] = []
    out_l: list[str] = []
    for k in order:
        if out_v and _fp_equald(values[k], out_v[-1]):
            continue
        out_v.append(values[k])
        out_l.append(labels[k])
    return LineList(np.array(out_v, dtype=float), out_l, units)


def _fp_equald(a: float, b: float) -> bool:
    return bool(abs(a - b) <= 2.0 * np.finfo(float).eps * max(abs(a), abs(b)))


# ------------------------------------------------------------------- identify

@dataclass
class Feature:
    pix: float
    fit: float
    user: float | None
    wts: float = 1.0
    label: str = ""
    fwidth: float | None = None
    ftype: str | None = None


@dataclass
class Identify:
    """The state identify keeps for one spectrum."""

    data: np.ndarray                        # the spectrum, 1-D
    linelist: LineList | None
    spec: icfit.FitSpec
    fwidth: float = 4.0
    ftype: str = "emission"
    cradius: float = 5.0
    threshold: float = 0.0
    minsep: float = 2.0
    match: float = -3.0
    maxfeatures: int = 50
    interp: str = "spline3"                 # c1d_params
    epsilon: float = 0.001
    features: list[Feature] = field(default_factory=list)
    fit: icfit.Fit | None = None
    shift: float = 0.0
    fitdata: np.ndarray = field(default=None)  # type: ignore[assignment,arg-type]
    stats: dict = field(default_factory=lambda: {"ncandidate": 0, "nmatch1": 0,
                                                 "nmatch2": 0})
    #: ID_NEWFEATURES: features added or replaced since the database was
    #: last written -- set by reading a record, cleared by writing one.
    newfeatures: bool = False
    current: int = -1                       #: ID_CURRENT: the last one newfeature touched

    def __post_init__(self):
        self.data = np.asarray(self.data, dtype=f32)
        self.npts = self.data.size
        self.pixdata = np.arange(1.0, self.npts + 1.0)
        if self.fitdata is None:
            self.fitdata = self.pixdata.copy()

    # -- the coordinate function
    def fitpt(self, pix: float) -> float:
        """id_fitpt."""
        value = pix if self.fit is None else float(self.fit(np.array([pix]))[0])
        return value + self.shift

    def update_fitdata(self) -> None:
        """id_fitdata: the fit at every pixel; not monotonic is an error."""
        if self.fit is None:
            fitdata = self.pixdata.copy()
        else:
            fitdata = np.asarray(self.fit(self.pixdata), dtype=float)
            d = np.diff(fitdata)
            if fitdata[1] > fitdata[0]:
                if (d[1:] < 0).any():
                    raise ValueError("Coordinate solution is not monotonic")
            elif (d[1:] > 0).any():
                raise ValueError("Coordinate solution is not monotonic")
        if self.shift != 0:
            fitdata = fitdata + self.shift
        self.fitdata = fitdata

    def fit_features(self) -> None:
        """id_fitfeatures."""
        for feature in self.features:
            feature.fit = self.fitpt(feature.pix)

    def fit_to_pix(self, coord: float) -> float | None:
        """fit_to_pix: the pixel the fit puts `coord` at, to 0.01 px."""
        fd = self.fitdata
        if self.fit is None:
            return coord - self.shift
        if self.fit.spec.function in ("legendre", "chebyshev") and \
                _cv_order(self.fit) == 2:
            dx = fd[0]
            return (coord - dx) / (fd[1] - dx) + 1
        if fd[0] < fd[-1]:
            if coord < fd[0] or coord > fd[-1]:
                return None
            i = 1
            while coord > fd[i - 1]:
                i += 1
            if fd[i - 1] == coord:
                return float(self.pixdata[i - 1])
            pix = i - 0.5
            dx = 1.0
            while dx > DXMIN:
                dx /= 2
                pix = pix + dx if self.fitpt(pix) < coord else pix - dx
        else:
            if coord < fd[-1] or coord > fd[0]:
                return None
            i = 1
            while coord < fd[i - 1]:
                i += 1
            if fd[i - 1] == coord:
                return float(self.pixdata[i - 1])
            pix = i - 0.5
            dx = 1.0
            while dx > DXMIN:
                dx /= 2
                pix = pix - dx if self.fitpt(pix) < coord else pix + dx
        return pix

    def fit_to_pix_many(self, coords: np.ndarray) -> np.ndarray:
        """fit_to_pix for many coordinates at once, step for step as the
        scalar one goes; NaN where it gives INDEF."""
        coords = np.asarray(coords, dtype=float)
        fd = self.fitdata
        if self.fit is None:
            return coords - self.shift
        if self.fit.spec.function in ("legendre", "chebyshev") and \
                _cv_order(self.fit) == 2:
            dx = fd[0]
            return (coords - dx) / (fd[1] - dx) + 1
        increasing = fd[0] < fd[-1]
        out = np.full(coords.shape, np.nan)
        if increasing:
            inside = (coords >= fd[0]) & (coords <= fd[-1])
            i = np.searchsorted(fd, coords, side="left") + 1     # first fd >= coord
        else:
            inside = (coords >= fd[-1]) & (coords <= fd[0])
            rev = fd[::-1]
            i = fd.size - np.searchsorted(rev, coords, side="right") + 1  # first fd <= coord
        i = np.clip(i, 1, fd.size)
        exact = inside & (fd[i - 1] == coords)
        out[exact] = self.pixdata[i[exact] - 1]
        todo = inside & ~exact
        pix = i[todo] - 0.5
        target = coords[todo]
        dx = 1.0
        while dx > DXMIN:
            dx /= 2
            below = (np.asarray(self.fit(pix), dtype=float) + self.shift) < target
            step = np.where(below, dx, -dx) if increasing else np.where(below, -dx, dx)
            pix = pix + step
        out[todo] = pix
        return out

    # -- features
    def center(self, x: float | None) -> float | None:
        """id_center."""
        if x is None:
            return None
        return center1d(x, self.data, self.fwidth, self.ftype, self.cradius,
                        self.threshold, self.interp, self.epsilon)

    def peak(self, pix: float) -> float:
        """id_peak: the value at the pixel above the mean of +/- fwidth."""
        d = self.data
        c = nint(pix)
        lo = max(1, nint(pix - self.fwidth))
        hi = min(self.npts, nint(pix + self.fwidth))
        return float(f32(d[c - 1] - f32((d[lo - 1] + d[hi - 1]) / f32(2))))

    def newfeature(self, pix, fit, user, wts=1.0, label="") -> int:
        """id_newfeature: insert in pixel order, or replace the nearest one
        closer than minsep if the new one fits better. Returns its index."""
        if pix is None:
            return -1
        # The nearest feature, the first of two equally near; they are kept
        # in pixel order, so a bisection finds it.
        delta, current = MAX_REAL, -1
        if self.features:
            pixes = [f.pix for f in self.features]
            k = bisect.bisect_left(pixes, pix)
            for i in (k - 1, k):
                if 0 <= i < len(pixes) and abs(pix - pixes[i]) < delta:
                    delta = abs(pix - pixes[i])
                    current = i
        new = Feature(pix, fit, user, wts, label)
        if delta >= self.minsep:
            index = len(self.features)
            while index > 0 and pix < self.features[index - 1].pix:
                index -= 1
            self.features.insert(index, new)
            self.newfeatures = True
            self.current = index
            return index
        old = self.features[current]
        if user is not None and old.user is not None and \
                abs(fit - user) < abs(old.fit - old.user):
            self.features[current] = new
            self.newfeatures = True
        self.current = current
        return current

    def match_line(self, value: float, diff: float):
        """id_match: the nearest line in the list within `diff` (negative:
        pixels), as (value, label) or (None, '')."""
        if self.linelist is None or len(self.linelist) == 0:
            return value, ""
        fd = self.fitdata
        delta = abs(diff * (fd[0] - fd[-1]) / (self.npts - 1)) if diff < 0 else diff
        ll = self.linelist.values
        nll = ll.size
        j = max(1, nint(math.sqrt(nll)))
        i = 0
        while i < nll and value > ll[i]:
            i += j
        i = max(0, min(i - 1, nll - 1))
        while i > 0 and value < ll[i]:
            i -= 1
        if i < nll - 1 and abs(value - ll[i]) > abs(value - ll[i + 1]):
            i += 1
        if abs(value - ll[i]) <= delta:
            return float(ll[i]), self.linelist.labels[i]
        return None, ""

    def locate(self) -> None:
        """id_linelist: centre every list line the fit puts on the spectrum,
        keep those that land within `match`, at most maxfeatures of the
        strongest, and add them."""
        if self.linelist is None:
            return
        st = self.stats
        st["ncandidate"] = st["nmatch1"] = st["nmatch2"] = 0
        fd = self.fitdata
        cd = (fd[0] - fd[-1]) / (self.npts - 1) if self.match < 0 else 1.0
        fit1, fit2 = min(fd[0], fd[-1]), max(fd[0], fd[-1])
        found: list[list] = []
        lastpix = -1
        minval = MAX_REAL
        nextpix = -1
        values = self.linelist.values
        lo = int(np.searchsorted(values, fit1, side="left"))
        hi = int(np.searchsorted(values, fit2, side="right"))
        guesses = self.fit_to_pix_many(values[lo:hi])
        centres = [self.center(None if np.isnan(g) else float(g)) for g in guesses]
        good = [k for k, c in enumerate(centres) if c is not None]
        fits = np.full(len(centres), np.nan)
        if good:
            fits[good] = np.asarray(self.fit(np.array([centres[k] for k in good])),
                                    dtype=float) + self.shift \
                if self.fit is not None else np.array([centres[k] for k in good]) + self.shift
        for offset in range(hi - lo):
            user = values[lo + offset]
            label = self.linelist.labels[lo + offset]
            st["ncandidate"] += 1
            pix = centres[offset]
            if pix is None:
                continue
            fit = float(fits[offset])
            diff = abs((fit - user) / cd)
            if diff > abs(self.match):
                continue
            st["nmatch1"] += 1
            if lastpix >= 0 and abs(pix - found[lastpix][0]) < 0.01:
                diff1 = abs(found[lastpix][1] - found[lastpix][2])
                if diff < diff1:
                    found[lastpix] = [pix, fit, float(user), label]
                continue
            st["nmatch2"] += 1
            peak = abs(self.peak(pix))
            if len(found) < self.maxfeatures:
                found.append([pix, fit, float(user), label])
                if peak < minval:
                    nextpix, minval = len(found) - 1, peak
                lastpix = len(found) - 1
            elif peak > minval:
                found[nextpix] = [pix, fit, float(user), label]
                lastpix = nextpix
                minval = MAX_REAL
                for k, item in enumerate(found):
                    p = abs(self.peak(item[0]))
                    if p < minval:
                        nextpix, minval = k, p
        first = -1
        for number, (pix, fit, user, label) in enumerate(found):
            self.newfeature(pix, fit, user, 1.0, label)
            if number == 0:
                first = self.current
        self.current = first      # cursave: the first one added, and it is left current

    def dofit(self, spec: icfit.FitSpec | None = None) -> None:
        """id_dofit, non-interactive: fit user coordinates against pixels
        (weights max(1, wts)); rejected features stay, with weight 0."""
        spec = spec or self.spec
        use = [f for f in self.features if f.pix is not None and f.user is not None]
        if len(use) <= 1:
            self.fit = None
            self.shift = 0.0
            return
        x = np.array([f.pix for f in use])
        y = np.array([f.user for f in use])
        fit = icfit.fit(x, y, spec, xrange=(1.0, float(self.npts)))
        if fit.model is None:
            raise ValueError(fit.message or "the coordinate function could not be fitted")
        rejected = ~fit.mask
        for feature, out in zip(use, rejected, strict=True):
            feature.wts = 0.0 if out else max(1.0, feature.wts)
        self.fit = fit
        self.shift = 0.0

    def prune(self) -> None:
        """Keep what aid_dofit keeps: features with no user coordinate or a
        weight."""
        self.features = [f for f in self.features if f.user is None or f.wts != 0]

    def rms(self) -> float:
        used = [f for f in self.features if f.user is not None and f.wts != 0]
        if not used:
            return float("nan")
        total = 0.0
        for f in used:
            total += (f.fit - float(f.user or 0.0)) ** 2
        return math.sqrt(total / len(used))


def _cv_order(fit) -> int:
    """CVORDER as the curve was made: polynomial terms clamped to the points."""
    order = int(fit.spec.order)
    n = int(np.count_nonzero(fit.mask))
    return min(order, n) if fit.spec.function in ("legendre", "chebyshev") else order


# ---------------------------------------------------------------------- peaks

def _is_local_max(index: int, d) -> bool:
    npts = d.size
    value = d[index - 1]
    i, nleft = index - 1, 0
    while i >= 1:
        if d[i - 1] != value:
            break
        nleft += 1
        i -= 1
    j, nright = index + 1, 0
    while j <= npts:
        if d[j - 1] != value:
            break
        nright += 1
        j += 1
    if i == 0 and j == npts + 1:
        return False
    if i == 0:
        return not d[j - 1] > value
    if j == npts + 1:
        return not d[i - 1] > value
    if d[i - 1] > value or d[j - 1] > value:
        return False
    return (nleft - nright) in (0, 1)


def _ipeaks(data, contrast, separation, edge, threshold):
    """find_ipeaks: local maxima, thresholded, ranked strongest first, the
    weaker of any two within `separation` marked None."""
    d = np.asarray(data, dtype=f32)
    npts = d.size
    x: list = [i for i in range(1, npts + 1) if _is_local_max(i, d)]
    if edge > 0:
        x = [v for v in x if edge < v <= npts - edge]
    y = [d[v - 1] for v in x]
    if x and (threshold is not None or contrast is not None):
        maxval = max(y)
        if threshold is None:
            lcut = f32(contrast * maxval)
        elif contrast is None:
            lcut = f32(threshold)
        else:
            lcut = max(f32(threshold), f32(contrast * maxval))
        keep = [k for k in range(len(x)) if not y[k] < lcut]
        x, y = [x[k] for k in keep], [y[k] for k in keep]
    rank = sorted(range(len(x)), key=lambda k: -y[k])
    for a in range(len(rank)):
        if x[rank[a]] is None:
            continue
        for b in range(a + 1, len(rank)):
            if x[rank[b]] is not None and abs(x[rank[a]] - x[rank[b]]) < separation:
                x[rank[b]] = None
    return x, rank


def find_peaks(data, contrast=None, separation=0, edge=0, nmax=1000, threshold=None):
    """peaks.gx find_peaks: the nmax strongest isolated peaks, in position
    order."""
    x, rank = _ipeaks(data, contrast, separation, edge, threshold)
    if nmax < len(x):
        n = 0
        for k in rank:
            if x[k] is None:
                continue
            n += 1
            if n > nmax:
                x[k] = None
    return [float(v) for v in x if v is not None]


def find_upeaks(data, contrast=None, separation=0, edge=0, nmax=1000, nbins=5,
                threshold=None):
    """peaks.gx find_upeaks: up to nmax isolated peaks spread over nbins
    circular bins, the strongest left in each bin taken in turn."""
    npts = np.asarray(data).size
    x, rank = _ipeaks(data, contrast, separation, edge, threshold)
    n = len(x)
    if nmax < n:
        width = min(npts - 1, nint((npts - 1) / (nbins - 0.5)))
        x2 = 1
        npeaks = 0
        xs: list = [None if v is None else float(v) for v in x]
        while True:
            x1 = x2
            x2 = (x1 + width) % npts + 1
            skipped = 0
            for k in rank:
                a = xs[k]
                if a is None or a < 0:
                    skipped += 1
                    continue
                if (x1 < x2 and x1 <= a <= x2) or (x1 >= x2 and (a <= x2 or a >= x1)):
                    xs[k] = -a
                    npeaks += 1
                    break
            if npeaks >= nmax or skipped == n:
                break
        x = [(-v if (v is not None and v < 1) else None) for v in xs]
    return [float(v) for v in x if v is not None]


# ------------------------------------------------------------- autoidentify

@dataclass
class AidPars:
    """aidpars, with the values autoidentify reads from it."""

    crpix: float | None = None
    crquad: float | None = None
    cddir: str = "sign"
    crsearch: float | None = None
    cdsearch: float | None = None
    ntarget: int = 100
    npattern: int = 5
    nneighbors: int = 10
    nbins: int = 6
    ndmax: int = 500
    aidord: int = 3
    maxnl: float = 0.02
    nfound: int = 6
    sigma: float = 0.05
    minratio: float = 0.1
    rms: float = 0.1
    fmatch: float = 0.2
    nbest: int = 3
    wrms: float = 0.34
    wfmatch: float = 0.33
    wftmatch: float = 0.33


@dataclass
class _Ev:
    bin1: int
    bin2: int
    x: list = field(default_factory=list)     # pixels (real)
    y: list = field(default_factory=list)     # coordinates (real)
    a: list = field(default_factory=list)     # dispersion origins
    b: list = field(default_factory=list)     # dispersion slopes
    nd: int = 0


@dataclass
class AidResult:
    found: bool
    best: float
    rms: float
    fmatch: float
    ftmatch: float
    crval: float | None
    cdelt: float | None


class _Aid:
    def __init__(self, ident: Identify, crval, cdelt, pars: AidPars):
        self.id = ident
        self.p = pars
        self.crval = crval
        self.cdelt = cdelt
        self.np = pars.npattern
        self.nb = pars.nbins
        self.cddir = pars.cddir
        self.best = MAX_REAL
        self.rms = self.fmatch = self.ftmatch = float("nan")
        self.saved: tuple | None = None

    # aid_target
    def target(self):
        idt, p = self.id, self.p
        nt = idt.npts
        self.crpix = (nt + 1) / 2.0 if p.crpix is None else p.crpix
        self.crquad = 0.0 if p.crquad is None else p.crquad
        self.crsearch, self.cdsearch = p.crsearch, p.cdsearch
        if self.crval is not None and self.cdelt is not None:
            dw = nt * self.cdelt
            if self.crsearch is None:
                self.crsearch = abs(0.1 * dw)
            elif self.crsearch < 0:
                self.crsearch = abs(self.crsearch * dw)
            if self.cdsearch is None:
                self.cdsearch = abs(0.1 * self.cdelt)
            elif self.cdsearch < 0:
                self.cdsearch = abs(self.cdsearch * self.cdelt)
            self.crsearch = max(abs(0.0001 * dw), self.crsearch)
            self.cdsearch = max(abs(0.0001 * self.cdelt), self.cdsearch)
            dwmax = 2 * self.crsearch + (nt - 1) * (abs(self.cdelt) + self.cdsearch)
            dwmin = (abs(self.cdelt) - self.cdsearch) * (nt - 1)
            dwmin = max(1.0e-1, dwmin / dwmax)
            self.nb = nint(1.0 / dwmin)
        data = idt.data if idt.ftype != "absorption" else -idt.data
        peaks = find_upeaks(data, None, int(idt.minsep), 0, p.ntarget, 5, None)
        peaks.sort(key=lambda v: float(-idt.data[nint(v) - 1]))
        xtf: list[float] = []
        for pix in peaks:
            c = idt.center(pix)
            if c is None:
                continue
            if any(abs(c - v) < 1.0 for v in xtf):
                continue
            xtf.append(c)
        if not xtf:
            raise RuntimeError("No target lines found")
        self.xtf = xtf
        self.xt = sorted(xtf)
        self.xtl = [v + self.crquad * (v - self.crpix) ** 2 for v in self.xt]
        self.nt = len(self.xt)

    # aid_reference, from the line list
    def reference(self, ev: _Ev, flip: bool):
        idt = self.id
        nt = idt.npts
        self.crmin, self.crmax = -MAX_DOUBLE, MAX_DOUBLE
        self.cdmin, self.cdmax = 0.0, MAX_DOUBLE
        if self.cdelt is None:
            self.cdsign = {"increasing": 1, "decreasing": -1}.get(
                self.cddir, -1 if flip else 1)
            if self.crval is not None:
                self.crmin = self.crval - self.crsearch
                self.crmax = self.crval + self.crsearch
            wa, wb = -MAX_DOUBLE, MAX_DOUBLE
        elif self.crval is None:
            cdelt = self._signed(flip)
            self.cdmin = abs(cdelt) - self.cdsearch
            self.cdmax = abs(cdelt) + self.cdsearch
            wa, wb = -MAX_DOUBLE, MAX_DOUBLE
        else:
            cdelt = self._signed(flip)
            self.crmin = self.crval - self.crsearch
            self.crmax = self.crval + self.crsearch
            self.cdmin = abs(cdelt) - self.cdsearch
            self.cdmax = abs(cdelt) + self.cdsearch
            if cdelt > 0:
                wa = self.crmin + (cdelt + self.cdsearch) * (1 - self.crpix)
                wb = self.crmax + (cdelt + self.cdsearch) * (nt - self.crpix)
            else:
                wa = self.crmin + (cdelt - self.cdsearch) * (nt - self.crpix)
                wb = self.crmax + (cdelt - self.cdsearch) * (1 - self.crpix)
        assert idt.linelist is not None
        ll = idt.linelist.values
        sel = ll[(ll >= wa) & (ll <= wb)]
        # IRAF stops at the first line past wb, which in a sorted list is
        # the same selection.
        if sel.size == 0:
            raise RuntimeError("No reference lines found")
        wa = sel[0]
        wb = sel[-1] - sel[0]
        wb = wb / ((ev.bin1 + 1) // 2)
        wa = wa + wb / 2 * (ev.bin2 - 1)
        wb = wa + wb
        sel = ll[(ll >= wa) & (ll <= wb)]
        if sel.size == 0:
            raise RuntimeError("No reference lines found")
        self.xr = [float(v) for v in sel]
        self.nr = len(self.xr)

    def _signed(self, flip):
        if self.cddir == "increasing":
            self.cdsign = 1
            return abs(self.cdelt)
        if self.cddir == "decreasing":
            self.cdsign = -1
            return -abs(self.cdelt)
        cdelt = -self.cdelt if flip else self.cdelt
        self.cdsign = -1 if cdelt < 0 else 1
        return cdelt

    # aid_autoid1
    def autoid1(self, ev: _Ev):
        nn = self.p.nneighbors
        x1, n1 = self.xr, self.nr
        x2, x3, n2 = self.xtl, self.xt, self.nt
        r1 = _ratios(x1, n1, 1, nn, self.p.sigma, self.p.minratio, 1)
        s1 = sorted(range(len(r1)), key=lambda k: r1[k][0])
        r2 = _ratios(x2, n2, self.cdsign, nn, self.p.sigma, self.p.minratio, 2)
        s2 = sorted(range(len(r2)), key=lambda k: r2[k][0])
        votes = self._votes(r1, s1, r2, s2, n1, n2)
        flat = [votes[i][j] for j in range(n2) for i in range(n1)]
        svotes = sorted(range(n1 * n2), key=lambda k: -flat[k])
        n = 0
        for n in range(1, n1 * n2 + 1):
            if flat[svotes[n - 1]] < 1:
                break
        else:
            n = n1 * n2 + 1
        n = max(3 * n2, n - 1)
        xs, ys, ws = self._select(x1, x2, x3, flat, svotes, n1, n2, n)
        ev.x, ev.y = xs, ys
        ev.a, ev.b = self._disp(ys, xs, ws)
        ev.nd = len(ev.a)

    def _votes(self, r1, s1, r2, s2, n1, n2):
        nn = self.p.nneighbors
        npat = max(3, min(self.np, n1 - 5))
        maxr = 1.0 / self.p.minratio
        votes = [[0] * n2 for _ in range(n1)]
        lines1 = [_lines(k + 1, n1, nn) for k in range(len(r1))]
        lines2 = [_lines(k + 1, n2, nn) for k in range(len(r2))]
        start = 0
        for j in range(len(s2)):
            rb1 = r2[s2[j]][0]
            if rb1 > maxr:
                break
            rb2 = r2[s2[j]][1]
            for i in range(start, len(s1)):
                ra = r1[s1[i]][0]
                if ra > rb2:
                    break
                if ra < rb1:
                    start = i + 1
                    continue
                a = list(lines1[s1[i]])
                b = list(lines2[s2[j]])
                self._addlines(r1, s1[i] + 1, r2, s2[j] + 1, nn, a, b, npat,
                               votes, n1, n2)
        return votes

    def _addlines(self, r1, s1, r2, s2, nn, a, b, npat, votes, n1, n2):
        """aid_addlines: grow the matched triplet line by line up to the
        pattern size, voting for every complete pattern."""
        na = min(a[0] + nn, n1)
        nb = min(b[0] + nn, n2)
        i1 = s1 - a[2]
        j1 = s2 - b[2]

        def grow(level, a, b):
            if level >= npat:
                self._vote(a, b, npat, votes, n2)
                return
            for av in range(a[level - 1] + 1, na + 1):
                for bv in range(b[level - 1] + 1, nb + 1):
                    i = i1 + av
                    j = j1 + bv
                    if r1[i - 1][0] < r2[j - 1][0] or r1[i - 1][0] > r2[j - 1][1]:
                        continue
                    grow(level + 1, a + [av], b + [bv])

        grow(3, a[:3], b[:3])

    def _vote(self, a, b, npat, votes, n2):
        if self.cdsign == 1:
            for k in range(npat):
                votes[a[k] - 1][b[k] - 1] += 1
        else:
            for k in range(npat):
                votes[a[k] - 1][n2 - b[k]] += 1

    def _select(self, x1, x2, x3, flat, svotes, n1, n2, ns):
        check = self.crmin > -MAX_DOUBLE / 10.0 and self.crmax < MAX_DOUBLE / 10.0
        xs, ys, ws = [], [], []
        for k in range(n1 * n2):
            if len(xs) >= ns:
                break
            i = svotes[k] % n1
            j = svotes[k] // n1
            if flat[svotes[k]] < 1:
                break
            if check:
                a = (x2[j] - self.crpix) * self.cdsign * self.cdmin
                b = (x2[j] - self.crpix) * self.cdsign * self.cdmax
                if x1[i] < self.crmin + min(a, b) or x1[i] > self.crmax + max(a, b):
                    continue
            xs.append(float(f32(x3[j])))
            ys.append(float(f32(x1[i])))
            ws.append(float(f32(flat[svotes[k]])))
        if not xs:
            raise RuntimeError("No matches found")
        return xs, ys, ws

    def _disp(self, x, y, v):
        """aid_disp: x are the coordinates, y the pixels (real). Sorted by
        coordinate in place -- the votes are not, as in IRAF.

        The point-gathering for each pair is done a row of pairs at a time,
        the sums accumulated in the same order as IRAF's loop (a cumulative
        sum is sequential, and adding zero changes nothing), so the numbers
        are the loop's to the last bit."""
        order = sorted(range(len(x)), key=lambda k: x[k])
        xs = [x[k] for k in order]
        ys = [y[k] for k in order]
        x[:], y[:] = xs, ys
        n = len(x)
        nd = self.p.ndmax
        if jit.available():
            W1k = np.zeros(nd)
            DWk = np.zeros(nd)
            NWk = np.zeros(nd, dtype=np.int64)
            NVk = np.zeros(nd, dtype=np.int64)
            m = _disp_kernel(np.array(x, dtype=float), np.array(y, dtype=float),
                             np.array(v, dtype=float), float(self.crpix),
                             float(self.cdmin), float(self.cdmax), float(self.crmin),
                             float(self.crmax), float(self.cdsign), nd, W1k, DWk, NWk, NVk)
            return [float(a) for a in W1k[:m]], [float(b) for b in DWk[:m]]
        X = np.array(x, dtype=float)
        Y = np.array(y, dtype=float)
        V = np.array(v, dtype=float)
        w1: list[float] = []
        dw: list[float] = []
        nw: list[int] = []
        nv: list[int] = []
        W1 = np.zeros(0)
        DW = np.zeros(0)
        crpix = self.crpix
        cdmin, cdmax, crmin, crmax, sign = (self.cdmin, self.cdmax, self.crmin,
                                            self.crmax, self.cdsign)
        kk = np.arange(n)
        x0, xn = X[0], X[-1]
        for i in range(n - 2):
            js = np.arange(i + 1, n - 1)
            ok = (X[js] != X[i]) & (Y[js] != Y[i])
            js = js[ok]
            if js.size == 0:
                continue
            bw = (X[js] - X[i]) / (Y[js] - Y[i])
            aw = X[i] - bw * Y[i]
            cw = aw + bw * crpix
            keep = ~((np.abs(bw) < cdmin) | (np.abs(bw) > cdmax) | (cw < crmin) |
                     (cw > crmax) | (sign * bw < 0))
            js, bw, aw = js[keep], bw[keep], aw[keep]
            if js.size == 0:
                continue
            resid = np.abs((X[None, :] - aw[:, None] - bw[:, None] * Y[None, :])
                           / bw[:, None])
            mask = ~(resid > 2.0) & (kk[None, :] > js[:, None])

            sumn = 2 + mask.sum(axis=1)
            sumv = (V[i] + V[js] + np.where(mask, V[None, :], 0.0).sum(axis=1)).astype(int)
            sumx = _sequential(mask, X[i] + X[js], X)
            sumy = _sequential(mask, Y[i] + Y[js], Y)
            sumyy = _sequential(mask, Y[i] * Y[i] + Y[js] * Y[js], Y * Y)
            sumxy = _sequential(mask, X[i] * Y[i] + X[js] * Y[js], X * Y)
            den = sumn * sumyy - sumy * sumy
            awf = (sumx * sumyy - sumy * sumxy) / den
            bwf = (sumn * sumxy - sumx * sumy) / den
            cwf = awf + bwf * crpix
            q = 0
            nq = js.size
            while q < nq:
                # Which earlier dispersion each remaining pair repeats, against
                # the list as it stands; recomputed only when the list changes.
                m = len(w1)
                if m:
                    rest = slice(q, nq)
                    same = (np.abs((x0 - awf[rest, None]) / bwf[rest, None]
                                   - (x0 - W1[None, :]) / DW[None, :]) < 2.0) & \
                           (np.abs((xn - awf[rest, None]) / bwf[rest, None]
                                   - (xn - W1[None, :]) / DW[None, :]) < 2.0)
                    hits = np.where(same.any(axis=1), same.argmax(axis=1), m)
                else:
                    hits = np.zeros(nq - q, dtype=int)
                changed = False
                for r in range(q, nq):
                    a_, b_ = float(awf[r]), float(bwf[r])
                    n_, v_ = int(sumn[r]), int(sumv[r])
                    k = int(hits[r - q])
                    if k < m:
                        if n_ > nw[k] or (n_ == nw[k] and v_ > nv[k]):
                            ell = k
                            while ell > 0:
                                if n_ < nw[ell - 1] or (n_ == nw[ell - 1] and
                                                        v_ < nv[ell - 1]):
                                    break
                                w1[ell], dw[ell] = w1[ell - 1], dw[ell - 1]
                                nw[ell], nv[ell] = nw[ell - 1], nv[ell - 1]
                                ell -= 1
                            w1[ell], dw[ell] = float(f32(a_)), float(f32(b_))
                            nw[ell], nv[ell] = n_, v_
                            changed = True
                        if changed:
                            q = r + 1
                            break
                        continue
                    c_ = float(cwf[r])
                    if abs(b_) < cdmin or abs(b_) > cdmax or c_ < crmin or \
                            c_ > crmax or sign * b_ < 0:
                        continue
                    k = 0
                    while k < m:
                        if n_ > nw[k] or (n_ == nw[k] and v_ > nv[k]):
                            break
                        k += 1
                    if k < nd:
                        if m < nd:
                            w1.append(0.0)
                            dw.append(0.0)
                            nw.append(0)
                            nv.append(0)
                            m += 1
                        for ell in range(m - 1, k, -1):
                            w1[ell], dw[ell] = w1[ell - 1], dw[ell - 1]
                            nw[ell], nv[ell] = nw[ell - 1], nv[ell - 1]
                        w1[k], dw[k] = float(f32(a_)), float(f32(b_))
                        nw[k], nv[k] = n_, v_
                        changed = True
                        q = r + 1
                        break
                if changed:
                    W1, DW = np.array(w1), np.array(dw)
                else:
                    q = nq
        return w1, dw

    # aid_eval
    def evaluate(self, ev: _Ev, nd: int):
        if nd > ev.nd:
            return None
        idt = self.id
        a = ev.a[nd - 1]
        b = ev.b[nd - 1]
        d = b if self.cdelt is None else self.cdelt
        saved_spec = idt.spec
        idt.spec = icfit.FitSpec(function="chebyshev", order=self.p.aidord, niterate=5,
                                 low_reject=2.0, high_reject=2.0)
        idt.features = []
        idt.fit = None
        idt.shift = 0.0
        idt.update_fitdata()
        for px, wy in zip(ev.x, ev.y, strict=True):
            if abs((wy - a - b * px) / b) < 2.0:
                idt.newfeature(px, px, wy, 1.0)
        best = None
        try:
            if len(idt.features) > 1:
                rms, fmatch, ftmatch, best = self.dofit(d)
                if best is not None and best < self.best:
                    self.fmatch, self.ftmatch, self.rms, self.best = (
                        fmatch, ftmatch, rms, best)
                    self.saved = ([replace(f) for f in idt.features], idt.fit)
        except (ValueError, RuntimeError, ZeroDivisionError):
            best = None
        idt.spec = saved_spec
        return best

    # aid_dofit
    def dofit(self, cdelt):
        idt, p = self.id, self.p
        maxfeatures = idt.maxfeatures
        idt.maxfeatures = 1000
        try:
            for k in range(1, 4):
                if len(idt.features) < 2:
                    raise ValueError("aid_dofit: not enough features")
                if k > 1:
                    idt.locate()
                idt.dofit()
                for order in range(p.aidord - 1, 1, -1):
                    user = idt.fitpt(1.0) - idt.shift
                    fit = (idt.fitpt(float(idt.npts // 2)) - idt.shift - user) / \
                          (idt.fitpt(float(idt.npts)) - idt.shift - user)
                    if abs(fit - 0.5) <= p.maxnl:
                        break
                    idt.dofit(replace(idt.spec, order=order))
                idt.update_fitdata()
                idt.fit_features()
                idt.prune()
        finally:
            idt.maxfeatures = maxfeatures
        if cdelt is None:
            return None, None, None, None
        nmin = 2
        nfound = p.nfound
        if len(idt.features) < nfound:
            raise ValueError("aid_dofit: not enough features")
        rms = sum((f.fit - f.user) ** 2 for f in idt.features)
        rms = math.sqrt(rms / max(1, len(idt.features) - nmin)) / abs(cdelt)
        rms = rms / idt.fwidth
        st = idt.stats
        st["ncandidate"] = max(nfound, st["ncandidate"] - (st["nmatch1"] - st["nmatch2"]))
        fmatch = 1 - float(f32(st["nmatch2"])) / st["ncandidate"]
        ntmatch = 0
        for x in self.xt:
            user, _ = idt.match_line(idt.fitpt(x), idt.match)
            if user is not None:
                ntmatch += 1
        ftmatch = 1 - float(f32(ntmatch)) / min(self.nt, st["ncandidate"])
        if p.rms > 0 and p.fmatch > 0:
            best = p.wrms * rms / p.rms
            best += p.wfmatch * fmatch / p.fmatch
            best += p.wftmatch * ftmatch / p.fmatch
        else:
            best = MAX_DOUBLE
        return rms, fmatch, ftmatch, best

    # aid_dofitf
    def dofitf(self):
        idt = self.id
        maxfeatures = idt.maxfeatures
        idt.maxfeatures = 1000
        try:
            for k in range(1, 4):
                if len(idt.features) < 2:
                    raise ValueError("aid_dofit: not enough features")
                if k > 1:
                    idt.locate()
                idt.dofit()
                idt.update_fitdata()
                idt.fit_features()
                if k < 3:
                    idt.prune()
        finally:
            idt.maxfeatures = maxfeatures


@jit.jit
def _disp_kernel(X, Y, V, crpix, cdmin, cdmax, crmin, crmax, sign, nd, W1, DW, NW, NV):
    """aid_disp's loops as IRAF runs them, compiled when numba is there.
    Returns how many dispersions are in W1/DW (single precision values)."""
    n = X.shape[0]
    m = 0
    for i in range(n - 2):
        for j in range(i + 1, n - 1):
            if X[j] == X[i] or Y[j] == Y[i]:
                continue
            bw = (X[j] - X[i]) / (Y[j] - Y[i])
            aw = X[i] - bw * Y[i]
            cw = aw + bw * crpix
            if abs(bw) < cdmin or abs(bw) > cdmax or cw < crmin or cw > crmax or \
                    sign * bw < 0.0:
                continue
            sumn = 2
            sumv = int(V[i] + V[j])
            sumx = X[i] + X[j]
            sumy = Y[i] + Y[j]
            sumyy = Y[i] * Y[i] + Y[j] * Y[j]
            sumxy = X[i] * Y[i] + X[j] * Y[j]
            for k in range(j + 1, n):
                if abs((X[k] - aw - bw * Y[k]) / bw) > 2.0:
                    continue
                sumn += 1
                sumv = int(sumv + V[k])
                sumx += X[k]
                sumy += Y[k]
                sumyy += Y[k] * Y[k]
                sumxy += X[k] * Y[k]
            den = sumn * sumyy - sumy * sumy
            aw = (sumx * sumyy - sumy * sumxy) / den
            bw = (sumn * sumxy - sumx * sumy) / den
            cw = aw + bw * crpix
            k = 0
            while k < m:
                if abs((X[0] - aw) / bw - (X[0] - W1[k]) / DW[k]) < 2.0 and \
                        abs((X[n - 1] - aw) / bw - (X[n - 1] - W1[k]) / DW[k]) < 2.0:
                    break
                k += 1
            if k < m:
                if sumn > NW[k] or (sumn == NW[k] and sumv > NV[k]):
                    ell = k
                    while ell > 0:
                        if sumn < NW[ell - 1] or (sumn == NW[ell - 1] and
                                                  sumv < NV[ell - 1]):
                            break
                        W1[ell] = W1[ell - 1]
                        DW[ell] = DW[ell - 1]
                        NW[ell] = NW[ell - 1]
                        NV[ell] = NV[ell - 1]
                        ell -= 1
                    W1[ell] = np.float64(np.float32(aw))
                    DW[ell] = np.float64(np.float32(bw))
                    NW[ell] = sumn
                    NV[ell] = sumv
                continue
            if abs(bw) < cdmin or abs(bw) > cdmax or cw < crmin or cw > crmax or \
                    sign * bw < 0.0:
                continue
            k = 0
            while k < m:
                if sumn > NW[k] or (sumn == NW[k] and sumv > NV[k]):
                    break
                k += 1
            if k < nd:
                if m < nd:
                    m += 1
                for ell in range(m - 1, k, -1):
                    W1[ell] = W1[ell - 1]
                    DW[ell] = DW[ell - 1]
                    NW[ell] = NW[ell - 1]
                    NV[ell] = NV[ell - 1]
                W1[k] = np.float64(np.float32(aw))
                DW[k] = np.float64(np.float32(bw))
                NW[k] = sumn
                NV[k] = sumv
    return m


def _sequential(mask, first, values):
    """first + the masked values, added one at a time in order, per row."""
    table = np.where(mask, values[None, :], 0.0)
    return np.cumsum(np.concatenate((first[:, None], table), axis=1), axis=1)[:, -1]


def _ratios(x, n, cdsign, nn, sig, minratio, nv):
    """aid_ratios: the ratios of the two spacings of every triplet of
    neighbours, in single precision; with nv=2 the interval the target's
    centring error allows."""
    minr = f32(minratio)
    maxr = f32(1 / f32(minratio))
    sig = f32(sig)
    out = []
    if cdsign == 1:
        for i in range(1, n - 1):
            xi = f32(x[i - 1])
            for j in range(i + 1, min(i + nn - 1, n - 1) + 1):
                xj = f32(x[j - 1])
                xij = f32(xj - xi)
                for k in range(j + 1, min(i + nn, n) + 1):
                    xk = f32(x[k - 1])
                    xjk = f32(xk - xj)
                    out.append(_ratio(xij, xjk, minr, maxr, sig, nv))
    else:
        for i in range(n, 2, -1):
            xi = f32(x[i - 1])
            for j in range(i - 1, max(i - nn + 1, 2) - 1, -1):
                xj = f32(x[j - 1])
                xij = f32(xi - xj)
                for k in range(j - 1, max(i - nn, 1) - 1, -1):
                    xk = f32(x[k - 1])
                    xjk = f32(xj - xk)
                    out.append(_ratio(xij, xjk, minr, maxr, sig, nv))
    return out


def _ratio(xij, xjk, minr, maxr, sig, nv):
    ratio = f32(xij / xjk)
    if ratio < minr or ratio > maxr:
        return (f32(1000.0), f32(1000.0))
    if nv == 1:
        return (ratio, ratio)
    err = f32(sig * f32(math.sqrt(f32(2 * (1 + ratio + ratio ** 2)))) / xjk)
    return (f32(ratio - err), f32(ratio + err))


def _lines(s, n, nn):
    """aid_lines: ratio index (1-based) to its three line indices."""
    k = s
    i = 1
    while True:
        ll = min(nn, n - i)
        ll = ll * (ll - 1) // 2
        if k <= ll:
            break
        k -= ll
        i += 1
    j = i + 1
    while True:
        ll = min(nn - 1, n - j)
        if k <= ll:
            break
        k -= ll
        j += 1
    return i, j, k + j


def autoidentify(ident: Identify, crval: float | None, cdelt: float | None,
                 pars: AidPars | None = None) -> AidResult:
    """aid_autoid on a fresh Identify: the features and fit it finds, or
    none, as IRAF leaves them."""
    pars = pars or AidPars()
    aid = _Aid(ident, crval, cdelt, pars)
    user_interp, user_eps = ident.interp, ident.epsilon
    ident.interp, ident.epsilon = "linear", 0.02          # c1d_params for the search
    ident.features, ident.fit, ident.shift = [], None, 0.0
    ident.update_fitdata()
    try:
        aid.target()
        cdflip = aid.cddir == "unknown" or (cdelt is None and aid.cddir == "sign")
        done = False
        evs: list[_Ev] = []
        last: _Ev | None = None
        nbest = 0
        while True:
            for i in (0, 1):
                for j in range(1, aid.nb + 1):
                    if j == 1:
                        nbins = (aid.nb + 2) // 2
                    elif j % 2 == 0:
                        nbins = (aid.nb + 2 - j) // 2
                    else:
                        nbins = (aid.nb + 1 + j) // 2
                    nbins = 2 * nbins - 1
                    for k in range(1, nbins + 1):
                        if k == 1:
                            bin_ = (nbins + 2) // 2
                        elif k % 2 == 0:
                            bin_ = (nbins + 2 - k) // 2
                        else:
                            bin_ = (nbins + 1 + k) // 2
                        if ((nbins - 1) // 2) % 2 == 0:
                            if bin_ % 2 == i:
                                continue
                        elif bin_ % 2 != i:
                            continue
                        ev = _Ev(nbins, bin_)
                        evs.append(ev)
                        last = ev
                        try:
                            aid.reference(ev, False)
                            aid.autoid1(ev)
                        except (RuntimeError, ValueError, ZeroDivisionError):
                            ev.nd = 0
                        evf = None
                        if cdflip:
                            evf = _Ev(nbins, bin_)
                            evs.append(evf)
                            try:
                                aid.reference(evf, True)
                                aid.autoid1(evf)
                            except (RuntimeError, ValueError, ZeroDivisionError):
                                evf.nd = 0
                        for ell in range(1, 6):
                            best = aid.evaluate(ev, ell)
                            if best is not None and best < 1:
                                nbest += 1
                                if nbest >= pars.nbest:
                                    done = True
                                    break
                            if evf is not None:
                                best = aid.evaluate(evf, ell)
                                if best is not None and best < 1:
                                    nbest += 1
                                    if nbest >= pars.nbest:
                                        done = True
                                        break
                        if done:
                            break
                    if done:
                        break
                if done:
                    break
            if not done:
                # IRAF's bound is the last unflipped evaluation's count,
                # and the loop runs over every evaluation of every pass.
                for jj in range(6, (last.nd if last is not None else 0) + 1):
                    for ev in evs:
                        best = aid.evaluate(ev, jj)
                        if best is not None and best < 1:
                            nbest += 1
                            if nbest >= pars.nbest:
                                done = True
                                break
                    if done:
                        break
            if done or aid.best <= 1 or aid.np <= 3:
                break
            aid.np -= 1

        if aid.saved is not None:
            ident.features = [replace(f) for f in aid.saved[0]]
            ident.fit = None
            with contextlib.suppress(ValueError, RuntimeError):
                aid.dofitf()
        else:
            ident.features, ident.fit = [], None
        found = aid.best <= 1
        if not found:
            ident.features, ident.fit, ident.shift = [], None, 0.0
        ident.update_fitdata()
    finally:
        ident.interp, ident.epsilon = user_interp, user_eps
    return AidResult(found, aid.best, aid.rms, aid.fmatch, aid.ftmatch,
                     crval, cdelt)


# ------------------------------------------------------------------- the data

def parse_section(section: str, shape: tuple[int, int]) -> tuple[str, int]:
    """identify's `section`: ('line' | 'column', number). 'middle line' is
    the middle image line -- a spectrum along x."""
    words = section.lower().split()
    if len(words) == 2 and words[0] in ("line", "column"):
        kind, where = words[0], words[1]
    elif len(words) == 2 and words[1].rstrip("s") in ("line", "column"):
        where, kind = words[0], words[1].rstrip("s")
    else:
        raise ValueError(f"cannot read section '{section}'. Write 'middle line', "
                         "'first column', 'line 120', ...")
    ny, nx = shape
    n = ny if kind == "line" else nx
    if where == "middle":
        index = (n + 1) // 2
    elif where == "first":
        index = 1
    elif where == "last":
        index = n
    else:
        index = int(where)
    if not 1 <= index <= n:
        raise ValueError(f"{kind} {index} is outside the image (1 to {n})")
    return kind, index


def arc_vector(data: np.ndarray, section: str = "middle line", nsum: int = 10):
    """The spectrum identify works on: a 1-D image as it is; for a 2-D one
    the sum of nsum lines or columns from index - (nsum - 1) / 2 on, fewer
    at the edges. Returns (vector, description)."""
    data = np.asarray(data, dtype=f32)
    if data.ndim == 1:
        return data, ""
    kind, index = parse_section(section, data.shape)
    n = data.shape[0] if kind == "line" else data.shape[1]
    nsum = max(1, int(nsum))
    # smw_gwattrs: from index - (nsum - 1) / 2 for nsum lines, cut at the
    # edges rather than moved in -- line 1 of a sum of 10 is six lines.
    lo = index - (nsum - 1) // 2
    hi = min(n, lo + nsum - 1)
    lo = max(1, lo)
    rows = data[lo - 1:hi] if kind == "line" else data[:, lo - 1:hi].T
    out = np.zeros(rows.shape[1], dtype=f32)
    for row in rows:
        out = (out + row).astype(f32)
    where = f"[*,{index}]" if kind == "line" else f"[{index},*]"
    return out, where


def spectrum_vector(apertures, section: str = "middle line"):
    """The aperture identify works on in an extracted spectrum (IRAF's
    multispec): the section counts the apertures as an image's lines --
    'middle line' is the middle one, 'line 2' the second, 'first line' the
    first. Returns (vector, " - Ap N", aperture)."""
    lines = len(apertures)
    npix = len(apertures[0].flux)
    words = section.lower().split()
    if not words or words[0].startswith("ap"):
        # harwood's own: 'aperture 3' names the aperture by its number.
        number = int(words[1]) if len(words) > 1 else apertures[(lines + 1) // 2 - 1].number
        found = [a for a in apertures if a.number == number]
        if not found:
            raise ValueError(f"there is no aperture {number}; the apertures are "
                             + ", ".join(str(a.number) for a in apertures))
        aperture = found[0]
    else:
        kind, index = parse_section(section, (lines, npix))
        if kind != "line":
            raise ValueError("an extracted spectrum's apertures are its lines: give "
                             "'middle line', 'line 2' or 'aperture 3'")
        aperture = apertures[index - 1]
    vector = np.asarray(aperture.flux, dtype=f32)
    return vector, f" - Ap {aperture.number}", aperture
