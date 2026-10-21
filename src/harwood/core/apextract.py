"""IRAF's apextract (noao.twodspec.apextract), ported: find, resize, trace,
background and extraction.

From iraf-community 2.18: apgetdata.x (the profile across the slit),
peaks.x and apfind.x (finding), apcenter.x with xtools center1d (centring),
aprecenter.x (moving apertures taken from a reference frame),
apylevel.x and apresize.x (aperture limits), aptrace.x (tracing),
apskyeval.x (background), apextract.x (the sum with fractional edge pixels),
approfile.x and apvariance.x (variance weighting and cleaning).

The frame is handled as IRAF handles it: with the aperture axis across and
the dispersion axis along. DISPAXIS=1 means the dispersion runs along the
image lines (x), so the apertures sit along y; DISPAXIS=2 the other way.
Positions are 1-based throughout.

Single precision where IRAF decides with it -- the edge weights, the sums,
the centring. The trace and background fits go through core.icfit, IRAF's
icfit, in double.

Not ported: fit2d (Marsh) profiles, subapertures.

Translated from IRAF, Copyright (c) 1986 Association of Universities for
Research in Astronomy Inc.; the notice it is distributed under is in
LICENSES/IRAF-COPYRIGHT.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

import numpy as np

from . import icfit
from .asi import Interpolant
from .center1d import center1d

f32 = np.float32
MAX_REAL = 3.4028235e38


def nint(value) -> int:
    """SPP's nint: half away from zero."""
    v = float(value)
    return int(np.floor(v + 0.5)) if v >= 0 else -int(np.floor(-v + 0.5))


def frame(data: np.ndarray, dispaxis: int) -> np.ndarray:
    """The frame as [dispersion, aperture axis], single precision."""
    data = np.asarray(data, dtype=f32)
    return data.T.copy() if dispaxis == 1 else data


# ------------------------------------------------------------------ apertures

@dataclass
class Trace:
    """The trace: a fit of the aperture centre's offset along the
    dispersion, evaluated clamped to its range as ap_cveval does."""

    fit: icfit.Fit | None = None
    xmin: float = 1.0
    xmax: float = 1.0

    def __call__(self, position):
        position = np.clip(np.asarray(position, dtype=float), self.xmin, self.xmax)
        if self.fit is None or self.fit.model is None:
            return np.zeros(np.shape(position)) + 0.0
        return self.fit(position)


@dataclass
class Aperture:
    """One aperture: AP_ID, AP_CEN, AP_LOW and AP_HIGH along the aperture
    axis, the line it was defined at, its trace and its background fit."""

    number: int
    centre: float
    low: float = -5.0
    high: float = 5.0
    line: float = 1.0
    trace: Trace = field(default_factory=Trace)
    background: icfit.FitSpec = field(default_factory=lambda: icfit.FitSpec(
        function="chebyshev", order=1, sample="-10:-6,6:10", naverage=-3,
        niterate=0, low_reject=3.0, high_reject=3.0, grow=0.0))

    def centre_at(self, position) -> np.ndarray:
        return self.centre + self.trace(position)

    def background_limits(self, length: int) -> tuple[float, float]:
        """ap_icset: the range the background sample reaches, xmin and xmax."""
        x1, x2 = self.low, self.high
        text = self.background.sample.replace(":", ",")
        for piece in text.split(","):
            piece = piece.strip()
            if piece == "*":
                x1, x2 = min(x1, -length), max(x2, length)
            else:
                try:
                    value = float(piece)
                except ValueError:
                    continue
                x1, x2 = min(x1, value), max(x2, value)
        return x1, x2


def to_record(ap: Aperture) -> dict:
    """An aperture as database/ap<image> keeps it: enough to use it again on
    another frame -- the trace as the points it was fitted to, the fit's
    parameters and the points deleted by hand, so refitting gives back the
    same curve, as an identify solution is kept."""
    bg = ap.background

    def g(value):           # ap_dbwrite's %g of a real: what a reference reads back
        return float(f"{float(f32(value)):.7g}")

    record = {"number": ap.number, "centre": g(ap.centre), "low": g(ap.low),
              "high": g(ap.high),
              "line": float(ap.line),
              "background": {"function": bg.function, "order": bg.order,
                             "sample": bg.sample, "naverage": bg.naverage,
                             "niterate": bg.niterate, "low_reject": bg.low_reject,
                             "high_reject": bg.high_reject, "grow": bg.grow},
              "trace": {"xmin": ap.trace.xmin, "xmax": ap.trace.xmax}}
    fit = ap.trace.fit
    if fit is not None and fit.model is not None:
        spec = fit.spec
        record["trace"].update({
            "function": spec.function, "order": spec.order, "sample": spec.sample,
            "naverage": spec.naverage, "niterate": spec.niterate,
            "low_reject": spec.low_reject, "high_reject": spec.high_reject,
            "grow": spec.grow, "x": [float(v) for v in fit.x],
            "y": [float(v) for v in fit.y],
            "deleted": [int(i) for i in np.flatnonzero(fit.deleted)]})
    return record


def from_record(record: dict, length: int) -> Aperture:
    """to_record backwards; `length` is the frame's along the dispersion,
    the range a trace is fitted over."""
    background = icfit.FitSpec(**record["background"]) if isinstance(
        record.get("background"), dict) else icfit.FitSpec(
        function="chebyshev", order=1, sample=str(record.get("background", "-10:-6,6:10")),
        naverage=-3, niterate=0, low_reject=3.0, high_reject=3.0, grow=0.0)
    saved = record.get("trace") or {}
    trace = Trace(None, float(saved.get("xmin", 1.0)), float(saved.get("xmax", 1.0)))
    if "x" in saved:
        spec = icfit.FitSpec(**{k: saved[k] for k in (
            "function", "order", "sample", "naverage", "niterate", "low_reject",
            "high_reject", "grow")})
        x = np.asarray(saved["x"], dtype=float)
        deleted = np.zeros(x.size, dtype=bool)
        deleted[np.asarray(saved.get("deleted", []), dtype=int)] = True
        trace.fit = icfit.fit(x, np.asarray(saved["y"], dtype=float), spec,
                              deleted=deleted, xrange=(1.0, float(length)))
    return Aperture(int(record["number"]), float(record["centre"]), float(record["low"]),
                    float(record["high"]), float(record.get("line", 1.0)), trace,
                    background)


def recenter(apertures: list[Aperture], profile, line: int, width: float = 5.0,
             radius: float = 10.0, threshold: float = 0.0, aprecenter: str = "",
             npeaks: float | None = None, shift: bool = True) -> tuple[int, float | None]:
    """ap_recenter: move apertures taken from another frame onto this one's
    profile. Each aperture's centre at `line` is centred with center1d; with
    `shift` (IRAF's default) every aperture moves by the median of those
    moves -- of the `npeaks` brightest, among the apertures `aprecenter`
    names -- keeping their spacing; without it, each moves by its own.
    Returns how many moved and, with shift, by how much."""
    from . import ranges

    naps = len(apertures)
    if naps < 1:
        return 0, None
    if npeaks is None:
        npeaks_n = naps
    elif npeaks < 1:
        npeaks_n = int(max(1.0, npeaks * naps))
    else:
        npeaks_n = int(npeaks)
    data = np.asarray(profile, dtype=f32)
    npts = data.size
    if npeaks_n == naps and not shift:
        moved = 0
        for ap in apertures:
            offset = f32(ap.trace(np.array([float(line)]))[0])
            centre = center1d(float(f32(f32(ap.centre) + offset)), data, width,
                              "emission", radius, threshold)
            if centre is not None:
                ap.centre = float(f32(f32(centre) - offset))
                moved += 1
        return moved, None
    wanted = ranges.parse(aprecenter)
    found = []                                      # (-peak, delta, index)
    for i, ap in enumerate(apertures):
        if ap.number not in wanted:
            continue
        offset = f32(ap.trace(np.array([float(line)]))[0])
        centre = f32(f32(ap.centre) + offset)
        c = center1d(float(centre), data, width, "emission", radius, threshold)
        if c is not None:
            k = max(1, min(npts, int(c + 0.5)))
            found.append((-float(data[k - 1]), float(f32(f32(c) - centre)), i))
    if not found or npeaks_n <= 0:
        return 0, None
    if npeaks_n < len(found):
        found = sorted(found, key=lambda item: item[0])[:npeaks_n]      # xt_sort3
    if shift:
        deltas = np.sort(np.array([d for _, d, _ in found], dtype=f32))
        j = deltas.size
        delta = float(f32((deltas[j // 2 - 1] + deltas[j // 2]) / f32(2))) if j % 2 == 0 \
            else float(deltas[j // 2])                                   # asokr
        for ap in apertures:
            ap.centre = float(f32(f32(ap.centre) + f32(delta)))
        return naps, delta
    for _, delta, i in found:
        apertures[i].centre = float(f32(f32(apertures[i].centre) + f32(delta)))
    return len(found), None


# ---------------------------------------------------------------- the profile

def getdata(data: np.ndarray, line: int | None = None, nsum: int = 10):
    """ap_getdata on a [dispersion, aperture] frame: the sum (nsum > 0) or
    median (nsum < 0) of |nsum| lines around `line`. Returns (profile, line)."""
    length = data.shape[0]
    line = length // 2 if line is None else max(1, min(length, int(line)))
    n = max(1, abs(nsum))
    i = max(1, line - n // 2)
    j = min(length, i + n - 1)
    i = max(1, j - n + 1)
    rows = np.asarray(data[i - 1:j], dtype=f32)
    if rows.shape[0] < 3 or nsum > 0:
        out = np.zeros(data.shape[1], dtype=f32)
        for row in rows:
            out = (out + row).astype(f32)
        return out, line
    return np.array([_amed(rows[:, k]) for k in range(rows.shape[1])], dtype=f32), line


def _amed(values):
    """amed: the (n + 1) / 2-th smallest value."""
    v = np.sort(np.asarray(values, dtype=f32))
    return f32(v[(v.size + 1) // 2 - 1])


# -------------------------------------------------------------------- finding

def find_peaks(data, contrast: float, edge: int, nmax: int, separation: float,
               threshold: float) -> list[int]:
    """peaks.x find_peaks with no previous peaks: positions (1-based) of up
    to nmax local maxima, strongest first, none within `separation` of a
    stronger one."""
    d = np.asarray(data, dtype=f32)
    npts = d.size
    x = [i for i in range(edge + 1, npts - edge + 1)
         if d[i - 1] >= threshold and _is_local_max(i, d)]
    y = [d[i - 1] for i in x]
    if x and contrast > 0:
        limit = contrast * max(y)
        keep = [k for k in range(len(x)) if y[k] < limit]
        x, y = [x[k] for k in keep], [y[k] for k in keep]
    rank = sorted(range(len(x)), key=lambda k: -y[k])
    alive = [True] * len(x)
    if separation > 0:
        for a in range(len(rank) - 1):
            if not alive[rank[a]]:
                continue
            for b in range(a + 1, len(rank)):
                if alive[rank[b]] and abs(x[rank[a]] - x[rank[b]]) < separation:
                    alive[rank[b]] = False
    out: list[int] = []
    for k in rank:
        if alive[k] and len(out) < nmax:
            out.append(x[k])
    return out


def _is_local_max(index: int, d) -> bool:
    """is_local_max, its loop bug included: the right-hand scan runs to the
    first different value even past the end, which here is the end."""
    npts = d.size
    value = d[index - 1]
    i = index - 1
    nleft = 0
    while i >= 1:
        if d[i - 1] != value:
            break
        nleft += 1
        i -= 1
    j = index + 1
    nright = 0
    while j <= npts:
        if d[j - 1] != value:
            break
        nright += 1
        j += 1
    if i == 0 and j > npts:
        return False
    if i == 0:
        return not (j <= npts and d[j - 1] > value)
    if j > npts:
        return not d[i - 1] > value
    if d[i - 1] > value or d[j - 1] > value:
        return False
    return (nleft - nright) in (0, 1)


def find(profile, nfind: int, minsep: float = 5.0, width: float = 5.0,
         radius: float = 10.0, threshold: float = 0.0,
         order: str = "increasing") -> list[float]:
    """ap_find: the centres (1-based) of up to nfind spectra, numbered in
    `order`. A negative nfind divides the profile evenly instead."""
    npts = profile.size
    if nfind == 0:
        return []
    if nfind > 0:
        peaks = find_peaks(profile, 0.0, 1, nfind + 2, minsep, -MAX_REAL)
        centres: list[float] = []
        for peak in peaks:
            if len(centres) >= nfind:
                break
            c = center1d(float(peak), profile, width, "emission", radius, threshold)
            if c is not None:
                centres.append(c)
    else:
        step = float(npts) / abs(nfind)
        centres = [(i - 0.5) * step for i in range(1, abs(nfind) + 1)]
    return sorted(centres, reverse=(order == "decreasing"))


# ------------------------------------------------------------------- resizing

def ylevel(profile, centre: float, level: float, peak: bool = True, bkg: bool = True,
           grow: float = 0.0, low: float = -MAX_REAL, high: float = MAX_REAL):
    """ap_ylevel: the aperture limits where the profile falls to `level`
    (of the peak, if `peak`), above a straight background between the
    minima either side (if `bkg`). Returns (low, high), unchanged where the
    level is never crossed."""
    d = np.concatenate(([f32(0)], np.asarray(profile, dtype=f32)))   # 1-based
    npts = d.size - 1
    centre = f32(centre)  # type: ignore[assignment]
    if centre < 1 or centre >= npts or level is None:
        return low, high
    if bkg:
        i1 = nint(centre)
        i2 = max(1, nint(centre + f32(max(low, -MAX_REAL))))
        k1 = i1
        while k1 > i2 and d[k1] <= d[k1 - 1]:
            k1 -= 1
        while k1 > i2 and d[k1] >= d[k1 - 1]:
            k1 -= 1
        i2 = min(npts, nint(centre + f32(min(high, MAX_REAL))))
        k2 = i1
        while k2 < i2 and d[k2] <= d[k2 + 1]:
            k2 += 1
        while k2 < i2 and d[k2] >= d[k2 + 1]:
            k2 += 1
        a = d[k1]
        b = f32((d[k2] - d[k1]) / f32(k2 - k1)) if k2 != k1 else f32(0)
    else:
        k1 = int(centre)
        a = b = f32(0)

    def level_at(k):
        return f32(d[k] - a - b * f32(k - k1))

    i1 = int(centre)
    i2 = i1 + 1
    y1, y2 = level_at(i1), level_at(i2)
    y = f32(y1 * (f32(i2) - centre) + y2 * (centre - f32(i1)))
    ycut = f32(level * y) if peak else f32(level)
    above = y > ycut
    j1 = i1
    while j1 >= 1:
        y1 = level_at(j1)
        if (y1 <= ycut) if above else (y1 >= ycut):
            break
        j1 -= 1
    if j1 >= 1:
        j2 = j1 + 1
        y2 = level_at(j2)
        x = f32((ycut + y2 * f32(j1) - y1 * f32(j2)) / (y2 - y1) - centre)
        low = max(low, float((1.0 + grow) * x))
    j2 = i2
    while j2 <= npts:
        y2 = level_at(j2)
        if (y2 <= ycut) if above else (y2 >= ycut):
            break
        j2 += 1
    if j2 <= npts:
        j1 = j2 - 1
        y1 = level_at(j1)
        x = f32((ycut + y2 * f32(j1) - y1 * f32(j2)) / (y2 - y1) - centre)
        high = min(high, float((1.0 + grow) * x))
    return low, high


def resize(apertures: list[Aperture], profile, line: int, llimit=None, ulimit=None,
           level: float | None = 0.1, peak: bool = True, bkg: bool = True,
           grow: float = 0.0, avglimits: bool = False) -> None:
    """ap_resize, in place."""
    npts = profile.size
    llimit = -npts if llimit is None else llimit
    ulimit = npts if ulimit is None else ulimit
    llimit, ulimit = min(llimit, ulimit), max(llimit, ulimit)
    if level is None:
        for ap in apertures:
            ap.low, ap.high = llimit, ulimit
        return
    for ap in apertures:
        centre = ap.centre + float(ap.trace(line))
        low, high = ylevel(profile, centre, level, peak, bkg, grow, llimit, ulimit)
        ap.low, ap.high = min(low, high), max(low, high)
    if avglimits and apertures:
        low = sum(ap.low for ap in apertures) / len(apertures)
        high = sum(ap.high for ap in apertures) / len(apertures)
        for ap in apertures:
            ap.low, ap.high = low, high


# -------------------------------------------------------------------- tracing

@dataclass
class Traced:
    positions: np.ndarray       # along the dispersion
    centres: np.ndarray         # across, absolute
    lost: list[int]             # dispersion positions where the trace was lost


def trace(data: np.ndarray, dispaxis: int, apertures: list[Aperture], line: int,
          step: int = 10, nsum: int = 10, nlost: int = 3, width: float = 5.0,
          radius: float = 10.0, threshold: float = 0.0,
          spec: icfit.FitSpec | None = None) -> list[Traced]:
    """ap_trace: follow each aperture with center1d from `line` outwards
    and fit its offset from the aperture centre (ic_fit, function etc. in
    `spec`). Sets each aperture's trace; returns the traced points."""
    spec = spec or icfit.FitSpec(function="legendre", order=2, sample="*", naverage=1,
                                 niterate=0, low_reject=3.0, high_reject=3.0, grow=0.0)
    nsum = max(1, abs(nsum))
    if dispaxis == 2:
        points = _ltrace(np.asarray(data, dtype=f32), apertures, line, step, nsum,
                         nlost, width, radius, threshold)
        length = data.shape[0]
    else:
        points = _ctrace(np.asarray(data, dtype=f32), apertures, line, step, nsum,
                         nlost, width, radius, threshold)
        length = data.shape[1]
    out = []
    for ap, (positions, centres, lost) in zip(apertures, points, strict=True):
        out.append(Traced(positions, centres, lost))
        if positions.size < 2:
            continue
        offsets = centres - ap.centre
        fit = icfit.fit(positions, offsets, spec, xrange=(1.0, float(length)))
        ap.trace = Trace(fit, *icfit.cv_range(positions, spec, (1.0, float(length))))
    return out


def _ltrace(data, apertures, start, step, nsum, nlost, width, radius, threshold):
    """ap_ltrace: dispersion along the lines' perpendicular (DISPAXIS=2),
    each step the sum of nsum image lines."""
    ny, nx = data.shape
    istart = (start - 1) // step + 1
    ntrace = istart + (ny - start) // step
    lines = [start + (i - istart) * step for i in range(1, ntrace + 1)]
    found = [[None] * ntrace for _ in apertures]
    lostat: list[list[int]] = [[] for _ in apertures]

    def summed(line):
        line1 = max(1, line - nsum // 2)
        line2 = min(ny, line1 + nsum - 1)
        line1 = max(1, line2 - nsum + 1)
        out = np.zeros(nx, dtype=f32)
        for row in data[line1 - 1:line2]:
            out = (out + row).astype(f32)
        return out

    for direction in (+1, -1):
        xc = [ap.centre + float(ap.trace(start)) for ap in apertures]
        lost = [0] * len(apertures)
        indices = range(istart, ntrace + 1) if direction > 0 else range(istart - 1, 0, -1)
        for i in indices:
            vector = summed(lines[i - 1])
            for j in range(len(apertures)):
                if lost[j] >= nlost:
                    continue
                c = center1d(xc[j], vector, width, "emission", radius, threshold)
                if c is None:
                    lost[j] += 1
                    lostat[j].append(lines[i - 1])
                else:
                    lost[j] = 0
                    xc[j] = c
                    found[j][i - 1] = c
    out = []
    for j in range(len(apertures)):
        keep = [i for i in range(ntrace) if found[j][i] is not None]
        out.append((np.array([lines[i] for i in keep], dtype=float),
                    np.array([found[j][i] for i in keep], dtype=float), lostat[j]))
    return out


def _ctrace(data, apertures, start, step, nsum, nlost, width, radius, threshold):
    """ap_ctrace: dispersion along the lines (DISPAXIS=1), each step the sum
    of nsum columns over a scrolling buffer of 5 x width lines."""
    ny, nx = data.shape
    nlines = int(5 * width)
    istart = (start - 1) // step + 1
    ntrace = istart + (nx - start) // step
    cols = [start + (i - istart) * step for i in range(1, ntrace + 1)]
    out = []
    line1 = line2 = 0
    for ap in apertures:
        found = [None] * ntrace
        lostat = []
        for direction in (+1, -1):
            lost = 0
            yc = f32(ap.centre + float(ap.trace(start)))
            indices = range(istart, ntrace + 1) if direction > 0 else range(istart - 1, 0, -1)
            for i in indices:
                if lost >= nlost:
                    continue
                if (yc - line1) < width or (line2 - yc) < width:
                    line1 = max(1, int(yc + f32(0.5) - f32(nlines // 2)))
                    line2 = min(ny, line1 + nlines - 1)
                    line1 = max(1, line2 - nlines + 1)
                col = cols[i - 1]
                col1 = max(1, col - nsum // 2)
                col2 = min(nx, col1 + nsum - 1)
                col1 = max(1, col2 - nsum + 1)
                vector = np.zeros(line2 - line1 + 1, dtype=f32)
                for column in data[line1 - 1:line2, col1 - 1:col2].T:
                    vector = (vector + column).astype(f32)
                c = center1d(float(yc - f32(line1) + f32(1)), vector, width, "emission",
                             radius, threshold)
                if c is None:
                    lost += 1
                    lostat.append(col)
                else:
                    lost = 0
                    yc = f32(f32(c) + f32(line1) - f32(1))
                    found[i - 1] = float(yc)
        keep = [i for i in range(ntrace) if found[i] is not None]
        out.append((np.array([cols[i] for i in keep], dtype=float),
                    np.array([found[i] for i in keep], dtype=float), lostat))
    return out


# ----------------------------------------------------------------- extraction

@dataclass
class Extracted:
    spec: np.ndarray            # the spectrum
    raw: np.ndarray | None      # unweighted, uncleaned (variance weighting)
    sky: np.ndarray | None      # the background summed over the aperture
    sigma: np.ndarray | None    # IRAF's sigma (variance weighting)
    log: list[str] = field(default_factory=list)


def extract(data: np.ndarray, apertures: list[Aperture], background: str = "none",
            skybox: int = 1, weights: str = "none", pfit: str = "fit1d",
            clean: bool = False, saturation: float | None = None,
            readnoise: float = 0.0, gain: float = 1.0, lsigma: float = 4.0,
            usigma: float = 4.0, nclean: float = 0.5,
            niterate: int = 5) -> list[Extracted]:
    """ap_extract on a [dispersion, aperture] frame."""
    if clean:
        weights = "variance"
    if weights == "variance" and pfit != "fit1d":
        raise ValueError("pfit=fit2d (Marsh) is not ported; use fit1d")
    nb, namax = data.shape
    interpbuf = 2                                   # II_LINEAR edges
    if weights == "none":
        gain_used, rdnoise = 1.0, 0.0
    else:
        gain_used, rdnoise = float(gain), float(readnoise)

    a1, a2, cmins, cmaxs = [], [], [], []
    for ap in apertures:
        shifts = [float(f32(ap.trace(j))) for j in range(1, nb + 1)]
        cmin, cmax = min(shifts), max(shifts)
        xmin, xmax = ap.low, ap.high
        if weights == "variance":
            xmin, xmax = xmin - 2, xmax + 2
        xmin, xmax = xmin - interpbuf, xmax + interpbuf
        if background != "none":
            bx1, bx2 = ap.background_limits(namax)
            xmin, xmax = min(xmin, bx1), max(xmax, bx2)
        a1.append(min(namax, max(1, nint(f32(ap.centre + xmin + cmin)))))
        a2.append(max(1, min(namax, nint(f32(ap.centre + xmax + cmax)))))
        cmins.append(cmin)
        cmaxs.append(cmax)
    amin, amax = min(a1), max(a2)
    dbuf = np.asarray(data[:, amin - 1:amax], dtype=f32)
    if weights == "variance" and gain_used != 1.0:
        dbuf = (dbuf * f32(gain_used)).astype(f32)
    c1 = amin

    results = []
    for k, ap in enumerate(apertures):
        if weights == "none":
            xmin = float(f32(f32(ap.centre + ap.low) - f32(interpbuf)))
            xmax = float(f32(f32(ap.centre + ap.high) + f32(interpbuf)))
            na1 = nint(xmax) - nint(xmin) + 1
            xs = np.array([nint(f32(xmin + f32(ap.trace(j)))) for j in range(1, nb + 1)])
        else:
            xmin = float(f32(ap.centre + ap.low + cmins[k])) - interpbuf
            xmax = float(f32(ap.centre + ap.high + cmaxs[k])) + interpbuf
            na1 = nint(xmax) - nint(xmin) + 1
            xs = np.full(nb, nint(xmin))
        sbuf = svar = specsky = None
        if background != "none":
            sbuf, svar, specsky = skyeval(ap, dbuf, c1, xs, na1, background, skybox,
                                          rdnoise)
        log: list[str] = []
        if weights == "none":
            spec = _sum(ap, dbuf, c1, xs, sbuf, na1)
            raw = sigma = None
        else:
            profile = _profile(ap, dbuf, c1, xs, sbuf, svar, na1, rdnoise, saturation,
                               clean, lsigma, usigma, niterate, log)
            spec, raw, sigma = _variance(ap, dbuf, c1, xs, sbuf, svar, profile, na1,
                                         rdnoise, saturation, clean, lsigma, usigma,
                                         nclean, log)
            if gain_used != 1.0:
                g = f32(gain_used)
                spec, raw, sigma = (spec / g).astype(f32), (raw / g).astype(f32), \
                    (sigma / g).astype(f32)
                if specsky is not None:
                    specsky = (specsky / g).astype(f32)
        results.append(Extracted(spec, raw, specsky, sigma, log))
    return results


def _edges(ap, nc, c1, xs_line, line):
    """The aperture's edges on one line, relative to the line's start xs."""
    s = f32(f32(ap.trace(line)) - f32(c1) + f32(1))
    low = f32(ap.centre + ap.low)
    high = f32(ap.centre + ap.high)
    x1 = f32(max(f32(0.5), f32(low + s)) + f32(c1) - f32(xs_line))
    x2 = f32(min(f32(nc + 0.49), f32(high + s)) + f32(c1) - f32(xs_line))
    return x1, x2, low, high, s


def _line_interp(dbuf_line, nc, c1, xs_line, x2):
    """ap_asifit: the data from xs - 1 on, padded with the buffer's end
    values where it runs over, and its linear interpolant. Element k of
    `pix` (k >= 0) is image pixel xs - 1 + k."""
    xs_d = xs_line - c1 + 1
    ix2 = nint(f32(min(f32(nc + 0.49), x2))) + 1 - (xs_d - 1)
    n = ix2 + 1
    idx = np.arange(n) + (xs_d - 2)
    pix = dbuf_line[np.clip(idx, 0, nc - 1)].astype(f32)
    return pix, Interpolant("linear", pix)


def edge_weights(asi: Interpolant, x1, x2):
    """ap_edge: the fractions of the end pixels inside (x1, x2), from the
    integral of the interpolant where it is positive."""
    x1, x2 = f32(x1), f32(x2)
    ix1, ix2 = nint(x1), nint(x2)
    if ix1 == ix2:
        wt1, wt2 = f32(x2 - x1), f32(0)
    else:
        wt1, wt2 = f32(f32(ix1) - x1 + f32(0.5)), f32(x2 - f32(ix2) + f32(0.5))
    if asi(float(ix1)) > 0:
        b = f32(asi.integral(ix1 - 0.5, ix1 + 0.5))
        if b > 0:
            a = f32(asi.integral(x1, x2) if ix1 == ix2 else asi.integral(x1, ix1 + 0.5))
            if 0 < a < b:
                wt1 = f32(a / b)
    if ix1 != ix2 and asi(float(ix2)) > 0:
        b = f32(asi.integral(ix2 - 0.5, ix2 + 0.5))
        if b > 0:
            a = f32(asi.integral(ix2 - 0.5, x2))
            if 0 < a < b:
                wt2 = f32(a / b)
    return wt1, wt2


def _sum(ap, dbuf, c1, xs, sbuf, nx):
    """ap_sum, one subaperture."""
    nb, nc = dbuf.shape
    spec = np.zeros(nb, dtype=f32)
    for iy in range(1, nb + 1):
        x1, x2, _, _, _ = _edges(ap, nc, c1, xs[iy - 1], iy)
        if x2 <= x1:
            continue
        pix, asi = _line_interp(dbuf[iy - 1], nc, c1, xs[iy - 1], x2 + f32(xs[iy - 1] - c1))
        ix1, ix2 = nint(x1), nint(x2)
        wt1, wt2 = edge_weights(asi, x1 + 1, x2 + 1)
        sval = f32(wt1 * pix[ix1] + wt2 * pix[ix2])
        for ix in range(ix1 + 1, ix2):
            sval = f32(sval + pix[ix])
        if sbuf is not None:
            sky = sbuf[iy - 1]
            skyval = f32(wt1 * sky[ix1 - 1] + wt2 * sky[ix2 - 1])
            for ix in range(ix1 + 1, ix2):
                skyval = f32(skyval + sky[ix - 1])
            sval = f32(sval - skyval)
        spec[iy - 1] = sval
    return spec


def _samples(text: str, centre: float):
    """ap_skyeval's reading of the sample string: a:b pairs only, relative
    to the aperture centre."""
    out = []
    for a, b in re.findall(r"(-?[\d.]+(?:[eE][-+]?\d+)?)\s*:\s*(-?[\d.]+(?:[eE][-+]?\d+)?)",
                           text):
        a, b = float(a), float(b)
        out.append((f32(centre + min(a, b)), f32(centre + max(a, b))))
        if len(out) == 20:
            break
    return out


def skyeval(ap, dbuf, c1, xs, nx, background, skybox, rdnoise):
    """ap_skyeval: the background under every pixel of the aperture
    (sbuf[line, pixel from xs]), its variance per line, and the background
    summed over the aperture."""
    nb, nc = dbuf.shape
    centre = f32(ap.centre)
    bx1, bx2 = ap.background_limits(10 ** 9)
    xmin = f32(centre + f32(min(ap.low, bx1)))
    xmax = f32(centre + f32(max(ap.high, bx2)))
    samples = _samples(ap.background.sample, ap.centre) or [(xmin, xmax)]
    sbuf = np.zeros((nb, nx), dtype=f32)
    svar = np.zeros(nb, dtype=f32)
    rd = f32(rdnoise)
    for iy in range(1, nb + 1):
        s = f32(ap.trace(iy))
        ix1 = max(c1, nint(f32(xmin + s)) - 1)
        ix2 = min(c1 + nc - 1, nint(f32(xmax + s)) + 1)
        nfit = ix2 - ix1 + 1
        if nfit < 3:
            continue
        data = dbuf[iy - 1, ix1 - c1:ix2 - c1 + 1]
        asi = Interpolant("linear", data) if background in ("average", "fit") else None
        avg = f32(0)
        nsky = 0
        ranges = [(f32(max(f32(ix1), f32(a + s)) - f32(ix1) + f32(1)),
                   f32(min(f32(ix2), f32(b + s)) - f32(ix1) + f32(1))) for a, b in samples]
        if background == "average":
            c = f32(0)
            for a, b in ranges:
                if b - a > 0:
                    avg = f32(avg + f32(asi.integral(a, b)))
                    c = f32(c + b - a)
                    nsky += nint(b) - nint(a) + 1
            if c > 0:
                avg = f32(avg / c)
            sbuf[iy - 1] = avg
        elif background in ("median", "minimum"):
            values = [data[j - 1] for a, b in ranges for j in range(nint(a), nint(b) + 1)]
            nsky = len(values)
            if background == "median":
                avg = _amed(values) if values else f32(0)
            else:
                avg = f32(min(values)) if values else f32(0)
            sbuf[iy - 1] = avg
        elif background == "fit":
            c = f32(centre + s)
            a = f32(f32(ix1) + c - f32(int(c)))
            xv, yv = [], []
            for _ in range(nfit - 1):
                xv.append(nint(f32(1000.0) * f32(a - c)) / 1000.0)
                yv.append(asi(float(f32(a - f32(ix1) + f32(1)))))
                a = f32(a + f32(1))
            fit = None
            try:
                fit = icfit.fit(np.array(xv), np.array(yv), ap.background,
                                xrange=ap.background_limits(10 ** 9))
            except ValueError:
                fit = None
            if fit is not None and fit.model is not None:
                lo, hi = icfit.cv_range(np.array(xv), ap.background,
                                        ap.background_limits(10 ** 9))
                positions = np.clip(xs[iy - 1] + np.arange(nx) - float(c), lo, hi)
                values = np.asarray(fit(positions), dtype=f32)
                sbuf[iy - 1] = values
                avg = f32(values.sum(dtype=np.float32) / f32(nx))
            for a, b in ranges:
                nsky += nint(b) - nint(a) + 1
        else:
            raise ValueError(f"background '{background}' is not one of none, "
                             "average, median, minimum, fit")
        if nsky > 1:
            svar[iy - 1] = max(f32(0), f32((rd + avg) / f32(nsky - 1)))

    if skybox > 1:
        _boxcar(sbuf, svar, skybox, background == "fit")

    specsky = np.zeros(nb, dtype=f32)
    for iy in range(1, nb + 1):
        x1, x2, _, _, _ = _edges(ap, nc, c1, xs[iy - 1], iy)
        if x1 >= x2:
            continue
        ix1, ix2 = nint(x1), nint(x2)
        row = sbuf[iy - 1]
        if ix1 == ix2:
            value = f32((x2 - x1) * row[ix1 - 1])
        else:
            value = f32((f32(ix1) - x1 + f32(0.5)) * row[ix1 - 1])
            value = f32(value + (x2 - f32(ix2) + f32(0.5)) * row[ix2 - 1])
        for ix in range(ix1 + 1, ix2):
            value = f32(value + row[ix - 1])
        specsky[iy - 1] = value
    return sbuf, svar, specsky


def _boxcar(sbuf, svar, skybox, fit):
    """ap_skyeval's running mean of skybox lines, ends held, as IRAF does
    it: the variance divided by skybox squared."""
    ny = sbuf.shape[0]

    def run(values, divisor):
        out = values.copy()
        avg = f32(0)
        for i in range(skybox):
            avg = f32(avg + values[i])
        iy = 0
        for i in range(skybox, ny):
            out[iy] = f32(avg / divisor)
            avg = f32(avg + values[i] - values[iy])
            iy += 1
        out[iy] = f32(avg / divisor)
        # shift so each mean sits at the middle of its box, ends held
        result = out.copy()
        i = ny - skybox            # 0-based index of IRAF's ny - skybox + 1
        k = ny - 1
        while k > ny - 1 - skybox // 2:
            result[k] = out[i]
            k -= 1
        while i > 0:
            result[k] = out[i]
            k -= 1
            i -= 1
        while k > 0:
            result[k] = out[0]
            k -= 1
        return result

    svar[:] = run(svar, f32(skybox ** 2))
    if fit:
        for column in range(sbuf.shape[1]):
            sbuf[:, column] = run(sbuf[:, column], f32(skybox))
    else:
        # IRAF smooths the first column and copies it across every line but
        # the first, whose other columns keep their own sky.
        level = run(sbuf[:, 0].copy(), f32(skybox))
        sbuf[1:] = level[1:, None]
        sbuf[0, 0] = level[0]


# ------------------------------------------------------- variance weighting

def _profile(ap, dbuf, c1, xs, sbuf, svar, nx, rdnoise, saturation, clean, lsigma,
             usigma, niterate, log):
    """ap_profile with ap_horne: the normalised profile, profile[line, ix]."""
    nb, nc = dbuf.shape
    rd2 = f32(rdnoise) ** 2
    sat = None if saturation is None else f32(saturation)
    vmin = f32(1) if rd2 == 0 else rd2
    iters = niterate if clean else 0
    profile = np.zeros((nb, nx), dtype=f32)
    spec = np.zeros(nb, dtype=f32)
    x1s = np.zeros(nb, dtype=f32)
    x2s = np.zeros(nb, dtype=f32)
    reject = np.zeros((nb, nx + 2), dtype=bool)      # True = usable, 1-based ix
    pixes = []
    nrej = 0
    for iy in range(1, nb + 1):
        x1, x2, _, _, _ = _edges(ap, nc, c1, xs[iy - 1], iy)
        x1s[iy - 1], x2s[iy - 1] = x1, x2
        ix1, ix2 = nint(x1), nint(x2)
        reject[iy - 1, max(1, ix1):min(nx, ix2) + 1] = True
        if x1 >= x2:
            pixes.append(None)
            continue
        pix, asi = _line_interp(dbuf[iy - 1], nc, c1, xs[iy - 1], x2 + f32(xs[iy - 1] - c1))
        pixes.append((pix, asi))
        sky = sbuf[iy - 1] if sbuf is not None else np.zeros(nx, dtype=f32)
        wt1, wt2 = edge_weights(asi, x1 + 1, x2 + 1)
        s = f32(0)
        for ix in range(ix1, ix2 + 1):
            if sat is not None and pix[ix] > sat:
                s = f32(0)
                nrej += 1
                break
            d = f32(pix[ix] - sky[ix - 1])
            if ix1 == ix2 or ix == ix1:
                d = f32(wt1 * d)
            elif ix == ix2:
                d = f32(wt2 * d)
            s = f32(s + d)
        if s > 0:
            for ix in range(ix1, ix2 + 1):
                profile[iy - 1, ix - 1] = max(f32(0), f32((pix[ix] - sky[ix - 1]) / s))
        spec[iy - 1] = s
    if nrej == nb:
        raise ValueError("All profiles contain saturated pixels")
    if nrej:
        log.append(f"EXTRACT: {nrej} profiles with saturated pixels in aperture {ap.number}")

    tfac = f32(2)
    nreject = 0
    for i in range(iters + 1):
        _horne(ap, dbuf, c1, xs, spec, sbuf, svar, reject, profile, nx, rd2, vmin, x1s)
        if i == iters:
            break
        chisq = f32(0)
        nsum = 0
        rrej = f32(0)
        dat = f32(0)
        ixrej = iyrej = 0
        nrej = 0
        for iy in range(1, nb + 1):
            s = spec[iy - 1]
            if s <= 0 or pixes[iy - 1] is None:
                continue
            pix, asi = pixes[iy - 1]
            sky = sbuf[iy - 1] if sbuf is not None else np.zeros(nx, dtype=f32)
            var0 = f32(rd2 + svar[iy - 1]) if sbuf is not None else rd2
            wt1, wt2 = edge_weights(asi, x1s[iy - 1] + 1, x2s[iy - 1] + 1)
            ix1, ix2 = nint(x1s[iy - 1]), nint(x2s[iy - 1])
            for ix in range(ix1, ix2 + 1):
                if not reject[iy - 1, ix]:
                    continue
                nsum += 1
                predict = max(f32(0), f32(s * profile[iy - 1, ix - 1] + sky[ix - 1]))
                var = max(vmin, f32(var0 + predict))
                resid = f32((pix[ix] - predict) / np.sqrt(var, dtype=f32))
                chisq = f32(chisq + resid ** 2)
                p = wt1 if (ix1 == ix2 or ix == ix1) else (wt2 if ix == ix2 else f32(1))
                if resid < -tfac * lsigma or resid > tfac * usigma:
                    spec[iy - 1] = f32(spec[iy - 1] - p * (pix[ix] - predict))
                    nrej += 1
                    reject[iy - 1, ix] = False
                elif abs(resid) > abs(rrej):
                    rrej = resid
                    dat = f32(p * (pix[ix] - predict))
                    ixrej, iyrej = ix, iy
        if nsum == 0:
            raise ValueError("All pixels rejected")
        tfac = f32(np.sqrt(chisq / f32(nsum)))
        if rrej < -tfac * lsigma or rrej > tfac * usigma:
            spec[iyrej - 1] = f32(spec[iyrej - 1] - dat)
            reject[iyrej - 1, ixrej] = False
            nrej += 1
        nreject += nrej
        if nrej == 0:
            break
    if clean:
        log.append(f"EXTRACT: {nreject} pixels rejected for profile from aperture "
                   f"{ap.number}")
    return profile


def _horne(ap, dbuf, c1, xs, spec, sbuf, svar, reject, profile, nx, rd2, vmin, x1s):
    """ap_horne: each column of the profile, divided by the spectrum
    estimate, fitted along the dispersion by a weighted cubic spline of
    2 (tilt + trace order) pieces."""
    nb = dbuf.shape[0]
    p, s = float(x1s.min()), float(x1s.max())
    ncoeff = 1
    fit = ap.trace.fit
    if fit is not None and fit.model is not None:
        ncoeff = _ncoeff(fit)
    order = 2 * (int(f32(s - p + 1)) + max(0, ncoeff - 2))
    lines = np.arange(1, nb + 1, dtype=float)
    spline = icfit.FitSpec(function="spline3", order=order, niterate=0)
    for ix in range(1, nx + 1):
        w = np.zeros(nb, dtype=f32)
        values = profile[:, ix - 1].copy()
        wmin = MAX_REAL
        for iy in range(1, nb + 1):
            sp = spec[iy - 1]
            if sp > 0 and reject[iy - 1, ix]:
                sk = sbuf[iy - 1, ix - 1] if sbuf is not None else f32(0)
                var0 = f32(rd2 + svar[iy - 1]) if sbuf is not None else rd2
                expected = f32(sp * profile[iy - 1, ix - 1] + sk)
                var = max(vmin, f32(var0 + max(f32(0), expected)))
                var = f32(sp ** 2 / var)
                wmin = min(wmin, float(var))
                w[iy - 1] = var
                col = xs[iy - 1] - 1 + ix          # image pixel
                values[iy - 1] = f32((dbuf[iy - 1, col - c1] - sk) / sp)
            else:
                w[iy - 1] = 0
        if wmin == MAX_REAL:
            w[:] = 1
        else:
            w = np.maximum(w, f32(wmin / 10.0))
        model = icfit._build(lines, values.astype(float), spline, w.astype(float),
                             xrange=(1.0, float(nb)))
        profile[:, ix - 1] = np.maximum(np.asarray(model(lines), dtype=f32), f32(0))


def _ncoeff(fit) -> int:
    """CVNCOEFF of the trace fit: terms for a polynomial, pieces + 3 for a
    cubic spline, pieces + 1 for a linear one."""
    function, order = fit.spec.function, int(fit.spec.order)
    if function == "spline3":
        return order + 3
    if function == "spline1":
        return order + 1
    return order


def _variance(ap, dbuf, c1, xs, sbuf, svar, profile, nx, rdnoise, saturation, clean,
              lsigma, usigma, nclean, log):
    """ap_variance: the variance-weighted spectrum, the raw one and sigma."""
    nb, nc = dbuf.shape
    rd2 = f32(rdnoise) ** 2
    sat = None if saturation is None else f32(saturation)
    vmin = f32(1) if rd2 == 0 else rd2
    if clean:
        niter = max(1, int(nclean * nx)) if nclean < 1 else max(1, int(min(nx, nclean)))
    else:
        niter = 0
    spec = np.zeros(nb, dtype=f32)
    raw = np.zeros(nb, dtype=f32)
    sigma = np.full(nb, -1, dtype=f32)
    nreject = 0
    total1 = total2 = f32(0)
    for iy in range(1, nb + 1):
        x1, x2, _, _, _ = _edges(ap, nc, c1, xs[iy - 1], iy)
        if x1 >= x2:
            continue
        ix1, ix2 = nint(x1), nint(x2)
        pix, asi = _line_interp(dbuf[iy - 1], nc, c1, xs[iy - 1], x2 + f32(xs[iy - 1] - c1))
        sky = sbuf[iy - 1] if sbuf is not None else np.zeros(nx, dtype=f32)
        var0 = f32(rd2 + svar[iy - 1]) if sbuf is not None else rd2
        wt1, wt2 = edge_weights(asi, x1 + 1, x2 + 1)
        prof = profile[iy - 1]
        span = range(ix1, ix2 + 1)
        wt = {}
        data1 = {}
        eplot = {}
        fplot = {}
        saturated = False
        total = sw = swv = f32(0)
        for ix in span:
            w = wt1 if (ix1 == ix2 or ix == ix1) else (wt2 if ix == ix2 else f32(1))
            dat = pix[ix]
            if sat is not None and dat > sat:
                saturated = True
            sk = sky[ix - 1]
            raw[iy - 1] = f32(raw[iy - 1] + w * (dat - sk))
            data1[ix] = f32(dat - sk)
            wt[ix] = w
            var = max(vmin, f32(var0 + max(f32(0), dat)))
            w = f32(prof[ix - 1] / var)
            var = f32(np.sqrt(var, dtype=f32))
            eplot[ix] = var
            total = f32(total + w * (dat - sk))
            sw = f32(sw + w * prof[ix - 1])
            swv = f32(swv + (w * var) ** 2)
        if sw > 0:
            spec[iy - 1] = f32(total / sw)
            sigma[iy - 1] = f32(np.sqrt(swv, dtype=f32) / abs(sw))
        else:
            spec[iy - 1], sigma[iy - 1] = f32(0), f32(-1)

        total = sw = swv = sum1 = sum2 = f32(0)
        for ix in span:
            sum1 = f32(sum1 + wt[ix] * data1[ix])
            if eplot[ix] <= 0:
                continue
            sk = sky[ix - 1]
            s = f32(max(f32(0), spec[iy - 1]) * prof[ix - 1])
            var = max(vmin, f32(var0 + f32(s + sk)))
            w = f32(prof[ix - 1] / var)
            var = f32(np.sqrt(var, dtype=f32))
            eplot[ix] = var
            fplot[ix] = s
            total = f32(total + w * data1[ix])
            sw = f32(sw + w * prof[ix - 1])
            swv = f32(swv + (w * var) ** 2)
        if sw > 0:
            spec[iy - 1] = f32(total / sw)
            sigma[iy - 1] = f32(np.sqrt(swv, dtype=f32) / abs(sw))
            sum2 = f32(sum2 + spec[iy - 1])
        else:
            spec[iy - 1], sigma[iy - 1] = f32(0), f32(-1)
            sum1 = sum2 = f32(0)

        nrej = 0
        for _ in range(niter):
            irej, rrej = 0, f32(0)
            for ix in span:
                if eplot[ix] <= 0:
                    continue
                s = f32(max(f32(0), spec[iy - 1]) * prof[ix - 1])
                sk = sky[ix - 1]
                var = f32(np.sqrt(max(vmin, f32(var0 + max(f32(0), f32(s + sk)))), dtype=f32))
                fplot[ix] = s
                eplot[ix] = var
                resid = f32((data1[ix] - fplot[ix]) / var)
                if abs(resid) > abs(rrej):
                    rrej, irej = resid, ix
            if rrej <= -lsigma or rrej >= usigma:
                eplot[irej] = -eplot[irej]
                data1[irej] = fplot[irej]
                nrej += 1
            else:
                break
            total = sw = swv = sum1 = sum2 = f32(0)
            for ix in span:
                sum1 = f32(sum1 + wt[ix] * data1[ix])
                if eplot[ix] <= 0:
                    continue
                w = f32(prof[ix - 1] / eplot[ix] ** 2)
                total = f32(total + w * data1[ix])
                sw = f32(sw + w * prof[ix - 1])
                swv = f32(swv + (w * eplot[ix]) ** 2)
            if sw > 0:
                spec[iy - 1] = f32(total / sw)
                sigma[iy - 1] = f32(np.sqrt(swv, dtype=f32) / abs(sw))
                sum2 = f32(sum2 + spec[iy - 1])
            else:
                spec[iy - 1], sigma[iy - 1] = f32(0), f32(-1)
                sum1 = sum2 = f32(0)
        nreject += nrej
        total1 = f32(total1 + sum1)
        total2 = f32(total2 + sum2)
        if saturated:
            sigma[iy - 1] = -sigma[iy - 1]

    if total1 * total2 <= 0:
        log.append(f"EXTRACT: WARNING - Aperture {ap.number}: total variance weighted "
                   f"flux {float(total2):g}, unweighted {float(total1):g}; bias factor "
                   "ignored")
    else:
        factor = f32(total1 / total2)
        spec = (spec * factor).astype(f32)
        sigma = (sigma * factor).astype(f32)
        log.append(f"EXTRACT: Aperture {ap.number} variance spectrum bias factor is "
                   f"{float(factor):g}")
    if clean:
        log.append(f"EXTRACT: {nreject} pixels rejected from aperture {ap.number}")
    return spec, raw, sigma
