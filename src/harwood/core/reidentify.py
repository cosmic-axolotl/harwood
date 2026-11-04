"""IRAF's reidentify (noao.onedspec), ported.

A solution identify made on one spectrum -- one aperture of an arc, or one
line of a long slit -- is carried to others: each reference feature is put
where the reference solution says its wavelength now falls (plus a shift),
centred on the new spectrum, and the solution refitted from the features
found. That is how one identified aperture becomes every aperture of the
arc (they share the lines, not the dispersion), how a long slit's solution
is traced from line to line, and how the night's other arcs get theirs.

From iraf-community 2.18, noao/onedspec/identify: t_reidentify.x
(ri_reference, ri_image, ri_reidentify, ri_loghdr), iddoshift.x and
idshift.x (id_shift1).

Translated from IRAF, Copyright (c) 1986 Association of Universities for
Research in Astronomy Inc.; the notice it is distributed under is in
LICENSES/IRAF-COPYRIGHT.
"""

from __future__ import annotations

import contextlib
import copy
import math
from dataclasses import dataclass, field, replace

import numpy as np

from . import icfit
from .identify import Feature, Identify

NBIN = 10


class SavedFit:
    """A curve IRAF saved (dcvsave's coefficients) where identify keeps an
    icfit fit: callable on pixels, with the function and order it has."""

    def __init__(self, coefficients, function: str, order: int) -> None:
        from .dispcor import Curve

        self.curve_values = [float(v) for v in coefficients]
        self.curve = Curve.from_saved(coefficients)
        self.spec = icfit.FitSpec(function=function, order=order)
        self.mask = np.ones(max(order, 2), dtype=bool)
        self.model = self

    def __call__(self, x):
        return np.array([self.curve(float(v)) for v in np.atleast_1d(x)])


@dataclass
class Reference:
    """One saved solution: what identify left for an aperture or a line."""

    key: tuple                              # (aperture or line, second index)
    features: list[Feature]
    fit: object
    shift: float
    spec: icfit.FitSpec
    fwidth: float = 4.0
    source: object = None                   # the database record
    newfeatures: bool = True                # ID_NEWFEATURES: read, not yet written
    current: int = 1                        # ID_CURRENT, kept with the rest by id_sid


def reference_from(record, key, fwidth: float = 4.0) -> Reference:
    """id_dbread: a database record as identify keeps it."""
    extra = record.extra
    spec = icfit.FitSpec(function=record.function, order=record.order,
                         sample=str(extra.get("sample", "*")),
                         naverage=int(extra.get("naverage", 1)),
                         niterate=int(extra.get("niterate", 0)),
                         low_reject=float(extra.get("low_reject", 3.0)),
                         high_reject=float(extra.get("high_reject", 3.0)),
                         grow=float(extra.get("grow", 0.0)))
    used = record.used if record.used is not None else np.ones(len(record), bool)
    fitted = record.fitted if record.fitted is not None else record.wavelength
    features = [Feature(float(p), float(f), None if not np.isfinite(w) else float(w),
                        1.0 if u else 0.0)
                for p, f, w, u in zip(record.pixel, fitted, record.wavelength, used,
                                      strict=True)]
    if extra.get("curve"):
        fit = SavedFit(extra["curve"], record.function, record.order)
    else:
        keep = [f for f in features if f.user is not None and f.wts != 0]
        xrange = extra.get("xrange")
        fit = icfit.fit(np.array([f.pix for f in keep]), np.array([f.user for f in keep]),
                        replace(spec, niterate=0),
                        xrange=tuple(xrange) if xrange else None)
    return Reference(key, features, fit, float(extra.get("shift", 0.0)), spec,
                     float(extra.get("fwidth", fwidth)), record)


@dataclass
class Result:
    """What ri_reidentify reports for one spectrum."""

    found: int                              # features found again
    total: int                              # reference features
    features: list[Feature] = field(default_factory=list)
    fit: object = None
    shift: float = 0.0
    pix_shift: float = 0.0
    fit_shift: float = 0.0
    z_shift: float = 0.0
    rms: float = float("nan")
    monotonic: bool = True

    @property
    def nfit(self) -> int:
        return sum(1 for f in self.features if f.wts > 0)


def shift1(ident: Identify, maxfeatures: int) -> float:
    """id_shift1: the mode of the differences between every reference
    feature and every peak of the new spectrum, in user coordinates."""
    from .identify import find_peaks

    f32 = np.float32
    peaks = find_peaks(ident.data, 0.0, int(ident.minsep), 0, maxfeatures, 0.0)
    ys = []
    for x in peaks:
        pix = ident.center(float(x))
        if pix is not None:
            ys.append(ident.fitpt(pix))
    diff = np.array([f32(y - f.fit) for f in ident.features for y in ys], dtype=f32)
    diff.sort()
    n = diff.size
    if n == 0:
        return 0.0
    dmin = diff[n - 1] - diff[0]
    pix = float(diff[0])
    for i in range(0, n - NBIN):
        d = diff[i + NBIN] - diff[i]
        if d < dmin:
            dmin = d
            pix = float(f32(diff[i] + f32(d / f32(2))))
    return pix


def doshift(ident: Identify) -> None:
    """id_doshift: the constant shift that minimises the residuals."""
    total, count = 0.0, 0
    for f in ident.features:
        if f.user is None or f.wts == 0:
            continue
        total += f.user - ident.fitpt(f.pix)
        count += 1
    if count:
        ident.shift += total / count


def _searched(shift: float | None) -> float:
    """id_shift's answer: IRAF carries on with an INDEF shift, which loses every
    feature; here that is said."""
    if shift is None:
        raise RuntimeError("the shift search found no solution")
    return shift


def reidentify(ident: Identify, fit_shift: float | None, search: float | None = 0.0,
               refit: bool = True, addfeatures: bool = False,
               maxfeatures: int = 50) -> Result:
    """ri_reidentify: `ident` holds the reference's features and solution
    on the new spectrum (its fitdata already computed). `fit_shift` None
    is IRAF's INDEF: the shift is then found from the spectrum (search
    INDEF) or taken as zero (search 0)."""
    from .identify import shift_search

    nfeatures1 = len(ident.features)
    if nfeatures1 == 0:
        shift = 0.0
        if search is not None and search != 0:
            shift = _searched(shift_search(ident, search, -0.05))
        elif addfeatures:
            ident.locate()
    elif fit_shift is None:
        if search is None:
            shift = shift1(ident, maxfeatures)
        elif search != 0:
            shift = _searched(shift_search(ident, search, -0.02))
        else:
            shift = 0.0
    else:
        shift = fit_shift
    nfeatures1 = len(ident.features)
    if nfeatures1 == 0:
        raise RuntimeError("No features in reference")
    old = [(f.pix, f.fit) for f in ident.features]
    pixels: list[float | None] = []
    for f in ident.features:
        pix = ident.fit_to_pix(f.fit + shift)
        pix = ident.center(pix)
        if pix is not None:
            f.fit = ident.fitpt(pix)
        pixels.append(pix)
    n = len(ident.features)
    for i in range(n - 1):
        if pixels[i] is None:
            continue
        for j in range(i + 1, n):
            if pixels[j] is None:
                continue
            if abs(pixels[i] - pixels[j]) < ident.minsep:
                fi, fj = ident.features[i], ident.features[j]
                if abs(fi.fit - fi.user) < abs(fj.fit - fj.user):
                    pixels[j] = None
                else:
                    pixels[i] = None
                    break
    kept = []
    pix_shift = fit_shift_out = z_shift = 0.0
    for f, pix, (pix0, fit0) in zip(ident.features, pixels, old, strict=True):
        if pix is None:
            continue
        pix_shift += pix - pix0
        fit_shift_out += f.fit - fit0
        if fit0 != 0:
            z_shift += (f.fit - fit0) / fit0
        f.pix = pix
        kept.append(f)
    ident.features = kept
    found = len(kept)
    pix_shift /= max(1, found)
    fit_shift_out /= max(1, found)
    z_shift /= max(1, found)
    monotonic = True
    if refit and ident.fit is not None and found > 1:
        if addfeatures and abs(pix_shift) > 0.1 * ident.npts:
            doshift(ident)
            ident.newfeatures = True
        else:
            ident.dofit()
    else:
        doshift(ident)
    try:
        ident.update_fitdata()
    except ValueError:
        monotonic = False
    ident.fit_features()
    if addfeatures:
        ident.locate()
        # Not only when locate added something: a reference read from the
        # database still counts as new until it is written, and IRAF refits.
        if ident.newfeatures:
            if refit and ident.fit is not None:
                ident.dofit()
            else:
                doshift(ident)
            try:
                ident.update_fitdata()
            except ValueError:
                monotonic = False
            ident.fit_features()
    return Result(found, nfeatures1, ident.features, ident.fit, ident.shift, pix_shift,
                  fit_shift_out, z_shift, ident.rms(), monotonic)


def log_line(name: str, result: Result) -> str:
    """ri_reidentify's line: '%s%23t%3d/%-3d %3d/%-3d %9.3g  %10.3g  %7.3g  %7.3g'."""
    from .iraffmt import printf

    def g(value, width):
        if value is None or (isinstance(value, float) and math.isnan(value)):
            return "INDEF".rjust(width)
        return printf(float(value), "g", 3, width).rjust(width)

    head = name.ljust(22) if len(name) < 22 else name
    return (f"{head}{result.found:3d}/{result.total:<3d} {result.nfit:3d}/"
            f"{len(result.features):<3d} {g(result.pix_shift, 9)}  "
            f"{g(result.fit_shift, 10)}  {g(result.z_shift, 7)}  {g(result.rms, 7)}")


def header(reference: str, image: str, refit: bool) -> list[str]:
    """ri_loghdr's column labels."""
    return [f"  Reference image = {reference}, New image = {image}, "
            f"Refit = {'yes' if refit else 'no'}",
            f"{'Image Data':>20}  {'Found':>7} {'Fit':>7} {'Pix Shift':>9}  "
            f"{'User Shift':>10}  {'Z Shift':>7}  {'RMS':>7}"]


def start(ref: Reference, data, linelist, params, restored: bool = False) -> Identify:
    """id_gdata/id_fitdata: the reference's features and solution, on the
    new spectrum.

    `restored` is a reference brought back by id_getap rather than the one
    in memory: id_gid then recomputes every feature's fit from the curve
    (id_fitfeatures), at the pixel as the database rounded it. The shift
    reidentify reports is measured from those fits, not the database's.
    """
    ident = Identify(data, linelist, ref.spec, fwidth=ref.fwidth, ftype=params.ftype,
                     cradius=params.cradius, threshold=params.threshold,
                     minsep=params.minsep, match=params.match,
                     maxfeatures=params.maxfeatures)
    ident.features = copy.deepcopy(ref.features)
    ident.fit = ref.fit
    ident.shift = ref.shift
    ident.newfeatures = ref.newfeatures
    ident.current = ref.current
    if restored and ident.fit is not None:
        ident.fit_features()
    with contextlib.suppress(ValueError):
        ident.update_fitdata()
    return ident
