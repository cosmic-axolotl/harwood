"""icfit -- interactive curve fitting, the piece dozens of tasks share.

The IRAF inventory lists `xtools.icfit` as a library "reused by dozens of
tasks", and that is the whole point: the overscan fit in `ccdproc`, the
dispersion solution in `identify`, the trace in `aptrace`, the surface in
`fitcoords`, the continuum in `continuum`, the sensitivity in `sensfunc` and
the flat response in `response` are all *the same interaction*. Learn the keys
once -- `f` to refit, `d` to delete a point, `:order 3` -- and they work
everywhere. That transfer is worth more than any single task.

So this is infrastructure, not a task. There is no Qt here: the fit is a
function of arrays, the interactive panel is a view onto it, and a task that
runs in batch gets exactly the same numbers as one driven by hand.

What is faithful to IRAF
------------------------
The parameter names and their meanings: `function`, `order`, `sample`,
`naverage`, `low_reject`, `high_reject`, `niterate`, `grow`. The fitting
functions: `legendre`, `chebyshev`, `spline1`, `spline3`. For the splines,
`order` is the number of polynomial pieces, as in IRAF -- not the polynomial
degree, which is the mistake to make here.

The fitting, the binning and the rejection (ic_fit, ic_reject, rg_wtbin) are
translated from IRAF, Copyright (c) 1986 Association of Universities for
Research in Astronomy Inc.; the notice it is distributed under is in
LICENSES/IRAF-COPYRIGHT.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field, replace
from typing import Literal

import numpy as np

Function = Literal["legendre", "chebyshev", "spline1", "spline3"]

FUNCTIONS = ("legendre", "chebyshev", "spline1", "spline3")


@dataclass
class FitSpec:
    """How to fit. The parameter set every fitting task exposes."""

    function: Function = "legendre"
    order: int = 1
    sample: str = "*"
    naverage: int = 1
    low_reject: float = 3.0
    high_reject: float = 3.0
    niterate: int = 0
    grow: float = 0.0

    def describe(self) -> str:
        return (f"{self.function} order {self.order}, sample {self.sample}, "
                f"reject {self.low_reject}/{self.high_reject} x{self.niterate}")


@dataclass
class Fit:
    """A fit and everything needed to understand or redraw it."""

    spec: FitSpec
    x: np.ndarray
    y: np.ndarray
    mask: np.ndarray                  #: True where the point was used
    deleted: np.ndarray               #: True where a human deleted it
    model: object = field(repr=False, default=None)
    message: str = ""
    sample: np.ndarray | None = field(repr=False, default=None)    #: in the sample range
    #: What was fitted, the sample's points averaged: x, y, weight, and what was rejected.
    binned: tuple | None = field(repr=False, default=None)

    def __call__(self, x: np.ndarray) -> np.ndarray:
        if self.model is None:
            return np.full(np.shape(x), np.nan)
        return np.asarray(self.model(np.asarray(x, dtype=float)), dtype=float)

    @property
    def values(self) -> np.ndarray:
        """The fit evaluated at the input points."""
        return self(self.x)

    @property
    def residuals(self) -> np.ndarray:
        return self.y - self.values

    @property
    def rms(self) -> float:
        """ic_rms, over what was fitted: averages, where the points are averaged, so
        averaging lowers it by what it took out."""
        if self.binned is not None:
            x, y, w, rejected = self.binned
            kept = (w != 0) & ~rejected
            if kept.sum() < 2 or self.model is None:
                return float("nan")
            return float(np.sqrt(np.mean((y[kept] - self(x[kept])) ** 2)))
        used = self.mask & ~self.deleted
        if used.sum() < 2:
            return float("nan")
        return float(np.sqrt(np.mean(self.residuals[used] ** 2)))

    @property
    def nrejected(self) -> int:
        """Those the rejection took: outside the sample is not rejected."""
        inside = np.ones(self.x.shape, dtype=bool) if self.sample is None else self.sample
        return int((~self.mask & ~self.deleted & inside).sum())

    def summary(self) -> str:
        return (f"{self.spec.function} order {self.spec.order}   "
                f"rms {self.rms:.4g}   "
                f"{int((self.mask & ~self.deleted).sum())} points, "
                f"{self.nrejected} rejected, {int(self.deleted.sum())} deleted")


def parse_sample(sample: str, x: np.ndarray) -> np.ndarray:
    """Which points the sample string includes.

    ``"*"`` is everything; otherwise a list of ``first:last`` ranges in the units
    of x, inclusive at both ends, separated by commas or spaces, as IRAF writes them::

        sample = "1:512"
        sample = "1:100,400:512"
        sample = " 1:100 400:512"
    """
    text = (sample or "*").strip()
    if text in ("*", ""):
        return np.ones(np.shape(x), dtype=bool)

    keep = np.zeros(np.shape(x), dtype=bool)
    for piece in re.split(r"[,\s]+", re.sub(r"\s*:\s*", ":", text)):
        if not piece:
            continue
        match = re.fullmatch(r"(-?[\d.]+)\s*:\s*(-?[\d.]+)", piece)
        if match:
            low, high = float(match.group(1)), float(match.group(2))
            if low > high:
                low, high = high, low
            keep |= (x >= low) & (x <= high)
            continue
        try:                                   # a bare value selects one point
            value = float(piece)
        except ValueError:
            raise ValueError(
                f"cannot read sample '{piece}'. Write ranges as first:last, "
                "separated by commas, or * for everything."
            ) from None
        keep |= np.isclose(x, value)
    if not keep.any():
        raise ValueError(f"sample '{sample}' selects no points; x runs from "
                         f"{np.min(x):g} to {np.max(x):g}")
    return keep


def average(x: np.ndarray, y: np.ndarray, naverage: int) -> tuple[np.ndarray, np.ndarray]:
    """Bin before fitting, as IRAF's `naverage` does (rg_wtbin).

    Positive averages groups of `naverage` points; negative takes the median of
    groups of `|naverage|`, which is what you want when the data has spikes --
    a cosmic ray on an overscan column, say. A short group at the end is kept
    only if it has at least max(min(n, 3), (n + 1) / 2) points, as in IRAF.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    groups = _groups([np.arange(x.size)], naverage)
    xb, yb, _ = _binned(x, y, np.ones(x.size), groups, naverage)
    return xb, yb


def _runs(selected: np.ndarray) -> list[np.ndarray]:
    """The sample as runs of consecutive indices -- IRAF's ranges."""
    index = np.flatnonzero(selected)
    if index.size == 0:
        return []
    breaks = np.flatnonzero(np.diff(index) != 1) + 1
    return np.split(index, breaks)


def _groups(runs: list[np.ndarray], naverage: int) -> list[np.ndarray]:
    """rg_wtbin's grouping: consecutive points within each range."""
    size = max(1, abs(int(naverage)))
    groups = []
    for run in runs:
        start, first = 0, True
        while start < run.size:
            n = min(size, run.size - start)
            if not first and n < max(min(size, 3), (size + 1) // 2):
                break
            groups.append(run[start:start + n])
            start += n
            first = False
    return groups


def _binned(x, y, w, groups, naverage):
    """Each group's x (weighted mean), y (weighted mean, or median for a
    negative naverage) and weight (the sum of its members')."""
    n = len(groups)
    xb, yb, wb = np.empty(n), np.empty(n), np.empty(n)
    for k, members in enumerate(groups):
        weights = w[members]
        total = weights.sum()
        xb[k] = ((x[members] * weights).sum() / total if total != 0
                 else x[members].mean())
        if naverage < 0 and members.size > 1:
            values = np.sort(y[members][weights > 0] if total > 0 else y[members])
            yb[k] = values[(values.size + 1) // 2 - 1]          # amed: lower median
        else:
            yb[k] = ((y[members] * weights).sum() / total if total != 0
                     else y[members].mean())
        wb[k] = total
    return xb, yb, wb


def _build(x: np.ndarray, y: np.ndarray, spec: FitSpec, weights=None,
           xrange: tuple[float, float] | None = None):
    """Fit one function to the points given, least squares weighted as IRAF's
    curfit weights them (by w, not sqrt(w)). Returns a callable.

    The order is clamped as ic_dosetup clamps it: a polynomial never has more
    terms than points, a spline never more pieces than its points allow, and a
    spline with no pieces left falls back to a polynomial through them all.
    """
    weights = np.ones_like(x) if weights is None else np.asarray(weights, dtype=float)
    keep = weights > 0
    x, y, weights = x[keep], y[keep], weights[keep]
    nfit = x.size
    order = max(1, int(spec.order))
    function = spec.function
    if function == "spline1":
        pieces = min(order, nfit - 1)
    elif function == "spline3":
        pieces = min(order, nfit - 3)
    else:
        pieces = 0
    if function in ("spline1", "spline3") and pieces <= 0:
        function, order = "legendre", nfit

    if function in ("legendre", "chebyshev"):
        degree = min(order, nfit) - 1          # IRAF counts terms; numpy counts degree
        if nfit < 1 or degree < 0:
            raise ValueError(
                f"{spec.function} order {order} needs points to fit; none are left. "
                "Lower the order or reject less."
            )
        series = {"legendre": np.polynomial.Legendre,
                  "chebyshev": np.polynomial.Chebyshev}[function]
        return series.fit(x, y, deg=degree, w=np.sqrt(weights))

    # Splines: `order` is the number of polynomial pieces, as in IRAF, laid
    # evenly over the range the caller evaluates -- not over whatever points
    # survived rejection, which would move the knots every iteration.
    try:
        from scipy.interpolate import LSQUnivariateSpline
    except ImportError as error:
        raise ValueError(
            f"'{spec.function}' needs scipy, which is part of harwood[phot]. "
            "Run: pip install harwood[phot] -- or use legendre or chebyshev, "
            "which need nothing extra."
        ) from error

    degree = 1 if function == "spline1" else 3
    sort = np.argsort(x)
    xs, ys, ws = x[sort], y[sort], weights[sort]
    low, high = xrange if xrange is not None else (float(xs[0]), float(xs[-1]))
    low, high = min(low, float(xs[0])), max(high, float(xs[-1]))
    knots = np.linspace(low, high, pieces + 1)[1:-1]
    if len(xs) <= degree + len(knots):
        raise ValueError(
            f"{spec.function} with {pieces} pieces needs more points; "
            f"{len(xs)} are left."
        )
    try:
        return LSQUnivariateSpline(xs, ys, knots, w=np.sqrt(ws), bbox=[low, high],
                                   k=degree, ext="const")
    except ValueError:
        # Some piece has no points in it. curfit solves what it can and says so.
        return _SingularSpline(function, pieces, low, high, xs, ys, ws)


class _SingularSpline:
    """A spline with a piece no point falls in, as cvsolve leaves it: the row of no
    information is dropped, and it is single precision, being curfit's."""

    def __init__(self, function, pieces, low, high, x, y, w) -> None:
        from .curfit import Curve

        self.curve = Curve(function, pieces, low, high)
        for point, value, weight in zip(x, y, w, strict=True):
            one = np.ones(1, dtype=np.float32) * np.float32(weight)
            self.curve.accumulate(np.array([point], dtype=np.float32),
                                  np.array([value], dtype=np.float32), one)
        self.curve.solve()

    def __call__(self, x) -> np.ndarray:
        x = np.atleast_1d(np.asarray(x, dtype=np.float32))
        return np.asarray(self.curve.vector_at(x), dtype=float)


def fit(x, y, spec: FitSpec | None = None, deleted=None,
        xrange: tuple[float, float] | None = None, weights=None, **overrides) -> Fit:
    """Fit, reject, refit -- IRAF's ic_fit, one call.

    In IRAF's order: the sample is cut into ranges, each range into groups of
    `naverage` points (rg_wtbin), and the fit and the rejection both work on
    those groups -- a rejected group takes all its points with it. The
    rejection sigma is the rms of the residuals of the points still in, and
    there is no rejection with fewer than five of them (ic_deviant).

    `deleted` marks points a human removed by hand; they never come back
    through rejection, which is the difference between "the algorithm dropped
    this" and "I decided this one is wrong".

    `xrange` is ic's xmin and xmax, over which spline pieces are laid; left
    out, the range of the data, as most tasks set it. `weights` are the points'
    own, which a weight of zero takes out as `deleted` does.
    """
    spec = replace(spec or FitSpec(), **overrides) if overrides else (spec or FitSpec())
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.shape != y.shape:
        raise ValueError(f"x has {x.size} points and y has {y.size}")

    deleted = (np.zeros(x.shape, dtype=bool) if deleted is None
               else np.asarray(deleted, dtype=bool))
    in_sample = parse_sample(spec.sample, x)
    usable = in_sample & ~deleted & np.isfinite(y)
    if usable.sum() < 2:
        return Fit(spec, x, y, usable, deleted, None,
                   "not enough points to fit; check the sample range", in_sample)

    weights = usable * (1.0 if weights is None else np.asarray(weights, dtype=float))
    groups = _groups(_runs(in_sample), spec.naverage)
    xb, yb, wb = _binned(x, np.where(np.isfinite(y), y, 0.0), weights, groups,
                         spec.naverage)
    if xrange is None:
        xrange = (float(np.min(x)), float(np.max(x)))
    rejected = np.zeros(len(groups), dtype=bool)

    def model_from(active):
        return _build(xb, yb, spec, np.where(active, wb, 0.0), xrange)

    def used_points():
        mask = np.zeros(x.shape, dtype=bool)
        for k, members in enumerate(groups):
            if not rejected[k] and wb[k] > 0:
                mask[members] = True
        return mask & usable

    def finished(mask, model, message):
        if isinstance(model, _SingularSpline) and not message:
            message = "Singular solution"
        return Fit(spec, x, y, mask, deleted, model, message, in_sample,
                   (xb, yb, wb, rejected.copy()))

    try:
        model = model_from(~rejected)
    except Exception as error:
        return finished(usable, None, str(error))

    pixgrow = _pixgrow(xb, spec.grow)
    for _ in range(max(0, int(spec.niterate))):
        if spec.low_reject <= 0 and spec.high_reject <= 0:
            break
        residual = yb - np.asarray(model(xb), dtype=float)
        live = (wb != 0) & ~rejected
        if live.sum() < 5:
            break
        sigma = float(np.sqrt(np.sum(residual[live] ** 2) / live.sum()))
        # Not in IRAF: once the fit is exact to rounding, sigma collapses and
        # a 3-sigma window narrows to nothing, so the next pass would reject
        # the whole sample. The fit was already right; stop.
        if sigma <= 1e-10 * max(float(np.max(np.abs(yb[live]))), 1.0):
            break
        low = -spec.low_reject * sigma if spec.low_reject > 0 else -np.inf
        high = spec.high_reject * sigma if spec.high_reject > 0 else np.inf
        new = np.zeros_like(rejected)
        for i in np.flatnonzero(live & ((residual >= high) | (residual <= low))):
            for j in range(max(0, i - pixgrow), min(len(xb), i + pixgrow + 1)):
                if (abs(xb[i] - xb[j]) <= spec.grow or j == i) and wb[j] != 0 \
                        and not rejected[j]:
                    new[j] = True
        if not new.any():
            break
        if (~(rejected | new) & (wb > 0)).sum() < _minimum_points(spec):
            return finished(used_points(), model,
                            f"stopped rejecting: another pass would leave fewer than "
                            f"{_minimum_points(spec)} points, which {spec.function} "
                            f"order {spec.order} needs")
        rejected |= new
        try:
            model = model_from(~rejected)
        except Exception as error:
            return finished(used_points(), None, str(error))

    return finished(used_points(), model, _clamped(spec, wb, rejected))


def cv_range(x, spec: FitSpec, xrange: tuple[float, float] | None = None
             ) -> tuple[float, float]:
    """The range IRAF's curve is defined over (ic_dosetup), which is where
    ap_cveval clamps it: for a polynomial, the range of the points fitted --
    after the sample and the averaging; for a spline, that and ic's xmin and
    xmax together."""
    x = np.asarray(x, dtype=float)
    groups = _groups(_runs(parse_sample(spec.sample, x)), spec.naverage)
    if len(groups) == x.size or len(groups) == 1:
        lo, hi = float(x.min()), float(x.max())
    else:
        centres = [float(x[members].mean()) for members in groups]
        lo, hi = min(centres), max(centres)
    if spec.function in ("legendre", "chebyshev"):
        return lo, hi
    if xrange is not None:
        lo, hi = min(lo, xrange[0]), max(hi, xrange[1])
    return lo, hi


def _clamped(spec: FitSpec, wb: np.ndarray, rejected: np.ndarray) -> str:
    """IRAF lowers an order the points cannot support, silently. harwood
    lowers it too -- the numbers are IRAF's -- but says so: a curve with as
    many terms as points goes through every one of them, noise included."""
    n = int(((wb > 0) & ~rejected).sum())
    order = max(1, int(spec.order))
    room = {"legendre": n, "chebyshev": n, "spline1": n - 1, "spline3": n - 3}
    if order <= room[spec.function]:
        return ""
    return (f"order {order} asks for more than {n} points can give, so it was "
            f"lowered as IRAF lowers it; the curve now passes through every "
            f"point. Lower the order, or widen the sample.")


def _pixgrow(x: np.ndarray, grow: float) -> int:
    """How many neighbours either side a growing radius can reach (ic_deviant)."""
    if grow <= 0 or x.size < 2:
        return 0
    steps = np.abs(np.diff(x))
    steps = steps[steps >= 0.0001]
    return int(np.max(grow / steps)) if steps.size else 0


def _minimum_points(spec: FitSpec) -> int:
    """How many points the chosen function needs to exist at all."""
    order = max(1, int(spec.order))
    if spec.function in ("legendre", "chebyshev"):
        return order
    degree = 1 if spec.function == "spline1" else 3
    return degree + max(0, order - 1) + 1


def _grow(x: np.ndarray, rejected: np.ndarray, radius: float) -> np.ndarray:
    """Reject the neighbours of a rejected point, as IRAF's `grow` does.

    A cosmic ray rarely lands on one pixel alone, and the pixels beside it are
    contaminated even when they survive the threshold on their own.
    """
    grown = rejected.copy()
    for position in x[rejected]:
        grown |= np.abs(x - position) <= radius
    return grown
