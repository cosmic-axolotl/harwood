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
        used = self.mask & ~self.deleted
        if used.sum() < 2:
            return float("nan")
        return float(np.sqrt(np.mean(self.residuals[used] ** 2)))

    @property
    def nrejected(self) -> int:
        return int((~self.mask & ~self.deleted).sum())

    def summary(self) -> str:
        return (f"{self.spec.function} order {self.spec.order}   "
                f"rms {self.rms:.4g}   "
                f"{int((self.mask & ~self.deleted).sum())} points, "
                f"{self.nrejected} rejected, {int(self.deleted.sum())} deleted")


def parse_sample(sample: str, x: np.ndarray) -> np.ndarray:
    """Which points the sample string includes.

    ``"*"`` is everything; otherwise a comma-separated list of ``first:last``
    ranges in the units of x, inclusive at both ends, as IRAF writes them::

        sample = "1:512"
        sample = "1:100,400:512"
    """
    text = (sample or "*").strip()
    if text in ("*", ""):
        return np.ones(np.shape(x), dtype=bool)

    keep = np.zeros(np.shape(x), dtype=bool)
    for piece in text.split(","):
        piece = piece.strip()
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
    """Bin before fitting, as IRAF's `naverage` does.

    Positive averages groups of `naverage` points; negative takes the median of
    groups of `|naverage|`, which is what you want when the data has spikes --
    a cosmic ray on an overscan column, say.
    """
    n = int(naverage)
    if abs(n) <= 1:
        return x, y
    size = abs(n)
    count = len(x) // size
    if count < 1:
        return x, y
    xs = x[: count * size].reshape(count, size)
    ys = y[: count * size].reshape(count, size)
    reducer = np.median if n < 0 else np.mean
    return xs.mean(axis=1), reducer(ys, axis=1)


def _build(x: np.ndarray, y: np.ndarray, spec: FitSpec):
    """Fit one function to the points given. Returns a callable."""
    order = max(1, int(spec.order))

    if spec.function in ("legendre", "chebyshev"):
        degree = order - 1          # IRAF counts terms; numpy counts degree
        if len(x) <= degree:
            raise ValueError(
                f"{spec.function} order {order} needs more than {degree} points; "
                f"{len(x)} are left. Lower the order or reject less."
            )
        series = {"legendre": np.polynomial.Legendre,
                  "chebyshev": np.polynomial.Chebyshev}[spec.function]
        return series.fit(x, y, deg=degree)

    # Splines: `order` is the number of polynomial pieces, as in IRAF.
    from scipy.interpolate import LSQUnivariateSpline

    degree = 1 if spec.function == "spline1" else 3
    pieces = max(1, order)
    order_index = np.argsort(x)
    xs, ys = x[order_index], y[order_index]
    if pieces == 1:
        knots = np.array([])
    else:
        edges = np.linspace(xs[0], xs[-1], pieces + 1)[1:-1]
        knots = np.unique(edges)
        knots = knots[(knots > xs[0]) & (knots < xs[-1])]
    if len(xs) <= degree + len(knots):
        raise ValueError(
            f"{spec.function} with {pieces} pieces needs more points; "
            f"{len(xs)} are left."
        )
    spline = LSQUnivariateSpline(xs, ys, knots, k=degree, ext="const")
    return spline


def fit(x, y, spec: FitSpec | None = None, deleted=None, **overrides) -> Fit:
    """Fit, reject, refit -- the whole `icfit` cycle in one call.

    `deleted` marks points a human removed by hand; they never come back
    through rejection, which is the difference between "the algorithm dropped
    this" and "I decided this one is wrong".
    """
    spec = replace(spec or FitSpec(), **overrides) if overrides else (spec or FitSpec())
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.shape != y.shape:
        raise ValueError(f"x has {x.size} points and y has {y.size}")

    deleted = (np.zeros(x.shape, dtype=bool) if deleted is None
               else np.asarray(deleted, dtype=bool))
    in_sample = parse_sample(spec.sample, x)
    mask = in_sample & ~deleted & np.isfinite(y)
    if mask.sum() < 2:
        return Fit(spec, x, y, mask, deleted, None,
                   "not enough points to fit; check the sample range")

    model = None
    for iteration in range(max(0, int(spec.niterate)) + 1):
        xs, ys = average(x[mask], y[mask], spec.naverage)
        try:
            model = _build(xs, ys, spec)
        except Exception as error:
            return Fit(spec, x, y, mask, deleted, None, str(error))

        if iteration >= int(spec.niterate):
            break
        residual = y - np.asarray(model(x), dtype=float)
        sigma = np.std(residual[mask], ddof=1) if mask.sum() > 2 else 0.0

        # Two guards that a fitter used by dozens of tasks cannot do without.
        #
        # The first: once the fit is essentially exact, sigma collapses towards
        # floating-point noise and a 3-sigma window narrows to nothing, so the
        # next pass rejects the entire sample. The fit was already good; stop.
        if sigma <= 0 or sigma <= 1e-10 * max(float(np.max(np.abs(y[mask]))), 1.0):
            break

        drop = mask.copy()
        drop &= ((spec.low_reject > 0) & (residual < -spec.low_reject * sigma)) | \
                ((spec.high_reject > 0) & (residual > spec.high_reject * sigma))
        if spec.grow > 0 and drop.any():
            drop = _grow(x, drop, spec.grow)
        drop &= in_sample & ~deleted
        if not drop.any():
            break

        # The second: never reject past the point where the function can still
        # be fitted. IRAF would fail on the next pass; saying so here, with the
        # fit that did work still in hand, is more use than an error.
        if (mask & ~drop).sum() < _minimum_points(spec):
            return Fit(spec, x, y, mask, deleted, model,
                       f"stopped rejecting: another pass would leave fewer than "
                       f"{_minimum_points(spec)} points, which {spec.function} "
                       f"order {spec.order} needs")
        mask &= ~drop

    return Fit(spec, x, y, mask, deleted, model)


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
