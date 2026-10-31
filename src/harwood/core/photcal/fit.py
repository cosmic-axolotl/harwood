"""fitparams: the transformation equations fitted to the standards.

From photcal/fitparams (fttrneq.x, ftref.x, ftweights.x, ftindef.x,
fteval.x) and xtools/inlfit (infit, inrefit, inreject, indeviant, inerrors,
inrms, and the log's ing_show, ing_errors, ing_results); the keys are in
fitsession.py.

For each equation, in order: the reference side is evaluated for every
observation; the weights are made (uniform, from the photometric errors,
or from the equation's weight expression); an observation whose reference
or fit side is undefined gets no weight, and its undefined values are
replaced by the last defined one in the column, so the fit never sees an
INDEF; then nlfit, and the rejection iterations, each a new fit from the
initial values. A converged fit's values replace the equation's, so an
equation that refers to it later sees them.

The quirks kept: a rejected point's neighbours within `grow` are found by
comparing x[i] with x[j] of the flattened variables array, not the points'
first variables; the sigma for rejection is the plain rms of the residuals
of the points with weight, and needs five of them; the rejected points'
weights are not put back in the weights the errors are computed from,
only zeroed for the errors themselves.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .. import iraffmt
from ..nlfit import DONE, NO_DEG_FREEDOM, NLFit, square
from .config import Config, Equation, SetEquation, evaluate

f32 = np.float32
MAX_REAL = f32(3.4028235e38)


@dataclass
class FitResult:
    """One equation's fit, what io_pcoeffs writes and the log shows."""
    eq: Equation
    stat: int
    params: np.ndarray
    errors: np.ndarray
    plist: list[int]
    variance: np.float32
    chisqr: np.float32
    scatter: np.float32
    rms: np.float32
    niter: int
    rejected: np.ndarray            # bool, per observation
    weights: np.ndarray             # the weights the errors were computed with
    y: np.ndarray                   # the reference side
    fit: np.ndarray                 # the fit side at the fitted parameters
    deleted: int
    settings: dict = field(default_factory=dict)

    @property
    def nrejected(self) -> int:
        return int(self.rejected.sum())


# ------------------------------------------------------------- weights


def weights_photometric(cfg: Config, eq: Equation, rows: np.ndarray) -> np.ndarray:
    """ft_wtphoterrs: 1 / the sum of the squared errors of the variables in
    the equation, each counted as often as it appears, those inside set
    equations (two levels deep) included."""
    nrows = len(rows)
    wts = np.ones(nrows, dtype=f32)
    terms: list[list] = []          # [error index, count], reference then fit

    def own(variables) -> list[list]:
        return [[cfg.error_index(var), count] for var, count in variables]

    def from_sets(code, found: list[list]) -> int:
        new = 0
        for op, arg in code:
            if op == "set":
                new += _set_errors(cfg, arg, found, 2)
        return new

    reference = own(eq.refvars)
    from_sets(eq.ref.code, reference)
    fitting = own(eq.fitvars)
    from_sets(eq.fit.code, fitting)
    terms = reference + fitting
    if not any(index is not None for index, _ in terms):
        return wts
    for n in range(nrows):
        errval = 0.0                                        # double
        for index, count in terms:
            if index is None:
                continue
            val = rows[n, index]
            if np.isnan(val) or val < 0:
                continue
            errval += float(f32(f32(count) * f32(val * val)))
        wts[n] = f32(0.0) if errval <= 0 else f32(1.0 / errval)
    return wts


def _set_errors(cfg: Config, seq: SetEquation, found: list[list], levels: int) -> int:
    """ft_scoeval / ft_coeval: the variables with errors in a set equation,
    added to `found` or counted again; a set equation inside it is looked
    into once more."""
    new = 0
    for op, arg in seq.eq.code:
        if op == "set":
            if levels > 1:
                new += _set_errors(cfg, arg, found, levels - 1)
        elif op == "var":
            var = (cfg.obsvars[arg] if arg < cfg.nobs else cfg.catvars[arg - cfg.nobs])
            index = cfg.error_index(var)
            if index is None:
                continue
            for item in found:
                if item[0] == index:
                    item[1] += 1
                    break
            else:
                found.append([index, 1])
                new += 1
    return new


def weights_equations(eq: Equation, rows: np.ndarray) -> np.ndarray:
    """ft_wteqns: the weight equation, at the initial parameter values; an
    undefined or negative weight is 0. `min` caps it and `max` raises it --
    the names are the other way round, and kept."""
    wts = np.ones(len(rows), dtype=f32)
    if not eq.weight:
        return wts
    for n, row in enumerate(rows):
        value = evaluate(eq.weight.code, row, eq.values)
        if np.isnan(value) or value < 0:
            wts[n] = 0
            continue
        if eq.wtsmin:
            low = evaluate(eq.wtsmin.code, row, eq.values)
            if not np.isnan(low) and low >= 0:
                value = min(value, low)
        if eq.wtsmax:
            high = evaluate(eq.wtsmax.code, row, eq.values)
            if not np.isnan(high) and high >= 0:
                value = max(value, high)
        wts[n] = value
    return wts


def _undefined(eq: Equation, rows: np.ndarray, y: np.ndarray, wts: np.ndarray) -> None:
    """ft_indef, in place: no weight where the reference or the fit side is
    undefined, and the undefined values replaced by the last defined one
    above them (the first in the column for the first rows)."""
    def fill(column: np.ndarray) -> None:
        defined = column[~np.isnan(column)]
        replace = defined[0] if len(defined) else f32(0.0)
        for n in range(len(column)):
            if np.isnan(column[n]):
                column[n] = replace
            else:
                replace = column[n]

    wts[np.isnan(y)] = 0
    fill(y)
    replace = np.zeros(rows.shape[1], dtype=f32)
    for i in range(rows.shape[1]):
        defined = rows[:, i][~np.isnan(rows[:, i])]
        if len(defined):
            replace[i] = defined[0]
    for n in range(len(rows)):
        # The rows above are already filled when this one is evaluated.
        if np.isnan(evaluate(eq.fit.code, rows[n], eq.values)):
            wts[n] = 0
        missing = np.isnan(rows[n])
        rows[n, missing] = replace[missing]
        replace = rows[n].copy()


# ------------------------------------------------------------------ fit


def _functions(eq: Equation, deltas: np.ndarray):
    """ft_func and ft_dfunc: the derivatives from the derivative equations
    where there are any, else centred differences of one delta."""
    fcode = eq.fit.code

    def func(x, p):
        return evaluate(fcode, x, p)

    def dfunc(x, p):
        z = evaluate(fcode, x, p)
        der = np.zeros(len(p), dtype=f32)
        q = np.array(p, dtype=f32)
        for i in range(len(p)):
            if eq.derivatives[i]:
                der[i] = evaluate(eq.derivatives[i].code, x, q)
            else:
                pi = q[i]
                q[i] = f32(pi + f32(f32(0.5) * deltas[i]))
                zplus = evaluate(fcode, x, q)
                q[i] = f32(pi - f32(f32(0.5) * deltas[i]))
                zminus = evaluate(fcode, x, q)
                der[i] = f32(f32(zplus - zminus) / deltas[i])
                q[i] = pi
        return z, der

    return func, dfunc


def _deviant(nl: NLFit, x: np.ndarray, y: np.ndarray, w: np.ndarray, rejected: np.ndarray,
             low: float, high: float, grow: float) -> tuple[int, int]:
    """in_deviant: reject the points beyond low and high sigma."""
    nreject = newreject = 0
    if low == 0 and high == 0:
        return 0, 0
    residuals = (y - nl.vector(x)).astype(f32)
    sigma, j = f32(0.0), 0
    for i in range(len(y)):
        if w[i] != 0 and not rejected[i]:
            sigma = f32(sigma + square(residuals[i]))
            j += 1
        elif rejected[i]:
            nreject += 1
    if j < 5:
        return nreject, newreject
    sigma = np.sqrt(f32(sigma / f32(j)))
    low_cut = f32(-f32(low) * sigma) if low > 0 else -MAX_REAL
    high_cut = f32(f32(high) * sigma) if high > 0 else MAX_REAL
    flat = x.reshape(-1)
    npts = len(y)
    grow = f32(grow)
    i = 0
    while i < npts:
        if w[i] == 0 or rejected[i]:
            i += 1
            continue
        if residuals[i] > high_cut or residuals[i] < low_cut:
            # 1-based in IRAF: i - grow and i + grow, truncated.
            i_min = max(1, int(f32(i + 1) - grow))
            i_max = min(npts, int(f32(i + 1) + grow))
            last = i
            for jj in range(i_min - 1, i_max):
                if (abs(f32(flat[i] - flat[jj])) <= grow and w[jj] != 0
                        and not rejected[jj]):
                    rejected[jj] = True
                    newreject += 1
                    last = jj
            i = last
        i += 1
    return nreject, newreject


class Fitter:
    """One equation's points, settings and fit: what an INLFIT descriptor
    holds. in_fit and in_errors are `run` and `result`, apart so that an
    interactive session can do one and look at the other any number of times,
    with `wts` and `plist` changed between them."""

    def __init__(self, cfg: Config, eq: Equation, rows: np.ndarray, weighting: str = "uniform",
                 addscatter: bool = True, tol: float = 3e-5, maxiter: int = 15,
                 nreject: int = 0, low_reject: float = 3.0, high_reject: float = 3.0,
                 grow: float = 0.0) -> None:
        self.eq = eq
        self.x = np.array(rows, dtype=f32)
        self.y = np.array([evaluate(eq.ref.code, row, ()) for row in self.x], dtype=f32)
        if weighting == "photometric":
            self.wts = weights_photometric(cfg, eq, rows)
        elif weighting == "equations":
            self.wts = weights_equations(eq, rows)
        else:
            self.wts = np.ones(len(rows), dtype=f32)
        _undefined(eq, self.x, self.y, self.wts)
        deltas = np.array([p.delta for p in eq.params], dtype=f32)
        self.func, self.dfunc = _functions(eq, deltas)
        self.values = np.array(eq.values, dtype=f32)
        self.plist = list(eq.plist)
        self.scatter = weighting != "uniform" and addscatter
        self.tol, self.maxiter = tol, maxiter
        self.nreject, self.low_reject, self.high_reject = nreject, low_reject, high_reject
        self.grow = grow
        self.rejected = np.zeros(len(self.y), dtype=bool)
        self.nl = self._fresh()
        self.stat = NO_DEG_FREEDOM

    def _fresh(self) -> NLFit:
        nl = NLFit(self.func, self.dfunc, self.values, [k - 1 for k in self.plist], self.tol,
                   self.maxiter)
        nl.w = self.wts.copy()
        nl.scatter = f32(0.0)
        return nl

    def run(self) -> int:
        """in_fit."""
        x, y = self.x, self.y
        npts = len(y)
        nfit = len(self.plist)
        wts1 = self.wts.copy()
        self.rejected = rejected = np.zeros(npts, dtype=bool)
        self.nl = nl = self._fresh()
        self.stat = NO_DEG_FREEDOM
        if not npts or npts - int(np.sum(self.wts <= 0)) < nfit:
            return self.stat
        self.stat = nl.fit(x, y, self.wts, scatter=self.scatter)
        self.wts = nl.w.copy()
        if self.low_reject > 0 or self.high_reject > 0:
            nrej = 0
            if self.nreject >= 1:
                w0 = wts1.copy()
                for _ in range(self.nreject):
                    old, new = _deviant(nl, x, y, wts1, rejected, self.low_reject,
                                        self.high_reject, self.grow)
                    nrej = old + new
                    if new == 0:
                        break
                    # in_refit, from the initial values
                    wts1[:] = w0
                    wts1[rejected] = 0
                    self.nl = nl = self._fresh()
                    if npts - int(np.sum(wts1 <= 0)) < nfit:
                        self.stat = NO_DEG_FREEDOM
                        continue
                    self.stat = nl.fit(x, y, wts1, scatter=self.scatter)
                    wts1[:] = nl.w
            if nrej > 0:
                self.wts = np.where(wts1 > 0, wts1, self.wts).astype(f32)
        return self.stat

    def settings(self) -> dict:
        return dict(low_reject=self.low_reject, high_reject=self.high_reject,
                    nreject=self.nreject, grow=self.grow, tol=self.tol, maxiter=self.maxiter)

    def result(self) -> FitResult:
        """in_errors, on the weights as they now stand."""
        nl = self.nl
        werr = self.wts.copy()
        werr[self.rejected] = 0
        zfit = nl.vector(self.x)
        variance, chisqr, errors = nl.statistics(self.y, zfit, werr)
        return FitResult(self.eq, int(self.stat), nl.param.copy(), errors, list(self.plist),
                         variance, chisqr, f32(getattr(nl, "scatter", 0.0)),
                         _rms(self.y, zfit, werr), nl.iter, self.rejected.copy(), werr, self.y,
                         zfit, int(np.sum(self.wts == 0)), self.settings())


def fit_equation(cfg: Config, eq: Equation, rows: np.ndarray, **settings) -> FitResult:
    """ft_trneq, non-interactive."""
    fitter = Fitter(cfg, eq, rows, **settings)
    fitter.run()
    result = fitter.result()
    if result.stat == DONE:
        eq.values[:] = result.params
    return result


def _rms(y, fit, w) -> np.float32:
    """in_rms: of the points with weight."""
    total, n = f32(0.0), 0
    for i in range(len(y)):
        if w[i] == 0:
            continue
        resid = f32(y[i] - fit[i])
        total = f32(total + f32(resid * resid))
        n += 1
    return np.sqrt(f32(total / f32(n))) if n else f32(0.0)


# ------------------------------------------------------------------ log


def _w(value, width: int, fmt: str = "g", decpl: int = 7) -> str:
    """IRAF's %{width}.{decpl}{fmt}, right-justified; INDEF as INDEF."""
    if value is None or np.isnan(value):
        return "INDEF".rjust(width)
    return iraffmt.printf(float(value), fmt, decpl, width).rjust(width)


def show_text(s: dict) -> str:
    """ing_show."""
    g = iraffmt.g
    return (f"low_reject    {g(s['low_reject'])}\nhigh_reject   {g(s['high_reject'])}\n"
            f"nreject       {s['nreject']}\ngrow          {g(s['grow'])}\n"
            f"tol           {g(float(f32(s['tol'])))}\nmaxiter       {s['maxiter']}\n\n")


def errors_text(result: FitResult) -> str:
    """ing_errors."""
    lines = ["", f"niterations        {result.niter}",
             f"total_points       {len(result.y)}",
             f"rejected           {result.nrejected}",
             f"deleted            {result.deleted}",
             f"standard deviation {_w(np.sqrt(result.variance), 10)}",
             f"reduced chi        {_w(np.sqrt(result.chisqr), 10)}"]
    average = (f32(0.0) if result.chisqr <= 0
               else np.sqrt(f32(max(result.variance, f32(0.0)) / result.chisqr)))
    lines += [f"average error      {_w(average, 10)}",
              f"average scatter    {_w(np.sqrt(result.scatter), 10)}",
              f"RMS                {_w(result.rms, 10)}", "",
              f"{'parameter':<10.10}  {'value':>14.14}  {'error':>14.14}"]
    for k, param in enumerate(result.eq.params):
        kind = "fit" if k + 1 in result.plist else "constant"
        lines.append(f"{param.name:<10.10}  {_w(result.params[k], 14, 'f', 7)}  "
                     f"{_w(result.errors[k], 14, 'f', 7)}    ({kind})")
    return "\n".join(lines) + "\n\n"


def log_fit_text(result: FitResult) -> str:
    """ing_show and ing_errors."""
    return show_text(result.settings) + errors_text(result)


def log_results_text(result: FitResult, names: list[str]) -> str:
    """ing_results: each point's reference value, fit, residual and sigma;
    INDEF for those without weight."""
    lines = ["", f"#{'objectid':>14.14} {'function':>14.14} {'fit':>14.14} "
                 f"{'residuals':>14.14} {'sigma':>14.14}"]
    for i, name in enumerate(names):
        w = result.weights[i]
        if w <= 0:
            values = [None] * 4
        else:
            values = [result.y[i], result.fit[i], f32(result.y[i] - result.fit[i]),
                      np.sqrt(f32(f32(1.0) / w))]
        lines.append(f" {name:>14.14} " + " ".join(_w(v, 14) for v in values))
    return "\n".join(lines) + "\n\n"


# --------------------------------------------------------------- the task


def read_inputs(cfg: Config, observations: list[str], catalogs: list[str], logfile: str = "",
                catdir: str = "", log_unmatched: bool = True):
    """t_fitparams up to the fit: the observations, and what IRAF logs of them.
    Raises ValueError with IRAF's message when there is nothing to fit."""
    from . import io

    logs: list[str] = []
    if logfile:
        logs.append(f"\n#{io.cnvtime().upper()}\n")
    catalog = None
    if cfg.mincol > 1 and cfg.ncat and catalogs:
        catalog = io.read_catalogs(cfg, catalogs, catdir)
        if not catalog:
            raise ValueError("ERROR: No data was read from the catalog files")
    obs = io.read_observations(cfg, observations, catalog)
    if logfile:
        text = "\n#UNMATCHED OBJECTS\n\n"
        if log_unmatched:
            text += "".join(f"File: {path}  Object: {cid} was unmatched\n"
                            for path, cid in obs.unmatched)
        logs.append(text + "\n")
    if not len(obs.rows):
        if catalog:
            raise ValueError("ERROR: No observations could be matched with "
                             "the catalog entries")
        raise ValueError("ERROR: No observations could be read from the "
                         "observations files")
    return obs, logs


def equation_log(result: FitResult, names: list[str], log_fit: bool, log_results: bool) -> str:
    """What ft_trneq logs of one equation."""
    text = ""
    if log_fit:
        text += f"#EQUATION: {result.eq.name}\n\n" + log_fit_text(result)
    if log_results:
        text += f"#RESULTS: {result.eq.name}\n\n" + log_results_text(result, names)
    return text


def fitparams(cfg: Config, observations: list[str], catalogs: list[str], output: str,
              logfile: str = "", catdir: str = "", log_unmatched: bool = True,
              log_fit: bool = False, log_results: bool = False, **fit) -> list[FitResult]:
    """t_fitparams: read, match, fit each equation and append its record to
    `output` (and, with a log file, what IRAF logs). Raises ValueError with
    IRAF's message when there is nothing to fit."""
    from . import io

    obs, logs = read_inputs(cfg, observations, catalogs, logfile, catdir, log_unmatched)
    results = []
    for eq in cfg.equations:
        result = fit_equation(cfg, eq, obs.rows.copy(), **fit)
        io.write_coeffs(output, eq, result, fit.get("weighting", "uniform"))
        if logfile:
            logs.append(equation_log(result, obs.names, log_fit, log_results))
        results.append(result)
    if logfile:
        with open(logfile, "a") as out:
            out.write("".join(logs))
    return results
