"""invertfit: the fitted transformation turned round, star by star.

From photcal/evaluate (t_invertfit.x, phinvert.x, pherrors.x, phcheck.x,
phprint.x, phminv.f). For each star the catalog variables are found that
make the fit sides of the equations equal their reference sides, by
Bevington's CHIFIT with the rms in place of chi-squared (there may be as
many unknowns as equations), from the catalog's mean values with steps of
its standard deviation, at most 0.1.

The errors are the change in the solution when each observed variable is
moved by its error (obserrors), or when each equation's reference side is
moved by its error equation (equations), added in quadrature.

The quirks kept, because they decide the digits: the copy meant to restore
the catalog variables when a step makes the fit worse copies the first
variables of the row, the observations, so the worse step stays; the rms
the step search started from is changed by it and the comparison after it
uses the changed one; a set equation is evaluated with the parameters of
the equation with its number; and the catalog's mean is summed from the
last star to the first, as IRAF's symbol table is walked.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .. import iraffmt
from ..nlfit import DONE
from . import io
from .config import INDEF, Config, evaluate

f32 = np.float32
EPSILONR = f32(1.192e-7)
MAX_REAL = f32(3.4028235e38)
MAX_NITER1 = 10
MAX_NITER2 = 50
DET_TOL = 1.0e-20
MIN_DELTA = f32(0.01)


class InvertError(ValueError):
    pass


# --------------------------------------------------------------- phminv


def phminv(array: np.ndarray, n: int) -> float:
    """phminv.f (Bevington's MATINV): invert the symmetric matrix in place,
    with full pivoting; the determinant, in single precision."""
    a = array
    det = f32(1.0)
    ik = [0] * n
    jk = [0] * n
    for k in range(n):
        while True:
            amax = 0.0
            ik[k] = jk[k] = k
            for i in range(k, n):
                for j in range(k, n):
                    if abs(amax) - abs(a[i, j]) <= 0:
                        amax = a[i, j]
                        ik[k], jk[k] = i, j
            i = ik[k]
            if i < k:
                continue
            if i > k:
                for j in range(n):
                    save = a[k, j]
                    a[k, j] = a[i, j]
                    a[i, j] = -save
            j = jk[k]
            if j < k:
                continue
            if j > k:
                for i in range(n):
                    save = a[i, k]
                    a[i, k] = a[i, j]
                    a[i, j] = -save
            break
        if amax == 0:
            det = f32(0.0)
            for i in range(n):
                if i != k:
                    a[i, k] = 0.0
            for j in range(n):
                if j != k:
                    a[k, j] = 0.0
            a[k, k] = 0.0
            det = f32(det * amax)
            continue
        for i in range(n):
            if i != k:
                a[i, k] = -a[i, k] / amax
        for i in range(n):
            for j in range(n):
                if i != k and j != k:
                    a[i, j] = a[i, j] + a[i, k] * a[k, j]
        for j in range(n):
            if j != k:
                a[k, j] = a[k, j] / amax
        a[k, k] = 1.0 / amax
        det = f32(det * amax)
    for k in range(n - 1, -1, -1):
        j = ik[k]
        if j > k:
            for i in range(n):
                save = a[i, k]
                a[i, k] = -a[i, j]
                a[i, j] = save
        i = jk[k]
        if i > k:
            for j in range(n):
                save = a[k, j]
                a[k, j] = -a[i, j]
                a[i, j] = save
    return det


# ------------------------------------------------------------ inversion


class _Solver:
    """phinvert.x's routines over one system: the active equations and
    catalog variables of a star."""

    def __init__(self, cfg: Config, params: list[np.ndarray]):
        self.cfg = cfg
        self.params = params
        self.nobs = cfg.nobs
        self.nstd = cfg.ncat

    def _fit(self, i: int, a: np.ndarray) -> np.float32:
        eq = self.cfg.equations[i]
        return evaluate(eq.fit.code, a, self.params[i])

    def _ref(self, i: int, a: np.ndarray) -> np.float32:
        eq = self.cfg.equations[i]
        return evaluate(eq.ref.code, a, self.params[i])

    def _sumsq(self, y, eqindex, a) -> np.float32:
        total = f32(0.0)
        for k, i in enumerate(eqindex):
            total = f32(total + f32(f32(y[k] - self._fit(i, a)) ** 2))
        return f32(total / f32(len(eqindex)))

    def accum(self, y, eqindex, a, deltaa, aindex):
        """ph_accum: the curvature matrix and gradient of the rms by
        differences, inverted; the new increments and the rms, or None."""
        nobs, nstd = self.nobs, self.nstd
        rms1 = f32(0.0)
        for k, i in enumerate(eqindex):
            yfit = self._fit(i, a)
            if np.isnan(yfit):
                return None, None, 0.0
            rms1 = f32(rms1 + f32(f32(y[k] - yfit) ** 2))
        rms1 = f32(rms1 / f32(len(eqindex)))
        active = [j for j in range(nstd) if aindex[j] != 0]
        n = len(active)
        alpha = np.zeros((n, n), dtype=np.float64)
        beta = np.zeros(n, dtype=f32)
        for nj, j in enumerate(active):
            aj = a[j + nobs]
            a[j + nobs] = f32(aj + deltaa[j])
            rms2 = self._sumsq(y, eqindex, a)
            alpha[nj, nj] = float(f32(rms2 - f32(f32(2.0) * rms1)))
            beta[nj] = -rms2
            for nk, k in enumerate(active):
                if nk == nj:
                    continue
                if nk < nj:
                    alpha[nk, nj] = (alpha[nk, nj] - float(rms2)) / 2.0
                    alpha[nj, nk] = alpha[nk, nj]
                    continue
                alpha[nj, nk] = float(f32(rms1 - rms2))
                ak = a[k + nobs]
                a[k + nobs] = f32(ak + deltaa[k])
                rms3 = self._sumsq(y, eqindex, a)
                alpha[nj, nk] = alpha[nj, nk] + float(rms3)
                a[k + nobs] = ak
            a[j + nobs] = f32(aj - deltaa[j])
            rms3 = self._sumsq(y, eqindex, a)
            a[j + nobs] = aj
            alpha[nj, nj] = (alpha[nj, nj] + float(rms3)) / 2.0
            beta[nj] = f32(f32(beta[nj] + rms3) / f32(4.0))
        for j in range(n):
            if alpha[j, j] > 0:
                continue
            alpha[j, j] = -alpha[j, j] if alpha[j, j] < 0 else 0.01
            for k in range(n):
                if k != j:
                    alpha[j, k] = alpha[k, j] = 0.0
        det = phminv(alpha, n)
        da = np.zeros(nstd, dtype=f32)
        for nj, j in enumerate(active):
            value = f32(0.0)
            for nk in range(n):
                value = f32(float(value) + float(beta[nk]) * alpha[nj, nk])
            da[j] = f32(f32(f32(0.2) * value) * deltaa[j])
        return rms1, da, det

    def incrms(self, y, eqindex, a, da, aindex, rms1):
        """ph_incrms: step along the increments until three rms values
        bracket the minimum, then a parabola through them. Returns the rms
        and rms1 as the search left it (IRAF's argument is changed)."""
        nobs = self.nobs
        active = [j for j in range(self.nstd) if aindex[j] != 0]

        def move(sign: float) -> None:
            for j in active:
                a[nobs + j] = f32(a[nobs + j] + f32(sign) * da[j])

        rms = rms1
        move(1.0)
        orms2 = MAX_REAL
        niter = 0
        while True:
            rms2 = self._sumsq(y, eqindex, a)
            if rms2 <= 0 or f32(rms1 - rms2) >= 0:
                break
            if orms2 < MAX_REAL and abs(f32(f32(rms2 - orms2) / orms2)) < EPSILONR:
                return rms, rms1
            orms2 = rms2
            niter += 1
            if niter >= MAX_NITER2:
                return rms, rms1
            move(-1.0)
        orms3 = MAX_REAL
        niter = 0
        while True:
            move(1.0)
            rms3 = self._sumsq(y, eqindex, a)
            if f32(rms3 - rms2) >= 0:
                break
            if orms3 < MAX_REAL and abs(f32(f32(rms3 - orms3) / orms3)) < EPSILONR:
                return rms, rms1
            niter += 1
            if niter >= MAX_NITER2:
                return rms, rms1
            orms3 = rms3
            rms1 = rms2
            rms2 = rms3
        if rms3 <= rms2:
            delta = f32(1.0)
        else:
            ratio = f32(f32(rms1 - rms2) / f32(rms3 - rms2))
            delta = f32(f32(f32(1.0) / f32(f32(1.0) + ratio)) + f32(0.5))
        for j in active:
            a[nobs + j] = f32(a[nobs + j] - f32(delta * da[j]))
        rms = self._sumsq(y, eqindex, a)
        if f32(rms2 - rms) < 0:
            for j in active:
                a[nobs + j] = f32(a[nobs + j] + f32(f32(delta - f32(1.0)) * da[j]))
            rms = rms2
        return rms, rms1

    def invert(self, a, deltaa, aindex, eqindex) -> bool:
        """ph_invert: True when a solution was found (a holds it)."""
        y = [self._ref(i, a) for i in eqindex]
        if any(np.isnan(v) for v in y):
            return False
        delta = np.array(deltaa, dtype=f32)
        niter = 0
        nstd = self.nstd
        while True:
            stdev1, da, det = self.accum(y, eqindex, a, delta, aindex)
            if stdev1 is None:
                return False
            if abs(det) < DET_TOL and niter > 0:
                break
            saved = a[:nstd].copy()               # the observations: IRAF's slip
            stdev2, stdev1 = self.incrms(y, eqindex, a, da, aindex, stdev1)
            if np.isnan(stdev2) or stdev2 >= stdev1:
                if niter == 0:
                    return False
                a[:nstd] = saved
                return True
            delta = -da
            small = np.abs(delta) < MIN_DELTA
            delta[small] = np.where(delta[small] < 0, -MIN_DELTA, MIN_DELTA)
            niter += 1
            rms = np.sqrt(stdev2)
            if niter == MAX_NITER1 or rms <= EPSILONR:
                break
        return True

    def errors_obs(self, a, eindex, deltaa, aindex, eqindex, sets, svals):
        """ph_ierval: each observed variable with an error moved by it."""
        nobs, nstd = self.nobs, self.nstd
        errors = np.zeros(nstd, dtype=f32)
        serrors = np.zeros(len(sets), dtype=f32)
        nerrors = 0
        for j in range(nobs):
            if eindex[j] is None:
                continue
            nerrors += 1
            err = a[eindex[j]]
            if np.isnan(err) or err <= 0:
                continue
            atemp = a[j]
            a[j] = f32(a[j] + err)
            afit = a[nobs:].copy()
            y = [self._ref(i, a) for i in eqindex]
            if any(np.isnan(v) for v in y):
                return 0, errors, serrors
            rms, da, _ = self.accum(y, eqindex, a, deltaa, aindex)
            if rms is not None:
                self.incrms(y, eqindex, a, da, aindex, rms)
                for k in range(nstd):
                    if aindex[k] > 0:
                        errors[k] = f32(errors[k] + f32(f32(a[nobs + k] - afit[k]) ** 2))
                a[j] = atemp
                for k, seq in enumerate(sets):
                    value = evaluate(seq.eq.code, a, self.params[0])
                    serrors[k] = f32(serrors[k] + f32(f32(value - svals[k]) ** 2))
            else:
                a[j] = atemp
            a[nobs:] = afit
        return nerrors, _roots(errors), _roots(serrors)

    def errors_equations(self, a, deltaa, aindex, eqindex, sets, svals):
        """ph_iereqn: each equation's reference side moved by its error
        equation."""
        nobs, nstd = self.nobs, self.nstd
        y, yerr = [], []
        nerrors = 0
        for i in eqindex:
            value = self._ref(i, a)
            if np.isnan(value):
                return 0, None, None
            y.append(value)
            code = self.cfg.equations[i].error
            if not code:
                yerr.append(f32(0.0))
                continue
            err = evaluate(code.code, a, self.params[i])
            if np.isnan(err):
                yerr.append(f32(0.0))
            else:
                yerr.append(err)
                nerrors += 1
        if nerrors <= 0:
            return 0, None, None
        errors = np.zeros(nstd, dtype=f32)
        serrors = np.zeros(len(sets), dtype=f32)
        for k in range(len(eqindex)):
            if yerr[k] <= 0:
                continue
            afit = a[nobs:].copy()
            y[k] = f32(y[k] + yerr[k])
            rms, da, _ = self.accum(y, eqindex, a, deltaa, aindex)
            if rms is not None:
                self.incrms(y, eqindex, a, da, aindex, rms)
                for j in range(nstd):
                    if aindex[j] != 0:
                        errors[j] = f32(errors[j] + f32(f32(a[nobs + j] - afit[j]) ** 2))
                for m, seq in enumerate(sets):
                    value = evaluate(seq.eq.code, a, self.params[0])
                    serrors[m] = f32(serrors[m] + f32(f32(value - svals[m]) ** 2))
            y[k] = f32(y[k] - yerr[k])
            a[nobs:] = afit
        return nerrors, _roots(errors), _roots(serrors)


def _roots(values: np.ndarray) -> np.ndarray:
    return np.where(values <= 0, f32(0.0), np.sqrt(np.maximum(values, 0))).astype(f32)


# ---------------------------------------------------------------- setup


@dataclass
class Plan:
    """What t_invertfit works out before the first star."""
    eqvars: np.ndarray            # [ncat, neq]: the catalog variable's row index + 1
    uservars: np.ndarray          # [ncat]: > 0 used directly, < 0 only in a set equation
    usererrs: list                # [nobs]: the error's row index, or None
    eqset: np.ndarray             # [max(1, nsets), neq]
    sets: list = field(default_factory=list)
    params: list = field(default_factory=list)
    warnings: list = field(default_factory=list)


def plan(cfg: Config, parameters: str) -> Plan:
    neq, ncat, nobs = len(cfg.equations), cfg.ncat, cfg.nobs
    eqvars = np.zeros((ncat, neq), dtype=int)
    uservars = np.zeros(ncat, dtype=int)
    usererrs: list = [None] * nobs
    params, warnings = [], []
    first_at = cfg.column_map("cat")
    first_obs = cfg.column_map("obs")
    for i, eq in enumerate(cfg.equations):
        catalog_in_ref = any(var.kind == "cat" for var, _ in eq.refvars)
        if catalog_in_ref or _sets_with_catalog(cfg, eq.ref.code):
            raise InvertError("Error: Cannot invert equations with catalog "
                              "variables in the function expression")
        for var, _ in eq.refvars + eq.fitvars:
            if var.kind == "cat":
                vcol = first_at[var.column][0] - nobs
                eqvars[vcol, i] = vcol + nobs + 1
                uservars[vcol] = vcol + nobs + 1
            else:
                vcol = first_obs[var.column][0]
                ecol = cfg.error_index(var)
                if ecol is not None:
                    usererrs[vcol] = ecol
        coeffs = io.read_coeffs(parameters, eq.name, len(eq.params)) \
            if Path(parameters).is_file() else None
        if coeffs is None or coeffs.stat is None:
            warnings.append(f"Warning: Error reading parameters for equation "
                            f"{eq.name} from {parameters}")
            params.append(np.full(len(eq.params), INDEF, dtype=f32))
            stat = None
        else:
            params.append(coeffs.values)
            stat = coeffs.stat
        if stat != DONE:
            warnings.append(f"Warning: The solution for equation {eq.name} "
                            "did not converge")
    if int(np.count_nonzero(uservars)) > neq:
        raise InvertError("Error: The number of equations is less than the "
                          "number of unknowns")
    maxnset = max(1, len(cfg.sets))
    eqset = np.zeros((maxnset, neq), dtype=object)
    sets: list = []
    if cfg.sets:
        for i, eq in enumerate(cfg.equations):
            _setvars(cfg, i, eq, eqvars, uservars, usererrs, eqset, sets)
        if int(np.count_nonzero(uservars > 0)) + len(sets) > neq:
            raise InvertError("Error: The number of equations is less than the "
                              "number of unknowns")
    return Plan(eqvars, uservars, usererrs, eqset, sets, params, warnings)


def _sets_with_catalog(cfg: Config, code: list) -> bool:
    """ph_seteqn: a set equation in the code with a catalog variable."""
    if not cfg.sets:
        return False
    for op, arg in code:
        if op == "set" and cfg.ncat and any(o == "var" and x >= cfg.nobs
                                            for o, x in arg.eq.code):
            return True
    return False


def _setvars(cfg, eqn, eq, eqvars, uservars, usererrs, eqset, sets) -> None:
    """ph_setvar / ph_catvar: a set equation in the fit side that brings a
    catalog variable not used directly is solved for too."""
    first_at = cfg.column_map("cat")
    first_obs = cfg.column_map("obs")
    for op, seq in eq.fit.code:
        if op != "set":
            continue
        brought = False
        for o, x in seq.eq.code:
            if o != "var":
                continue
            if x >= cfg.nobs:
                var = cfg.catvars[x - cfg.nobs]
                vcol = first_at[var.column][0] - cfg.nobs
                if uservars[vcol] <= 0:
                    eqvars[vcol, eqn] = -(vcol + cfg.nobs + 1)
                    uservars[vcol] = -(vcol + cfg.nobs + 1)
                    brought = True
            else:
                var = cfg.obsvars[x]
                ecol = cfg.error_index(var)
                if ecol is not None:
                    usererrs[first_obs[var.column][0]] = ecol
        if brought:
            if seq in sets:
                eqset[sets.index(seq), eqn] = seq
            else:
                eqset[len(sets), eqn] = seq
                sets.append(seq)


def catalog_means(cfg: Config, catalog: dict | None,
                  uservars) -> tuple[np.ndarray, np.ndarray]:
    """ph_avstdvars: the catalog's mean and (at most 0.1) standard deviation
    of each variable used, summed from the last star to the first."""
    ncat = cfg.ncat
    avg = np.zeros(ncat, dtype=f32)
    if catalog is None:
        return avg, np.full(ncat, f32(0.1), dtype=f32)
    sigma = np.zeros(ncat, dtype=f32)
    count = np.zeros(ncat, dtype=int)
    for row in reversed(list(catalog.values())):
        for i in range(ncat):
            if uservars[i] == 0 or np.isnan(row[i]):
                continue
            avg[i] = f32(avg[i] + row[i])
            sigma[i] = f32(sigma[i] + f32(row[i] ** 2))
            count[i] += 1
    for i in range(ncat):
        if count[i] <= 1:
            sigma[i] = f32(0.1)
            continue
        avg[i] = f32(avg[i] / f32(count[i]))
        s = f32(f32(sigma[i] - f32(f32(avg[i] * avg[i]) * f32(count[i])))
                / f32(count[i] - 1))
        sigma[i] = f32(0.1) if s <= 0 else min(f32(0.1), np.sqrt(s))
    return avg, sigma


def _check(solver: _Solver, a, plan: Plan):
    """ph_objcheck: the equations defined for this star, the variables they
    need, and whether there are enough of them."""
    neq = len(solver.cfg.equations)
    eqindex = [i for i in range(neq) if not np.isnan(solver._ref(i, a))]
    if not eqindex:
        return None
    vindex = np.zeros(solver.nstd, dtype=int)
    for i in eqindex:
        for j in range(solver.nstd):
            if plan.eqvars[j, i] != 0:
                vindex[j] = plan.eqvars[j, i]
    nset = sum(1 for j in range(plan.eqset.shape[0])
               if any(plan.eqset[j, i] != 0 for i in eqindex))
    ncat = int(np.count_nonzero(vindex > 0))
    if ncat + nset > len(eqindex):
        return None
    return vindex, eqindex


# --------------------------------------------------------------- output


_FIELD = re.compile(r"%(-?)(\d*)(?:\.(\d+))?([a-zA-Z])")


def _format(fmt: str, values: list) -> str:
    """IRAF's printf for the output line: %s and the number formats, INDEF
    written as INDEF in the field's width."""
    out, pos = [], 0
    fmt = fmt.replace("\\n", "\n").replace("\\t", "\t")
    for k, match in enumerate(_FIELD.finditer(fmt)):
        out.append(fmt[pos:match.start()].replace("%%", "%"))
        pos = match.end()
        left, width, decpl, kind = match.groups()
        value = values[k] if k < len(values) else ""
        if kind == "s":
            text = str(value)
        elif value is None or (isinstance(value, float | np.floating)
                               and np.isnan(value)):
            text = "INDEF"
        elif kind in "fgeFGE":
            text = iraffmt.printf(float(value), kind.lower(),
                                  int(decpl) if decpl else None)
        elif kind == "d":
            text = str(int(value))
        else:
            text = str(value)
        if width:
            text = text.ljust(int(width)) if left else text.rjust(int(width))
        out.append(text)
    out.append(fmt[pos:])
    return "".join(out)


def default_format(getid: bool, ncols: int) -> str:
    """ph_oformatstr."""
    parts = ["%-10s "] if getid else []
    first = 2 if getid else 1
    for i in range(first, ncols + 1):
        parts.append("%-7.3f\n" if i == ncols else "%-7.3f ")
    return "".join(parts)


def print_list(cfg: Config, names: list[str]) -> list[tuple]:
    """ph_mkplist: (name, row index, None) for a catalog or observed variable,
    (name, None, the equation) for a set equation; any other name is left out."""
    printed = []
    for name in names:
        sym = cfg.symbols.get(name)
        if sym is None:
            continue
        if hasattr(sym, "column") and sym.kind in ("cat", "obs"):
            printed.append((name, cfg.column_map(sym.kind)[sym.column][0], None))
        elif hasattr(sym, "eq") and not hasattr(sym, "fit"):
            printed.append((name, None, sym))
    return printed


def _fields(fmt: str) -> int:
    """ph_ofields: the % fields, %% not counted."""
    return len(_FIELD.findall(fmt.replace("%%", "")))


# ----------------------------------------------------------------- task


def invertfit(cfg: Config, observations: list[str], parameters: str, calib: str,
              catalogs: list[str] | None = None, catdir: str = "",
              errors: str = "obserrors", objects: str = "all", print_vars: list[str] = (),
              format: str = "", append: bool = False) -> list[str]:
    """t_invertfit: write the calibrated file; returns the warnings."""
    plan_ = plan(cfg, parameters)
    nobs, ncat = cfg.nobs, cfg.ncat
    getid = cfg.mincol > 1
    matchid = getid and bool(catalogs)
    printed = print_list(cfg, print_vars)
    residuals = (not getid) or (matchid and objects != "program")

    head = ["", f"# {io.cnvtime()}", "# List of observations files:"]
    head += [f"#\t\t{name}" for name in observations]
    if matchid:
        head.append("# Number of catalog files:")
        head += [f"#\t\t{name}" for name in catalogs]
    head += [f"# Config:\t{cfg.path}", f"# Parameters:\t{parameters}", "#"]
    if matchid:
        head.append({"all": "# Computed indices for program and standard objects",
                     "program": "# Computed indices for program objects only",
                     "standards": "# Computed indices for standard objects only"}[objects])
    else:
        head.append("# Computed indices for program and standard objects")
    head += ["#", "# Columns: "]
    ncols = 0
    if getid:
        ncols = 1
        head.append("#\t1\tobject id")
    for name, _, _ in printed:
        ncols += 1
        head.append(f"#\t{ncols}\t{name}")
    witherr = errors != "undefined"
    for i, var in enumerate(cfg.catvars):
        if plan_.uservars[i] <= 0:
            continue
        for label in ([var.name] + ([f"error({var.name})"] if witherr else [])
                      + ([f"resid({var.name})"] if residuals else [])):
            ncols += 1
            head.append(f"#\t{ncols}\t{label}")
    for seq in plan_.sets:
        for label in ([seq.name] + ([f"error({seq.name})"] if witherr else [])
                      + ([f"resid({seq.name})"] if residuals else [])):
            ncols += 1
            head.append(f"#\t{ncols}\t{label}")
    warnings = list(plan_.warnings)
    if not format:
        format = default_format(getid, ncols)
    elif _fields(format) != ncols:
        warnings.append("Warning: The number of format string fields does not "
                        "match the number of output columns")
        format = default_format(getid, ncols)

    catalog = io.read_catalogs(cfg, catalogs, catdir) if matchid else None
    avg, dtvars = catalog_means(cfg, catalog, plan_.uservars)
    solver = _Solver(cfg, plan_.params)
    nset = len(plan_.sets)
    lines = []
    for path in observations:
        for uid, row in io.read_program(cfg, path, catalog, objects):
            out: list = [uid] if getid else []
            for _, index, seq in printed:
                out.append(row[index] if index is not None
                           else evaluate(seq.eq.code, row, plan_.params[0]))
            a = row.copy()
            a[nobs:] = avg
            fitted = np.full(ncat, INDEF, dtype=f32)
            ervars = np.full(ncat, INDEF, dtype=f32)
            svars = np.full(nset, INDEF, dtype=f32)
            servars = np.full(nset, INDEF, dtype=f32)
            checked = _check(solver, a, plan_)
            if checked is not None:
                vindex, eqindex = checked
                if solver.invert(a, dtvars, vindex, eqindex):
                    for i in range(ncat):
                        if vindex[i] == 0:
                            a[nobs + i] = INDEF
                    fitted = a[nobs:].copy()
                    for k, seq in enumerate(plan_.sets):
                        svars[k] = evaluate(seq.eq.code, a, plan_.params[k])
                    if errors == "equations":
                        n, e, s = solver.errors_equations(a, dtvars, vindex, eqindex,
                                                          plan_.sets, svars)
                    elif errors == "obserrors":
                        n, e, s = solver.errors_obs(a, plan_.usererrs, dtvars, vindex,
                                                    eqindex, plan_.sets, svars)
                    else:
                        n = 0
                    if n > 0:
                        ervars, servars = e, s
                    fitted = a[nobs:].copy()
            for i in range(ncat):
                if plan_.uservars[i] <= 0:
                    continue
                out.append(fitted[i])
                if witherr:
                    out.append(INDEF if np.isnan(fitted[i]) else ervars[i])
                if residuals:
                    known = row[nobs + i]
                    out.append(INDEF if np.isnan(known) or np.isnan(fitted[i])
                               else f32(known - fitted[i]))
            for k, seq in enumerate(plan_.sets):
                out.append(svars[k])
                if witherr:
                    out.append(INDEF if np.isnan(svars[k]) else servars[k])
                if residuals:
                    value = evaluate(seq.eq.code, row, plan_.params[k])
                    out.append(INDEF if np.isnan(svars[k]) or np.isnan(value)
                               else f32(value - svars[k]))
            lines.append(_format(format, out))
    mode = "a" if append else "w"
    with open(calib, mode) as handle:
        handle.write("\n".join(head) + "\n\n\n" + "".join(lines))
    return warnings
