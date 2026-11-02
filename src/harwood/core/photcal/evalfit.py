"""evalfit: the fitted equations evaluated as they stand, star by star.

From photcal/evaluate (t_evalfit.x, ph_erval in pherrors.x, phprint.x). Where
invertfit solves the equations for the catalog variables, this one only works
out each equation's right side, so it needs equations written the other way
round, the catalog magnitude on the left. A catalog variable on the right side
is INDEF for a star not in a catalog, and so is the result.

The quirks kept, because they decide the digits: an error from the observations
moves every observed variable that has an error column, as declared for any
equation, but a variable reached only through a set equation is not one of them;
an error equation is evaluated even for a star whose fit is INDEF; and a set
equation in `print` is evaluated with the first equation's parameters.

Two differences, both where IRAF has no useful answer. An error that takes an
expression out of its domain is infinite in IRAF, printed as 0.00E309, and INDEF
here. A configuration with no catalog section stops IRAF with "Out of memory";
here its stars have no id, as t_evalfit.x means them to.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from ..nlfit import DONE, square
from . import io
from .config import INDEF, Config, evaluate
from .invert import _fields, _format, default_format, print_list

f32 = np.float32


class EvalError(ValueError):
    pass


@dataclass
class Evaluated:
    stars: int = 0
    undefined: int = 0                 #: stars with an equation that came to INDEF
    warnings: list[str] = field(default_factory=list)


def _parameters(cfg: Config, parameters: str, warnings: list[str]) -> list[np.ndarray]:
    """Each equation's values from the parameters file. As in t_evalfit.x, a
    record that cannot be read leaves the status of the one before it."""
    if not Path(parameters).is_file():
        raise EvalError(f"Cannot open file ({parameters})")
    params = []
    stat = DONE
    for eq in cfg.equations:
        coeffs = io.read_coeffs(parameters, eq.name, len(eq.params))
        if coeffs is None or coeffs.stat is None:
            warnings.append(f"Warning: Error reading parameters for equation {eq.name} "
                            f"from {parameters}")
            params.append(np.full(len(eq.params), INDEF, dtype=f32))
        else:
            params.append(coeffs.values)
            stat = coeffs.stat
        if stat != DONE:
            warnings.append(f"Warning: The solution for equation {eq.name} did not converge")
    return params


def _error_columns(cfg: Config) -> list[int | None]:
    """usererrs: for each observed variable, the row index of its error."""
    errors: list[int | None] = [None] * cfg.nobs
    first = cfg.column_map("obs")
    for eq in cfg.equations:
        for var, _ in eq.refvars + eq.fitvars:
            if var.kind == "obs":
                column = cfg.error_index(var)
                if column is not None:
                    errors[first[var.column][0]] = column
    return errors


def _error(code: list, fit, params, row: np.ndarray, errors: list) -> np.float32:
    """ph_erval: the changes in the fit when each observed variable is moved by
    its error, in quadrature; INDEF when none has an error column."""
    total = f32(0.0)
    nerrors = 0
    with np.errstate(over="ignore"):
        for i, column in enumerate(errors):
            if column is None:
                continue
            nerrors += 1
            err = row[column]
            if np.isnan(err) or err <= 0:
                continue
            saved = row[i]
            row[i] = f32(saved + err)
            total = f32(total + square(evaluate(code, row, params) - fit))
            row[i] = saved
    return f32(np.sqrt(total)) if nerrors else INDEF


def evalfit(cfg: Config, observations: list[str], parameters: str, calib: str,
            catalogs: list[str] | None = None, catdir: str = "",
            errors: str = "obserrors", objects: str = "all", print_vars: list[str] = (),
            format: str = "", append: bool = False) -> Evaluated:
    """t_evalfit: write the calibrated file."""
    done = Evaluated()
    params = _parameters(cfg, parameters, done.warnings)
    getid = cfg.mincol > 1
    matchid = getid and bool(catalogs)
    printed = print_list(cfg, print_vars)
    residuals = (not getid) or (matchid and objects != "program")
    witherr = errors != "undefined"

    head = ["", f"# {io.cnvtime()}", "# List of observations files:"]
    head += [f"#\t\t{name}" for name in observations]
    if matchid:
        head.append("# List of catalog files:")
        head += [f"#\t\t{name}" for name in catalogs]
    head += [f"# Config:\t{cfg.path}", f"# Parameters:\t{parameters}", "#"]
    if matchid:
        head.append({"all": "# Computed indices for program and standard objects",
                     "program": "# Computed indices for program objects only",
                     "standards": "# Computed indices for standard objects only"}[objects])
    else:
        head.append("# Computed indices for program and standard objects")
    head += ["#", "# Columns:"]
    ncols = 0
    if getid:
        ncols = 1
        head.append("#\t1\tobject id")
    for name, _, _ in printed:
        ncols += 1
        head.append(f"#\t{ncols}\t{name}")
    for eq in cfg.equations:
        for label in ([eq.ref.text] + ([f"error({eq.ref.text})"] if witherr else [])
                      + ([f"resid({eq.ref.text})"] if residuals else [])):
            ncols += 1
            head.append(f"#\t{ncols}\t{label}")

    if not format:
        format = default_format(getid, ncols)
    elif _fields(format) != ncols:
        done.warnings.append("Warning: The number of formatstr string fields does not "
                             "match the number of output columns")
        format = default_format(getid, ncols)

    catalog = io.read_catalogs(cfg, catalogs, catdir) if matchid else None
    columns = _error_columns(cfg)
    lines = []
    for path in observations:
        if not Path(path).is_file():
            done.warnings.append(f"Warning: Cannot open file ({path})")
            continue
        for uid, row in io.read_program(cfg, path, catalog, objects):
            out: list = [uid] if getid else []
            for _, index, seq in printed:
                out.append(row[index] if index is not None
                           else evaluate(seq.eq.code, row, params[0]))
            undefined = False
            for eq, values in zip(cfg.equations, params, strict=True):
                fit = evaluate(eq.fit.code, row, values)
                undefined = undefined or bool(np.isnan(fit))
                out.append(fit)
                if errors == "obserrors":
                    out.append(INDEF if np.isnan(fit)
                               else _error(eq.fit.code, fit, values, row, columns))
                elif errors == "equations":
                    out.append(evaluate(eq.error.code, row, values) if eq.error else INDEF)
                if residuals:
                    ref = evaluate(eq.ref.code, row, values)
                    out.append(INDEF if np.isnan(fit) or np.isnan(ref) else f32(ref - fit))
            lines.append(_format(format, out))
            done.stars += 1
            done.undefined += undefined
    with open(calib, "a" if append else "w", encoding="utf-8") as handle:
        handle.write("\n".join(head) + "\n\n\n" + "".join(lines))
    return done
