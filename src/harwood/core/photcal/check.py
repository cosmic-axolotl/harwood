"""chkconfig: compile a configuration file and say what it declares.

From photcal/parser/t_chkconfig.x. The parser's errors and warnings come
between the compilation banners as pr_error prints them; the verbose
listing of variables, parameters and equations only when there were no
errors; the counters always.

Translated from IRAF, Copyright (c) 1986 Association of Universities for
Research in Astronomy Inc.; the notice it is distributed under is in
LICENSES/IRAF-COPYRIGHT.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .. import iraffmt
from .config import Config, ConfigError, Expr, _Parser


@dataclass
class Report:
    text: str                     # what chkconfig prints on the standard output
    stderr: str                   # yyerror's "syntax error", when there was one
    config: Config | None         # the parsed configuration, when it compiled
    nerrors: int
    nwarnings: int


def _i(value: int | None) -> str:
    return "INDEF" if value is None else str(value)


def _x(expr: Expr | None) -> str:
    return expr.text if expr else ""


def _columns(cfg: Config, kind: str) -> tuple[int | None, int | None]:
    """pr_exit's first and last input column of a section, error and
    weight columns included."""
    cols = []
    for var in (cfg.catvars if kind == "cat" else cfg.obsvars):
        cols += [c for c in (var.column, var.errcol, var.wtscol) if c is not None]
    return (min(cols), max(cols)) if cols else (None, None)


def check(path: str | Path, verbose: bool = False, text: str | None = None) -> Report:
    if text is None:
        text = Path(path).read_text(errors="replace")
    parser = _Parser(text, str(path))
    try:
        cfg = parser.parse()
        done = True
    except ConfigError:
        cfg = parser.cfg
        done = False

    out = ["\n\n** Beginning of compilation **\n\n"]
    out += [message + "\n" for message in parser.messages]
    out.append("\n** End of compilation **\n\n")
    if done and verbose:
        out += _listing(cfg)

    nfit = sum(p.kind == "fit" for p in cfg.params)
    cat = _columns(cfg, "cat") if done else (None, None)
    obs = _columns(cfg, "obs") if done else (None, None)
    nwarnings = len(cfg.warnings)
    nerrors = len(parser.errors)
    out += [
        f"Catalog input variables         = {cfg.ncat}\n",
        f"First catalog column            = {_i(cat[0])}\n",
        f"Last catalog column             = {_i(cat[1])}\n\n",
        f"Observational input variables   = {cfg.nobs}\n",
        f"First observational column      = {_i(obs[0])}\n",
        f"Last observational column       = {_i(obs[1])}\n\n",
        f"Fitting parameters              = {nfit}\n",
        f"Constant parameters             = {len(cfg.params) - nfit}\n\n",
        f"Auxiliary (set) equations       = {len(cfg.sets)}\n",
        f"Transformation equations        = {len(cfg.equations)}\n\n",
        f"Warnings                        = {nwarnings}\n",
        f"Errors                          = {nerrors}\n"]
    return Report("".join(out), "syntax error\n" if parser.syntax_error else "",
                  cfg if done else None, nerrors, nwarnings)


def _listing(cfg: Config) -> list[str]:
    out = []
    for title, table in (("CATALOG", cfg.catvars), ("OBSERVATIONAL", cfg.obsvars)):
        if not table:
            continue
        out.append(f"\n{title} VARIABLES, COLUMNS, AND ERROR COLUMNS:\n\n")
        out += [f"{v.number:2d}  {v.name}\t{v.column}\t{_i(v.errcol)}\n"
                for v in table if not v.spare]

    if cfg.params:
        out.append("\nFIT AND CONSTANT PARAMETER VALUES:\n\n")
        out += [f"{i:2d}  {p.name}\t{iraffmt.g(float(p.value))}\t"
                f"{'(constant)' if p.kind == 'const' else ''}\n"
                for i, p in enumerate(cfg.params, start=1)]

    if cfg.sets:
        out.append("\nAUXILIARY (SET) EQUATIONS:\n\n")
        for s in cfg.sets:
            out.append(f"{s.number:2d} {s.name} = {s.eq.text}\n")
            out.append(f"   error = {_x(s.error)},  min = {_x(s.errmin)},  "
                       f"max = {_x(s.errmax)}\n")
            out.append("\n")

    if cfg.equations:
        out.append("\nTRANSFORMATION EQUATIONS:\n\n")
        for eq in cfg.equations:
            out.append(f"{eq.number:2d} {eq.name}:  {eq.ref.text} = {eq.fit.text}\n")
            for param, der in zip(eq.params, eq.derivatives, strict=True):
                if der is not None:
                    out.append(f"   derivative ({eq.name}, {param.name}) = {der.text}\n")
                else:
                    out.append(f"   delta({eq.name}, {param.name}) = "
                               f"{iraffmt.g(float(param.delta))}\n")
            out.append(f"   error = {_x(eq.error)},  min = {_x(eq.errmin)},  "
                       f"max = {_x(eq.errmax)}\n")
            out.append(f"   weight = {_x(eq.weight)},  min = {_x(eq.wtsmin)},  "
                       f"max = {_x(eq.wtsmax)}\n")
            out.append(f"   plot  x = {_x(eq.xplot)},  y = {_x(eq.yplot)}\n")
            out.append("\n")
        out.append("\n")
    return out
