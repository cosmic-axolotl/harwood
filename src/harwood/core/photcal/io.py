"""photcal's catalog and observation files, and the fitted parameters file.

From photcal/io (iogetline.x, iolineid.x, iocat.x, ioobs.x, iocoeffs.x).

A record is a line and the lines after it that start with "* " -- obsfile
writes each filter of a star on its own line -- and its columns are counted
through them all. Lines starting with # and empty lines are skipped. The
first column is the object's id when the configuration has a catalog
section: upper-cased, and with everything but letters, digits, + - and _
taken out, so "PG 2331+055" and "pg2331+055" are the same star. A token
that does not start with a number leaves its variable undefined; a catalog
star given twice keeps its first line.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .. import iraffmt
from .config import INDEF, Config, Equation, ctor

f32 = np.float32


class ContinuationError(ValueError):
    pass


def records(path: str | Path):
    """io_getline: the logical lines of a file."""
    current = None
    for raw in Path(path).read_text(errors="replace", encoding="utf-8").split("\n"):
        line = raw.rstrip("\r")
        if not line or line.startswith("#"):
            continue
        if line.startswith("* "):
            if current is None:
                raise ContinuationError(
                    f"{path}: Continuation character found in first line")
            current += line[1:]
            continue
        if current is not None:
            yield current
        current = line
    if current is not None:
        yield current


def words(line: str) -> list[str]:
    """ctowrd, one after the other: blank-separated, or a quoted string."""
    out, i, n = [], 0, len(line)
    while True:
        while i < n and line[i] in " \t":
            i += 1
        if i >= n:
            return out
        if line[i] == '"':
            end = line.find('"', i + 1)
            end = n if end < 0 else end
            out.append(line[i + 1:end])
            i = end + 1
        else:
            start = i
            while i < n and line[i] not in " \t":
                i += 1
            out.append(line[start:i])


def lineid(token: str) -> str:
    """io_lineid's compressed id."""
    up = token.upper()
    return "".join(ch for ch in up if ch.isdigit() or ("A" <= ch <= "Z") or ch in "+-_")


def catalog_path(name: str, catdir: str = "") -> str:
    """A catalog in catdir as name.dat, or the file itself."""
    candidate = f"{catdir}{name}.dat"
    return candidate if Path(candidate).is_file() else name


def read_catalogs(cfg: Config, files: list[str], catdir: str = "") -> dict[str, np.ndarray]:
    """io_gcatdat: id -> the catalog variables (NaN where undefined)."""
    columns = cfg.column_map("cat")
    table: dict[str, np.ndarray] = {}
    for name in files:
        for line in records(catalog_path(name, catdir)):
            tokens = words(line)
            if not tokens:
                continue
            cid = lineid(tokens[0])
            if not cid or cid in table:
                continue
            row = np.full(cfg.ncat, INDEF, dtype=f32)
            for col, token in enumerate(tokens[1:], start=2):
                value = ctor(token)
                if value is not None:
                    for index in columns.get(col, ()):
                        row[index - cfg.nobs] = value
            table[cid] = row
    return table


@dataclass
class Observations:
    """io_gcatobs: the observations matched to the catalog, one row each
    (observation variables, then the star's catalog variables)."""
    rows: np.ndarray
    names: list[str]
    unmatched: list[tuple[str, str]] = field(default_factory=list)


def read_observations(cfg: Config, files: list[str],
                      catalog: dict[str, np.ndarray] | None) -> Observations:
    getid = cfg.mincol > 1
    columns = cfg.column_map("obs")
    rows, names, unmatched = [], [], []
    for path in files:
        for line in records(path):
            tokens = words(line)
            first = 1
            cid = ""
            if getid:
                if not tokens or not lineid(tokens[0]):
                    continue
                cid = lineid(tokens[0])
                first = 2
                tokens = tokens[1:]
            star = None
            if catalog is not None:
                star = catalog.get(cid)
                if star is None:
                    unmatched.append((str(path), cid))
                    continue
            row = np.full(cfg.nvars, INDEF, dtype=f32)
            for col, token in enumerate(tokens, start=first):
                value = ctor(token)
                if value is not None:
                    for index in columns.get(col, ()):
                        row[index] = value
            if star is not None:
                row[cfg.nobs:] = star
            rows.append(row)
            names.append(cid)
    table = (np.array(rows, dtype=f32) if rows
             else np.zeros((0, cfg.nvars), dtype=f32))
    return Observations(table, names, unmatched)


def read_program(cfg: Config, path: str, catalog: dict[str, np.ndarray] | None,
                 objects: str = "all"):
    """io_gobs, record by record: (the id as written, the row). With a
    catalog the row ends with the star's catalog values, copied in after
    the last observation variable the line set -- so a line that leaves its
    last ones undefined shifts them, as IRAF does."""
    getid = cfg.mincol > 1
    columns = cfg.column_map("obs")
    for line in records(path):
        tokens = words(line)
        first, uid = 1, ""
        if getid:
            if not tokens or not lineid(tokens[0]):
                continue
            uid, cid = tokens[0], lineid(tokens[0])
            first = 2
            tokens = tokens[1:]
        star = None
        if catalog is not None:
            star = catalog.get(cid)
            if objects == "program" and star is not None:
                continue
            if objects == "standards" and star is None:
                continue
        row = np.full(cfg.nvars, INDEF, dtype=f32)
        nread = 0
        for col, token in enumerate(tokens, start=first):
            value = ctor(token)
            if value is not None:
                for index in columns.get(col, ()):
                    row[index] = value
                    nread = max(nread, index + 1)
        if star is not None:
            row[nread:] = star[:cfg.nvars - nread]
        yield uid, row


# ------------------------------------------------------- parameters file


def cnvtime(when: float | None = None) -> str:
    """cnvtime: Wed 15:49:56 21-Oct-2026."""
    return iraffmt.cnvtime(when)


STATUS = {0: "Solution converged", 1: "Singular matrix", 2: "Too few points",
          3: "Solution did not converge"}


def g(value) -> str:
    return iraffmt.g(float(value))


def write_coeffs(path: str | Path, eq: Equation, result, weighting: str) -> None:
    """io_pcoeffs: append the equation's record to the parameters file."""
    variance, chisqr = f32(result.variance), f32(result.chisqr)
    avsq = f32(0.0) if chisqr <= 0 else f32(variance / chisqr)
    scatter = f32(result.scatter) if result.scatter > 0 else f32(0.0)
    if weighting == "uniform":
        weights = "uniform"
    elif weighting == "photometric":
        weights = "photometric"
    else:
        weights = eq.weight.text if eq.weight else "uniform"
    lines = [f"# {cnvtime()}", f"begin\t{eq.name}",
             f"\tstatus\t{result.stat}\t({STATUS.get(result.stat, 'Unknown error code')})",
             f"\tvariance\t{g(variance)}",
             f"\tstdeviation\t{g(np.sqrt(variance) if variance > 0 else 0.0)}",
             f"\tavsqerror\t{g(avsq)}",
             f"\taverror\t\t{g(np.sqrt(avsq))}",
             f"\tavsqscatter\t{g(scatter)}",
             f"\tavscatter\t{g(np.sqrt(scatter))}",
             f"\tchisqr\t\t{g(chisqr)}",
             f"\tmsq\t\t{g(f32(result.rms) * f32(result.rms))}",
             f"\trms\t\t{g(result.rms)}",
             f"\treference\t{eq.ref.text}",
             f"\tfitting\t\t{eq.fit.text}",
             f"\tweights\t\t{weights}",
             f"\tparameters\t{len(eq.params)}"]
    for k, param in enumerate(eq.params, start=1):
        lines.append(f"\t\t{param.name}\t({'fit' if k in result.plist else 'constant'})")
    lines.append(f"\tderivatives\t{len(eq.params)}")
    for k, param in enumerate(eq.params):
        der = eq.derivatives[k]
        lines.append(f"\t\t{der.text if der else g(param.delta)}")
    lines.append(f"\tvalues\t{len(eq.params)}")
    lines += [f"\t\t{g(v)}" for v in result.params]
    lines.append(f"\terrors\t{len(eq.params)}")
    lines += [f"\t\t{g(v)}" for v in result.errors]
    with open(path, "a", encoding="utf-8") as out:
        out.write("\n".join(lines) + "\n\n")


@dataclass
class Coeffs:
    stat: int | None
    chisqr: float
    rms: float
    values: np.ndarray
    errors: np.ndarray


def read_coeffs(path: str | Path, name: str, nparams: int) -> Coeffs | None:
    """io_gcoeffs: the last record for the equation (dtlocate finds the
    last one written), or None."""
    text = Path(path).read_text(errors="replace", encoding="utf-8").split("\n")
    start = None
    for k, line in enumerate(text):
        fields = line.split()
        if len(fields) >= 2 and fields[0] == "begin" and fields[1] == name:
            start = k
    if start is None:
        return None
    record: dict[str, list[str]] = {}
    k = start + 1
    while k < len(text):
        fields = text[k].split()
        if fields and fields[0] == "begin":
            break
        if fields and not text[k].startswith("#"):
            key = fields[0]
            if key in ("values", "errors", "parameters", "derivatives"):
                n = int(fields[1]) if len(fields) > 1 else 0
                record[key] = [text[k + 1 + j].strip() for j in range(n)
                               if k + 1 + j < len(text)]
                k += n
            else:
                record[key] = fields[1:]
        k += 1

    def number(key: str) -> float:
        value = ctor(record[key][0]) if record.get(key) else None
        return float("nan") if value is None else value

    def array(key: str) -> np.ndarray:
        values = [ctor(v) for v in record.get(key, [])]
        return np.array([np.nan if v is None else v for v in values], dtype=f32)

    status = record.get("status")
    values, errors = array("values"), array("errors")
    if len(values) != nparams:
        return Coeffs(None, np.nan, np.nan, np.full(nparams, INDEF, dtype=f32),
                      np.full(nparams, INDEF, dtype=f32))
    return Coeffs(int(status[0]) if status else None, number("chisqr"),
                  number("rms"), values, errors)
