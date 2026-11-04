"""mkconfig: a configuration file put together from its three sections.

From photcal/mkconfig.cl and mkconfig/t_config.x: the catalog section
comes from a format file (f<catalog>.dat, here or in
the standards directory), the observation section from the observations'
format file (f<observations>.dat, as mknobsfile and obsfile write it), the
transformation section from a template (t<transform>.dat); each may also be
named as the file itself. The three are copied one after the other, as
IRAF copies them. A template configuration is copied whole instead. A
section given as STDIN is typed at the terminal, at IRAF's prompts (typed()).

The standards directory is photcal$catalogs/, harwood's copy of IRAF's
catalogs, format files and transformation templates (Landolt, Elias,
UKIRT, NICMOS ...).

Translated from IRAF, Copyright (c) 1986 Association of Universities for
Research in Astronomy Inc.; the notice it is distributed under is in
LICENSES/IRAF-COPYRIGHT.
"""

from __future__ import annotations

import re
import shutil
from dataclasses import dataclass
from pathlib import Path

CATALOGS = Path(__file__).resolve().parent.parent.parent / "data" / "iraf" / "catalogs"
LOGICAL = "photcal$catalogs/"


def catdir(text: str) -> str:
    """The standards directory, photcal$catalogs/ meaning harwood's copy;
    as in the CL, file names are appended to it as they are."""
    if text.startswith("photcal$catalogs"):
        return str(CATALOGS) + "/" + text[len("photcal$catalogs"):].lstrip("/")
    return text


@dataclass
class Sources:
    catalog: str | None
    observations: str
    transform: str


def _find(name: str, candidates: list[str], where: str) -> str:
    for path in candidates:
        if Path(path).is_file():
            return path
    raise FileNotFoundError(f"Cannot find {name} in {where}")


def _own(name: str) -> str:
    """f<name>.dat next to the name."""
    head, tail = Path(name).parent, Path(name).name
    return str(head / f"f{tail}.dat") if str(head) != "." else f"f{tail}.dat"


def catalog_file(catalog: str, standards: str = LOGICAL) -> str:
    return _find(catalog, [_own(catalog), f"{catdir(standards)}f{catalog}.dat", catalog],
                 "current or standards directory")


def observations_file(observations: str) -> str:
    return _find(observations, [_own(observations), observations], "current directory")


def transform_file(transform: str, standards: str = LOGICAL) -> str:
    cdir = catdir(standards)
    return _find(transform, [f"t{transform}.dat", f"{cdir}t{transform}.dat", transform],
                 "current or standards directory")


def sources(catalog: str | None, observations: str, transform: str,
            standards: str = LOGICAL) -> Sources:
    """mkconfig.cl's search for each section's file."""
    return Sources(catalog_file(catalog, standards) if catalog else None,
                   observations_file(observations), transform_file(transform, standards))


def mkconfig(config: str | Path, catalog: str | None = None, observations: str = "",
             transform: str = "", template: str = "",
             standards: str = LOGICAL) -> Sources | None:
    """Write the configuration. Returns the section files used, or None
    when a template was copied."""
    config = Path(config)
    if config.exists():
        raise FileExistsError(f"The configuration file {config} already exists")
    if template and Path(template).is_file():
        shutil.copyfile(template, config)
        return None
    found = sources(catalog, observations, transform, standards)
    parts = [found.catalog, found.observations, found.transform]
    config.write_bytes(b"".join(Path(p).read_bytes() for p in parts if p))
    return found


# ------------------------------------------------------------ typed (STDIN)
#
# t_config.x when a section is STDIN: ph_rcsection and ph_rosection ask for the
# columns, ph_rtsection for the equations, their parameters to fit and those held
# constant. Each prompt and message is IRAF's, and so are its slips: a label
# verified to another is cut to the new one's length, not replaced by it; a
# parameter verified to another keeps the old value; the fitted and the constant
# parameters each have room for 160 characters in all, past which the line is
# cut and no more equations are asked for; a line longer than 78 characters is
# broken at the 78th, wherever that falls.

MAXLENLINE = 78          # CFG_MAXLENLINE
MAXLENPARAM = 160        # CFG_MAXLENPARAM, the fit and const string files
MAXLENTRANS = 16000      # CFG_MAXLENTRANS
HELP = Path(__file__).resolve().parent.parent.parent / "data" / "iraf" / "mkconfig"
_LEXNUM = re.compile(r"[-+]?(?:\d+\.?\d*|\.\d+)|INDEF")


class _Overflow(Exception):
    pass


class _StrFile:
    """stropen on a fixed buffer: what does not fit is cut, and the write fails."""

    def __init__(self, size: int):
        self.size, self.text = size, ""

    def putline(self, line: str) -> None:
        line = line.split("\0", 1)[0]
        room = self.size - len(self.text)
        self.text += line[:room]
        if len(line) > room:
            raise _Overflow


class _Out:
    """putline to the terminal or to the configuration being written."""

    def __init__(self, write):
        self.write = write

    def putline(self, line: str) -> None:
        self.write(line.split("\0", 1)[0])


def is_ident(text: str) -> bool:
    """ph_isident: a letter, then letters and digits."""
    return bool(text) and text[0].isascii() and text[0].isalpha() and all(
        c.isascii() and c.isalnum() for c in text)


def _help(term, name: str) -> None:
    """pagefile: the help, its standout marks dropped."""
    text = (HELP / name).read_text(encoding="utf-8")
    term.printf(text.replace("\x0e", "").replace("\x0f", ""))


def _gargi(got) -> int | None:
    """gargi: an integer from the next field; INDEF is INDEFI."""
    before = got.n
    value = got.r()
    if got.n == before:
        return None
    return -2147483647 if value is None else int(value)


def _rcolnumber(term, helpfile: str) -> tuple[int | None, str, int]:
    """ph_rcolnumber: a column name and number; None at the end of the input."""
    term.printf("Enter column definition (name number, ?=help, <EOF>=quit entry): ")
    got = term.scan()
    if got is None:
        return None, "", 0
    column = got.wrd() or ""
    number = _gargi(got)
    if column == "?":
        _help(term, helpfile)
        return 0, column, 0
    if not is_ident(column) and "error(" not in column:
        term.printf(f"<Error> {column} is not a legal column name\n")
        return 0, column, 0
    if got.n != 2:
        term.printf("<Error> Cannot decode the column number\n")
        return 0, column, 0
    if number <= 1:
        term.printf("<Error> Column 1 is reserved for the object name\n")
        return 0, column, 0
    return 2, column, number


def _vcolnumber(term, column: str, number: int) -> tuple[str, int]:
    """ph_vcolnumber: a name that is not an identifier, or a number not past the
    first column, keeps the old one."""
    term.printf(f"\tVerify ({column}  {number}) (name number, <CR>=ok): ")
    got = term.scan()
    if got is None:
        term.printf("\n")
        return column, number
    if not got.line:
        return column, number
    new = got.wrd() or ""
    newnumber = _gargi(got)
    if not is_ident(new):
        new = column
    if got.n != 2 or newnumber <= 1:
        newnumber = number
    return new, newnumber


def ask_columns(term, keyword: str, verify: bool = False) -> str:
    """ph_rcsection / ph_rosection: the section, a column at a time, until the
    end of the input."""
    out = [f"\n{keyword}\n\n"]
    what = ("THE STANDARD STAR CATALOG" if keyword == "catalog"
            else "THE OBSERVATIONS FILE")
    helpfile = "catsection.key" if keyword == "catalog" else "obsection.key"
    term.printf(f"\nENTER {what} FORMAT DESCRIPTION\n\n")
    while True:
        stat, column, number = _rcolnumber(term, helpfile)
        if stat is None:
            break
        if stat <= 0:
            continue
        if verify:
            column, number = _vcolnumber(term, column, number)
        out.append(f"{column}  {number}\n")
    term.printf("\n")
    return "".join(out)


def _write(sink, text: str, length: int, line: list[str]) -> None:
    """ph_write: `length` characters of `text` into the line, which goes out
    whenever it reaches 78."""
    text = text.ljust(length, "\0")[:length]
    ip = 0
    while True:
        n = min(length - ip, MAXLENLINE - len(line))
        line.extend(text[ip:ip + n])
        ip += n
        if len(line) >= MAXLENLINE:
            sink.putline("".join(line) + "\n")
            line.clear()
        if ip >= length:
            break


def _wequation(sink, label: str, nlabel: int, expr: str) -> None:
    """ph_wequation: `label : expression`, broken at 78 characters."""
    line: list[str] = []
    _write(sink, label, nlabel, line)
    _write(sink, " : ", 3, line)
    _write(sink, expr, len(expr), line)
    if line:
        sink.putline("".join(line) + "\n")


def _wparam(sink, keyword: str, params: str) -> None:
    """ph_wparam: `fit  a = 1, b = 2`, broken at 78 characters."""
    line: list[str] = []
    _write(sink, keyword, len(keyword), line)
    _write(sink, "  ", 2, line)
    _write(sink, params, len(params), line)
    if line:
        sink.putline("".join(line) + "\n")


class _Label:
    """The label's buffer: a new word overwrites the old one's start, as gargwrd
    writes into the same space each time."""

    def __init__(self):
        self.buf = ""

    def set(self, word: str) -> None:
        self.buf = word + "\0" + self.buf[len(word) + 1:]


def _rlabel(term, label: _Label, verify: bool) -> int | None:
    """ph_rlabel: the length of the label, None at the end of the input. With
    verify, the new label's length is returned and the old label kept."""
    term.printf("Enter label (e.g. VFIT) (label, ?=help, <EOF>=quit entry): ")
    got = term.scan()
    if got is None:
        term.printf("\n")
        return None
    word = got.wrd() or ""
    label.set(word)
    if word == "?":
        _help(term, "transection.key")
        return 0
    if not is_ident(word):
        term.printf(f"<Error> {word} is not a legal label\n")
        return 0
    if not verify:
        return len(word)
    term.printf(f"\tVerify ({word}) (label, <CR>=ok): ")
    got = term.scan()
    if got is None:
        term.printf("\n")
        new = ""
    else:
        new = got.wrd() or ""
    if not new or not is_ident(new):
        new = word
    return len(new)


def _rexpr(term, neq: int, verify: bool) -> str | None:
    """ph_rexpr: the equation, over lines ending in a backslash; None at the end
    of the input."""
    term.printf("Enter equation (equation, equation\\=continue, ?=help, "
                "<EOF>=quit entry):\n")
    expr = ""
    while True:
        got = term.scan()
        if got is None:
            term.printf("\n")
            return None
        chunk = got.line
        if not chunk:
            break
        if chunk == "?":
            _help(term, "transection.key")
            term.printf(f"Equation {neq} enter equation (?=help, \\=continue, "
                        "EOF=quit):\n")
            continue
        expr += chunk
        if expr.endswith("\\"):
            expr = expr[:-1]
        else:
            break
    if not verify:
        return expr
    term.printf("Verify (equation, <CR>=ok):\n")
    new = ""
    for ip in range(0, len(expr), MAXLENLINE - 2):
        piece = expr[ip:ip + MAXLENLINE - 2]
        term.printf(piece + "\n")
        got = term.scan()
        if got is None:
            term.printf("\n")
            typed = ""
        else:
            typed = got.line[:MAXLENLINE]
        new += typed or piece
    return new or expr


def _rpar(term, keyword: str, verify: bool) -> tuple[int | None, str, str]:
    """ph_rpar: a parameter's name and value. With verify a new name replaces the
    old, a new value does not."""
    term.printf(f"Enter {keyword} (name value, ?=help, <EOF>=quit entry):")
    got = term.scan()
    if got is None:
        term.printf("\n")
        return None, "", ""
    param = got.wrd() or ""
    number = got.wrd() or ""
    if param == "?":
        _help(term, "transection.key")
        return 0, param, number
    if not is_ident(param):
        term.printf(f"<Error> {param} is not a legal parameter name\n")
        return 0, param, number
    if got.n != 2:
        term.printf("<Error> Cannot decode the parameter value\n")
        return 0, param, number
    if not _LEXNUM.match(number):
        term.printf(f"<Error> The parameter value {number} is not a legal number\n")
        return 0, param, number
    if not verify:
        return 2, param, number
    term.printf(f"\tVerify ({param} {number}) (name value, <CR>=ok):")
    got = term.scan()
    if got is not None and got.line:
        new = got.wrd() or ""
        if is_ident(new):
            param = new
    return 2, param, number


def _rparams(term, what: str, neq: int, label: _Label, nlabel: int, expr: str,
             verify: bool) -> str:
    """The loop over ph_rlist: `name = value, ...` until the end of the input;
    a line that is not a parameter shows the equation again."""
    term.printf(f"\nEnter initial values for the parameters to be {what} in "
                f"EQUATION {neq}\n\n")
    listed = ""
    nparam = 1
    while True:
        keyword = (f"parameter {nparam}" if what == "fit"
                   else f"parameter{nparam} and value")
        stat, param, number = _rpar(term, keyword, verify)
        if stat is None:
            break
        if stat <= 0:
            term.printf("\n")
            _wequation(_Out(term.printf), label.buf, nlabel, expr)
            term.printf("\n")
            continue
        listed += f"{param} = {number}, "
        nparam += 1
    return listed[:-2] if listed else ""


def ask_equations(term, verify: bool = False) -> str:
    """ph_rtsection: the equations, each with its label, its parameters to fit
    and those held constant."""
    trans, fit, const = (_StrFile(MAXLENTRANS), _StrFile(MAXLENPARAM),
                         _StrFile(MAXLENPARAM))
    label = _Label()
    term.printf("\nENTER THE TRANSFORMATION EQUATIONS\n")
    neq = 0
    while True:
        term.printf(f"\nEnter the label and functional form for EQUATION {neq + 1}\n\n")
        nlabel = _rlabel(term, label, verify)
        if nlabel is None:
            break
        if nlabel <= 0:
            continue
        expr = _rexpr(term, neq + 1, verify)
        if expr is None:
            continue
        if not expr:
            term.printf("<Error> The function expression is blank\n")
            continue
        try:
            _wequation(trans, label.buf, nlabel, expr)
            fitted = _rparams(term, "fit", neq + 1, label, nlabel, expr, verify)
            if fitted:
                _wparam(fit, "fit", fitted)
            held = _rparams(term, "held constant", neq + 1, label, nlabel, expr,
                            verify)
            if held:
                _wparam(const, "const", held)
        except _Overflow:
            break
        neq += 1
    term.printf("\n")
    return ("\ntransformation\n\n" + fit.text + "\n" + const.text + "\n"
            + trans.text)


def typed(config: str | Path, catalog: str | None, observations: str, transform: str,
          term, verify: bool = False, standards: str = LOGICAL) -> Sources:
    """t_config with any of the sections STDIN: those typed at the terminal, the
    others copied from their files, in the order of the file."""
    config = Path(config)
    if config.exists():
        raise FileExistsError(f"The configuration file {config} already exists")
    found = Sources(
        catalog_file(catalog, standards) if catalog and catalog != "STDIN" else None,
        observations_file(observations) if observations != "STDIN" else "",
        transform_file(transform, standards) if transform != "STDIN" else "")
    text = []
    for keyword, given, path in (("catalog", catalog, found.catalog),
                                 ("observation", observations, found.observations),
                                 ("transformation", transform, found.transform)):
        if given == "STDIN":
            text.append(ask_equations(term, verify) if keyword == "transformation"
                        else ask_columns(term, keyword, verify))
        elif path:
            text.append(Path(path).read_text(encoding="utf-8"))
    config.write_text("".join(text), encoding="utf-8", newline="")
    return found
