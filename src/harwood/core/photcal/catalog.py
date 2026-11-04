"""mkcatalog: a catalog of standard stars, and the format file for mkconfig.

From photcal/mkcatalog (t_catalog.x, phrecord.x), with the column names,
widths and records given at once instead of typed at prompts. A new
catalog starts with its header -- title, number of columns, the record
length, the column names and widths -- and gets a format file f<name>.dat
declaring the columns for the configuration's catalog section; an
existing one, read back from its header, has records added at the end.

Fields are written left-justified in their width plus a two-blank gap, cut
to the width as IRAF cuts them; a record longer than a line is broken
between fields onto continuation lines starting with "* ".

What IRAF does and harwood does not: a catalog of more than ten columns
loses the name of the eleventh -- t_catalog.x keeps a pointer into the
name buffer across the realloc that grows it.

Translated from IRAF, Copyright (c) 1986 Association of Universities for
Research in Astronomy Inc.; the notice it is distributed under is in
LICENSES/IRAF-COPYRIGHT.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

KYCATALOG = "CATALOG:"
KYNCOLS = "NCOLS:"
KYHDRLENGTH = "HDRLENGTH:"
IDCOLWIDTH = 15
COLWIDTH = 10
MAXCOLWIDTH = 77                 # 79 in IRAF, which counts the gap in it
SZGAP = 2
COMMENTSTR = "# "
CONTSTR = "* "
BLANKSTR = "  "
MAXCH_PERLINE = 1022             # SZ_LINE - 1


@dataclass
class Catalog:
    path: str
    title: str
    names: list[str]
    widths: list[int]
    created: bool = False
    format_file: str = ""
    nrecords: int = 0
    truncated: list[str] = field(default_factory=list)


def is_ident(name: str) -> bool:
    """ph_isident: a letter, then letters and digits."""
    return bool(name) and name[0].isalpha() and name.isascii() and name.isalnum()


def legal_column(name: str) -> bool:
    return is_ident(name) or "error(" in name


def _twidths(widths: list[int]) -> list[int]:
    return [w + SZGAP for w in widths[:-1]] + widths[-1:]


def _field(text: str, width: int, twidth: int) -> str:
    """%*.*s with -twidth, width: cut to the width, padded to the field."""
    return f"{text[:width]:<{twidth}}"


def pdrecord(record: str, twidths: list[int], rectype: str,
             maxch: int = MAXCH_PERLINE) -> str:
    """ph_pdrecord: the record behind its type marker, broken into lines of
    whole fields no longer than maxch."""
    out = [f"{rectype:<2.2}"]
    nlines, ip, length = 1, 0, len(rectype)
    i = 0
    while i < len(twidths):
        length += twidths[i]
        if length < maxch:
            i += 1
            continue
        if length > maxch:
            length -= twidths[i]
            i -= 1
        chunk = length - len(COMMENTSTR)
        if nlines > 1:
            out.append(f"{CONTSTR:<2.2}")
        out.append(f"{record[ip:ip + chunk]:<{chunk}}\n")
        ip += chunk
        nlines += 1
        length = len(COMMENTSTR)
        i += 1
    if ip < len(record):
        out.append((CONTSTR if nlines > 1 else "") + record[ip:] + "\n")
    return "".join(out)


def header(title: str, names: list[str], widths: list[int]) -> str:
    """ph_whdr, the blank line that ends the header included."""
    tw = _twidths(widths)
    labels = "".join(_field(n, w, t) for n, w, t in zip(names, widths, tw, strict=True))
    sizes = "".join(f"{w:<{t}d}" for w, t in zip(widths, tw, strict=True))
    return (f"{COMMENTSTR}{KYCATALOG} {title}\n"
            f"{COMMENTSTR}{KYNCOLS} {len(names)}\n"
            f"{COMMENTSTR}{KYHDRLENGTH} {sum(tw)}\n"
            f"{COMMENTSTR}\n"
            + pdrecord(labels, tw, COMMENTSTR) + pdrecord(sizes, tw, COMMENTSTR) + "\n")


def data_record(values: list[str], widths: list[int]) -> str:
    """One record of ph_rdata: values missing at the end are INDEF; no
    values at all is a blank line."""
    if not values:
        return "\n"
    values = values + ["INDEF"] * (len(widths) - len(values))
    tw = _twidths(widths)
    record = "".join(_field(v, w, t) for v, w, t in zip(values, widths, tw, strict=True))
    return pdrecord(record, tw, BLANKSTR)


def format_file(names: list[str], widths: list[int]) -> str:
    """The catalog section for mkconfig: every column but the id, the
    names padded to the widest column."""
    imax = max(widths)
    return ("# Declare the catalog variables\n\ncatalog\n\n"
            + "".join(f"{n[:imax]:<{imax}}  {i}\n"
                      for i, n in enumerate(names[1:], start=2))
            + "\n")


def read_header(path: str | Path) -> tuple[str, list[str], list[int]] | None:
    """ph_open and ph_rhdr: the title, names and widths of a catalog made
    by mkcatalog, or None when it was not."""
    lines = Path(path).read_text(errors="replace", encoding="utf-8").split("\n")
    title, ncols, size = "", 0, 0
    k = 0
    while k < len(lines):
        words = lines[k].split()
        k += 1
        if not words or words[0][0] != "#":
            break
        if len(words) < 2:
            continue
        if KYCATALOG in words[1]:
            title = lines[k - 1].split(words[1], 1)[1].lstrip()
        elif KYNCOLS in words[1]:
            ncols = int(words[2]) if len(words) == 3 and words[2].isdigit() else 0
        elif KYHDRLENGTH in words[1]:
            size = int(words[2]) if len(words) == 3 and words[2].isdigit() else 0
            break
    if ncols <= 1:
        return None
    records = []
    while k < len(lines) and len(records) < 2:
        record = ""
        while k < len(lines):
            line = lines[k]
            k += 1
            if not line:
                return None
            if line == COMMENTSTR:
                continue
            record += line[2:] if line.startswith(CONTSTR) else line
            if len(record) > size + len(COMMENTSTR) - 1:
                break
        words = record.split()
        if not words or words[0][0] != "#":
            return None
        records.append(words[1:])
    if len(records) < 2 or len(records[0]) < ncols:
        return None
    try:
        widths = [int(w) for w in records[1][:ncols]]
    except ValueError:
        return None
    return title, records[0][:ncols], widths


def mkcatalog(path: str | Path, names: list[str] | None = None,
              widths: list[int] | None = None,
              records: list[list[str]] | None = None) -> Catalog:
    """Make the catalog (and its format file) or add to it. Raises
    ValueError with what is wrong."""
    title = str(path)             # the name as given, as IRAF titles it
    path = Path(path)
    records = records or []
    if path.exists():
        found = read_header(path)
        if found is None:
            raise ValueError(f"{path} was not created with mkcatalog")
        otitle, onames, owidths = found
        if title not in otitle:
            raise ValueError(f"the title {title} does not match the catalog's, {otitle}")
        if names and names != onames:
            raise ValueError(f"{path} already has the columns {', '.join(onames)}")
        catalog = Catalog(str(path), otitle, onames, owidths)
        text = ""
    else:
        if not names:
            raise ValueError(f"{path} does not exist and no columns were given")
        if len(names) < 2:
            raise ValueError(f"catalog {title} has no data columns")
        bad = [n for n in names if not legal_column(n)]
        if bad:
            raise ValueError("not a legal column name: " + ", ".join(bad))
        if widths is None:
            widths = [IDCOLWIDTH] + [COLWIDTH] * (len(names) - 1)
        if len(widths) != len(names):
            raise ValueError(f"{len(names)} columns but {len(widths)} widths")
        if any(not 1 <= w <= MAXCOLWIDTH for w in widths):
            raise ValueError(f"a column width must be 1 to {MAXCOLWIDTH}")
        catalog = Catalog(str(path), title, list(names), list(widths), created=True,
                          format_file=str(path.with_name(f"f{path.name}.dat")))
        text = header(title, catalog.names, catalog.widths)
    for values in records:
        if len(values) > len(catalog.names):
            raise ValueError(f"{len(values)} values for {len(catalog.names)} columns: "
                             + " ".join(values))
        catalog.truncated += [v for v, w in zip(values, catalog.widths, strict=False)
                              if len(v) > w]
        text += data_record(values, catalog.widths)
        catalog.nrecords += bool(values)
    if catalog.created:
        path.write_text(text, encoding="utf-8")
        Path(catalog.format_file).write_text(format_file(catalog.names, catalog.widths),
                                             encoding="utf-8")
    else:
        with path.open("a", encoding="utf-8") as out:
            out.write(text)
    return catalog


# ------------------------------------------------- typed at the terminal
#
# t_catalog.x as IRAF runs it at the terminal: a new catalog's columns asked for
# one by one (ph_setcols), the records value by value (ph_rdata), and an old
# catalog's records reviewed one by one before more are added (ph_review). A
# Terminal (entry.py) is where the prompts go and the lines come from; gargstr
# takes a line as it is typed, its leading blanks included.


def ask_columns(term, verify: bool = False) -> tuple[list[str], list[int]]:
    """ph_setcols: the id column, then each data column and its error column, a
    name and a width each, until the end of the input."""
    names: list[str] = []
    widths: list[int] = []
    errcol = False
    term.printf("\n")
    while True:
        n = len(names)
        if n == 0:
            term.printf("Enter the id column name (name, <CR>=ID, <EOF>=quit entry): ")
        elif not errcol:
            term.printf(f"Enter a name for column {n + 1} (name, <CR>=COL{n + 1}, "
                        "<EOF>=quit entry): ")
        else:
            term.printf(f"Enter a name for error column {n + 1} (name, "
                        f"<CR>=error({names[-1]}), <->=skip): ")
        got = term.scan()
        if got is None:
            term.printf("\n")
            break
        name = (got.wrd() or "")[:MAXCOLWIDTH + 2]
        if not name:
            name = "ID" if n == 0 else (f"COL{n + 1}" if not errcol
                                        else f"error({names[-1]})")
        elif not is_ident(name) and "error(" not in name:
            if name.startswith("-"):
                errcol = False
            else:
                term.printf(f"{name} is not a legal column name - skipping\n")
            continue
        if n == 0:
            term.printf(f"\tEnter width of id column (width, <CR>={IDCOLWIDTH}): ")
        else:
            term.printf(f"\tEnter width of column {n + 1} (width, <CR>={COLWIDTH}): ")
        width = IDCOLWIDTH if n == 0 else COLWIDTH
        got = term.scan()
        if got is not None:
            value = _int(got)
            if value is not None:
                width = value
        names.append(name)
        widths.append(width)
        errcol = len(names) > 1 and not errcol
    term.printf("\n")
    if not verify or len(names) < 2:
        return names, widths
    for i in range(len(names)):
        term.printf(f"Verify name of column {i + 1} ({names[i]}) (name, <CR>=ok): ")
        old = names[i]
        got = term.scan()
        if got is None:
            term.printf("\n")
        else:
            new = got.wrd()
            if new and (is_ident(new) or "error(" in new):
                names[i] = new
        term.printf(f"\tOld name: {old}  New name: {names[i]}\n")
        term.printf(f"Verify the width of column {i + 1} ({widths[i]}) (width, <CR>=ok): ")
        old_width = widths[i]
        got = term.scan()
        if got is not None:
            value = _int(got)
            if value is not None:
                widths[i] = value
        term.printf(f"\tOld width: {old_width}  New width: {widths[i]}\n")
    term.printf("\n")
    return names, widths


def _int(got) -> int | None:
    """gargi: an integer, the word it starts."""
    import re

    word = got.wrd()
    found = re.match(r"[-+]?\d+", word or "")
    return int(found.group()) if found else None


def _line(got) -> str:
    """gargstr: the rest of the line, as typed."""
    return got.line if got is not None else ""


def ask_records(term, names: list[str], widths: list[int], verify: bool = False) -> str:
    """ph_rdata: records typed value by value, a blank id a blank line, the end of the
    input the end -- a record it cuts short is not written."""
    tw = _twidths(widths)
    out: list[str] = []
    while True:
        values: list[str] = []
        ended = False
        for i, name in enumerate(names):
            if i == 0:
                term.printf(f"\nEnter {name} (value, <CR>=blank line, <EOF>=quit entry): ")
            else:
                term.printf(f"\tEnter {name} (value, <CR>=INDEF): ")
            got = term.scan()
            if got is None:
                ended = True
                if i == 0:
                    break
                value = ""
            else:
                value = _line(got)
            if not value:
                if i == 0:
                    break
                value = "INDEF"
            values.append(value)
        if ended:
            break
        if not values:
            out.append("\n")
            continue
        record = "".join(_field(v, w, t) for v, w, t in zip(values, widths, tw, strict=True))
        if verify:
            fields = record.split()
            kept = []
            for i, name in enumerate(names):
                if i >= len(fields):
                    break
                old = fields[i]
                lead = "\n" if i == 0 else "\t"
                term.printf(f"{lead}Verify {name} ({old}) (value, <CR>=ok): ")
                got = term.scan()
                if got is None:
                    term.printf("\n")
                    new = ""
                else:
                    new = _line(got)
                if not new:
                    new = old
                else:
                    term.printf(f"\t{name} old: {old} new: {new}\n")
                kept.append(new)
            record = "".join(_field(v, w, t) for v, w, t in
                             zip(kept, widths, tw, strict=False))
        out.append(pdrecord(record, tw, BLANKSTR))
    term.printf("\n")
    return "".join(out)


def data_lines(path: str | Path) -> list[str]:
    """The lines after the header: what ph_rhdr leaves for ph_gdrecord."""
    lines = Path(path).read_text(errors="replace", encoding="utf-8").split("\n")
    if lines and lines[-1] == "":
        lines.pop()
    k = 0
    while k < len(lines) and lines[k].split()[:1] == ["#"]:
        k += 1
    return lines[k + 1:]


def records(lines: list[str]) -> list[str]:
    """ph_gdrecord: the records, a line and the continuation lines (* ) after it."""
    out: list[str] = []
    for line in lines:
        if line.startswith(CONTSTR) and out:
            out[-1] += line[2:]
        else:
            out.append(line)
    return out


def review(term, lines: list[str], names: list[str], widths: list[int]) -> str:
    """ph_review: each record shown value by value, to be kept or changed; the end of
    the input keeps the rest as they were."""
    tw = _twidths(widths)
    out: list[str] = []
    pending = records(lines)
    while pending:
        record = pending.pop(0)
        if not record:
            out.append("\n")
            continue
        if record.startswith(COMMENTSTR):
            out.append(record + "\n")
            continue
        fields = record.split()
        kept: list[str] = []
        ended = False
        for i, name in enumerate(names):
            if i >= len(fields):
                break
            old = fields[i]
            if i == 0:
                term.printf(f"\nReview OBJECT {name} ({old}) (value, <CR>=ok, "
                            "<EOF>=quit review): ")
            else:
                term.printf(f"\tReview {name} ({old}) (value, <CR>=ok): ")
            got = term.scan()
            if got is None:
                ended = True
                if i == 0:
                    break
                new = ""
            else:
                new = _line(got)
            if not new:
                new = old
            else:
                term.printf(f"\t{name} old: {old} new: {new}\n")
            kept.append(new)
        if ended:
            out.append(pdrecord(record[len(COMMENTSTR):], tw, BLANKSTR))
            break
        if not kept:
            out.append("\n")
        else:
            out.append(pdrecord("".join(_field(v, w, t) for v, w, t in
                                        zip(kept, widths, tw, strict=False)), tw, BLANKSTR))
    for record in pending:
        out.append("\n" if not record else pdrecord(record[len(COMMENTSTR):], tw, BLANKSTR))
    term.printf("\n")
    return "".join(out)


def session(path: str | Path, term, review_records: bool = False,
            verify: bool = False) -> tuple[Catalog | None, list[str]]:
    """t_catalog at the terminal: what it says on the standard error comes back, as
    written there; the catalog is None when there is none to add to."""
    notes: list[str] = []
    title = str(path)
    path = Path(path)
    if not path.exists():
        fmt = path.with_name(f"f{path.name}.dat")
        names, widths = ask_columns(term, verify)
        if len(names) <= 1:
            missing = "no id column" if not names else "no data columns"
            notes.append(f"\n<Error> Catalog {title} in file {path} has {missing}\n\n")
            path.write_text("", encoding="utf-8")
            fmt.write_text("", encoding="utf-8")
            return None, notes
        term.printf(f"\nCatalog {title} in file {path} has {len(names)} columns\n")
        for i, (name, width) in enumerate(zip(names, widths, strict=True), start=1):
            term.printf(f"\tColumn {i}:  {_field(name, width, width)}\n")
        term.printf("\n")
        text = header(title, names, widths) + ask_records(term, names, widths, verify)
        path.write_text(text, encoding="utf-8")
        fmt.write_text(format_file(names, widths), encoding="utf-8")
        return Catalog(str(path), title, names, widths, created=True,
                       format_file=str(fmt)), notes
    found = read_header(path)
    if found is None:
        notes.append(f"\n<Warning> File {path} was not created with MKCATALOG\n")
        return None, notes
    otitle, names, widths = found
    if title not in otitle:
        notes.append(f"\n<Error> The input title {title} does not match the catalog "
                     f"title  {otitle}.\n")
        return None, notes
    term.printf(f"\nCatalog {title} in file {path} has {len(names)} columns\n")
    for i, name in enumerate(names, start=1):
        term.printf(f"\tColumn {i}:  {name}\n")
    term.printf("\n")
    if review_records:
        text = (header(title, names, widths) + review(term, data_lines(path), names, widths)
                + ask_records(term, names, widths, verify))
        path.write_text(text, encoding="utf-8")
    else:
        with path.open("a", encoding="utf-8") as out:
            out.write(ask_records(term, names, widths, verify))
    return Catalog(str(path), otitle, names, widths), notes
