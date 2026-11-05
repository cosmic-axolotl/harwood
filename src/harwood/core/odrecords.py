"""onedspec's record format: a root name and record numbers, root.0001, root.0002...,
the names of the spectra IRAF's old instruments wrote one to a file.

From noao/onedspec/odropenp.x and onedspec's own copy of xtools' ranges.x
(decode_ranges, get_next_number), which is not the rng_open of aperture lists: the
numbers are whole, from 0, given as 1-5, 7, 10-20x2, and come out in increasing
order, each once, whatever order they were given in. More than a thousand of them
is an error, so a range left open, 998-, is too. Of a root's extension only one of
three letters ending in h, .imh, or two ending in l, .pl, is taken off -- .fits
stays: root.fits.0001.

Translated from IRAF, Copyright (c) 1986 Association of Universities for Research in
Astronomy Inc.; the notice it is distributed under is in LICENSES/IRAF-COPYRIGHT.
"""

from __future__ import annotations

from . import advice

FIRST, LAST, STEP, EOLIST = 0, 2 ** 31 - 1, 1, -1     #: onedspec's ranges.x
MAXRECS = 1000


def _ctoi(text: str, ip: int) -> tuple[int | None, int]:
    start = ip
    while ip < len(text) and text[ip].isdigit():
        ip += 1
    return (int(text[start:ip]) if ip > start else None), ip


def decode_ranges(text: str, max_ranges: int = 100) -> tuple[list[tuple[int, int, int]], int]:
    """decode_ranges: the (first, last, step) triples and how many values they hold."""
    text = str(text)
    ip, nvalues, out = 0, 0, []

    def skip(ip):
        while ip < len(text) and (text[ip] in " \t" or text[ip] == ","):
            ip += 1
        return ip

    def at(ip):
        return text[ip] if ip < len(text) else ""

    for nrange in range(1, max_ranges):
        first, last, step = FIRST, LAST, STEP
        ip = skip(ip)
        if at(ip) == "":
            if nrange == 1:
                return [(first, last, step)], LAST
            return out, nvalues
        if at(ip).isdigit():
            first, ip = _ctoi(text, ip)
        elif at(ip) not in "-x":
            _bad()
        ip = skip(ip)
        if at(ip) == "x":
            pass
        elif at(ip) == "-":
            ip = skip(ip + 1)
            if at(ip).isdigit():
                last, ip = _ctoi(text, ip)
            elif at(ip) not in ("", "x"):
                _bad()
        else:
            last = first
        ip = skip(ip)
        if at(ip) == "x":
            ip = skip(ip + 1)
            if at(ip).isdigit():
                value, ip = _ctoi(text, ip)
                step = value if value is not None else step
            elif at(ip) not in ("", "-"):
                _bad()
        out.append((first, last, step))
        nvalues += abs(last - first) // step + 1
    _bad()


def _bad():
    advice.fail("Bad record specification",
                "give the records as 1-5,7 or 10-20x2: whole numbers, from 1.")


def get_next_number(ranges, number: int) -> int | None:
    """get_next_number: the smallest number of the list above `number`, or None."""
    number += 1
    best = None
    for first, last, step in ranges:
        if first == EOLIST:
            break
        lo, hi = min(first, last), max(first, last)
        if step == 0:
            _bad()                  # IRAF's, unchecked, divides by it
        if lo <= number <= hi:
            remainder = (number - lo) % step
            if remainder == 0:
                return number
            if number - remainder + step <= hi:
                best = number - remainder + step
        elif lo > number:
            best = lo if best is None else min(best, lo)
    return best


def _root(name: str) -> str:
    """imgimage and odr_openp: the section off, and a .?h or .?l extension."""
    name = name.split("[")[0]
    if len(name) > 3 and name.endswith("h") and name[-4] == ".":
        return name[:-4]
    if len(name) > 2 and name.endswith("l") and name[-3] == ".":
        return name[:-3]
    return name


def names(roots: list[str], records: str) -> list[str]:
    """odr_openp: each root with each record, root.0001; the roots themselves when no
    records are given."""
    if not str(records).strip():
        return list(roots)
    ranges, count = decode_ranges(records)
    if count > MAXRECS:
        advice.fail("Too many records", f"give at most {MAXRECS} record numbers.")
    out = []
    for root in roots:
        stem, rec = _root(root), -1
        while True:
            rec = get_next_number(ranges, rec)
            if rec is None:
                break
            out.append(f"{stem}.{rec:04d}")
    return out
