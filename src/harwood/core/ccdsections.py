"""ccdproc's arithmetic: its sections, the overscan vector and the conversions of a
calibration, in real and in IRAF's order.

From iraf-community 2.18: noao/imred/ccdred/src/{setsections,settrim,setoverscan,setzero,
readcor,scancor}.x and generic/proc.x. See LICENSES/IRAF-COPYRIGHT.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import advice
from .jit import jit as _jit

REAL = np.float32


def section(text, c1: int, c2: int, l1: int, l2: int) -> tuple[int, int, int, int, int, int]:
    """ccd_section: [c1:c2:cs,l1:l2:ls], and what it leaves out stays as given. Steps
    last, for the caller to refuse."""
    cs = ls = 1
    body = str(text or "").strip()
    if not body:
        return c1, c2, l1, l2, cs, ls
    if not (body.startswith("[") and body.endswith("]")):
        advice.fail(f"error in the section '{text}'", "write it as [x1:x2,y1:y2]")
    parts = body[1:-1].split(",")
    values = [[c1, c2, cs], [l1, l2, ls]]
    for axis, part in enumerate(parts[:2]):
        pieces = part.strip().split(":")
        low, high, step = values[axis]
        try:
            head = pieces[0].strip()
            if head in ("*", ""):
                rest = pieces[1:]
            elif head == "-*":
                low, high = high, low
                rest = pieces[1:]
            else:
                low = int(head)
                high = int(pieces[1]) if len(pieces) > 1 else low
                rest = pieces[2:]
            if rest:
                step = int(rest[0])
        except (ValueError, IndexError):
            advice.fail(f"error in the section '{text}'", "write it as [x1:x2,y1:y2]")
        values[axis] = [low, high, step]
    return (values[0][0], values[0][1], values[1][0], values[1][1], values[0][2], values[1][2])


@dataclass
class Sections:
    """IN, OUT and CCD of the structure ccdproc keeps, after set_sections and set_trim."""

    nc: int
    nl: int
    in_c1: int
    in_c2: int
    in_l1: int
    in_l2: int
    out_c1: int
    out_c2: int
    out_l1: int
    out_l2: int
    ccd_c1: int
    ccd_c2: int
    ccd_l1: int
    ccd_l2: int
    trim_c1: int
    trim_c2: int
    trim_l1: int
    trim_l2: int
    bias_c1: int
    bias_c2: int
    bias_l1: int
    bias_l2: int
    trimmed: bool = False


def sections(header, shape, biassec, trimsec) -> Sections:
    """set_sections."""
    nl, nc = shape
    c1, c2, l1, l2, cs, ls = section(header.get("DATASEC", ""), 1, nc, 1, nl)
    if c1 < 1 or c2 > nc or l1 < 1 or l2 > nl or cs != 1 or ls != 1:
        advice.fail(f"error in the DATASEC {header.get('DATASEC')}",
                    f"the image is {nc} by {nl}: a section of it, with no step")
    in_c1, in_c2, in_l1, in_l2 = c1, c2, l1, l2
    tc1, tc2, tl1, tl2, cs, ls = section(trimsec, in_c1, in_c2, in_l1, in_l2)
    if cs != 1 or ls != 1:
        advice.fail(f"error in the trim section {trimsec}", "a section with no step")
    bc1, bc2, bl1, bl2, cs, ls = section(biassec, 1, nc, 1, nl)
    if cs != 1 or ls != 1:
        advice.fail(f"error in the bias section {biassec}", "a section with no step")
    cc1, cc2, cl1, cl2, cs, ls = section(header.get("CCDSEC", ""), 1, in_c2 - in_c1 + 1, 1,
                                         in_l2 - in_l1 + 1)
    if cs != 1 or ls != 1:
        advice.fail(f"error in the CCDSEC {header.get('CCDSEC')}", "a section with no step")
    if in_c2 - in_c1 != cc2 - cc1 or in_l2 - in_l1 != cl2 - cl1:
        advice.fail("the sizes of DATASEC and CCDSEC do not agree",
                    f"{header.get('DATASEC')} and {header.get('CCDSEC')}")
    return Sections(nc, nl, in_c1, in_c2, in_l1, in_l2, in_c1, in_c2, in_l1, in_l2, cc1, cc2,
                    cl1, cl2, tc1, tc2, tl1, tl2, bc1, bc2, bl1, bl2)


def trim(sec: Sections) -> None:
    """set_trim, after checking the section is inside the image."""
    if sec.trim_c1 < 1 or sec.trim_c2 > sec.nc or sec.trim_l1 < 1 or sec.trim_l2 > sec.nl:
        advice.fail(f"error in the trim section: the image is {sec.nc} by {sec.nl}, the "
                    f"section [{sec.trim_c1}:{sec.trim_c2},{sec.trim_l1}:{sec.trim_l2}]",
                    "give one inside it, with --trimsec")
    xt1 = max(0, sec.trim_c1 - sec.in_c1)
    xt2 = min(0, sec.trim_c2 - sec.in_c2)
    yt1 = max(0, sec.trim_l1 - sec.in_l1)
    yt2 = min(0, sec.trim_l2 - sec.in_l2)
    sec.ccd_c1 += xt1
    sec.ccd_c2 += xt2
    sec.ccd_l1 += yt1
    sec.ccd_l2 += yt2
    sec.in_c1 += xt1
    sec.in_c2 += xt2
    sec.in_l1 += yt1
    sec.in_l2 += yt2
    sec.out_c1 = sec.in_c1 - sec.trim_c1 + 1
    sec.out_c2 = sec.in_c2 - sec.trim_c1 + 1
    sec.out_l1 = sec.in_l1 - sec.trim_l1 + 1
    sec.out_l2 = sec.in_l2 - sec.trim_l1 + 1
    sec.trimmed = True


def update_header(header, sec: Sections, shape) -> None:
    """set_header's sections, and the physical system set_sections and mw_shift leave."""
    nl, nc = shape
    if (sec.out_c1, sec.out_c2, sec.out_l1, sec.out_l2) != (1, nc, 1, nl):
        header["DATASEC"] = f"[{sec.out_c1}:{sec.out_c2},{sec.out_l1}:{sec.out_l2}]"
    else:
        header.remove("DATASEC", ignore_missing=True)
    header["CCDSEC"] = f"[{sec.ccd_c1}:{sec.ccd_c2},{sec.ccd_l1}:{sec.ccd_l2}]"
    moved = {1: sec.in_c1 - sec.ccd_c1, 2: sec.in_l1 - sec.ccd_l1}
    if sec.trimmed:
        header.remove("TRIMSEC", ignore_missing=True)
        header.remove("BIASSEC", ignore_missing=True)
        c1 = max(1, sec.bias_c1 - sec.trim_c1 + 1)
        c2 = min(nc, sec.bias_c2 - sec.trim_c1 + 1)
        l1 = max(1, sec.bias_l1 - sec.trim_l1 + 1)
        l2 = min(nl, sec.bias_l2 - sec.trim_l1 + 1)
        if c1 <= c2 and l1 <= l2:
            header["BIASSEC"] = f"[{c1}:{c2},{l1}:{l2}]"
    shift = {1: 1 - sec.in_c1 if sec.trimmed else 0, 2: 1 - sec.in_l1 if sec.trimmed else 0}
    for axis in (1, 2):
        old = float(header.get(f"LTV{axis}", 0.0))
        ltv = moved[axis] + shift[axis]
        if f"CRPIX{axis}" in header:
            header[f"CRPIX{axis}"] = float(header[f"CRPIX{axis}"]) - old + ltv
        if ltv or f"LTV{axis}" in header:
            header[f"LTV{axis}"] = float(ltv)


def asum(values) -> np.float32:
    """asumr: the sum of a line, in real, in order."""
    return REAL(np.add.accumulate(np.asarray(values, REAL), dtype=REAL)[-1])


def overscan_vector(raw, sec: Sections, line_axis: bool) -> tuple[np.ndarray, np.ndarray]:
    """setoverscan's vector of the means across the strip, trimmed: (x, values). Along
    the lines with readaxis line, along the columns with readaxis column; the strip's
    limits in the long way are replaced by the trim's."""
    raw = np.asarray(raw, REAL)
    if line_axis:
        navg = sec.bias_c2 - sec.bias_c1 + 1
        strip = raw[:, sec.bias_c1 - 1:sec.bias_c2]
        values = np.add.accumulate(strip, axis=1, dtype=REAL)[:, -1]
        start, stop = sec.in_l1, sec.in_l2
    else:
        navg = sec.bias_l2 - sec.bias_l1 + 1
        strip = raw[sec.bias_l1 - 1:sec.bias_l2, :]
        values = np.add.accumulate(strip, axis=0, dtype=REAL)[-1, :]
        start, stop = sec.in_c1, sec.in_c2
    if navg > 1:
        values = (values / REAL(navg)).astype(REAL)
    x = np.arange(start, stop + 1, dtype=REAL)
    return x, np.array(values[start - 1:stop], REAL)


def find_overscan(strip, kind: str) -> np.ndarray:
    """find_overscan, a value for each line of the strip[lines, columns]: its mean, its
    median, or its mean without the highest and the lowest."""
    strip = np.asarray(strip, REAL)
    npix = strip.shape[1]
    if kind == "median":
        return np.sort(strip, axis=1)[:, (npix + 1) // 2 - 1]
    total = np.add.accumulate(strip, axis=1, dtype=REAL)[:, -1]
    if kind == "mean":
        return (total / REAL(npix)).astype(REAL)
    if npix <= 2:
        advice.fail("the minmax overscan needs more than two columns in --biassec",
                    "use mean, or a wider strip")
    low, high = strip.min(axis=1), strip.max(axis=1)
    return (((total - low).astype(REAL) - high).astype(REAL) / REAL(npix - 2)).astype(REAL)


def cal_region(cal_shape, cal_header, sec: Sections, name: str):
    """What set_zero, set_dark, set_flat, set_illum and set_fringe work out: the slices of
    a calibration for this frame's CCD section, in numpy order. One line, or one column,
    goes for the whole frame that way, its slice just that one."""
    nl, nc = cal_shape
    c1, c2, l1, l2, cs, ls = section(cal_header.get("DATASEC", ""), 1, nc, 1, nl)
    if c1 < 1 or c2 > nc or l1 < 1 or l2 > nl or cs != 1 or ls != 1:
        advice.fail(f"data section error: image={name}[{nc},{nl}], "
                    f"datasec=[{c1}:{c2},{l1}:{l2}]", "fix the DATASEC of the calibration")
    data_c1, data_l1 = c1, l1
    c1, c2, l1, l2, cs, ls = section(cal_header.get("CCDSEC", ""), c1, c2, l1, l2)
    if nc == 1:
        c1, c2 = sec.ccd_c1, sec.ccd_c2
    if nl == 1:
        l1, l2 = sec.ccd_l1, sec.ccd_l2
    if c1 > sec.ccd_c1 or c2 < sec.ccd_c2 or l1 > sec.ccd_l1 or l2 < sec.ccd_l2:
        advice.fail(f"CCD section error: input=[{sec.ccd_c1}:{sec.ccd_c2},{sec.ccd_l1}:"
                    f"{sec.ccd_l2}], {name}=[{c1}:{c2},{l1}:{l2}]",
                    "the calibration has to cover the frame: was it trimmed more?")
    first_c = sec.ccd_c1 - c1 + data_c1
    last_c = sec.ccd_c2 - c1 + data_c1
    first_l = sec.ccd_l1 - l1 + data_l1
    last_l = sec.ccd_l2 - l1 + data_l1
    return (slice(0, 1) if nl == 1 else slice(first_l - 1, last_l),
            slice(0, 1) if nc == 1 else slice(first_c - 1, last_c))


def readcor(data, header, line_axis: bool):
    """readcor: a zero averaged across the readout axis, into one line or one column."""
    nl, nc = data.shape
    c1, c2, l1, l2, cs, ls = section(header.get("DATASEC", ""), 1, nc, 1, nl)
    if c1 < 1 or c2 > nc or l1 < 1 or l2 > nl or cs != 1 or ls != 1:
        advice.fail("error in the DATASEC of the zero", "a section of it, with no step")
    in_c1, in_c2, in_l1, in_l2 = c1, c2, l1, l2
    c1, c2, l1, l2, cs, ls = section(header.get("CCDSEC", ""), c1, c2, l1, l2)
    if cs != 1 or ls != 1:
        advice.fail("error in the CCDSEC of the zero", "a section with no step")
    if in_c2 - in_c1 != c2 - c1 or in_l2 - in_l1 != l2 - l1:
        advice.fail("the sizes of DATASEC and CCDSEC do not agree", "fix them in the header")
    data = np.asarray(data, REAL)
    if line_axis:
        out = np.zeros((1, nc), REAL)
        total = np.add.accumulate(data[in_l1 - 1:in_l2, in_c1 - 1:in_c2], axis=0,
                                  dtype=REAL)[-1]
        out[0, in_c1 - 1:in_c2] = (total / REAL(in_l2 - in_l1 + 1)).astype(REAL)
        header["DATASEC"] = f"[{in_c1}:{in_c2},1:1]"
        header["CCDSEC"] = f"[{c1}:{c2},*]"
        return out
    out = np.zeros((nl, 1), REAL)
    width = in_c2 - in_c1 + 1
    inner = data[in_l1 - 1:in_l2, in_c1 - 1:in_c2]
    total = np.add.accumulate(inner, axis=1, dtype=REAL)[:, -1]
    out[in_l1 - 1:in_l2, 0] = (total / REAL(width)).astype(REAL)
    header["DATASEC"] = f"[1:1,{in_l1}:{in_l2}]"
    header["CCDSEC"] = f"[*,{l1}:{l2}]"
    return out


@_jit
def _shortscan_column(line, out, nscani, nl, replace, minreplace, mean):
    """scancor's shortscan along a column: one line. Its loops go by the number of lines,
    where the columns are meant, and only reach as far as the line goes."""
    real = np.float32
    nscanr = real(nscani)
    total = real(0.0)
    j = 0
    k = 0
    while j < nscani:
        total = real(total + real(line[j]))
        j += 1
    value = real(total / nscanr)
    out[0] = max(real(minreplace), value) if replace else value
    mean = real(mean + out[0])
    at = 1
    limit = min(nl, line.size)
    while j < limit:
        total = real(real(total + real(line[j])) - real(line[k]))
        value = real(total / nscanr)
        out[at] = max(real(minreplace), value) if replace else value
        mean = real(mean + out[at])
        j += 1
        k += 1
        at += 1
    while at < limit:
        total = real(total - real(line[k]))
        value = real(total / nscanr)
        out[at] = max(real(minreplace), value) if replace else value
        mean = real(mean + out[at])
        k += 1
        at += 1
    return mean


def shortscan(data, header, nscan: int, minreplace, line_axis: bool):
    """scancor's shortscan: the moving average of the scan, with the ramps at each end.
    (the image, its CCDMEAN). IRAF's nscan here is the steps, and one more makes the rows."""
    len2, len1 = data.shape
    c1, c2, l1, l2, cs, ls = section(header.get("DATASEC", ""), 1, len1, 1, len2)
    if c1 < 1 or c2 > len1 or l1 < 1 or l2 > len2 or cs != 1 or ls != 1:
        advice.fail("error in the DATASEC of the calibration", "a section with no step")
    nc, nl = c2 - c1 + 1, l2 - l1 + 1
    data = np.asarray(data, REAL)
    out = data.copy()
    replace = minreplace is not None
    floor = REAL(minreplace if replace else 0.0)
    mean = REAL(0.0)
    if line_axis:
        nscani = max(1, min(nscan, nl) + 1)
        nscanr = REAL(nscani)
        running = np.zeros(nc, REAL)
        lines = data[l1 - 1:l2, c1 - 1:c2]

        def emit(where):
            nonlocal mean
            value = (running / nscanr).astype(REAL)
            if replace:
                value = np.maximum(value, floor)
            out[where + l1 - 1, c1 - 1:c2] = value
            mean = REAL(mean + asum(value))

        for j in range(nscani):
            running = (running + lines[j]).astype(REAL)
        emit(0)
        where = 1
        k = 0
        for j in range(nscani, nl):
            running = (running - lines[k]).astype(REAL)
            running = (running + lines[j]).astype(REAL)
            emit(where)
            k += 1
            where += 1
        while where < nl:
            running = (running - lines[k]).astype(REAL)
            emit(where)
            k += 1
            where += 1
    else:
        nscani = max(1, min(nscan, nc) + 1)
        for i in range(nl):
            start = 2 * (c1 - 1)
            line = data[i + l1 - 1, start:]
            target = out[i + l1 - 1, start:]
            mean = REAL(_shortscan_column(line, target, nscani, nl, replace, floor, mean))
    return out, float(REAL(REAL(mean / REAL(nc)) / REAL(nl)))


def longscan(data, header, line_axis: bool):
    """scancor's longscan: one line or one column across the readout, and a CCDMEAN that
    divides by the other axis, as IRAF's does."""
    nl, nc = data.shape
    c1, c2, l1, l2, _, _ = section(header.get("DATASEC", ""), 1, nc, 1, nl)
    out = readcor(data, header, line_axis)
    if line_axis:
        mean = REAL(asum(out[0, c1 - 1:c2]) / REAL(l2 - l1 + 1))
    else:
        mean = REAL(asum(out[:l2 - l1 + 1, 0]) / REAL(c2 - c1 + 1))
    return out, float(mean)


def mean_of(data) -> float:
    """ccdmean: the sum of the lines' sums, in real, over the size."""
    nl, nc = data.shape
    total = REAL(0.0)
    for line in np.asarray(data, REAL):
        total = REAL(total + asum(line))
    return float(REAL(total / REAL(nc * nl)))
