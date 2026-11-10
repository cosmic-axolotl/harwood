"""imutil -- the small tasks a reduction leans on constantly.

`imheader`, `hselect`, `imstatistics`, `imarith`. None of them is hard; all of
them are what turns a directory of FITS files into a data set you can reason
about. `hselect` in particular has no command-line equivalent in the Python
ecosystem, and it is how a night of observing gets sorted.
"""

from __future__ import annotations

import math
import re
from pathlib import Path
from typing import Literal

from pydantic import Field

from ..core import advice
from ..core.registry import register
from ..core.task import (
    Step,
    Task,
    TaskParams,
    TaskResult,
    Template,
    format_field,
    uncertainty_field,
)

WHY = {
    "read headers": "Only the headers are read, not the pixels, so this is quick "
                    "over a whole night. The header is where the telescope wrote "
                    "down what it was doing, and it is the only record of that.",
    "print": "The summary picks the keywords a reduction turns on -- image type, "
             "exposure, filter, object -- because those are what decide which "
             "frame is which.",
    "evaluate": "A boolean expression over the header, evaluated per frame. This "
                "is how a night gets sorted: 'IMAGETYP == \'zero\'' finds the "
                "biases, 'EXPTIME > 30 and FILTER == \'V\'' finds the long V "
                "exposures. The namespace holds the header's own keywords and "
                "nothing else -- a header expression is a filter, not a place to "
                "run code.",
    "select": "The keywords asked for, for every frame that matched.",
    "measure": "Statistics over the pixels. The median resists a bright star "
               "where the mean does not; the mode is the most common value and is "
               "what a flat field is normalised by; the spread says how much "
               "noise there is to work against.",
    "threshold": "Pixels outside these limits are left out of every statistic. "
                 "Saturated pixels carry no information and would otherwise drag "
                 "the mean and inflate the spread.",
    "clip": "Iterative rejection before measuring, so a handful of stars do not "
            "decide what the sky level is.",
    "compute": "Arithmetic pixel by pixel. A constant operand applies everywhere; "
               "an image operand has to be the same shape, because two frames of "
               "different sizes are not two frames of the same thing.",
    "copy": "Pixels and header together. A section copies part of a frame, which "
            "is how a subraster becomes an image in its own right.",
    "delete": "Files go for good. The confirmation exists because a pattern that "
              "matches more than was meant is how a night's work disappears.",
    "expand": "What an image template actually matches, before anything is run "
              "on it. Worth doing first when the template renames things.",
    "write": "Written as 32-bit floating point, whatever came in. A reduced frame "
             "holds fractions of a count and negative sky residuals, and neither "
             "fits in the integers a detector writes.",
    "plot": "A cut through the image, plotted. This is how the sections get "
            "decided: you look at where the illuminated pixels stop and the "
            "overscan begins, rather than trusting that the header is right "
            "about it.",
    "expression": "The expression is evaluated once for each line of the output. An image "
                  "with fewer lines starts over, so a line is applied to a whole frame, "
                  "and a function of a vector, as median or hiv, is a function of a line.",
    "expression_type": "The output's pixels are the expression's type: short * short wraps "
                       "at 16 bits as it does in IRAF, and --intype real or --outtype say "
                       "otherwise.",
    "extremes": "The highest pixel says whether anything saturated, the lowest whether "
                "a bias went wrong. They come from the header when it has them, "
                "because reading every pixel of a night takes time.",
    "list": "A pixel per line is what another program can read, or what a column of "
            "numbers in a script needs. A section keeps it short.",
}


class ImHeaderParams(TaskParams):
    input: list[str] = Field(..., description="Images to inspect. Accepts globs and @lists.")
    longheader: bool = Field(
        False, description="Print the whole header instead of a one-line summary."
    )
    keywords: list[str] | None = Field(
        None, description="Print only these keywords, in this order."
    )


@register
class ImHeader(Task[ImHeaderParams]):
    name = "imheader"
    package = "imutil"
    description = "Print image headers, or selected keywords."
    Params = ImHeaderParams

    def plan(self, params: ImHeaderParams) -> list[Step]:
        return [Step("read headers", f"{len(params.input)} patterns",
                     why=WHY["read headers"]),
                Step("print", "whole header" if params.longheader else "summary",
                     why=WHY["print"])]

    def execute(self, params: ImHeaderParams) -> TaskResult:
        from ..core import images

        rows = []
        for spec in images.expand(params.input):
            hdul, hdu = images.open_hdu(spec)
            try:
                header = images.inherited(hdul, hdu)
                if params.keywords:
                    row = {key: header.get(key) for key in params.keywords}
                elif params.longheader:
                    row = dict(header.items())
                else:
                    shape = "x".join(str(n) for n in reversed(hdu.data.shape)) \
                        if getattr(hdu, "data", None) is not None else "no pixels"
                    row = {"shape": shape, "type": header.get("IMAGETYP", ""),
                           "exptime": header.get("EXPTIME", ""),
                           # FILTERS is what KPNO wrote; FILTER is what most
                           # else writes. Looking for only one of them leaves
                           # the column blank on half the data in the world.
                           "filter": _first(header, "FILTER", "FILTERS",
                                            "FILTER1", "INSFILTE"),
                           "object": header.get("OBJECT", "")}
            finally:
                hdul.close()
            rows.append({"image": str(spec), **row})
        return TaskResult(data={"rows": rows, "images": len(rows)})


class HSelectParams(TaskParams):
    input: list[str] = Field(..., description="Images to consider. Accepts globs and @lists.")
    fields: list[str] = Field(
        ...,
        description="Keywords to print for each image that matches, e.g. "
                    "'$I,OBJECT,EXPTIME'; $I is the image's name, as in IRAF.",
    )
    expr: str = Field(
        "yes",
        description=(
            "Boolean expression over the header, evaluated as IRAF evaluates it: "
            "\"imagetyp == 'zero'\", \"exptime > 30 && filters == 'V'\", "
            "\"object ?= 'M92*'\". Keywords in any case; 'yes' selects everything."
        ),
    )
    missing: str = Field(
        "INDEF", description="Printed for a keyword the header does not have."
    )


@register
class HSelect(Task[HSelectParams]):
    name = "hselect"
    package = "imutil"
    description = "Select images by a boolean expression over their headers."
    Params = HSelectParams

    def plan(self, params: HSelectParams) -> list[Step]:
        return [Step("evaluate", f"{params.expr!r} over each header",
                     why=WHY["evaluate"]),
                Step("print", ", ".join(params.fields), why=WHY["select"])]

    def execute(self, params: HSelectParams) -> TaskResult:
        from ..core import evexpr, images

        expr = _iraf_words(params.expr)
        matched, lines, absent, skipped = [], [], set(), 0
        for spec in images.expand(params.input):
            header = images.read_header(spec)
            label = str(spec)
            getop = _Edit(header, label, "", False, False, [], [])._getop(label)[0]
            try:
                selected = evexpr.evaluate(expr, getop)
            except evexpr.ExprError as error:
                found = re.search(r"not found \((.+)\)", str(error))
                if found:
                    # IRAF stops the whole task here; one frame without the
                    # keyword simply is not selected.
                    absent.add(found[1])
                    skipped += 1
                    continue
                advice.fail(f"cannot evaluate {params.expr!r}: {error}",
                            "write it as IRAF does: imagetyp == 'zero' && exptime > 30; "
                            "a keyword with - in it is @\"DATE-OBS\"")
            if selected.type != "bool":
                advice.fail(f"expression must be boolean: {params.expr!r} is a "
                            f"{selected.type}", "compare it: exptime > 0, object ?= 'M92*'")
            if not selected.value:
                continue
            fields = [key for field in params.fields for name in str(field).split(",")
                      if name.strip() for key in _expand(header, name.strip())]
            row = {"image": label}
            out = []
            for key in fields:
                value = label if key == "$I" else image_parameter(header, label,
                                                                   key.lstrip("$"))
                if value is None:
                    out.append("\t" + params.missing)
                    row[key] = params.missing
                else:
                    out.append(("\t" if out else "") + _quoted(value))
                    row[key] = value
            lines.append("".join(out))
            matched.append(row)
        warnings = []
        if absent:
            names = ", ".join(sorted(absent))
            plural = "s" if len(absent) > 1 else ""
            warnings.append(advice.warn(
                (f"{skipped} image{'s' if skipped > 1 else ''} without the keyword{plural} "
                 f"{names} were not selected" if matched else
                 f"no image has the keyword{plural} {names}, so none was selected"),
                "check the spelling against a header: harwood imheader IMAGE "
                "--longheader lists every keyword"))
        return TaskResult(data={"rows": matched, "matched": len(matched)},
                          warnings=warnings, stdout=lines)


def _iraf_words(expr: str) -> str:
    """Python's and, or and not, outside quotes, as IRAF's && || ! -- a
    harwood courtesy; IRAF has only the symbols."""
    out, word = [], re.compile(r"(?<![\w$])(and|or|not)(?![\w$])")
    for piece in re.split(r"""('[^']*'|"[^"]*")""", expr):
        if piece[:1] in ("'", '"'):
            out.append(piece)
        else:
            out.append(word.sub(lambda m: {"and": "&&", "or": "||", "not": "!"}[m[1]],
                                piece))
    return "".join(out)


def _first(header, *keywords: str) -> str:
    """The first of these keywords the header actually has.

    Observatories disagree about names -- KPNO wrote FILTERS, most write
    FILTER -- and looking for only one of them leaves the column blank on half
    the data there is.
    """
    for keyword in keywords:
        value = header.get(keyword)
        if value not in (None, ""):
            return str(value).strip()
    return ""


class ImStatParams(TaskParams):
    input: list[str] = Field(
        ..., description="Images to measure. Accepts globs and @lists."
    )
    fields: str = Field(
        "image,npix,mean,stddev,min,max",
        description="Which statistics to report, comma separated. Choose from "
                    "image, npix, mean, midpt, mode, stddev, skew, kurtosis, "
                    "min, max.",
    )
    section: str | None = Field(
        None, description="Section to measure, 1-based and (x,y), e.g. '[100:400,*]'."
    )
    lower: float | None = Field(
        None, description="Ignore pixels below this value."
    )
    upper: float | None = Field(
        None, description="Ignore pixels above this value."
    )
    nclip: int = Field(0, ge=0, description="Sigma-clipping iterations before measuring.")
    lsigma: float = Field(
        3.0, ge=0, description="Clipping threshold below the mean, in sigma."
    )
    usigma: float = Field(
        3.0, ge=0, description="Clipping threshold above the mean, in sigma -- "
                               "IRAF's usigma."
    )
    binwidth: float = Field(
        0.1, gt=0,
        description="Histogram bin width for the mode, in sigma. IRAF's own.",
    )


#: Every statistic imstatistics knows how to report.
STATISTICS = ("image", "npix", "mean", "midpt", "mode", "stddev", "skew",
              "kurtosis", "min", "max")


@register
class ImStatistics(Task[ImStatParams]):
    name = "imstatistics"
    package = "imutil"
    description = "Pixel statistics, with optional clipping and a choice of fields."
    Params = ImStatParams

    def plan(self, params: ImStatParams) -> list[Step]:
        steps = [Step("read", f"{len(params.input)} patterns"
                              + (f", section {params.section}" if params.section else ""),
                      why=WHY["read headers"])]
        if params.lower is not None or params.upper is not None:
            steps.append(Step("threshold", f"keep {params.lower} .. {params.upper}",
                              why=WHY["threshold"]))
        if params.nclip:
            steps.append(Step("clip", f"{params.nclip} iterations at "
                                      f"-{params.lsigma}/+{params.usigma} sigma",
                              why=WHY["clip"]))
        steps.append(Step("measure", params.fields, why=WHY["measure"]))
        return steps

    def execute(self, params: ImStatParams) -> TaskResult:

        from ..core import images

        wanted = [f.strip() for f in params.fields.split(",") if f.strip()]
        unknown = [f for f in wanted if f not in STATISTICS]
        if unknown:
            raise ValueError(
                f"no statistic called {', '.join(unknown)}. "
                f"Choose from {', '.join(STATISTICS)}."
            )

        rows = []
        for spec in images.expand(params.input):
            if params.section:
                spec = images.ImageSpec(spec.path, spec.extension, params.section)
            data, _ = images.read(spec)
            row = imstat(data, wanted, params.lower, params.upper, params.nclip,
                         params.lsigma, params.usigma, params.binwidth)
            if row["npix"] == 0:
                raise ValueError(f"'{spec}' has no pixels left after the limits")
            row = {"image": str(spec), **row}
            rows.append({field: row[field] for field in wanted})
        return TaskResult(data={"rows": rows}, steps=self.plan(params))


MAX_REAL = 3.4028235e38


def imstat(data, fields=("npix", "mean", "stddev", "min", "max"), lower=None,
           upper=None, nclip=0, lsigma=3.0, usigma=3.0, binwidth=0.1) -> dict:
    """IRAF's imstatistics (images.imutil, t_imstat.x), as a function.

    The pixels are read as IRAF reads them, in single precision; the sums are
    double, the results single, as IRAF stores them. Clipping recomputes the
    statistics from the pixels within mean -/+ lsigma/usigma sigma of the
    previous pass, and stops once a pass keeps as many as the one before.
    midpt and mode come from a histogram -- of every pixel in the image, not
    only the clipped ones, between the last pass's min and max, in bins of
    binwidth sigma -- with the median interpolated in its integral and the
    mode in a parabola through the peak bin. They are IRAF's numbers, which
    is the point; a true median would not be.

    Translated from IRAF, Copyright (c) 1986 Association of Universities for
    Research in Astronomy Inc.; the notice is in LICENSES/IRAF-COPYRIGHT.
    """
    import numpy as np

    f32 = np.float32
    x = np.asarray(data, dtype=np.float32).ravel()
    x = x[np.isfinite(x)]
    fields = set(fields)
    want_hist = "midpt" in fields or "mode" in fields

    def accumulate(low, high):
        if low is None and high is None:
            values = x
        else:
            lo = -MAX_REAL if low is None else f32(low)
            hi = MAX_REAL if high is None else f32(high)
            values = x[(x >= lo) & (x <= hi)]
        v = values.astype(np.float64)
        stats = {"npix": int(v.size), "mean": float("nan"), "stddev": float("nan"),
                 "skew": float("nan"), "kurtosis": float("nan"),
                 "min": float(values.min()) if v.size else float("nan"),
                 "max": float(values.max()) if v.size else float("nan"),
                 "midpt": float("nan"), "mode": float("nan")}
        n = v.size
        if n <= 0:
            return stats
        v2 = v * v
        sumx, sumx2, sumx3, sumx4 = v.sum(), v2.sum(), (v2 * v).sum(), (v2 * v2).sum()
        mean = sumx / n
        stats["mean"] = float(f32(mean))
        if n < 2:
            return stats
        var = (sumx2 - sumx * mean) / (n - 1)
        if var <= 0:
            stats["stddev"] = 0.0
            return stats
        stdev = np.sqrt(var)
        stats["stddev"] = float(f32(stdev))
        # IRAF's skew uses its single-precision copy of the mean in one term
        # and the double in the others; the cancellation that follows is in
        # its numbers, so it is in these.
        stats["skew"] = float(f32((sumx3 - 3.0 * float(f32(mean)) * sumx2
                                   + 3.0 * mean * mean * sumx - n * mean ** 3)
                                  / n / stdev / stdev / stdev))
        stats["kurtosis"] = float(f32((sumx4 - 4.0 * mean * sumx3
                                       + 6.0 * mean * mean * sumx2
                                       - 4.0 * mean ** 3 * sumx + n * mean ** 4)
                                      / n / stdev ** 4 - 3.0))
        return stats

    low, high = lower, upper
    previous = None
    for i in range(int(nclip) + 1):
        stats = accumulate(low, high)
        if i < nclip:
            mean, sigma = stats["mean"], stats["stddev"]
            defined = np.isfinite(mean) and np.isfinite(sigma)
            low = mean - lsigma * sigma if defined and lsigma > 0 else -MAX_REAL
            high = mean + usigma * sigma if defined and usigma > 0 else MAX_REAL
            if lower is not None:
                low = max(low, lower)
            if upper is not None:
                high = min(high, upper)
            if i > 0 and stats["npix"] == previous:
                break
            previous = stats["npix"]

    if want_hist and stats["npix"] > 0 and np.isfinite(stats["stddev"]):
        hwidth = f32(f32(binwidth) * f32(stats["stddev"]))
        if binwidth > 0 and hwidth > 0:
            hmin, hmax = f32(stats["min"]), f32(stats["max"])
            nbins = int(f32(hmax - hmin) / hwidth + f32(1.0))
            if nbins >= 3:
                hgm = _ahgm(x, nbins, hmin, hmax)
                stats["midpt"] = _hmedian(hgm, nbins, hwidth, hmin)
                stats["mode"] = _hmode(hgm, nbins, hwidth, hmin)
    return stats


def _ahgm(x, nbins, z1, z2):
    """vops ahgm: bins of (z2 - z1) / (nbins - 1), whatever width was meant."""
    import numpy as np

    f32 = np.float32
    dz = f32(f32(nbins - 1) / f32(z2 - z1))
    inside = x[(x >= z1) & (x <= z2)]
    if abs(dz - 1.0) < 2 * np.finfo(np.float32).eps:
        bins = (inside - z1).astype(np.int64)
    else:
        bins = ((inside - z1) * dz).astype(np.int64)
    return np.bincount(bins, minlength=nbins)[:nbins]


def _hmedian(hgm, nbins, hwidth, hmin):
    """ist_hmedian: where the integral of the histogram crosses one half."""
    import numpy as np

    f32 = np.float32
    integral = np.cumsum(hgm).astype(np.float32)
    integral = integral / integral[-1]
    lo = 0
    for i in range(1, nbins + 1):
        if integral[i - 1] > 0.5:
            break
        lo = i
    hi = lo + 1
    h1 = f32(hmin + f32(lo) * hwidth)
    hdiff = integral[hi - 1] if lo == 0 else f32(integral[hi - 1] - integral[lo - 1])
    if hdiff == 0:
        return float(h1)
    if lo == 0:
        return float(f32(h1 + f32(0.5) / hdiff * hwidth))
    return float(f32(h1 + (f32(0.5) - integral[lo - 1]) / hdiff * hwidth))


def _hmode(hgm, nbins, hwidth, hmin):
    """ist_hmode: the peak bin, refined by a parabola through its neighbours."""
    import numpy as np

    f32 = np.float32
    if nbins == 1:
        return float(f32(hmin + f32(0.5) * hwidth))
    if nbins == 2:
        if hgm[0] > hgm[1]:
            return float(f32(hmin + f32(0.5) * hwidth))
        if hgm[1] > hgm[0]:
            return float(f32(hmin + f32(1.5) * hwidth))
        return float(f32(hmin + hwidth))
    bpeak = int(np.argmax(hgm)) + 1          # the first highest, as IRAF's scan
    if bpeak == 1:
        return float(f32(hmin + f32(0.5) * hwidth))
    if bpeak == nbins:
        return float(f32(hmin + f32(nbins - 0.5) * hwidth))
    bpeak -= 1
    dh1 = f32(hgm[bpeak] - hgm[bpeak - 1])
    dh2 = f32(hgm[bpeak] - hgm[bpeak + 1])
    denom = f32(dh1 + dh2)
    if denom == 0:
        return float(f32(hmin + f32(bpeak + 0.5) * hwidth))
    position = f32(bpeak + 1 + f32(0.5) * (dh1 - dh2) / denom)
    return float(f32(hmin + (position - f32(0.5)) * hwidth))


def mode(values, width: float | None = None) -> float:
    """The mode as IRAF's imstatistics finds it: bins of `width` (a tenth of
    the standard deviation by default, IRAF's binwidth), the peak bin, and a
    parabola through it and its neighbours."""
    import numpy as np

    data = np.asarray(values, dtype=np.float32).ravel()
    sigma = float(data.astype(np.float64).std(ddof=1)) if data.size > 1 else 0.0
    if sigma <= 0:
        return float(data[0]) if data.size else float("nan")
    binwidth = 0.1 if width is None else width / sigma
    return imstat(data, ("mode",), binwidth=binwidth)["mode"]


class ImArithParams(TaskParams):
    operand1: str = Field(
        ..., description="First operand: images, an @list, or a number."
    )
    op: Literal["+", "-", "*", "/"] = Field(..., description="Operation to apply.")
    operand2: str = Field(
        ..., description="Second operand: images, an @list, or a number. A single "
                         "image or number is applied to every input."
    )
    result: Template = Field(
        ..., description="Where to write: one name per input, an @list, a "
                         "directory, or a template like '%tr%n%'."
    )
    divzero: float = Field(
        0.0, description="Value written where a division by zero would happen; the "
                         "DQ marks those pixels 'no data'."
    )
    uncertainty: Literal["VAR", "ERR", "UNCERT"] = uncertainty_field()
    format: Literal["harwood", "iraf"] = format_field()


@register
class ImArith(Task[ImArithParams]):
    name = "imarith"
    package = "imutil"
    description = "Arithmetic between images, lists of images, and constants."
    Params = ImArithParams

    def plan(self, params: ImArithParams) -> list[Step]:
        left = _operand_list(params.operand1)
        right = _operand_list(params.operand2)
        detail = f"{len(left)} image(s) {params.op} " + (
            params.operand2 if len(right) <= 1 else f"{len(right)} image(s)")
        return [Step("compute", detail, why=WHY["compute"]),
                Step("write", params.result, why=WHY["write"])]

    def execute(self, params: ImArithParams) -> TaskResult:
        from astropy.io import fits

        from ..core import images, mef, provenance

        left = _operand_list(params.operand1)
        right = _operand_list(params.operand2)
        if len(right) not in (1, len(left)):
            raise ValueError(
                f"operand1 has {len(left)} images and operand2 has {len(right)}. "
                "The second operand must be a single image or number, or match "
                "the first one for one."
            )
        destinations = (images.outputs_for([_as_spec(name) for name in left],
                                           params.result)
                        if not _is_number(params.operand1)
                        else [Path(params.result)])

        outputs, history, warnings = [], [], []
        for index, first in enumerate(left):
            second = right[0] if len(right) == 1 else right[index]
            a, header_a, va, qa = _read(first)
            b, header_b, vb, qb = _read(second)
            header = (header_a if header_a is not None else header_b)

            out, var, quality = _arith(a, b, va, vb, qa, qb, params.op, params.divzero)
            header = header.copy() if header is not None else fits.Header()
            line = f"imarith: {first} {params.op} {second}"
            provenance.stamp_header(header, [line])
            destination = Path(destinations[index])
            destination.parent.mkdir(parents=True, exist_ok=True)
            written, notes = mef.save(destination, out, header, var, quality,
                                      params.uncertainty, params.format)
            outputs.extend(written)
            if notes and not warnings:
                warnings.extend(notes)
            history.append(line)

        return TaskResult(outputs=outputs, history=history, steps=self.plan(params),
                          warnings=warnings, data={"images": len(left)})


def _is_number(text: str) -> bool:
    try:
        float(text)
    except ValueError:
        return False
    return True


def _operand_list(text: str) -> list[str]:
    """One operand, expanded. A number stays a single item."""
    from ..core import images

    if _is_number(text):
        return [text]
    return [str(spec) for spec in images.expand(text)]


def _as_spec(name: str):
    from ..core import images

    return images.parse(name)


def _read(text: str):
    """An operand: (value, header, variance, DQ). A number is exact: no
    header, no variance, nothing wrong with it."""
    from ..core import mef

    if _is_number(text):
        return float(text), None, 0.0, 0
    return mef.load(text)


def _arith(a, b, va, vb, qa, qb, op: str, divzero: float):
    """imarith on values with their variances (None: unknown) and DQ. Sums
    and differences add variances; products and quotients propagate them to
    first order. The DQ is what either operand said, and a division by zero
    is no data."""
    import numpy as np

    from ..core import dq

    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    a, b = np.broadcast_arrays(a, b)
    known = va is not None and vb is not None
    var = None
    quality = (np.zeros(a.shape, dtype=dq.DTYPE) | np.asarray(qa, dtype=dq.DTYPE)
               | np.asarray(qb, dtype=dq.DTYPE))
    with np.errstate(divide="ignore", invalid="ignore"):
        if op in "+-":
            out = a + b if op == "+" else a - b
            var = va + vb if known else None
        elif op == "*":
            out = a * b
            var = b ** 2 * va + a ** 2 * vb if known else None
        else:
            zero = b == 0
            safe = np.where(zero, 1.0, b)
            out = np.where(zero, divzero, a / safe)
            if known:
                var = np.where(zero, 0.0, va / safe ** 2 + a ** 2 * vb / safe ** 4)
            quality = quality | np.where(zero, dq.NO_DATA, 0).astype(dq.DTYPE)
    if var is not None:
        var = np.broadcast_to(np.asarray(var, dtype=float), a.shape)
    return out, var, quality


class ImCopyParams(TaskParams):
    input: list[str] = Field(
        ..., description="Images to copy. Sections are honoured: 'obj.fits[1:100,*]' "
                         "copies just that part."
    )
    output: Template = Field(
        ..., description="Where to write: one name per input, a directory, or a "
                         "template like '%m%tr%'."
    )
    verbose: bool = Field(True, description="Say what was copied.")


@register
class ImCopy(Task[ImCopyParams]):
    name = "imcopy"
    package = "imutil"
    description = "Copy images, or sections of them."
    Params = ImCopyParams

    def plan(self, params: ImCopyParams) -> list[Step]:
        from ..core import images

        specs = images.expand(params.input)
        return [Step("copy", f"{len(specs)} image(s), first {specs[0]}",
                     why=WHY["copy"]),
                Step("write", params.output, why=WHY["write"])]

    def execute(self, params: ImCopyParams) -> TaskResult:
        import numpy as np
        from astropy.io import fits

        from ..core import images, provenance

        specs = images.expand(params.input)
        destinations = images.outputs_for(specs, params.output)
        rows, outputs = [], []
        for spec, destination in zip(specs, destinations, strict=True):
            data, header = images.read(spec)
            provenance.stamp_header(header, [f"imcopy: from {spec}"])
            destination.parent.mkdir(parents=True, exist_ok=True)
            if destination.suffix == ".pl":
                from ..core import plio

                plio.write(destination, data.astype(np.int64), header=header)
            else:
                fits.PrimaryHDU(np.asarray(data, dtype=np.float32),
                                header).writeto(destination, overwrite=True)
            outputs.append(destination)
            rows.append({"from": str(spec), "to": str(destination),
                         "shape": "x".join(str(n) for n in reversed(data.shape))})
        return TaskResult(outputs=outputs, steps=self.plan(params),
                          data={"rows": rows} if params.verbose
                          else {"images": len(outputs)})


class ImDeleteParams(TaskParams):
    input: list[str] = Field(..., description="Images to delete. Accepts globs and @lists.")
    verify: bool = Field(
        True,
        description="Ask before deleting. Leave it on: a glob that matches more than "
                    "you meant is how a night's work disappears.",
    )


@register
class ImDelete(Task[ImDeleteParams]):
    name = "imdelete"
    package = "imutil"
    description = "Delete images."
    Params = ImDeleteParams

    def plan(self, params: ImDeleteParams) -> list[Step]:
        from ..core import images

        specs = images.expand(params.input)
        return [Step("delete", ", ".join(str(s.path) for s in specs[:6])
                     + (f" and {len(specs) - 6} more" if len(specs) > 6 else ""),
                     why=WHY["delete"])]

    def execute(self, params: ImDeleteParams) -> TaskResult:
        from ..core import images

        specs = images.expand(params.input)
        paths = sorted({spec.path for spec in specs})
        # The plan has to be built before the files go: it describes them by
        # expanding the pattern, and afterwards the pattern matches nothing.
        steps = self.plan(params)
        if params.verify:
            raise RuntimeError(
                f"{len(paths)} image(s) would be deleted: "
                + ", ".join(p.name for p in paths[:8])
                + (f" and {len(paths) - 8} more" if len(paths) > 8 else "")
                + ". Run again with --no-verify to actually delete them."
            )
        for path in paths:
            path.unlink()
        return TaskResult(data={"deleted": len(paths),
                                "names": [p.name for p in paths]},
                          steps=steps)


class SectionsParams(TaskParams):
    template: str = Field(
        ..., description="Image template to expand. Globs, @lists, comma lists and "
                         "substitutions like '%m%tr%92*.fits' all work."
    )
    option: Literal["fullname", "root", "section"] = Field(
        "fullname", description="What to print for each match."
    )


@register
class Sections(Task[SectionsParams]):
    name = "sections"
    package = "imutil"
    description = "Expand an image template and show what it matches."
    Params = SectionsParams

    def plan(self, params: SectionsParams) -> list[Step]:
        return [Step("expand", params.template, why=WHY["expand"])]

    def execute(self, params: SectionsParams) -> TaskResult:
        from ..core import images

        text = params.template
        if text.startswith("%"):
            parts = text.split("%")
            source = parts[1] + "%".join(parts[3:])
            specs = images.expand(source)
            names = [str(images.substitute(s.path, "%".join(parts[:3]) + "%"))
                     for s in specs]
        else:
            specs = images.expand(text)
            if params.option == "root":
                names = [s.path.stem for s in specs]
            elif params.option == "section":
                names = [s.section or "" for s in specs]
            else:
                names = [str(s) for s in specs]

        return TaskResult(data={"rows": [{"name": n} for n in names],
                                "matched": len(names)},
                          steps=self.plan(params))


class ImplotParams(TaskParams):
    image: list[str] = Field(..., description="Images to plot; n and m step through "
                                              "them. Sections are honoured.",
                             json_schema_extra={"positional": True})
    line: int | None = Field(
        None, description="Image line to plot first. The middle one by default."
    )
    column: int | None = Field(
        None, description="Image column to plot first, instead of a line."
    )
    nsum: int = Field(
        1, ge=1, description="How many lines or columns l and c average, as :a sets it."
    )
    wcs: Literal["logical", "physical", "world"] = Field(
        "logical", description="Coordinates along the plot: pixels, physical pixels "
                               "(through LTV/LTM), or world (CRVAL/CD). A harwood "
                               "spectrum is a line per aperture: world is its "
                               "wavelength, and across, the aperture number.")
    step: int = Field(0, ge=0, description="Lines or columns j and k move by. 0: a tenth "
                                           "of the image.")
    coords: str = Field(
        "", description="Graphics cursor commands instead of the keyboard: 'x y 1 key' "
                        "or 'x y 1 :command' per line, IRAF's form. Runs without a window "
                        "and prints what the keys would have.")


@register
class Implot(Task[ImplotParams]):
    """IRAF's implot, pkg/plot/t_implot.x: its keys and colon commands
    through one engine (core.implot), in the window or from a cursor file."""

    name = "implot"
    needs_window = True
    package = "plot"
    description = "Plot lines and columns of an image, interactively."
    Params = ImplotParams

    def plan(self, params: ImplotParams) -> list[Step]:
        what = (f"column {params.column}" if params.column
                else f"line {params.line}" if params.line else "the middle line")
        return [Step("read", ", ".join(params.image), why=WHY["read headers"]),
                Step("plot", what + (f", {params.nsum} averaged"
                                     if params.nsum > 1 else ""),
                     why=WHY["plot"]),
                Step("wait", f"commands from {params.coords}" if params.coords else
                     "keys: l c a e j k m n o p r s space, : commands, q",
                     why="Lower case picks what to plot, a colon command takes a "
                         "range, and e expands a corner at a time -- the keys the "
                         "tutorials tell you to press.")]

    def execute(self, params: ImplotParams) -> TaskResult:
        from ..core import images
        from ..core import implot as engine_module

        names = [str(spec) for spec in images.expand(params.image)]
        if not names:
            advice.fail(f"'{' '.join(params.image)}' matches no image",
                        "check the name, or run `harwood sections` on the template",
                        FileNotFoundError)
        engine = engine_module.Implot(names, line=params.line, wcs=params.wcs,
                                      step=params.step)
        engine.p_navg = engine.navg = params.nsum
        if params.column is not None:
            engine.colon(f":c {params.column}")
        elif params.nsum > 1:
            engine.colon(f":l {engine.line}")
        if params.coords:
            if not Path(params.coords).exists():
                advice.fail(f"cursor file '{params.coords}' does not exist",
                            "write 'x y 1 key' lines, or leave --coords empty for the "
                            "window", FileNotFoundError)
            for line in Path(params.coords).read_text(encoding="utf-8").splitlines():
                words = line.split(None, 3)
                if len(words) < 4 or line.lstrip().startswith("#"):
                    continue
                x, y, key = float(words[0]), float(words[1]), words[3].strip()
                if key.startswith(":"):
                    engine.colon(key)
                else:
                    engine.key({"\\040": " ", "\\40": " "}.get(key, key[:1]), x, y)
                if engine.done:
                    break
            messages = engine.messages
        else:
            from ..viewer.implot_panel import run

            first = engine.data
            messages = run(first, name=names[0], images=names, line=engine.line,
                           wcs=params.wcs, step=params.step)
        data: dict = {"shape": f"{engine.ncols}x{engine.nlines}"}
        if messages:
            data["text"] = list(messages)
        return TaskResult(data=data, steps=self.plan(params))


HEDWHY = {
    "read": "Every image the template matches. A keyword edit is nearly always "
            "wanted on a whole set -- all the flats, all of one night -- and "
            "doing it one file at a time is how half a set ends up different "
            "from the other half.",
    "edit": "The keyword is set, added or deleted. What it was before goes to "
            "HISTORY: a header changed with no record of what it said is a "
            "file nobody can retrace, and headers are edited precisely when "
            "something was wrong, which is exactly when the old value matters.",
    "write": "In place. There is no output copy, because a header edit on a "
             "hundred frames that wrote a hundred copies would be worse than "
             "the problem -- but --no-update shows what would happen without "
             "touching anything, and it is worth using first.",
}


class HEditParams(TaskParams):
    images: list[str] = Field(
        ..., description="Images to edit. Globs and @lists expand.",
        json_schema_extra={"positional": True},
    )
    fields: list[str] = Field(
        ..., description="Keywords to edit, e.g. OBJECT, or several: 'EXPTIME,DARKTIME'. "
                         "* and ? match, in any case: 'CCD*'."
    )
    value: str = Field(
        "",
        description="The new value. In parentheses it is an expression over the header, "
                    "as in IRAF: '(EXPTIME * 2)', '(OBJECT // \" B\")', '($ + 1)'. A "
                    "single . prints the current value instead.",
    )
    comment: str = Field("", description="Comment for the card.")
    add: bool = Field(
        False, description="Add the keyword when it is not there (and edit it when it is)."
    )
    addonly: bool = Field(
        False, description="Add the keyword only where it is not there; leave existing "
                           "values alone."
    )
    delete: bool = Field(False, description="Delete the keyword instead of setting it.")
    verify: bool = Field(
        False, description="Ask before each change, as IRAF does by default: Enter "
                           "accepts, n skips, anything else is the value to use."
    )
    show: bool = Field(True, description="Print each change as IRAF does: "
                                         "'image,KEY: old -> new'.")
    update: bool = Field(
        True, description="Write the change. Off, it only shows what would happen."
    )
    extension: int = Field(0, ge=0, description="Which HDU's header to edit.")


#: Cards that describe the file rather than the data: never matched by a
#: wildcard, never edited.
_STRUCTURAL = re.compile(r"^(SIMPLE|BITPIX|NAXIS\d*|EXTEND|XTENSION|PCOUNT|GCOUNT|BSCALE"
                         r"|BZERO|END|COMMENT|HISTORY|)$")


@register
class HEdit(Task[HEditParams]):
    """IRAF's hedit, from images/imutil/src/hedit.x, with its expression
    evaluator: the same fields, the same values, the same lines printed.
    Where IRAF would write a card that is not valid FITS -- text without
    quotes in a numeric or logical card -- harwood writes the valid one."""

    name = "hedit"
    package = "imutil"
    description = "Edit header keywords, over one image or a whole night."
    Params = HEditParams

    def plan(self, params: HEditParams) -> list[Step]:
        what = ("delete" if params.delete
                else "print" if params.value.strip() == "."
                else f"add only where absent: {params.value!r}" if params.addonly
                else f"set to {params.value!r}" if params.value != ""
                else "show")
        return [
            Step("read", ", ".join(params.images), why=HEDWHY["read"]),
            Step("edit", f"{', '.join(params.fields)}: {what}", why=HEDWHY["edit"]),
            Step("write", "in place" if params.update
                 else "nothing -- --no-update was given", why=HEDWHY["write"],
                 skipped=not params.update,
                 reason="" if params.update else "showing only"),
        ]

    def execute(self, params: HEditParams) -> TaskResult:
        from astropy.io import fits

        from ..core import images

        specs = images.expand(params.images)
        if not specs:
            raise ValueError(f"nothing matched {', '.join(params.images)}.")
        if params.delete and params.value:
            raise ValueError(
                "--delete removes the keyword and --value sets it; they cannot "
                "both apply. Drop one.")
        value = params.value.strip()
        printing = value == "." and not (params.add or params.addonly or params.delete)
        update = params.update and not printing
        show = params.show and not printing
        verify = params.verify and not printing
        wanted = [name.strip() for field in params.fields
                  for name in str(field).split(",") if name.strip()]
        steps = self.plan(params)
        by_action = {step.action: step for step in steps}

        rows, lines, warnings, changed = [], [], [], []
        missing = 0
        for spec in specs:
            path, label = spec.path, str(spec)
            with fits.open(path, memmap=False) as hdus:
                header = hdus[params.extension].header.copy()
            edit = _Edit(header, label, params.comment, show, verify, lines, rows)
            if params.add or params.addonly:
                for field in wanted:
                    present = _find(header, field)
                    if present is None:
                        edit.add(field, value)
                    elif params.addonly:
                        warnings.append(advice.note(
                            f"parameter {label},{field.upper()} already exists",
                            "--add edits keywords that are there as well"))
                        rows.append(_row(label, present, header.get(present),
                                         "kept -- already there"))
                    else:
                        edit.change(present, value)
                touched = bool(wanted)
            else:
                matched = [key for field in wanted for key in _expand(header, field)]
                for field in wanted:
                    if not _expand(header, field):
                        missing += 1
                        rows.append(_row(label, field.upper(), None,
                                         "absent" if params.delete or printing
                                         else "absent -- --add would create it"))
                for key in dict.fromkeys(matched):
                    if params.delete:
                        edit.delete(key)
                    elif printing or value == "":
                        edit.show(key)
                    else:
                        edit.change(key, value)
                touched = bool(matched)
            if update and touched and _confirm(verify, label):
                if edit.changed:
                    for line in edit.history:
                        header.add_history(line)
                    with fits.open(path, mode="update") as hdus:
                        hdus[params.extension].header = header
                    changed.append(Path(path))
                if show:
                    lines.append(f"{label} updated")

        by_action["read"].measured = {"images": len(specs), "keywords": len(wanted)}
        actions: dict[str, int] = {}
        for row in rows:
            actions[row["action"]] = actions.get(row["action"], 0) + 1
        by_action["edit"].measured = actions

        if missing and not params.add and not params.addonly and not params.delete \
                and value not in ("", "."):
            warnings.append(advice.warn(
                f"{missing} of {len(rows)} were not there and were left alone. "
                "hedit will not invent a keyword, because a typo in a keyword "
                "name would otherwise silently add a card nobody reads.",
                "add --add if you meant to create them. If you did not, check "
                "the spelling against 'harwood imheader <image> --long'."))
        if not update and not printing:
            warnings.append(advice.note(
                "--no-update was given, so nothing was written. The table "
                "above is what would have happened.",
                "run it again without --no-update to apply it."))
        data: dict = {"rows": rows}
        if lines:
            data = {"text": lines, "rows": rows}
        return TaskResult(outputs=changed, steps=steps, warnings=warnings,
                          data=data, history=[])


def _confirm(verify: bool, label: str) -> bool:
    """hedit's 'update image ? (yes): ' -- Enter or y writes, anything else
    does not."""
    if not verify:
        return True
    answer = _ask(f"update {label} ? (yes): ")
    return answer is not None and (answer.strip() == "" or answer.strip().startswith("y"))


def _ask(prompt: str) -> str | None:
    import sys

    sys.stderr.write(prompt)
    sys.stderr.flush()
    line = sys.stdin.readline()
    return None if line == "" else line.rstrip("\n")


def _find(header, field: str) -> str | None:
    """The header's own spelling of a keyword, matched in any case."""
    upper = field.upper()
    return upper if upper in header and not _STRUCTURAL.match(upper) else None


def _expand(header, field: str) -> list[str]:
    """imofnlu: the fields a template names, IRAF's standard ones (i_naxis,
    i_naxis1, ...) first and then the header's own keywords, in order. A
    name with $ in front (as $I) passes through; a name without wildcards
    must exist."""
    if field.startswith("$"):
        return [field]
    if not any(ch in field for ch in "*?["):
        if field.lower().removeprefix("i_") in _STANDARD:
            return [field]
        found = _find(header, field)
        return [found] if found else []
    from ..core import evexpr

    pattern = "^{" + field.replace("*", "?*") + "}$"
    naxis = int(header.get("NAXIS", 0))
    standard = ["naxis"] + [f"naxis{axis}" for axis in range(1, naxis + 1)]
    names = [f"i_{name}" for name in standard if evexpr.patmatch(name, pattern)]
    return names + [str(key) for key in dict.fromkeys(header.keys())
                    if not _STRUCTURAL.match(str(key)) and evexpr.patmatch(str(key), pattern)]


_STANDARD = {"naxis", "naxis1", "naxis2", "naxis3", "naxis4", "naxis5", "naxis6", "naxis7",
             "pixtype", "title", "pixfile", "minpixval", "maxpixval"}


def _quoted(text: str) -> str:
    """he_pargstr: a value with blanks in it is printed in double quotes."""
    return f'"{text}"' if any(ch in " \t" for ch in text) else text.replace("\n", "\\n")


class _Edit:
    """The edits on one header, as hedit.x makes them, recording what IRAF
    prints and what harwood keeps in HISTORY."""

    def __init__(self, header, label: str, comment: str, show: bool, verify: bool,
                 lines: list, rows: list) -> None:
        self.header, self.label, self.comment = header, label, comment
        self.show_lines, self.verify, self.lines, self.rows = show, verify, lines, rows
        self.history: list[str] = []
        self.changed = False

    def _old(self, key: str) -> str:
        if key == "$I":
            return self.label
        return image_parameter(self.header, self.label, key.lstrip("$")) or ""

    def _getop(self, field: str):
        from ..core import evexpr
        from ..core.evexpr import Operand

        def getop(name: str) -> Operand:
            if name == ".":
                return Operand("str", ".")
            if name == "$F":
                return Operand("str", field)
            if name == "$I":
                return Operand("str", self.label)
            if name == "$T":
                import time

                return Operand("int", int(time.time()) - 315532800 - time.timezone)
            key = field if name == "$" else name
            return _operand(self.header, self.label, key)

        return getop, evexpr

    def evaluate(self, field: str, value: str):
        getop, evexpr = self._getop(field)
        try:
            return evexpr.evaluate(value, getop)
        except evexpr.ExprError as error:
            advice.fail(f"{self.label},{field}: cannot evaluate {value!r}: {error}",
                        "an expression is in parentheses and uses the header's keywords; "
                        "a keyword with - in its name is written @\"DATE-OBS\"")

    def show(self, key: str) -> None:
        old = self._old(key)
        self.lines.append(f"{self.label},{key} = {_quoted(old)}")
        self.rows.append(_row(self.label, key, old if old != "" else None, "shown"))

    def change(self, key: str, value: str) -> None:
        if key.startswith("$") or key.lower().removeprefix("i_") in _STANDARD:
            self.lines.append(f"cannot update {self.label},{key}")
            return
        old = self._old(key)
        if value.startswith("("):
            new = self.evaluate(key, value)
            from ..core import evexpr

            text = evexpr.encode(new)
        else:
            text = _literal(value)
        if text == ".":
            self.show(key)
            return
        if self.verify:
            answer = _ask(f"{self.label},{key} ({_quoted(old)} -> {_quoted(text)}): ")
            if answer is None or answer in ("n", "no"):
                return
            if answer.startswith("\\"):
                text = answer[1:]
            elif answer not in ("", "y", "yes"):
                text = answer
        _put(self.header, key, text, self.comment)
        self.changed = True
        self.history.append(f"hedit: {key} = {text}, was {old}")
        if self.show_lines:
            self.lines.append(f"{self.label},{key}: {_quoted(old)} -> {_quoted(text)}")
        self.rows.append(_row(self.label, key, old, "set"))

    def add(self, field: str, value: str) -> None:
        from ..core import evexpr

        key = field.upper()
        literal = _literal(value)
        kind, length = evexpr.lexnum(value)
        if value.startswith("(") or (kind != evexpr.NON and length == len(value)):
            o = self.evaluate(key, value)
        else:
            o = evexpr.Operand("str", literal)
        text = evexpr.encode(o)
        _put(self.header, key, text, self.comment, kind=o.type)
        self.changed = True
        self.history.append(f"hedit: {key} = {text}, added")
        if self.show_lines:
            self.lines.append(f"add {self.label},{key} = {_quoted(text)}")
        self.rows.append(_row(self.label, key, None, "added"))

    def delete(self, key: str) -> None:
        if key.startswith("$") or key.lower().removeprefix("i_") in _STANDARD:
            self.lines.append(f"Warning: Cannot delete image header parameter ({key})")
            return
        if self.verify:
            answer = _ask(f"delete {self.label},{key} ? (yes): ")
            if answer is None or not (answer.strip() == "" or answer.strip().startswith("y")):
                return
        old = self._old(key)
        del self.header[key]
        self.changed = True
        self.history.append(f"hedit: deleted {key} (was {old})")
        if self.show_lines:
            self.lines.append(f"{self.label},{key} deleted")
        self.rows.append(_row(self.label, key, old, "deleted"))


def _literal(value: str) -> str:
    """A literal value as typed. Quotes around the whole of it make it a
    string, without the quotes -- the shell has usually eaten the ones IRAF's
    command language would have."""
    if len(value) >= 2 and value[0] == value[-1] and value[0] in "'\"":
        return value[1:-1]
    return value


def _operand(header, label: str, key: str):
    """he_getfield: a header value as an operand of its own type -- real in
    single precision, as IRAF reads it."""
    import numpy as np

    from ..core.evexpr import ExprError, Operand

    name = key.lower().removeprefix("i_")
    if name in _STANDARD:
        text = image_parameter(header, label, name)
        if name in ("title", "pixfile"):
            return Operand("str", text)
        return Operand("int", int(text)) if name != "minpixval" and name != "maxpixval" \
            else Operand("real", np.float32(0.0))
    upper = key.upper()
    if upper not in header:
        raise ExprError(f"image header parameter not found ({key})")
    value = header[upper]
    if isinstance(value, bool):
        return Operand("bool", value)
    if isinstance(value, int):
        return Operand("int", value)
    if isinstance(value, float):
        return Operand("real", np.float32(value))
    return Operand("str", str(value).rstrip())


def _put(header, key: str, text: str, comment: str = "", kind: str | None = None) -> None:
    """impstr / imadd: the card as IRAF writes it, kept valid FITS.

    An existing string card stays a string. Otherwise text that is a number
    is written as the number exactly as given (5., 1.334567); T and F, and
    yes or no in a logical card, are logical; anything else becomes a string
    -- where IRAF would write it unquoted and leave a card no FITS reader
    accepts."""
    from astropy.io import fits

    from ..core import evexpr

    key = key.upper()
    old = header.cards[key] if key in header else None
    keep = comment or (old.comment if old is not None else "")
    if kind is None:
        if old is not None and isinstance(old.value, str):
            kind = "str"
        else:
            number, length = evexpr.lexnum(text)
            if number != evexpr.NON and length == len(text.strip()) and ":" not in text:
                kind = "number"
            elif text in ("T", "F") or (old is not None and isinstance(old.value, bool)
                                        and text in ("yes", "no")):
                kind = "bool"
            else:
                kind = "str"
    elif kind in ("int", "real"):
        kind = "number"
    if len(key) > 8:
        value = (text in ("T", "yes") if kind == "bool" else
                 float(text) if kind == "number" and not text.lstrip("-").isdigit() else
                 int(text) if kind == "number" else text)
        header[key] = (value, keep) if keep else value
        return
    if kind == "bool":
        field = f"{'T' if text in ('T', 'yes') else 'F':>20}"
    elif kind == "number":
        field = f"{text.strip():>20}"
    else:
        field = "'" + f"{text.replace(chr(39), chr(39) * 2):<8}" + "'"
    image = f"{key:<8}= {field}" + (f" / {keep}" if keep else "")
    card = fits.Card.fromstring(image[:80])
    if old is not None:
        index = header.index(key)
        del header[index]
        header.insert(index, card)
    else:
        header.append(card)


def _row(path, keyword, before, action: str) -> dict:
    return {"image": Path(str(path)).name, "keyword": keyword,
            "was": "-" if before is None else str(before)[:28],
            "action": action}


class ImGetsParams(TaskParams):
    image: str = Field(
        ..., description="Image to read; obj.fits[1] names an extension.",
        json_schema_extra={"positional": True}
    )
    param: str = Field(
        ..., description="Keyword to get, in any case. IRAF's standard fields work "
                         "too: naxis, naxis1..7, pixtype, title, pixfile, with or "
                         "without i_ in front.",
        json_schema_extra={"positional": True}
    )
    default: str = Field(
        "0", description="What to print when the keyword is not there; IRAF's is 0."
    )
    extension: int | None = Field(None, ge=0, description="Which HDU's header to read, "
                                                          "if not named in the image.")


#: IMIO's pixel types, as i_pixtype reports them (TY_SHORT, TY_INT, ...).
_PIXTYPE = {8: 3, 16: 3, 32: 4, -32: 6, -64: 7}


def image_parameter(header, filename: str, param: str) -> str | None:
    """imgstr: a header parameter as IRAF's imgets prints it -- the value
    as written in the card (15, 5.581000E3, T), a string without its
    trailing blanks -- or one of IMIO's standard fields. None when absent."""
    key = param.strip().lower()
    field = key[2:] if key.startswith("i_") else key
    naxis = int(header.get("NAXIS", 0))
    if field == "naxis":
        return str(naxis)
    if field.startswith("naxis") and field[5:].isdigit() and 1 <= int(field[5:]) <= 7:
        axis = int(field[5:])
        return str(header.get(f"NAXIS{axis}", 1)) if axis <= naxis else "1"
    if field == "pixtype":
        bitpix = int(header.get("BITPIX", -32))
        bscale, bzero = float(header.get("BSCALE", 1)), float(header.get("BZERO", 0))
        if bitpix == 16 and bscale == 1 and bzero == 32768:
            return "11"                                   # TY_USHORT
        if bscale != 1 or bzero != 0:
            return "6"                                    # scaled: TY_REAL
        return str(_PIXTYPE.get(bitpix, 6))
    if field == "title":
        return str(header.get("OBJECT", "")).rstrip()
    if field == "pixfile":
        return Path(filename).name
    if field in ("minpixval", "maxpixval"):
        # IRAF keeps limits only when it knows them current, which a FITS
        # file never tells it: it answers 0.
        return "0."
    keyword = param.strip().upper()
    if keyword in ("SIMPLE", "BITPIX", "EXTEND", "XTENSION", "PCOUNT", "GCOUNT"):
        return None                                       # the FITS kernel's own
    if keyword not in header:
        return None
    value = header[keyword]
    if isinstance(value, str):
        return value.rstrip()
    card = header.cards[keyword]
    raw = card.image[10:] if len(card.image) >= 10 else ""
    text = raw.split("/", 1)[0].strip()
    return text or str(value)


@register
class ImGets(Task[ImGetsParams]):
    name = "imgets"
    package = "imutil"
    description = "Get one header keyword, for a script to read."
    Params = ImGetsParams

    def plan(self, params: ImGetsParams) -> list[Step]:
        return [
            Step("get", f"{params.param} from {params.image}",
                 why="One keyword, one value, nothing else printed -- so a "
                     "shell loop can use it. hselect is for asking many "
                     "images a question; this is for asking one image one "
                     "thing and getting an answer a script can read."),
        ]

    def execute(self, params: ImGetsParams) -> TaskResult:
        from ..core import images

        spec = images.parse(params.image)
        if params.extension is not None and spec.extension is None:
            spec.extension = params.extension
        header = images.read_header(spec)
        value = image_parameter(header, str(spec.path), params.param)
        found = value is not None
        answer = value if found else params.default

        steps = self.plan(params)
        steps[0].measured = {"value": answer,
                             "found": "yes" if found else "no, the default was used"}
        return TaskResult(
            outputs=[], steps=steps, stdout=[answer],
            warnings=[] if found else [advice.warn(
                f"image header parameter not found ({params.param}); printing "
                f"{params.default!r} instead, as IRAF does",
                f"'harwood imheader {params.image} --long' lists what the header has")],
            data={"value": answer, "found": found}, history=[])


# ------------------------------------------------------------------ imhistogram


def histogram(data, integer: bool, z1=None, z2=None, binwidth=None, nbins: int = 512,
              autoscale: bool = True, top_closed: bool = False,
              hist_type: str = "normal"):
    """imhistogram.x: the histogram as IRAF builds it. Returns the bin
    centres, the counts, and the final z1, z2, nbins and bin width -- or
    None when the image has no data range."""
    import numpy as np

    f32 = np.float32
    values = np.asarray(data, dtype=f32).ravel()
    values = values[np.isfinite(values)]
    if values.size == 0:
        return None
    lo = f32(values.min() if z1 is None else z1)
    hi = f32(values.max() if z2 is None else z2)
    if lo > hi:
        lo, hi = hi, lo
    if binwidth is not None:
        nbins = _nint(f32(f32(hi - lo) / f32(binwidth)))
        hi = f32(lo + f32(f32(nbins) * f32(binwidth)))
    if integer:
        lo_i, hi_i = _nint(lo), _nint(hi)
        if autoscale:
            nlevels = hi_i - lo_i
            nwide = max(1, _nint(f32(f32(nlevels) / f32(nbins))))
            nbins = max(1, _nint(f32(f32(nlevels) / f32(nwide))))
            hi_i = lo_i + nbins * nwide
        if lo_i == hi_i:
            return None
        lo, hi = f32(lo_i), f32(hi_i)
        ints = np.round(values).astype(np.int64)
        inside = ints[(ints >= lo_i) & (ints <= hi_i)]
        dz = f32(f32(nbins) / f32(hi_i - lo_i))
        offsets = (inside - lo_i)
        bins = (offsets if abs(dz - 1) < 2 * 1.192e-7
                else (offsets.astype(f32) * dz).astype(np.int64))
    else:
        if lo == hi:
            return None
        inside = values[(values >= lo) & (values <= hi)]
        dz = f32(f32(nbins) / f32(hi - lo))
        bins = ((inside - lo).astype(f32) if abs(dz - 1) < 2 * 1.192e-7
                else ((inside - lo).astype(f32) * dz).astype(f32)).astype(np.int64)
    counts = np.bincount(bins, minlength=nbins + 1)[:nbins + 1].astype(np.int64)
    if top_closed:
        counts[nbins - 1] += counts[nbins]
    counts = counts[:nbins]
    width = f32(f32(hi - lo) / f32(nbins))
    if hist_type == "cumulative":
        counts = np.cumsum(counts)
    elif hist_type in ("difference", "second_difference"):
        times = 1 if hist_type == "difference" else 2
        for _ in range(times):
            counts = np.append(counts[1:] - counts[:-1], 0)[:nbins]
            nbins -= 1
        shift = f32(width / f32(2)) if times == 1 else width
        lo, hi = f32(lo + shift), f32(hi - shift)
        counts = counts[:nbins]
    centres = np.empty(nbins, dtype=f32)
    start = f32(lo + f32(width / f32(2)))
    for i in range(nbins):
        centres[i] = start
        start = f32(start + width)
    return centres, counts, float(lo), float(hi), nbins, float(width)


def _nint(value) -> int:
    v = float(value)
    return int(math.floor(v + 0.5)) if v >= 0 else -int(math.floor(0.5 - v))


class ImHistogramParams(TaskParams):
    image: str = Field(..., description="Image, or image section, to histogram.")
    z1: float | None = Field(None, description="Lowest value histogrammed. Empty is "
                             "IRAF's INDEF: the image minimum.")
    z2: float | None = Field(None, description="Highest value. Empty: the maximum.")
    binwidth: float | None = Field(None, gt=0, description="Bin width in data units; "
                                   "when given it sets the number of bins.")
    nbins: int = Field(512, ge=1, description="Number of bins, when binwidth is empty.")
    autoscale: bool = Field(True, description="For integer images, make each bin hold "
                            "a whole number of values (no comb).")
    top_closed: bool = Field(False, description="Count the pixels equal to z2 in the "
                             "top bin (bins are otherwise half open).")
    hist_type: Literal["normal", "cumulative", "difference", "second_difference"] = Field(
        "normal", description="The histogram, its running sum, or its first or "
                              "second difference.")
    listout: bool = Field(False, description="List each bin's centre and count instead "
                                             "of plotting.")
    plot_type: Literal["line", "box"] = Field(
        "line", description="line joins the bin centres; box draws each bin as a step.")
    logy: bool = Field(True, description="Plot the counts on a logarithmic axis.")
    device: str = Field(
        "stdgraph", description="Where the plot goes: stdgraph is the window when there "
                                "is a screen, and an SVG file in this run's directory "
                                "when there is not; a name ending in .svg writes there.")


@register
class ImHistogram(Task[ImHistogramParams]):
    """IRAF's imhistogram: the histogram plotted, or with --listout each
    bin's centre and count, as IRAF lists them.

    Integer images get IRAF's treatment -- limits rounded to integers and,
    with autoscale, a bin width that is a whole number of values -- which is
    decided by the image's pixel type, not by its values."""

    name = "imhistogram"
    package = "imutil"
    description = "Histogram of an image's pixel values."
    Params = ImHistogramParams

    def plan(self, params: ImHistogramParams) -> list[Step]:
        bins = (f"bins {params.binwidth:g} wide" if params.binwidth
                else f"{params.nbins} bins")
        return [Step("histogram", f"{params.image}: {params.hist_type}, {bins}",
                     why="The histogram is how the sky level, the saturation and "
                         "a bad bias show at a glance.")]

    def execute(self, params: ImHistogramParams) -> TaskResult:
        import numpy as np

        from ..core import examine, images

        data, header = images.read(params.image)
        integer = examine.native(data, header) is not data
        made = histogram(data, integer, params.z1, params.z2, params.binwidth,
                         params.nbins, params.autoscale, params.top_closed,
                         params.hist_type)
        if made is None:
            advice.fail(f"image '{params.image}' has no data range",
                        "give z1 and z2 around the values it does have")
        centres, counts, z1, z2, nbins, width = made
        rows = [{"value": float(c), "count": int(n)}
                for c, n in zip(centres, counts, strict=True)]
        summary = {"z1": z1, "z2": z2, "nbins": nbins, "width": width}
        if params.listout:
            return TaskResult(data={"rows": rows, **summary}, steps=self.plan(params))

        from ..core import graph, iraffmt

        kind = {"normal": "Histogram", "cumulative": "Cumulative histogram",
                "difference": "Difference histogram",
                "second_difference": "Second difference histogram"}[params.hist_type]
        title = str(header.get("OBJECT", "")).strip()
        plot = graph.Graph(
            [f"harwood -- {kind} of {params.image} = {title}",
             f"From z1={iraffmt.g(z1)} to z2={iraffmt.g(z2)}, nbins={nbins}, "
             f"width={iraffmt.g(width)}"],
            xlabel="value", ylabel="count", logy=params.logy)
        values = np.asarray(counts, dtype=float)
        if params.plot_type == "line":
            plot.line(np.asarray(centres, dtype=float), values)
        else:
            plot.steps(z1 + width * np.arange(nbins + 1), values)
        plot.scale_to(float(z1), float(z2))
        where = _plot(plot, params.device, f"imhistogram · {params.image}",
                      Path(str(images.parse(params.image).path)).stem + ".hist.svg")
        return TaskResult(data={"bins": rows, **summary}, steps=self.plan(params),
                          outputs=[where] if where else [])


def _plot(plot, device: str, title: str, name: str):
    """IRAF's stdgraph: the window when there is a screen to put one on,
    otherwise (or for a device ending in .svg) a file. Returns the file."""
    from ..core import display, output

    if device.endswith(".svg"):
        return plot.write(device)
    if device not in ("stdgraph", ""):
        advice.fail(f"no graphics device '{device}'",
                    "use stdgraph (the window, or a file without a screen) or a name "
                    "ending in .svg")
    if display._autodetect() == "native":
        from ..viewer.graph_panel import run

        run(plot, title)
        return None
    return plot.write(output.directory("imhistogram") / name)


# ------------------------------------------------------------------- imrename

#: Extensions a FITS image may carry, and those of IRAF's other image formats,
#: which a rename cannot turn a FITS file into.
_FITS_SUFFIXES = (".fits", ".fit", ".fts")
_OTHER_IMAGE_SUFFIXES = (".imh", ".hhh", ".pl", ".qp")


def _renamed(old: Path, new: str) -> Path:
    """iki_rename's new name: the new root with the old image's extension,
    unless the new name already carries an extension valid for it."""
    path = Path(new)
    suffix = path.suffix.lower()
    if suffix in _FITS_SUFFIXES or path.name.lower().endswith(".fits.gz"):
        return path
    if suffix in _OTHER_IMAGE_SUFFIXES:
        path = path.with_suffix("")
    old_suffix = "".join(old.suffixes[-2:]) if old.name.endswith(".gz") else old.suffix
    return path.with_name(path.name + old_suffix)


class ImRenameParams(TaskParams):
    oldnames: list[str] = Field(..., description="Images to rename. Globs and @lists.")
    newnames: list[str] = Field(..., description="New names, one per image -- or one "
                                "directory, to move them there keeping their names. A "
                                "name without an extension keeps the old one's.")
    verbose: bool = Field(False, description="Report each rename.")


@register
class ImRename(Task[ImRenameParams]):
    """IRAF's imrename: rename images, or move them into a directory.

    As in IRAF, an existing image is never overwritten (renaming an image to
    itself is allowed). Unlike IRAF, which warns and goes on, harwood checks
    every name first and renames nothing when one would clobber: a night half
    renamed is harder to sort out than one not renamed at all."""

    name = "imrename"
    package = "imutil"
    description = "Rename images, or move them into a directory."
    Params = ImRenameParams

    def _pairs(self, params: ImRenameParams) -> list[tuple[Path, Path]]:
        from ..core import images

        specs = images.expand(params.oldnames)
        if not specs:
            advice.fail(f"'{' '.join(params.oldnames)}' matches no image",
                        "check the names, or run 'harwood sections' on the template",
                        FileNotFoundError)
        for spec in specs:
            if spec.section or spec.extension is not None:
                advice.fail(f"'{spec}' is part of a file; a rename moves whole files",
                            "name the file itself, without the [...]")
        if len(params.newnames) == 1 and Path(params.newnames[0]).is_dir():
            directory = Path(params.newnames[0])
            return [(spec.path, directory / spec.path.name) for spec in specs]
        if len(params.newnames) != len(specs):
            advice.fail(f"different number of old ({len(specs)}) and new "
                        f"({len(params.newnames)}) image names",
                        "give one new name per image, or a single directory")
        pairs = [(spec.path, _renamed(spec.path, new))
                 for spec, new in zip(specs, params.newnames, strict=True)]
        targets = [new for _, new in pairs]
        clashes = [str(new) for old, new in pairs
                   if new.exists() and new.resolve() != old.resolve()]
        clashes += sorted({str(t) for t in targets if targets.count(t) > 1})
        if clashes:
            advice.fail("renaming would overwrite " + ", ".join(clashes[:6])
                        + (f" and {len(clashes) - 6} more" if len(clashes) > 6 else ""),
                        "choose new names, or delete those images first (imdelete)")
        return pairs

    def plan(self, params: ImRenameParams) -> list[Step]:
        pairs = self._pairs(params)
        shown = ", ".join(f"{old.name} -> {new}" for old, new in pairs[:4])
        return [Step("rename", shown + (f" and {len(pairs) - 4} more"
                                        if len(pairs) > 4 else ""),
                     why="Names are how a night is found again; IRAF's imrename never "
                         "overwrites an image, and neither does this.")]

    def execute(self, params: ImRenameParams) -> TaskResult:
        pairs = self._pairs(params)
        steps = self.plan(params)
        for old, new in pairs:
            if new.resolve() != old.resolve():
                new.parent.mkdir(parents=True, exist_ok=True)
                old.rename(new)
        rows = [{"from": str(old), "to": str(new)} for old, new in pairs]
        return TaskResult(outputs=[new for _, new in pairs], steps=steps,
                          data={"rows": rows} if params.verbose
                          else {"renamed": len(pairs)})


# ------------------------------------------------------------------ minmax

class MinMaxParams(TaskParams):
    images: list[str] = Field(
        ..., description="Images to examine. Globs, @lists and sections work.",
        json_schema_extra={"positional": True},
    )
    force: bool = Field(False, description="Look at the pixels even when the header "
                                           "already has DATAMIN or DATAMAX.")
    update: bool = Field(True, description="Write DATAMIN and DATAMAX in the header. "
                                           "Never done for a section.")
    verbose: bool = Field(True, description="Print each image's values.")


def _kept_axes(spec, shape: tuple[int, ...]) -> list[int]:
    """The FITS axes left of a section, as imio sees it: an axis cut down to
    one pixel is no longer an axis."""
    axes = list(range(1, len(shape) + 1))
    if not spec.section:
        return axes
    return [axis for axis in axes if shape[len(shape) - axis] > 1] or [1]


def _extremes(data, bitpix: int):
    """im_vminmax: the values as the pixel type holds them and, for each, where
    it is first met, line by line. Without a pixel with a value, None."""
    import numpy as np

    values = data.astype(np.float32) if bitpix == -32 else data
    finite = np.isfinite(values)
    if not finite.any():
        return None, None, int(values.size)
    low = int(np.argmin(np.where(finite, values, np.inf)))
    high = int(np.argmax(np.where(finite, values, -np.inf)))
    return (float(values.flat[low]), low), (float(values.flat[high]), high), \
        int(values.size - finite.sum())


def _pixel(index: int, shape: tuple[int, ...], axes: list[int]) -> str:
    import numpy as np

    where = np.unravel_index(index, shape)
    return "[" + ",".join(str(int(where[len(shape) - axis]) + 1) for axis in axes) + "]"


@register
class MinMax(Task[MinMaxParams]):
    """IRAF's minmax: the extreme pixels of an image and where they are, from
    the pixels, or from DATAMIN and DATAMAX when the header has either."""

    name = "minmax"
    package = "imutil"
    description = "Minimum and maximum pixel of an image, and where they are."
    Params = MinMaxParams

    def plan(self, params: MinMaxParams) -> list[Step]:
        return [Step("measure", ", ".join(params.images), why=WHY["extremes"])]

    def execute(self, params: MinMaxParams) -> TaskResult:
        import numpy as np

        from ..core import images, iraffmt

        f32 = np.float32
        rows, lines, warnings, changed = [], [], [], []
        for spec in images.expand(params.images):
            if not spec.path.exists():
                warnings.append(advice.warn(f"Cannot open image ({spec})",
                                            "check the name; the others were examined"))
                continue
            header = images.read_header(spec)
            row = {"image": str(spec), "minval": None, "maxval": None,
                   "minpix": "", "maxpix": ""}
            # Equal limits, or one missing and the other 0, say nothing to IRAF.
            limits = header.get("DATAMIN", 0.0) != header.get("DATAMAX", 0.0)
            if int(header.get("NAXIS", 0)) == 0:
                pass
            elif limits and not params.force and not spec.section:
                # A real in IRAF's image header, whatever the card says.
                row["minval"] = float(f32(header.get("DATAMIN", 0.0)))
                row["maxval"] = float(f32(header.get("DATAMAX", 0.0)))
            else:
                data, read = images.read(spec)
                low, high, bad = _extremes(data, int(read.get("BITPIX", -32)))
                axes = _kept_axes(spec, data.shape)
                if bad:
                    warnings.append(advice.note(
                        f"{spec}: {bad} pixels are NaN or infinite and were left out",
                        "IRAF reads them as 0, so its minimum or maximum can be that 0"))
                if low is not None:
                    row.update(minval=low[0], maxval=high[0],
                               minpix=_pixel(low[1], data.shape, axes),
                               maxpix=_pixel(high[1], data.shape, axes))
                    if params.update and not spec.section:
                        try:
                            # IRAF writes them with %E, seven figures.
                            images.set_keywords(spec, {
                                "DATAMIN": float(f"{float(f32(low[0])):.6e}"),
                                "DATAMAX": float(f"{float(f32(high[0])):.6e}")})
                            changed.append(spec.path)
                        except PermissionError:
                            pass
            rows.append(row)
            lines.append(f"    {spec} {row['minpix']} {iraffmt.g(row['minval'], True)} "
                         f"{row['maxpix']} {iraffmt.g(row['maxval'], True)}")
        last = rows[-1] if rows else {}
        return TaskResult(data={"rows": rows, **{k: last[k] for k in last if k != "image"}},
                          stdout=lines if params.verbose else [], warnings=warnings,
                          outputs=changed, steps=self.plan(params))


# -------------------------------------------------------------- listpixels

class ListPixelsParams(TaskParams):
    images: list[str] = Field(
        ..., description="Images to list. Globs, @lists and sections work.",
        json_schema_extra={"positional": True},
    )
    wcs: Literal["logical", "physical", "world", ""] = Field(
        "logical", description="The coordinates to print: logical, the pixel in the image "
                               "or section; physical, in the original image; world, what "
                               "the header's WCS says.")
    formats: str = Field(
        "", description="A format for each axis, as in IRAF: '%8.2f %h'. An axis without "
                        "one gets %0.15g, or the WCS's format when --wcs world.")
    verbose: bool = Field(False, description="Print a line naming each image first.")


@register
class ListPixels(Task[ListPixelsParams]):
    """IRAF's listpixels: one line for each pixel, its coordinates and value.
    The lines go to standard output alone, for a file or another program."""

    name = "listpixels"
    package = "imutil"
    description = "Each pixel of an image as a line of text: coordinates and value."
    Params = ListPixelsParams

    def plan(self, params: ListPixelsParams) -> list[Step]:
        return [Step("list", f"{', '.join(params.images)} in {params.wcs or 'logical'}",
                     why=WHY["list"])]

    def execute(self, params: ListPixelsParams) -> TaskResult:
        import numpy as np

        from ..core import images, iraffmt, mwcs
        from ..core.implot import _wat_attributes

        wcs = params.wcs or "logical"
        words = params.formats.split()
        given = [iraffmt.Format(word) for word in words]
        specs = images.expand(params.images)
        lines, warnings = [], []
        for spec in specs:
            if not spec.path.exists():
                advice.fail(f"Cannot open image ({spec})", "check the name")
            data, header = images.read(spec)
            axes = _kept_axes(spec, data.shape)
            if params.verbose:
                lines += ["", f"#Image: {spec}  Wcs: {wcs}", ""]
            grid = np.ones((data.ndim, data.size))
            indices = np.indices(data.shape).reshape(data.ndim, -1) + 1.0
            for axis in axes:
                grid[axis - 1] = indices[data.ndim - axis]
            coords = mwcs.transform(header, wcs, grid)
            columns = []
            for i, axis in enumerate(axes):
                if i < len(given) and given[i].text:
                    fmt = given[i]
                else:
                    own = wcs == "world" and _wat_attributes(header, axis).get("format")
                    fmt = iraffmt.Format(own or "%0.15g")
                cache: dict[float, str] = {}
                columns.append([cache[c] if c in cache else cache.setdefault(c, fmt(c))
                                for c in coords[axis - 1].tolist()])
            double = int(header.get("BITPIX", -32)) == -64
            values = data if double else data.astype(np.float32)
            finite = np.isfinite(values)
            if not finite.all():
                warnings.append(advice.note(
                    f"{spec}: {int(finite.size - finite.sum())} pixels are NaN or infinite, "
                    "listed as INDEF",
                    "IRAF reads them as 0, so its list has a 0 there"))
            texts: dict[float, str] = {}
            printed = []
            for value in values.ravel().tolist():
                printed.append(texts[value] if value in texts
                               else texts.setdefault(value, iraffmt.g(value, double)))
            lines += [" ".join(row) + "  " + text
                      for row, text in zip(zip(*columns, strict=True), printed, strict=True)]
        return TaskResult(data={"images": len(specs)}, stdout=lines, warnings=warnings,
                          steps=self.plan(params))


# ------------------------------------------------------------------ imexpr

class ImExprParams(TaskParams):
    expr: str = Field(
        ..., description="The expression, or @file. a to z are the operands, I J K the "
                         "pixel's coordinates.",
        json_schema_extra={"positional": True},
    )
    output: str = Field(
        ..., description="The output image; with a section, written into an existing image.",
        json_schema_extra={"positional": True},
    )
    dims: str = Field("auto", description="The output's axes, as 512,512. By default "
                                          "those the input images have in common.")
    intype: str = Field("auto", description="The least type an input image is read as: int, "
                                            "real, double. By default its own.")
    outtype: str = Field("auto", description="The pixel type of the output: ushort, short, "
                                             "int, long, real, double, ref, or by default "
                                             "the expression's.")
    refim: str = Field("auto", description="The operand whose header goes to the output. By "
                                           "default the first image of most dimensions.")
    rangecheck: bool = Field(True, description="Division by zero, and the log or root of what "
                                               "has none, give 0 instead of stopping.")
    exprdb: str = Field("none", description="A file of macros for the expression.")
    a: str | None = Field(None, description="Operand a: an image, a number, or b.KEYWORD.")
    b: str | None = Field(None, description="Operand b.")
    c: str | None = Field(None, description="Operand c.")
    d: str | None = Field(None, description="Operand d.")
    e: str | None = Field(None, description="Operand e.")
    f: str | None = Field(None, description="Operand f.")
    g: str | None = Field(None, description="Operand g.")
    h: str | None = Field(None, description="Operand h.")
    i: str | None = Field(None, description="Operand i.")
    j: str | None = Field(None, description="Operand j.")
    k: str | None = Field(None, description="Operand k.")
    l: str | None = Field(None, description="Operand l.")  # noqa: E741
    m: str | None = Field(None, description="Operand m.")
    n: str | None = Field(None, description="Operand n.")
    o: str | None = Field(None, description="Operand o.")
    p: str | None = Field(None, description="Operand p.")
    q: str | None = Field(None, description="Operand q.")
    r: str | None = Field(None, description="Operand r.")
    s: str | None = Field(None, description="Operand s.")
    t: str | None = Field(None, description="Operand t.")
    u: str | None = Field(None, description="Operand u.")
    v: str | None = Field(None, description="Operand v.")
    w: str | None = Field(None, description="Operand w.")
    x: str | None = Field(None, description="Operand x.")
    y: str | None = Field(None, description="Operand y.")
    z: str | None = Field(None, description="Operand z.")


@register
class ImExpr(Task[ImExprParams]):
    """IRAF's imexpr: an image from an expression over images, numbers and keywords,
    evaluated a line at a time with IRAF's types."""

    name = "imexpr"
    package = "imutil"
    description = "An image computed from an expression, IRAF's types and functions."
    Params = ImExprParams

    def plan(self, params: ImExprParams) -> list[Step]:
        return [Step("evaluate", params.expr, why=WHY["expression"]),
                Step("write", params.output, why=WHY["expression_type"])]

    def execute(self, params: ImExprParams) -> TaskResult:
        import numpy as np
        from astropy.io import fits

        from ..core import evvexpr, images, imexpr, provenance

        values = {c: getattr(params, c) for c in "abcdefghijklmnopqrstuvwxyz"}
        try:
            result = imexpr.run(params.expr, values, params.output, params.dims, params.intype,
                                params.outtype, params.refim, params.rangecheck, params.exprdb)
        except evvexpr.ExprError as e:
            advice.fail(f"in the expression: {e}", f"the expression was: {params.expr}")
        target = images.parse(params.output)
        line = f"imexpr: {result.expression}"
        if target.path.suffix == ".pl" and not target.section:
            from ..core import plio

            header = (None if result.header is None
                      else images.forget_limits(result.header.copy()))
            plio.write(target.path, result.data, header=header)
        elif target.section:
            with fits.open(str(target.path), mode="update", memmap=False) as hdul:
                hdu = hdul[images._resolve_extension(hdul, target.extension)]
                section = hdu.data[result.section]
                hdu.data[result.section] = result.data.reshape(section.shape)
                provenance.stamp_header(hdu.header, [line])
        else:
            header = fits.Header()
            if result.header is not None:
                from ..core import mef

                header = mef.merged(*mef.split(result.header))
                for keyword in ("BSCALE", "BZERO", "BLANK"):
                    header.remove(keyword, ignore_missing=True)
            provenance.stamp_header(header, [line])
            target.path.parent.mkdir(parents=True, exist_ok=True)
            fits.PrimaryHDU(np.asarray(result.data), header).writeto(target.path,
                                                                     overwrite=True)
        shape = "x".join(str(n) for n in reversed(result.data.shape))
        return TaskResult(outputs=[target.path], steps=self.plan(params), history=[line],
                          data={"expression": result.expression, "shape": shape,
                                "type": result.type})


# ------------------------------------------------------------------ imreplace

IMRWHY = {
    "read": "Each image the template matches. imreplace is meant for a whole set at "
            "once -- the same bad values out of every frame of a night.",
    "window": "Pixels from lower to upper, both included; a side left empty has no "
              "limit, and neither side means every pixel. In an integer image the "
              "limits are made integers as IRAF makes them: upper truncated, lower "
              "truncated when whole and raised by one when not -- for a negative "
              "limit that is not whole, that is one off where it reads (-100.5 "
              "starts at -99). The value is truncated toward zero too.",
    "radius": "Every pixel within the radius of one in the window is replaced as "
              "well: the edge of a bad region, which is bad without being far off.",
    "write": "The pixel type is kept, as IRAF keeps it in place. The input is not "
             "modified unless --output names it.",
}


class ImReplaceParams(TaskParams):
    images: list[str] = Field(
        ..., description="Images to edit. Globs, @lists and an image section work; a "
                         "section changes only its pixels.",
        json_schema_extra={"positional": True},
    )
    value: float = Field(
        ..., description="The value the pixels get. A negative one goes after --.",
        json_schema_extra={"positional": True},
    )
    lower: float | None = Field(None, description="Lower limit of the window. Empty is "
                                                  "IRAF's INDEF: no lower limit.")
    upper: float | None = Field(None, description="Upper limit of the window. Empty is "
                                                  "IRAF's INDEF: no upper limit.")
    radius: float = Field(0.0, ge=0.0, description="Also replace every pixel within this "
                                                   "many pixels of one in the window. "
                                                   "Below 1, none.")
    imaginary: float = Field(0.0, description="IRAF's imaginary part, for a complex "
                                              "image. FITS images are not complex: it "
                                              "changes nothing, and is kept for epar.")
    output: str = Field(
        "", description="Where to write: a name for one image, a,b,c or @list one per "
                        "image, %old%new%, a directory (ending in /) or a prefix. Left "
                        "out, this run's own directory. IRAF edits in place; name the "
                        "input itself to do the same.")
    uncertainty: Literal["VAR", "ERR", "UNCERT"] = uncertainty_field()
    format: Literal["harwood", "iraf"] = format_field()


@register
class ImReplace(Task[ImReplaceParams]):
    """IRAF's imreplace, from images/imutil/src/imrep.gx: pixels in a window, and
    with a radius those near them, set to a value. What the window is in an
    integer image, how value lands in it, and what the radius writes are IRAF's,
    defects included (core/imreplace.py lists them); checked pixel for pixel
    against IRAF 2.18 on every pixel type it writes."""

    name = "imreplace"
    package = "imutil"
    description = "Replace the pixels in a window of values by a constant."
    Params = ImReplaceParams

    def plan(self, params: ImReplaceParams) -> list[Step]:
        low = "INDEF" if params.lower is None else f"{params.lower:g}"
        high = "INDEF" if params.upper is None else f"{params.upper:g}"
        steps = [Step("read", ", ".join(params.images), why=IMRWHY["read"]),
                 Step("window", f"{low} to {high} -> {params.value:g}",
                      why=IMRWHY["window"])]
        if params.radius >= 1.0:
            steps.append(Step("radius", f"{params.radius:g} pixels", why=IMRWHY["radius"]))
        steps.append(Step("write", params.output or "this run's directory",
                          why=IMRWHY["write"]))
        return steps

    def execute(self, params: ImReplaceParams) -> TaskResult:
        import dataclasses

        import numpy as np

        from ..core import images, imreplace, mef, output, pixels
        from .crutil import _destination, _named

        specs = images.expand(params.images)
        if not specs:
            advice.fail(f"'{' '.join(params.images)}' matches no image",
                        "check the name, or run `harwood sections` on the template",
                        FileNotFoundError)
        named = _named(params.output, specs)
        written, rows, warnings, total = [], [], [], 0
        for spec in specs:
            hdul, hdu = images.open_hdu(spec)
            try:
                stored = np.array(hdu.data)
                header = images.inherited(hdul, hdu)
                index = hdul.index(hdu)
            finally:
                hdul.close()
            # astropy rewrites BITPIX once it has scaled the data: the file's own.
            from astropy.io import fits

            bitpix = int(fits.getheader(spec.path, index).get("BITPIX", 0))
            scaled = bitpix > 0 and stored.dtype.kind == "f"
            whole = dataclasses.replace(spec, section=None)
            _, _, var, quality = mef.load(whole)
            kind = imreplace.working_type(stored.dtype)
            if stored.dtype.kind == "u" and stored.dtype.itemsize == 1 or scaled:
                warnings.append(advice.note(
                    f"{spec}: IRAF cannot write {'a scaled' if scaled else 'an 8-bit'} "
                    "image back in place and stops; harwood writes it as IRAF reads it, "
                    f"{'real' if scaled else 'short'}.",
                    "nothing to do: the values are the ones IRAF would compare; only "
                    "the file's pixel type differs."))
                target = np.float32 if scaled else np.int16
            else:
                target = stored.dtype.newbyteorder("=")
            work = np.array(stored, dtype=imreplace.DTYPES[kind])
            where = (Ellipsis,)
            if spec.section:
                where = pixels.parse_section(spec.section.strip("[]"), work.shape)
            before = work[where].copy()
            work[where] = imreplace.replace(before, params.value, params.lower,
                                            params.upper, params.radius, kind)
            note = _imreplace_unwritten(work[where].shape, params)
            if note and not any(w.message == note.message for w in warnings):
                warnings.append(note)
            changed = int(np.count_nonzero(work[where] != before))
            total += changed
            result = work.astype(target)
            low = "INDEF" if params.lower is None else f"{params.lower:g}"
            high = "INDEF" if params.upper is None else f"{params.upper:g}"
            header.add_history(f"imreplace: {changed} pixels from {low} to {high}"
                               + (f" and within {params.radius:g}"
                                  if params.radius >= 1 else "")
                               + f" set to {params.value:g}")
            name = Path(spec.path).name
            stem = name[:-5] if name.endswith(".fits") else name
            destination = named.get(str(spec.path)) or _destination(
                params.output, stem, ".fits", spec, output)
            files, notes = mef.save(destination, result, header, var, quality,
                                    params.uncertainty, params.format, keep_type=True)
            written.extend(files)
            warnings.extend(n for n in notes if n not in warnings)
            rows.append({"image": str(spec), "type": kind, "changed": changed})

        steps = self.plan(params)
        steps[1].measured = {"images": len(specs), "pixels changed": total}
        return TaskResult(outputs=written, steps=steps, data={"rows": rows},
                          warnings=warnings, history=[f"imreplace: {total} pixels"])


def _imreplace_unwritten(shape, params: ImReplaceParams):
    """The radius cases where IRAF writes less than it marked, said: harwood writes
    what IRAF writes, and the image stays as it was where IRAF leaves it."""
    import numpy as np

    if params.radius < 1.0 or (params.lower is None and params.upper is None):
        return None
    nradius = int(np.float32(params.radius))
    if len(shape) == 1 or shape[-2] <= nradius:
        what = "a 1-D image" if len(shape) == 1 else \
            f"an image of {shape[-2]} lines with a radius of {params.radius:g}"
        return advice.warn(
            f"IRAF's imreplace writes nothing on {what}: a line is written only "
            "once the one int(radius) further on has been read. harwood does the "
            "same, and the image is left as it was.",
            "use a radius smaller than the number of lines, or imexpr for a 1-D "
            "image")
    if len(shape) > 2:
        return advice.warn(
            "IRAF's imreplace with a radius on a cube leaves the last lines of the "
            "last band as they were (its line counter goes back to 0 at each band's "
            "end, so they are never written), and the other bands' last lines "
            "written late. harwood does the same.",
            "replace band by band, with a section such as [*,*,2]")
    return None


# ---------------------------------------------------------------------- imsum

IMSWHY = {
    "read": "Every image, which must all have the same shape. The first one's header "
            "is the output's, as IRAF's NEW_COPY makes it.",
    "types": "What the pixels are added in, and what the output is written as, decide "
             "the result: an integer calculation wraps a sum too large for it and "
             "truncates an average, and a real result written as an integer image is "
             "truncated. By default the calculation is in the highest type of the "
             "inputs (a short and an unsigned short together, int), and the output in "
             "that.",
    "reject": "The lowest and highest values of each pixel are left out before adding: "
              "a number from 1 on, a fraction of the images below 1. The median keeps "
              "the middle one -- of an even number, the upper of the two in the middle.",
    "write": "IRAF's imsum keeps no uncertainty, and neither does this: imcombine adds "
             "the variance planes of a harwood night.",
}


class ImSumParams(TaskParams):
    input: list[str] = Field(
        ..., description="Images to add. Globs, @lists, a,b,c and sections work.",
        json_schema_extra={"positional": True},
    )
    output: str = Field(..., description="The output image, its extension written.",
                        json_schema_extra={"positional": True})
    title: str = Field("", description="Title (OBJECT) for the output. Empty keeps the "
                                       "first image's.")
    hparams: str = Field("", description="Header keywords to add up (or average) as "
                                         "well, a,b,c with * and ? patterns; at most "
                                         "ten, as in IRAF. Not with the median.")
    pixtype: str = Field("", description="Output pixel type: ushort, short, int, long, "
                                         "real or double, or its first letter. Empty is "
                                         "the calculation's.")
    calctype: str = Field("", description="Calculation type: short, int, long, real or "
                                          "double, or its first letter. Empty is the "
                                          "highest of the inputs'.")
    option: Literal["sum", "average", "median"] = Field(
        "sum", description="Add the pixels, average them, or take their median.")
    low_reject: float = Field(0.0, ge=0.0, description="Low pixels to reject: a number, "
                                                       "or a fraction of the images "
                                                       "below 1.")
    high_reject: float = Field(0.0, ge=0.0, description="High pixels to reject: a "
                                                        "number, or a fraction of the "
                                                        "images below 1.")
    verbose: bool = Field(False, description="Print IRAF's log of the operation.")
    format: Literal["harwood", "iraf"] = format_field()


@register
class ImSum(Task[ImSumParams]):
    """IRAF's imsum, from images/imutil/src/t_imsum.x and imsum.gx: the sum, average
    or median of images with the highest and lowest pixels rejected, in the
    calculation and pixel types IRAF chooses, with the rejection, the median and
    the fifteen-at-a-time sum as IRAF does them (core/imsum.py lists what
    follows). Checked pixel for pixel against IRAF 2.18."""

    name = "imsum"
    package = "imutil"
    description = "Sum, average or median of images, with high and low rejection."
    Params = ImSumParams

    def plan(self, params: ImSumParams) -> list[Step]:
        rejected = (params.option == "median" or params.low_reject or params.high_reject)
        steps = [Step("read", ", ".join(params.input), why=IMSWHY["read"]),
                 Step("types", f"calculation {params.calctype or 'from the inputs'}, "
                               f"output {params.pixtype or 'as the calculation'}",
                      why=IMSWHY["types"])]
        if rejected:
            steps.append(Step("reject", "the median" if params.option == "median" else
                              f"{params.low_reject:g} low, {params.high_reject:g} high",
                              why=IMSWHY["reject"]))
        steps.append(Step("write", params.output, why=IMSWHY["write"]))
        return steps

    def execute(self, params: ImSumParams) -> TaskResult:
        import numpy as np
        from astropy.io import fits

        from ..core import images, imsum, mef, pixels

        specs = images.expand(params.input)
        if not specs:
            advice.fail("No input images in list",
                        f"'{' '.join(params.input)}' matches nothing: check the names, "
                        "or run `harwood sections` on the template", FileNotFoundError)
        try:
            option = params.option
            nlow, nhigh = imsum.rejections(option, len(specs), params.low_reject,
                                           params.high_reject)
            frames, kinds, header, uncertain = [], [], None, []
            for spec in specs:
                hdul, hdu = images.open_hdu(spec)
                try:
                    data = np.array(hdu.data)
                    if header is None:
                        header = images.inherited(hdul, hdu)
                    index = hdul.index(hdu)
                    if any(h.name in ("VAR", "ERR", "UNCERT") for h in hdul):
                        uncertain.append(str(spec))
                finally:
                    hdul.close()
                bitpix = int(fits.getheader(spec.path, index).get("BITPIX", 0))
                if spec.section:
                    data = data[pixels.parse_section(spec.section.strip("[]"), data.shape)]
                frames.append(data)
                scaled = bitpix > 0 and data.dtype.kind == "f"
                kinds.append(imsum.pixel_type(data.dtype, scaled))
            if len({frame.shape for frame in frames}) > 1:
                raise imsum.ImsumError("Images have different dimensions or sizes")
            calc, pix = imsum.types(kinds, params.calctype, params.pixtype)
        except imsum.ImsumError as error:
            fix = {"Unrecognized datatype": "give short, int, long, real or double for "
                   "--calctype (and ushort too for --pixtype), or the first letter",
                   "Images have different dimensions or sizes": "give images of one "
                   "shape, or sections of one size",
                   "Number of pixels rejected >= number of images": "reject fewer, so "
                   "that at least one value of each pixel is left"}.get(str(error), "")
            advice.fail(str(error), fix)

        text = []
        if params.verbose:
            text = ["IMSUM:", "  Input images:", *(f"    {s}" for s in specs),
                    f"  Output image: {params.output}",
                    f"  Header parameters: {params.hparams}",
                    f"  Output pixel datatype: {imsum.NAMES[pix]}",
                    f"  Calculation type: {imsum.NAMES[calc]}",
                    f"  Option: {option}",
                    f"  Low rejection: {nlow}", f"  High rejection: {nhigh}"]
        work = [imsum.read_as(frame, kind, calc) for frame, kind in zip(frames, kinds,
                                                                        strict=True)]
        result = imsum.combine(work, calc, pix, option, nlow, nhigh)

        if params.title:
            header["OBJECT"] = params.title
        if params.hparams.strip() and option in ("sum", "average"):
            _imsum_hparams(header, specs, params.hparams, option)
        header.add_history(f"imsum: {option} of {len(specs)} images"
                           + (f", {nlow} low and {nhigh} high rejected"
                              if (nlow or nhigh) and option != "median" else ""))
        files, notes = mef.save(params.output, result, header, None, None,
                                format=params.format, keep_type=True)
        warnings = list(notes)
        if uncertain:
            warnings.append(advice.note(
                f"{len(uncertain)} of the inputs carry an uncertainty plane, which imsum "
                "leaves behind, as IRAF's has nowhere to keep one.",
                "use imcombine to add the frames with their variance"))
        steps = self.plan(params)
        steps[1].measured = {"calculation": imsum.NAMES[calc], "output": imsum.NAMES[pix]}
        data = {"images": len(specs), "calculation": imsum.NAMES[calc],
                "pixtype": imsum.NAMES[pix], "low rejected": nlow, "high rejected": nhigh}
        if text:
            data["text"] = text
        return TaskResult(outputs=files, steps=steps, data=data, warnings=warnings,
                          history=[f"imsum: {option} of {len(specs)} images"])


def _imsum_hparams(header, specs, patterns: str, option: str) -> None:
    """imsum_hparam: the first ten keywords of the output's header the patterns
    match, summed over the images (averaged over all of them, rejection or not)
    and written as imputd writes them."""
    import fnmatch

    from ..core import images, imsum

    wanted = [p.strip().upper() for p in patterns.split(",") if p.strip()]
    fields = []
    for key in header:
        if key in ("", "COMMENT", "HISTORY") or _STRUCTURAL.match(key):
            continue
        if any(fnmatch.fnmatchcase(key.upper(), p) for p in wanted) and key not in fields:
            fields.append(key)
        if len(fields) == 10:
            break
    totals = dict.fromkeys(fields, 0.0)
    for spec in specs:
        own = images.read_header(spec)
        for key in fields:
            if key not in own:
                advice.fail(f"Image header parameter not found ({key} in {spec})",
                            "give --hparams keywords every input has", KeyError)
            try:
                totals[key] += imsum.hparam_value(own.cards[key])
            except imsum.ImsumError:
                advice.fail(f"{key} in {spec} is not a number: {own[key]!r}",
                            "give --hparams keywords with numeric values", ValueError)
    for key, total in totals.items():
        value = total / len(specs) if option == "average" else total
        header[key] = float(imsum.hparam_format(value))
