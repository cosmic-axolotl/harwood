"""imutil -- the small tasks a reduction leans on constantly.

`imheader`, `hselect`, `imstatistics`, `imarith`. None of them is hard; all of
them are what turns a directory of FITS files into a data set you can reason
about. `hselect` in particular has no command-line equivalent in the Python
ecosystem, and it is how a night of observing gets sorted.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Literal

from pydantic import Field

from ..core import advice
from ..core.registry import register
from ..core.task import Step, Task, TaskParams, TaskResult, Template

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
                header = hdu.header
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
        description="Keywords to print for each image that matches, e.g. 'OBJECT,EXPTIME'.",
    )
    expr: str = Field(
        "yes",
        description=(
            "Boolean expression over the header, e.g. \"IMAGETYP == 'zero'\" or "
            "\"EXPTIME > 30 and FILTER == 'V'\". 'yes' selects everything. "
            "Keywords are Python names here, and the comparison is Python's."
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
        from ..core import images

        matched = []
        for spec in images.expand(params.input):
            hdul, hdu = images.open_hdu(spec)
            try:
                header = dict(hdu.header.items())
            finally:
                hdul.close()
            if not _matches(params.expr, header):
                continue
            matched.append(
                {"image": str(spec),
                 **{field: header.get(field, params.missing) for field in params.fields}}
            )
        return TaskResult(data={"rows": matched, "matched": len(matched)})


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


def _matches(expr: str, header: dict) -> bool:
    """Evaluate a header expression.

    The namespace holds only the header's own keywords, and builtins are
    removed: a header expression is a filter, not a place to run code.
    """
    if expr.strip().lower() in ("yes", "true", "1", ""):
        return True
    namespace = {str(key): value for key, value in header.items() if str(key).isidentifier()}
    try:
        return bool(eval(expr, {"__builtins__": {}}, namespace))  # noqa: S307
    except NameError:
        return False           # a keyword this image does not have
    except Exception as error:
        raise ValueError(f"cannot evaluate expression {expr!r}: {error}") from None


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
        0.0, description="Value written where a division by zero would happen."
    )


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
        import numpy as np
        from astropy.io import fits

        from ..core import images, provenance

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

        outputs, history = [], []
        for index, first in enumerate(left):
            second = right[0] if len(right) == 1 else right[index]
            a, header_a = _read(first)
            b, header_b = _read(second)
            header = (header_a if header_a is not None else header_b)

            with np.errstate(divide="ignore", invalid="ignore"):
                if params.op == "+":
                    out = a + b
                elif params.op == "-":
                    out = a - b
                elif params.op == "*":
                    out = a * b
                else:
                    out = np.where(np.asarray(b) != 0,
                                   np.divide(a, b, out=np.zeros_like(
                                       np.broadcast_arrays(np.asarray(a, float),
                                                           np.asarray(b, float))[0]),
                                       where=np.asarray(b) != 0),
                                   params.divzero)

            header = header.copy() if header is not None else fits.Header()
            line = f"imarith: {first} {params.op} {second}"
            provenance.stamp_header(header, [line])
            destination = Path(destinations[index])
            destination.parent.mkdir(parents=True, exist_ok=True)
            fits.PrimaryHDU(data=np.asarray(out, dtype=np.float32),
                            header=header).writeto(destination, overwrite=True)
            outputs.append(destination)
            history.append(line)

        return TaskResult(outputs=outputs, history=history, steps=self.plan(params),
                          data={"images": len(outputs)})


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
    from ..core import images

    if _is_number(text):
        return float(text), None
    return images.read(text)


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
    image: str = Field(..., description="Image to plot. Sections are honoured.")
    line: int | None = Field(
        None, description="Image line to plot first. The middle one by default."
    )
    column: int | None = Field(
        None, description="Image column to plot first, instead of a line."
    )
    nsum: int = Field(
        1, ge=1, description="How many lines or columns to average for the first plot."
    )


@register
class Implot(Task[ImplotParams]):
    name = "implot"
    package = "plot"
    description = "Plot lines and columns of an image, interactively."
    Params = ImplotParams

    def plan(self, params: ImplotParams) -> list[Step]:
        what = (f"column {params.column}" if params.column
                else f"line {params.line}" if params.line else "the middle line")
        return [Step("read", params.image, why=WHY["read headers"]),
                Step("plot", what + (f", {params.nsum} averaged"
                                     if params.nsum > 1 else ""),
                     why=WHY["plot"]),
                Step("wait", "keys: l c :l :c e r C o q",
                     why="Lower case picks what to plot, a colon command takes a "
                         "range, and e expands a corner at a time -- the keys the "
                         "tutorials tell you to press.")]

    def execute(self, params: ImplotParams) -> TaskResult:
        from ..core import images
        from ..viewer.implot_panel import run

        data, _ = images.read(params.image)
        if data.ndim != 2:
            raise ValueError(f"'{params.image}' is {data.ndim}-dimensional; "
                             "implot plots cuts through a 2-D image")
        run(data, name=str(params.image))
        return TaskResult(data={"shape": "x".join(str(n) for n in reversed(data.shape))},
                          steps=self.plan(params))


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
        ..., description="Keywords to edit, e.g. 'DISPAXIS' or 'OBJECT'."
    )
    value: str = Field(
        "",
        description="What to set it to. Numbers become numbers and yes/no "
                    "become booleans, unless quoted.",
    )
    comment: str = Field("", description="Comment for the card.")
    add: bool = Field(
        False, description="Add the keyword when it is not there. Without this "
                           "a missing keyword is reported, not created."
    )
    delete: bool = Field(False, description="Delete the keyword instead of setting it.")
    update: bool = Field(
        True, description="Write the change. Off, it only shows what would happen."
    )
    extension: int = Field(0, ge=0, description="Which HDU's header to edit.")


@register
class HEdit(Task[HEditParams]):
    name = "hedit"
    package = "imutil"
    description = "Edit header keywords, over one image or a whole night."
    Params = HEditParams

    def plan(self, params: HEditParams) -> list[Step]:
        what = ("delete" if params.delete
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

        paths = [spec.path for spec in images.expand(params.images)]
        if not paths:
            raise ValueError(f"nothing matched {', '.join(params.images)}.")
        if params.delete and params.value:
            raise ValueError(
                "--delete removes the keyword and --value sets it; they cannot "
                "both apply. Drop one.")

        wanted = [name.strip().upper() for field in params.fields
                  for name in str(field).split(",") if name.strip()]
        steps = self.plan(params)
        by_action = {step.action: step for step in steps}

        rows, warnings, changed = [], [], []
        missing = 0
        for path in paths:
            mode = "update" if params.update else "readonly"
            with fits.open(path, mode=mode) as hdus:
                header = hdus[params.extension].header
                touched = False
                for keyword in wanted:
                    before = header.get(keyword)
                    if params.delete:
                        if before is None:
                            missing += 1
                            rows.append(_row(path, keyword, before, "absent"))
                            continue
                        if params.update:
                            del header[keyword]
                            header.add_history(
                                f"hedit: deleted {keyword} (was {before!r})")
                            touched = True
                        rows.append(_row(path, keyword, before, "deleted"))
                        continue

                    if params.value == "":
                        rows.append(_row(path, keyword, before,
                                         "absent" if before is None else "unchanged"))
                        if before is None:
                            missing += 1
                        continue

                    if before is None and not params.add:
                        missing += 1
                        rows.append(_row(path, keyword, before,
                                         "absent -- --add would create it"))
                        continue

                    value = _typed(params.value)
                    if params.update:
                        header[keyword] = ((value, params.comment) if params.comment
                                           else value)
                        header.add_history(
                            f"hedit: {keyword} = {value!r}"
                            + (f", was {before!r}" if before is not None else
                               ", added"))
                        touched = True
                    rows.append(_row(path, keyword, before,
                                     "added" if before is None else "set"))
                if touched:
                    changed.append(Path(path))

        by_action["read"].measured = {"images": len(paths),
                                      "keywords": len(wanted)}
        actions = {}
        for row in rows:
            actions[row["action"]] = actions.get(row["action"], 0) + 1
        by_action["edit"].measured = actions

        if missing and not params.add and not params.delete and params.value:
            warnings.append(advice.warn(
                f"{missing} of {len(rows)} were not there and were left alone. "
                "hedit will not invent a keyword, because a typo in a keyword "
                "name would otherwise silently add a card nobody reads.",
                "add --add if you meant to create them. If you did not, check "
                "the spelling against 'harwood imheader <image> --long'."))
        if not params.update:
            warnings.append(advice.note(
                "--no-update was given, so nothing was written. The table "
                "above is what would have happened.",
                "run it again without --no-update to apply it."))

        return TaskResult(outputs=changed, steps=steps, warnings=warnings,
                          data={"rows": rows}, history=[])


def _row(path, keyword, before, action: str) -> dict:
    return {"image": Path(path).name, "keyword": keyword,
            "was": "-" if before is None else str(before)[:28],
            "action": action}


def _typed(text: str):
    """`3` is a number, `yes` is a boolean, `'3'` is the string three.

    IRAF's headers are typed and its hedit guesses the same way. Quoting is
    the escape hatch, and it is needed more often than it looks: an OBJECT of
    `4211` is a name, not a count.
    """
    if len(text) >= 2 and text[0] == text[-1] and text[0] in "'\"":
        return text[1:-1]
    lowered = text.strip().lower()
    if lowered in ("yes", "true"):
        return True
    if lowered in ("no", "false"):
        return False
    try:
        return int(text)
    except ValueError:
        pass
    try:
        return float(text)
    except ValueError:
        return text


class ImGetsParams(TaskParams):
    image: str = Field(
        ..., description="Image to read.", json_schema_extra={"positional": True}
    )
    param: str = Field(
        ..., description="Keyword to get.", json_schema_extra={"positional": True}
    )
    default: str = Field(
        "0", description="What to return when the keyword is not there."
    )
    extension: int = Field(0, ge=0, description="Which HDU's header to read.")


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
        from astropy.io import fits

        header = fits.getheader(params.image, ext=params.extension)
        keyword = params.param.strip().upper()
        value = header.get(keyword)
        found = value is not None

        steps = self.plan(params)
        steps[0].measured = {
            "value": str(value) if found else params.default,
            "found": "yes" if found else "no, the default was used",
        }
        return TaskResult(
            outputs=[], steps=steps,
            warnings=[] if found else
            [f"'{params.image}' has no {keyword}; the answer is the default, "
             f"{params.default!r}. hselect lists what the header does have."],
            data={"value": str(value) if found else params.default,
                  "found": found},
            history=[])


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
        for _ in range(1 if hist_type == "difference" else 2):
            counts = np.append(counts[1:] - counts[:-1], 0)[:nbins]
            lo, hi = f32(lo + f32(width / f32(2))), f32(hi - f32(width / f32(2)))
            nbins -= 1
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


@register
class ImHistogram(Task[ImHistogramParams]):
    """IRAF's imhistogram, listed (listout = yes): each bin's centre and count.

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
        return TaskResult(data={"rows": rows, "z1": z1, "z2": z2, "nbins": nbins,
                                "width": width}, steps=self.plan(params))


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
