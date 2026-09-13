"""imutil -- the small tasks a reduction leans on constantly.

`imheader`, `hselect`, `imstatistics`, `imarith`. None of them is hard; all of
them are what turns a directory of FITS files into a data set you can reason
about. `hselect` in particular has no command-line equivalent in the Python
ecosystem, and it is how a night of observing gets sorted.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import Field

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
    hsigma: float = Field(
        3.0, ge=0, description="Clipping threshold above the mean, in sigma."
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
                                      f"-{params.lsigma}/+{params.hsigma} sigma",
                              why=WHY["clip"]))
        steps.append(Step("measure", params.fields, why=WHY["measure"]))
        return steps

    def execute(self, params: ImStatParams) -> TaskResult:
        import numpy as np

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
            values = data[np.isfinite(data)]
            if params.lower is not None:
                values = values[values >= params.lower]
            if params.upper is not None:
                values = values[values <= params.upper]
            if values.size == 0:
                raise ValueError(f"'{spec}' has no pixels left after the limits")

            for _ in range(params.nclip):
                mean, sigma = values.mean(), values.std(ddof=1)
                if sigma <= 0:
                    break
                keep = (values > mean - params.lsigma * sigma) & \
                       (values < mean + params.hsigma * sigma)
                if keep.all():
                    break
                values = values[keep]

            rows.append(_measure(str(spec), values, wanted, params.binwidth))
        return TaskResult(data={"rows": rows}, steps=self.plan(params))


def _measure(name, values, wanted, binwidth):
    import numpy as np

    mean = float(values.mean())
    sigma = float(values.std(ddof=1)) if values.size > 1 else 0.0
    row = {}
    for field in wanted:
        if field == "image":
            row["image"] = name
        elif field == "npix":
            row["npix"] = int(values.size)
        elif field == "mean":
            row["mean"] = mean
        elif field == "midpt":
            row["midpt"] = float(np.median(values))
        elif field == "mode":
            row["mode"] = mode(values, binwidth * sigma if sigma > 0 else None)
        elif field == "stddev":
            row["stddev"] = sigma
        elif field == "skew":
            row["skew"] = (float(np.mean(((values - mean) / sigma) ** 3))
                           if sigma > 0 else 0.0)
        elif field == "kurtosis":
            row["kurtosis"] = (float(np.mean(((values - mean) / sigma) ** 4)) - 3.0
                               if sigma > 0 else 0.0)
        elif field == "min":
            row["min"] = float(values.min())
        elif field == "max":
            row["max"] = float(values.max())
    return row


def mode(values, width: float | None = None) -> float:
    """The most common value, as IRAF's imstatistics computes it.

    A histogram, the fullest bin, then a parabolic refinement using the two
    neighbours.

    The bin width is the whole game, and it is why this takes a width rather
    than a bin count. Bins a tenth of a sigma wide -- IRAF's default, and this
    one -- smooth the histogram enough that the fullest bin is the real peak.
    Make them much finer and the fullest bin is whichever one noise happened to
    favour: on the KPNO flat field, 512 bins across the range gives 1330 where
    a tenth of a sigma gives 1313, and 1313 is the answer.

    Even so, on a broad distribution the mode is a noisy statistic. It is a
    good normalisation constant and a poor thing to compare between programs.
    """
    import numpy as np

    data = np.asarray(values, dtype=float).ravel()
    low, high = float(data.min()), float(data.max())
    if high <= low:
        return low
    if width is None:
        sigma = float(data.std(ddof=1)) if data.size > 1 else 0.0
        width = 0.1 * sigma if sigma > 0 else (high - low) / 512
    bins = max(8, min(8192, int((high - low) / width)))
    counts, edges = np.histogram(data, bins=bins, range=(low, high))
    peak = int(counts.argmax())
    step = (high - low) / bins
    centre = low + (peak + 0.5) * step
    if 0 < peak < bins - 1:
        left = float(counts[peak] - counts[peak - 1])
        right = float(counts[peak] - counts[peak + 1])
        if left + right != 0:
            centre += step / 2 * (left - right) / (left + right)
    return float(centre)


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
            warnings.append(
                f"{missing} of {len(rows)} were not there and were left alone. "
                "--add creates them; without it hedit will not invent a "
                "keyword, because a typo in a keyword name would otherwise "
                "silently add a card nobody reads.")
        if not params.update:
            warnings.append(
                "--no-update was given, so nothing was written. The table "
                "above is what would have happened.")

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
