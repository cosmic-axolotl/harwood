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
from ..core.task import Step, Task, TaskParams, TaskResult


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
        return [Step("read headers", f"{len(params.input)} patterns"),
                Step("print", "whole header" if params.longheader else "summary")]

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
                           "filter": header.get("FILTER", ""),
                           "object": header.get("OBJECT", "")}
            finally:
                hdul.close()
            rows.append({"image": str(spec), **row})
        return TaskResult(data={"rows": rows, "images": len(rows)})


class HSelectParams(TaskParams):
    input: list[str] = Field(..., description="Images to consider. Accepts globs and @lists.")
    fields: list[str] = Field(
        ..., description="Keywords to print for each image that matches, e.g. 'OBJECT,EXPTIME'."
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
        return [Step("evaluate", f"{params.expr!r} over each header"),
                Step("print", ", ".join(params.fields))]

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
    input: list[str] = Field(..., description="Images to measure. Accepts globs and @lists.")
    section: str | None = Field(
        None, description="Section to measure, 1-based and (x,y), e.g. '[100:400,*]'."
    )
    nclip: int = Field(0, ge=0, description="Sigma-clipping iterations before measuring.")
    lsigma: float = Field(3.0, ge=0, description="Clipping threshold below the mean, in sigma.")
    hsigma: float = Field(3.0, ge=0, description="Clipping threshold above the mean, in sigma.")


@register
class ImStatistics(Task[ImStatParams]):
    name = "imstatistics"
    package = "imutil"
    description = "Pixel statistics, with optional iterative clipping."
    Params = ImStatParams

    def plan(self, params: ImStatParams) -> list[Step]:
        steps = [Step("read", f"{len(params.input)} patterns"
                              + (f", section {params.section}" if params.section else ""))]
        if params.nclip:
            steps.append(Step("clip", f"{params.nclip} iterations at "
                                      f"-{params.lsigma}/+{params.hsigma} sigma"))
        steps.append(Step("measure", "npix, mean, midpt, stddev, min, max"))
        return steps

    def execute(self, params: ImStatParams) -> TaskResult:
        import numpy as np

        from ..core import images

        rows = []
        for spec in images.expand(params.input):
            if params.section:
                spec = images.ImageSpec(spec.path, spec.extension, params.section)
            data, _ = images.read(spec)
            values = data[np.isfinite(data)]
            for _ in range(params.nclip):
                mean, sigma = values.mean(), values.std(ddof=1)
                keep = (values > mean - params.lsigma * sigma) & \
                       (values < mean + params.hsigma * sigma)
                if keep.all():
                    break
                values = values[keep]
            rows.append({
                "image": str(spec),
                "npix": int(values.size),
                "mean": float(values.mean()),
                "midpt": float(np.median(values)),
                "stddev": float(values.std(ddof=1)),
                "min": float(values.min()),
                "max": float(values.max()),
            })
        return TaskResult(data={"rows": rows})


class ImArithParams(TaskParams):
    operand1: str = Field(..., description="First operand: an image, or a number.")
    op: Literal["+", "-", "*", "/"] = Field(..., description="Operation to apply.")
    operand2: str = Field(..., description="Second operand: an image, or a number.")
    result: Path = Field(..., description="Image to write.")
    divzero: float = Field(
        0.0, description="Value written where a division by zero would happen."
    )


@register
class ImArith(Task[ImArithParams]):
    name = "imarith"
    package = "imutil"
    description = "Arithmetic between images and constants."
    Params = ImArithParams

    def plan(self, params: ImArithParams) -> list[Step]:
        return [Step("compute", f"{params.operand1} {params.op} {params.operand2}"),
                Step("write", str(params.result))]

    def execute(self, params: ImArithParams) -> TaskResult:
        import numpy as np
        from astropy.io import fits

        from ..core import images, provenance

        left, header = _operand(params.operand1)
        right, header2 = _operand(params.operand2)
        header = header if header is not None else header2

        with np.errstate(divide="ignore", invalid="ignore"):
            if params.op == "+":
                out = left + right
            elif params.op == "-":
                out = left - right
            elif params.op == "*":
                out = left * right
            else:
                out = np.divide(left, right, out=np.full_like(np.broadcast_arrays(
                    np.asarray(left, dtype=float), np.asarray(right, dtype=float))[0],
                    params.divzero), where=np.asarray(right) != 0)

        header = header.copy() if header is not None else fits.Header()
        history = [f"imarith: {params.operand1} {params.op} {params.operand2}"]
        provenance.stamp_header(header, history)
        params.result.parent.mkdir(parents=True, exist_ok=True)
        fits.PrimaryHDU(data=np.asarray(out, dtype=np.float32),
                        header=header).writeto(params.result, overwrite=True)
        return TaskResult(outputs=[params.result], history=history,
                          data={"mean": float(np.nanmean(out))}, steps=self.plan(params))


def _operand(text: str):
    from ..core import images

    try:
        return float(text), None
    except ValueError:
        return images.read(text)
