"""imfilter -- IRAF's images.imfilter: filters over a moving window.

`median` is the one the spectroscopy course needs: a lamp's lines separated
from its noise by the median of a 7 x 1 window along the dispersion
(`median cuar_btpf cuar_median 7 1`), and a frame's cosmic rays by a small
square one. The arithmetic is `core.median`, a literal port of IRAF's.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import Field

from ..core import advice
from ..core.registry import register
from ..core.task import Step, Task, TaskParams, TaskResult, Template, format_field

REFLECT_FIX = ("a reflected boundary reaches no further than the image is wide: use "
               "--boundary nearest, or a smaller window")

WHY = {
    "window": "Each pixel becomes the median of the window around it -- the middle "
              "value, which one deviant pixel cannot move as it moves a mean. A window "
              "of even size is made odd by one, as IRAF makes it, so that it has a "
              "centre.",
    "reject": "Values outside the limits are left out of each window's median: a "
              "saturated star or a bad column that would otherwise take part.",
    "boundary": "Near the edge the window reaches past the image, and the pixels it "
                "finds there are the boundary extension's: the nearest edge pixel, a "
                "constant, the image reflected about its edge, or wrapped round. Off "
                "an image section they are the image's own pixels.",
    "write": "In the input's pixel type, as IRAF writes it: a median that is not "
             "whole is truncated in an integer image.",
}


class MedianParams(TaskParams):
    input: list[str] = Field(
        ..., description="Images to filter. Globs, @lists, a,b,c and sections work.",
        json_schema_extra={"positional": True},
    )
    output: Template = Field(
        ..., description="One output per input: names, an @list, a directory or a "
                         "template like '%btpf%median%'.",
        json_schema_extra={"positional": True},
    )
    xwindow: int = Field(..., ge=1, description="Width of the window; an even one is "
                                                "made odd by adding one.",
                         json_schema_extra={"positional": True})
    ywindow: int = Field(..., ge=1, description="Height of the window; an even one is "
                                                "made odd by adding one.",
                         json_schema_extra={"positional": True})
    zloreject: float | None = Field(None, description="Values below this stay out of "
                                                      "the median. Empty is IRAF's "
                                                      "INDEF: none.")
    zhireject: float | None = Field(None, description="Values above this stay out of "
                                                      "the median. Empty is IRAF's "
                                                      "INDEF: none.")
    boundary: Literal["nearest", "constant", "reflect", "wrap"] = Field(
        "nearest", description="What the window finds past the image's edge.")
    constant: float = Field(0.0, description="The value past the edge with "
                                             "--boundary constant.")
    verbose: bool = Field(True, description="Say what was filtered, as IRAF does.")
    format: Literal["harwood", "iraf"] = format_field()


@register
class Median(Task[MedianParams]):
    """IRAF's median, from images/imfilter/src/median.x, ported line for line --
    its sorted linked list walked back and forth over the image, the boundary
    extension and image sections as IMIO reads them, and the defects that come
    with them (core/median.py says which). Checked pixel for pixel against IRAF
    2.18."""

    name = "median"
    package = "imfilter"
    description = "Median filter images over a rectangular window."
    Params = MedianParams

    def plan(self, params: MedianParams) -> list[Step]:
        from ..core import median

        xbox, ybox = median.boxes(params.xwindow, params.ywindow, 2)
        steps = [Step("window", f"{xbox}x{ybox}", why=WHY["window"])]
        if params.zloreject is not None or params.zhireject is not None:
            low = "INDEF" if params.zloreject is None else f"{params.zloreject:g}"
            high = "INDEF" if params.zhireject is None else f"{params.zhireject:g}"
            steps.append(Step("reject", f"below {low}, above {high}", why=WHY["reject"]))
        steps.append(Step("boundary", params.boundary
                          + (f" {params.constant:g}" if params.boundary == "constant"
                             else ""), why=WHY["boundary"]))
        steps.append(Step("write", params.output, why=WHY["write"]))
        return steps

    def execute(self, params: MedianParams) -> TaskResult:
        import numpy as np

        from ..core import median
        from ..core.iraffmt import printf

        text, written, warnings, rows = [], [], [], []
        for spec, destination in _pairs(params.input, params.output, "%btpf%median%"):
            frame = _read(spec)
            xbox, ybox = median.boxes(params.xwindow, params.ywindow, frame.real.ndim)
            if params.verbose:
                text.append(f"{xbox}x{ybox} Box median filter {spec} to {destination}")
                if params.zloreject is not None or params.zhireject is not None:
                    low = "INDEF" if params.zloreject is None else \
                        printf(float(np.float32(params.zloreject)))
                    high = "INDEF" if params.zhireject is None else \
                        printf(float(np.float32(params.zhireject)))
                    text.append(f"    Pixels < {low} or > {high} excluded from the "
                                "median filter")
            try:
                result = median.filter2d(frame.real, params.xwindow, params.ywindow,
                                         params.zloreject, params.zhireject,
                                         params.boundary, frame.constant(params.constant),
                                         section=frame.section)
            except median.MedianError as error:
                advice.fail(f"Error median filtering image: {spec}: {error}",
                            REFLECT_FIX)
            frame.header.add_history(f"median: {xbox}x{ybox} box from {spec}")
            _save(frame, result, destination, params.format, written, warnings)
            rows.append({"image": str(spec), "window": f"{xbox}x{ybox}",
                         "output": str(destination)})
        steps = self.plan(params)
        data: dict = {"rows": rows}
        if text:
            data["text"] = text
        return TaskResult(outputs=written, steps=steps, data=data, warnings=warnings,
                          history=[f"median: {len(rows)} image(s)"])


# ------------------------------------------------------------- shared reading

class _Frame:
    """One input as IRAF's filters read it: the pixels as stored, their IRAF
    type, as reals (NaN and infinities as 0, by the IEEE input), the section
    asked for, and the header the output copies."""

    def __init__(self, stored, header, kind, section):
        import numpy as np

        from ..core import imsum

        self.stored, self.header, self.kind, self.section = stored, header, kind, section
        self.store = imsum.STORE[kind]
        self.real = imsum.convert(imsum.ieee_in(stored), np.float32)
        self.integer = np.dtype(self.store).kind in "iu"

    def constant(self, value: float) -> float:
        """IM_OOBPIX, which goes through the image's own pixel type (impakr)."""
        import numpy as np

        from ..core import imsum

        return float(imsum.convert(imsum.convert(
            np.array([np.float32(value)]), self.store), np.float32)[0])


def _read(spec) -> _Frame:
    import numpy as np
    from astropy.io import fits

    from ..core import images, imsum, pixels

    whole = images.ImageSpec(spec.path, spec.extension)
    hdul, hdu = images.open_hdu(whole)
    try:
        stored = np.array(hdu.data)
        header = images.inherited(hdul, hdu)
        index = hdul.index(hdu)
    finally:
        hdul.close()
    raw = fits.getheader(spec.path, index)
    scaled = int(raw.get("BITPIX", 0)) > 0 and stored.dtype.kind == "f"
    kind = "real" if scaled else imsum.pixel_type(stored.dtype)
    section = (pixels.parse_section(spec.section.strip("[]"), stored.shape)
               if spec.section else None)
    frame = _Frame(stored, header, kind, section)
    frame.raw = raw
    return frame


def _pairs(inputs, output: str, example: str):
    """Each input with its output, the counts checked as IRAF checks them."""
    from ..core import images

    specs = images.expand(inputs)
    if not specs:
        advice.fail(f"'{' '.join(inputs)}' matches no image",
                    "check the name, or run `harwood sections` on the template",
                    FileNotFoundError)
    target = output.strip()
    if target.startswith("@"):
        count = len(images.expand(target))
    elif "%" in target or target.endswith("/") or Path(target).is_dir():
        count = len(specs)
    else:
        count = len([name for name in target.split(",") if name.strip()])
    if count != len(specs):
        advice.fail("Number of input and output images not the same.",
                    f"give one output for each of the {len(specs)} inputs, a "
                    f"directory (ending in /) or a template like '{example}'")
    return list(zip(specs, images.outputs_for(specs, output), strict=True))


def _save(frame: _Frame, result, destination, fmt: str, written: list, warnings: list):
    """The result in the input's pixel type, as IRAF's NEW_COPY writes it."""
    from ..core import imsum, mef

    files, notes = mef.save(destination, imsum.convert(result, frame.store), frame.header,
                            None, None, format=fmt, keep_type=True)
    written.extend(files)
    warnings.extend(n for n in notes if n not in warnings)


# ------------------------------------------------------------------- fmedian

FWHY = {
    "digitize": "The pixels are made integers first, zmin..zmax mapped onto "
                "hmin..hmax, so that the median can be kept as a histogram as the "
                "window moves. That is what makes fmedian fast, and also what it costs: "
                "the output is quantized to (zmax - zmin)/(hmax - hmin) of the data.",
}


class FmedianParams(TaskParams):
    input: list[str] = Field(
        ..., description="Images to filter. Globs, @lists, a,b,c and sections work.",
        json_schema_extra={"positional": True})
    output: Template = Field(
        ..., description="One output per input: names, an @list, a directory or a "
                         "template like '%obj%fmed%'.",
        json_schema_extra={"positional": True})
    xwindow: int = Field(..., ge=1, description="Width of the window; an even one is "
                                                "made odd by adding one.",
                         json_schema_extra={"positional": True})
    ywindow: int = Field(..., ge=1, description="Height of the window; an even one is "
                                                "made odd by adding one.",
                         json_schema_extra={"positional": True})
    hmin: int = Field(-32768, description="The integer zmin is mapped to.")
    hmax: int = Field(32767, description="The integer zmax is mapped to.")
    zmin: float | None = Field(None, description="The value mapped to hmin. Empty is "
                                                 "IRAF's INDEF: the image's minimum.")
    zmax: float | None = Field(None, description="The value mapped to hmax. Empty is "
                                                 "IRAF's INDEF: the image's maximum.")
    zloreject: float | None = Field(None, description="Values below this stay out of "
                                                      "the median. Empty: none.")
    zhireject: float | None = Field(None, description="Values above this stay out of "
                                                      "the median. Empty: none.")
    unmap: bool = Field(True, description="Map the medians back to zmin..zmax; off, "
                                          "the output is the histogram's integers.")
    boundary: Literal["nearest", "constant", "reflect", "wrap"] = Field(
        "nearest", description="What the window finds past the image's edge.")
    constant: float = Field(0.0, description="The value past the edge with "
                                             "--boundary constant.")
    verbose: bool = Field(True, description="Say what was filtered, as IRAF does.")
    format: Literal["harwood", "iraf"] = format_field()


@register
class Fmedian(Task[FmedianParams]):
    """IRAF's fmedian, from images/imfilter/src/fmedian.x: the median of each
    window kept as a running histogram of the digitized pixels, ported line for
    line (core/fmedian.py says what follows). Checked pixel for pixel against
    IRAF 2.18."""

    name = "fmedian"
    package = "imfilter"
    description = "Fast median filter, over a histogram of digitized pixels."
    Params = FmedianParams

    def plan(self, params: FmedianParams) -> list[Step]:
        from ..core import median

        xbox, ybox = median.boxes(params.xwindow, params.ywindow, 2)
        low = "the minimum" if params.zmin is None else f"{params.zmin:g}"
        high = "the maximum" if params.zmax is None else f"{params.zmax:g}"
        steps = [Step("digitize", f"{low} to {high} onto {params.hmin} to {params.hmax}",
                      why=FWHY["digitize"]),
                 Step("window", f"{xbox}x{ybox}", why=WHY["window"])]
        if params.zloreject is not None or params.zhireject is not None:
            steps.append(Step("reject", f"below {params.zloreject}, above "
                              f"{params.zhireject}", why=WHY["reject"]))
        steps.append(Step("boundary", params.boundary, why=WHY["boundary"]))
        steps.append(Step("write", params.output, why=WHY["write"]))
        return steps

    def execute(self, params: FmedianParams) -> TaskResult:
        import numpy as np

        from ..core import fmedian, median
        from ..core.iraffmt import printf
        from ..core.scombine import fp_equalr

        text, written, warnings, rows = [], [], [], []
        for spec, destination in _pairs(params.input, params.output, "%obj%fmed%"):
            frame = _read(spec)
            xbox, ybox = median.boxes(params.xwindow, params.ywindow, 2)
            constant = frame.constant(params.constant)
            if params.verbose:
                text.append(f"{xbox}x{ybox} Box median filter {spec} to {destination}")
            try:
                result, low, high = fmedian.filter2d(
                    frame.real, params.xwindow, params.ywindow, frame.raw, params.hmin,
                    params.hmax, params.zmin, params.zmax, params.zloreject,
                    params.zhireject, params.unmap, params.boundary, constant,
                    section=frame.section, integer=frame.integer)
            except fmedian.FmedianError as error:
                advice.fail(f"Error median filtering image: {spec}: {error}", REFLECT_FIX)
            if params.verbose:
                z1 = 1.6e38 if params.zmin is None else float(np.float32(params.zmin))
                z2 = 1.6e38 if params.zmax is None else float(np.float32(params.zmax))
                if not fp_equalr(z1, float(params.hmin)) and \
                        not fp_equalr(z2, float(params.hmax)):
                    text.append(f"    Pixels from {printf(low)} to {printf(high)} mapped "
                                f"to integers from {params.hmin} to {params.hmax}")
                if params.zloreject is not None or params.zhireject is not None:
                    lo = "INDEF" if params.zloreject is None else \
                        printf(float(np.float32(params.zloreject)))
                    hi = "INDEF" if params.zhireject is None else \
                        printf(float(np.float32(params.zhireject)))
                    text.append(f"    Pixels < {lo} or > {hi} excluded from the "
                                "median filter")
            frame.header.add_history(f"fmedian: {xbox}x{ybox} box from {spec}")
            _save(frame, result, destination, params.format, written, warnings)
            rows.append({"image": str(spec), "window": f"{xbox}x{ybox}",
                         "zmin": low, "zmax": high, "output": str(destination)})
        data: dict = {"rows": rows}
        if text:
            data["text"] = text
        return TaskResult(outputs=written, steps=self.plan(params), data=data,
                          warnings=warnings, history=[f"fmedian: {len(rows)} image(s)"])


# -------------------------------------------------------------------- boxcar

BWHY = {
    "window": "Each pixel becomes the mean of the box around it. The box is not made "
              "odd: an even side reaches a pixel further one way than the other, which "
              "shifts the image by half a pixel, as IRAF's does.",
}


class BoxcarParams(TaskParams):
    input: list[str] = Field(
        ..., description="Images to smooth. Globs, @lists, a,b,c and sections work.",
        json_schema_extra={"positional": True})
    output: Template = Field(
        ..., description="One output per input: names, an @list, a directory or a "
                         "template like '%im%b%'.",
        json_schema_extra={"positional": True})
    xwindow: int = Field(..., ge=1, description="Width of the box, in pixels.",
                         json_schema_extra={"positional": True})
    ywindow: int = Field(..., ge=1, description="Height of the box, in pixels.",
                         json_schema_extra={"positional": True})
    boundary: Literal["nearest", "constant", "reflect", "wrap"] = Field(
        "nearest", description="What the box finds past the image's edge.")
    constant: float = Field(0.0, description="The value past the edge with "
                                             "--boundary constant.")
    format: Literal["harwood", "iraf"] = format_field()


@register
class Boxcar(Task[BoxcarParams]):
    """IRAF's boxcar, from images/imfilter/src/boxcar.x: the mean over a box, as
    running sums in single precision (core/boxcar.py). Checked pixel for pixel
    against IRAF 2.18."""

    name = "boxcar"
    package = "imfilter"
    description = "Smooth images with the mean over a box."
    Params = BoxcarParams

    def plan(self, params: BoxcarParams) -> list[Step]:
        return [Step("window", f"{params.xwindow}x{params.ywindow}", why=BWHY["window"]),
                Step("boundary", params.boundary, why=WHY["boundary"]),
                Step("write", params.output, why=WHY["write"])]

    def execute(self, params: BoxcarParams) -> TaskResult:
        from ..core import boxcar

        written, warnings, rows = [], [], []
        for spec, destination in _pairs(params.input, params.output, "%im%b%"):
            frame = _read(spec)
            try:
                result = boxcar.smooth(frame.real, params.xwindow, params.ywindow,
                                       params.boundary, frame.constant(params.constant),
                                       section=frame.section)
            except boxcar.BoxcarError as error:
                fix = ("boxcar smooths 1-D and 2-D images: give a plane of the cube, "
                       "such as [*,*,1]" if "dimension" in str(error) else REFLECT_FIX)
                advice.fail(f"Error smoothing image: {spec}: {error}", fix)
            frame.header.add_history(f"boxcar: {params.xwindow}x{params.ywindow} box "
                                     f"from {spec}")
            _save(frame, result, destination, params.format, written, warnings)
            rows.append({"image": str(spec), "box": f"{params.xwindow}x{params.ywindow}",
                         "output": str(destination)})
        return TaskResult(outputs=written, steps=self.plan(params), data={"rows": rows},
                          warnings=warnings, history=[f"boxcar: {len(rows)} image(s)"])
