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
        from astropy.io import fits

        from ..core import images, imsum, median, mef, pixels
        from ..core.iraffmt import printf

        specs = images.expand(params.input)
        if not specs:
            advice.fail(f"'{' '.join(params.input)}' matches no image",
                        "check the name, or run `harwood sections` on the template",
                        FileNotFoundError)
        target = params.output.strip()
        if target.startswith("@"):
            count = len(images.expand(target))
        elif "%" in target or target.endswith("/") or Path(target).is_dir():
            count = len(specs)
        else:
            count = len([name for name in target.split(",") if name.strip()])
        if count != len(specs):
            advice.fail("Number of input and output images not the same.",
                        f"give one output for each of the {len(specs)} inputs, a "
                        "directory (ending in /) or a template like '%btpf%median%'")
        destinations = images.outputs_for(specs, params.output)
        text, written, warnings, rows = [], [], [], []
        for spec, destination in zip(specs, destinations, strict=True):
            whole = images.parse(str(spec.path)) if spec.extension is None else \
                images.ImageSpec(spec.path, spec.extension)
            hdul, hdu = images.open_hdu(whole)
            try:
                stored = np.array(hdu.data)
                header = images.inherited(hdul, hdu)
                index = hdul.index(hdu)
            finally:
                hdul.close()
            bitpix = int(fits.getheader(spec.path, index).get("BITPIX", 0))
            scaled = bitpix > 0 and stored.dtype.kind == "f"
            kind = "real" if scaled else imsum.pixel_type(stored.dtype)
            store = imsum.STORE[kind]
            real = imsum.convert(imsum.ieee_in(stored), np.float32)
            section = (pixels.parse_section(spec.section.strip("[]"), stored.shape)
                       if spec.section else None)
            xbox, ybox = median.boxes(params.xwindow, params.ywindow, stored.ndim)
            if params.verbose:
                text.append(f"{xbox}x{ybox} Box median filter {spec} to {destination}")
                if params.zloreject is not None or params.zhireject is not None:
                    low = "INDEF" if params.zloreject is None else \
                        printf(float(np.float32(params.zloreject)))
                    high = "INDEF" if params.zhireject is None else \
                        printf(float(np.float32(params.zhireject)))
                    text.append(f"    Pixels < {low} or > {high} excluded from the "
                                "median filter")
            # IM_OOBPIX goes through the image's own pixel type (impakr).
            constant = float(imsum.convert(imsum.convert(
                np.array([np.float32(params.constant)]), store), np.float32)[0])
            try:
                result = median.filter2d(real, params.xwindow, params.ywindow,
                                         params.zloreject, params.zhireject,
                                         params.boundary, constant, section=section)
            except median.MedianError as error:
                advice.fail(f"Error median filtering image: {spec}: {error}",
                            "a reflected boundary reaches no further than the image "
                            "is wide: use --boundary nearest, or a smaller window")
            header.add_history(f"median: {xbox}x{ybox} box from {spec}")
            files, notes = mef.save(destination, imsum.convert(result, store), header,
                                    None, None, format=params.format, keep_type=True)
            written.extend(files)
            warnings.extend(n for n in notes if n not in warnings)
            rows.append({"image": str(spec), "window": f"{xbox}x{ybox}",
                         "output": str(destination)})
        steps = self.plan(params)
        data: dict = {"rows": rows}
        if text:
            data["text"] = text
        return TaskResult(outputs=written, steps=steps, data=data, warnings=warnings,
                          history=[f"median: {len(specs)} image(s)"])
