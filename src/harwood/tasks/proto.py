"""proto -- the small utilities IRAF kept in its prototype package.

`fixpix` is the one a reduction needs: bad columns, hot pixels and the
occasional dead row replaced by interpolating across them, with a mask that
says which pixels they were. The arithmetic is `core.fixpix`, which ccdproc's
fixpix step uses too, so a frame fixed by either is fixed the same way -- the
way IRAF fixes it.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import Field

from ..core import advice
from ..core.registry import register
from ..core.task import Step, Task, TaskParams, TaskResult, format_field, uncertainty_field

WHY = {
    "mask": "Non-zero marks a bad pixel, as in IRAF; the value can also say "
            "which way to interpolate (linterp, cinterp). BPM takes the mask "
            "named in each image's own BPM keyword, so a whole night can be "
            "fixed with the masks its headers already point to; DQ takes the "
            "image's own DQ plane -- whatever cosmicrays, ccdproc and the "
            "rest have flagged in it.",
    "fix": "Each bad region is interpolated across its narrowest side: a bad "
           "column along the lines, a bad row along the columns, a single "
           "pixel along its line. The replacement is linear between the good "
           "pixels either side, so it follows the sky's gradient -- and is "
           "still not a measurement, which is why the DQ keeps the pixel "
           "flagged. The variance is interpolated the same way: the noise a "
           "pixel there would have had.",
    "write": "The pixel type is kept: an integer image is rounded back to "
             "integers, as IRAF writes it. The input is not modified unless "
             "--output names it.",
}


class FixpixParams(TaskParams):
    images: list[str] = Field(
        ..., description="Images to fix. Globs and @lists expand.",
        json_schema_extra={"positional": True},
    )
    masks: list[str] = Field(
        ..., description="Bad pixel masks: one for all the images, or one each. "
                         "BPM uses the mask in each image's BPM keyword; DQ the "
                         "pixels its own DQ flags. A text file of 'x y' or "
                         "'x1 x2 y1 y2' lines is accepted too.",
    )
    linterp: int | None = Field(
        None, ge=1, description="Mask value that forces interpolation along "
                                "lines. Empty is IRAF's INDEF: the narrowest side.")
    cinterp: int | None = Field(
        None, ge=1, description="Mask value that forces interpolation along "
                                "columns. Empty is IRAF's INDEF.")
    verbose: bool = Field(False, description="Print 'FIXPIX: image ... with mask ...' "
                                             "for each image, as IRAF does.")
    pixels: bool = Field(
        False, description="With --verbose, list every pixel replaced: x, y, the value "
                           "it had, the value it got, and the good pixels it came from.")
    output: str = Field(
        "", description="Where to write: a directory or a prefix. Left out, this "
                        "run's own directory. IRAF fixes in place; name the "
                        "input's own directory to do the same.")
    uncertainty: Literal["VAR", "ERR", "UNCERT"] = uncertainty_field()
    format: Literal["harwood", "iraf"] = format_field()


@register
class Fixpix(Task[FixpixParams]):
    """Replace the pixels a mask marks by interpolating across them, as
    IRAF's proto.fixpix does: across the narrowest side of each bad region,
    linearly between the good pixels either side, with the edge taking its
    one good neighbour. The arithmetic is a port of IRAF's own (xtfixpix),
    checked pixel for pixel against IRAF 2.18 on a mask of every shape it
    decides differently on.
    """

    name = "fixpix"
    package = "proto"
    description = "Replace bad pixels by interpolating across them, as IRAF's fixpix."
    Params = FixpixParams

    def plan(self, params: FixpixParams) -> list[Step]:
        forced = []
        if params.linterp is not None:
            forced.append(f"{params.linterp} along lines")
        if params.cinterp is not None:
            forced.append(f"{params.cinterp} along columns")
        return [Step("mask", ", ".join(params.masks), why=WHY["mask"]),
                Step("fix", "across the narrowest side"
                            + (f"; mask value {' and '.join(forced)}" if forced else ""),
                     why=WHY["fix"]),
                Step("write", params.output or "this run's directory", why=WHY["write"])]

    def execute(self, params: FixpixParams) -> TaskResult:
        import numpy as np

        from ..core import dq, fixpix, images, mef, output
        from .crutil import _destination

        specs = images.expand(params.images)
        if not specs:
            advice.fail(f"'{' '.join(params.images)}' matches no image",
                        "check the name, or run `harwood sections` on the template",
                        FileNotFoundError)
        if len(params.masks) not in (1, len(specs)):
            advice.fail(f"{len(params.masks)} masks for {len(specs)} images",
                        "give one mask for all of them, or one for each, in order")

        rows, written, total, text, warnings = [], [], 0, [], []
        for number, spec in enumerate(specs):
            hdul, hdu = images.open_hdu(spec)
            try:
                data, header = np.array(hdu.data), images.inherited(hdul, hdu)
            finally:
                hdul.close()
            _, _, var, quality = mef.load(spec)
            mask_name = params.masks[0 if len(params.masks) == 1 else number]
            if mask_name.strip().upper() == "DQ":
                mask, where = (quality != 0).astype(np.int32), "DQ"
            else:
                mask, where = _mask(mask_name, header, spec, data.shape)
            listing: list | None = [] if params.verbose and params.pixels else None
            fixed, summary = fixpix.fix(data, mask, params.linterp, params.cinterp, listing)
            if var is not None:
                var, _ = fixpix.fix(var, mask, params.linterp, params.cinterp)
            # A pixel the mask named and the DQ had no reason for is a bad
            # pixel; one the DQ already explains keeps its reason.
            quality = quality | np.where((mask != 0) & (quality == 0), dq.BAD_PIXEL,
                                         0).astype(dq.DTYPE)
            if params.verbose:
                text.append(f"FIXPIX: image {spec} with mask {where}")
                text += fixpix.format_listing(listing or [],
                                              double=data.dtype == np.float64)

            header.add_history(f"fixpix: {summary['pixels']} pixels from {where} "
                               f"({summary['along lines']} along lines, "
                               f"{summary['along columns']} along columns)")
            _mark(header, where)
            name = Path(spec.path).name
            stem = name[:-5] if name.endswith(".fits") else name
            destination = _destination(params.output, stem, ".fits", spec, output)
            files, notes = mef.save(destination, fixed, header, var, quality,
                                    params.uncertainty, params.format)
            written.extend(files)
            if notes and not warnings:
                warnings.extend(notes)
            rows.append({"image": str(spec), "mask": where, **summary})
            total += summary["pixels"]

        steps = self.plan(params)
        steps[1].measured = {"images": len(specs), "pixels replaced": total}
        data = {"text": text, "rows": rows} if text else {"rows": rows}
        return TaskResult(outputs=written, steps=steps, data=data, warnings=warnings,
                          history=[f"fixpix: {total} pixels"])


def _mark(header, mask: str) -> None:
    """IRAF's record: FIXPIX = '<date> Bad pixel file is <mask>', or FIXPIX02
    and on when the frame was fixed before. The date is cnvdate's, local."""
    import time

    now = time.localtime()
    stamp = time.strftime("%b", now) + f" {now.tm_mday:2d} {now.tm_hour:2d}:{now.tm_min:02d}"
    value = f"{stamp} Bad pixel file is {mask}"
    if "FIXPIX" not in header:
        header["FIXPIX"] = value
        return
    for number in range(2, 100):
        if f"FIXPIX{number:02d}" not in header:
            header[f"FIXPIX{number:02d}"] = value
            return


def _mask(name: str, header, spec, shape):
    """The mask for one image, named as every IRAF mask is (core.pixmask):
    an image or .pl, BPM or !KEY for the file a keyword names, a section, a
    text file of 'x y' and 'x1 x2 y1 y2' lines; ^ inverts. A relative name
    not found here is looked for beside the image."""
    from ..core import pixmask

    resolved, _ = pixmask.resolve(name, header)
    if not resolved:
        keyword = name.lstrip("^").lstrip("!")
        advice.fail(f"{spec} has no {keyword} keyword",
                    "name the mask instead, or add the keyword with hedit")
    beside = Path(spec.path).parent
    workdir = "." if (Path(resolved).exists() or resolved.startswith("[")
                      or resolved == "EMPTY" or not (beside / resolved).exists()) else beside
    mask = pixmask.read(name, header, shape, workdir)
    return mask, resolved
