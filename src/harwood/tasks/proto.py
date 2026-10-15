"""proto -- the small utilities IRAF kept in its prototype package.

`fixpix` is the one a reduction needs: bad columns, hot pixels and the
occasional dead row replaced by interpolating across them, with a mask that
says which pixels they were. The arithmetic is `core.fixpix`, which ccdproc's
fixpix step uses too, so a frame fixed by either is fixed the same way -- the
way IRAF fixes it.
"""

from __future__ import annotations

from pathlib import Path

from pydantic import Field

from ..core import advice
from ..core.registry import register
from ..core.task import Step, Task, TaskParams, TaskResult

WHY = {
    "mask": "Non-zero marks a bad pixel, as in IRAF; the value can also say "
            "which way to interpolate (linterp, cinterp). BPM takes the mask "
            "named in each image's own BPM keyword, so a whole night can be "
            "fixed with the masks its headers already point to.",
    "fix": "Each bad region is interpolated across its narrowest side: a bad "
           "column along the lines, a bad row along the columns, a single "
           "pixel along its line. The replacement is linear between the good "
           "pixels either side, so it follows the sky's gradient -- and is "
           "still not a measurement, which is why the mask stays with the frame.",
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
                         "BPM uses the mask in each image's BPM keyword. A text "
                         "file of 'x y' or 'x1 x2 y1 y2' lines is accepted too.",
    )
    linterp: int | None = Field(
        None, ge=1, description="Mask value that forces interpolation along "
                                "lines. Empty is IRAF's INDEF: the narrowest side.")
    cinterp: int | None = Field(
        None, ge=1, description="Mask value that forces interpolation along "
                                "columns. Empty is IRAF's INDEF.")
    output: str = Field(
        "", description="Where to write: a directory or a prefix. Left out, this "
                        "run's own directory. IRAF fixes in place; name the "
                        "input's own directory to do the same.")


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
        from astropy.io import fits

        from ..core import fixpix, images, output
        from .crutil import _destination

        specs = images.expand(params.images)
        if not specs:
            advice.fail(f"'{' '.join(params.images)}' matches no image",
                        "check the name, or run `harwood sections` on the template",
                        FileNotFoundError)
        if len(params.masks) not in (1, len(specs)):
            advice.fail(f"{len(params.masks)} masks for {len(specs)} images",
                        "give one mask for all of them, or one for each, in order")

        rows, written, total = [], [], 0
        for number, spec in enumerate(specs):
            hdul, hdu = images.open_hdu(spec)
            try:
                data, header = np.array(hdu.data), hdu.header.copy()
            finally:
                hdul.close()
            mask_name = params.masks[0 if len(params.masks) == 1 else number]
            mask, where = _mask(mask_name, header, spec, data.shape)
            fixed, summary = fixpix.fix(data, mask, params.linterp, params.cinterp)

            header.add_history(f"fixpix: {summary['pixels']} pixels from {where} "
                               f"({summary['along lines']} along lines, "
                               f"{summary['along columns']} along columns)")
            name = Path(spec.path).name
            stem = name[:-5] if name.endswith(".fits") else name
            destination = _destination(params.output, stem, ".fits", spec, output)
            fits.PrimaryHDU(fixed, header).writeto(destination, overwrite=True)
            written.append(Path(destination))
            rows.append({"image": str(spec), "mask": where, **summary})
            total += summary["pixels"]

        steps = self.plan(params)
        steps[1].measured = {"images": len(specs), "pixels replaced": total}
        return TaskResult(outputs=written, steps=steps, data={"rows": rows},
                          history=[f"fixpix: {total} pixels"])


def _mask(name: str, header, spec, shape):
    """The mask for one image: an image, the BPM keyword's, or a text list."""
    import numpy as np

    from ..core import images

    if name.upper() == "BPM":
        if "BPM" not in header:
            advice.fail(f"{spec} has no BPM keyword",
                        "name the mask instead of BPM, or add BPM with hedit")
        name = str(header["BPM"])
        candidate = Path(spec.path).parent / name
        if not Path(name).exists() and candidate.exists():
            name = str(candidate)
    path = Path(images.parse(name).path)
    if not path.exists():
        advice.fail(f"mask '{name}' does not exist", "check the name, or the BPM keyword",
                    FileNotFoundError)
    if path.suffix.lower() in (".fits", ".fit", ".fz", ".gz", ".pl", ".imh"):
        mask, _ = images.read(name)
        return np.trunc(mask).astype(np.int64), name

    # IRAF's text format: 'x y' for a pixel, 'x1 x2 y1 y2' for a region.
    mask = np.zeros(shape, np.int64)
    for number, line in enumerate(path.read_text().splitlines(), 1):
        fields = line.split("#", 1)[0].split()
        if not fields:
            continue
        try:
            values = [int(float(v)) for v in fields]
        except ValueError:
            advice.fail(f"{path}:{number}: '{line.strip()}' is not numbers",
                        "write 'x y' for a pixel or 'x1 x2 y1 y2' for a region")
        if len(values) == 2:
            x1 = x2 = values[0]
            y1 = y2 = values[1]
        elif len(values) == 4:
            x1, x2, y1, y2 = values
        else:
            advice.fail(f"{path}:{number}: '{line.strip()}' has {len(values)} numbers",
                        "write 'x y' for a pixel or 'x1 x2 y1 y2' for a region")
        mask[max(1, y1) - 1:min(shape[0], y2), max(1, x1) - 1:min(shape[1], x2)] = 1
    return mask, name
