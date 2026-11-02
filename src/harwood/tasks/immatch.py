"""immatch -- putting two frames of the same field on the same coordinates.

Nothing that compares magnitudes across frames works until this does. On the
1987 M92 frames the telescope moved 0.75 by 1.75 pixels between two exposures
taken minutes apart, and nearly five pixels between the B and V sequences --
enough that a two-pixel matching radius finds almost nothing and quietly
returns the few stars near the pivot.

Three tasks, IRAF's names and IRAF's division of labour:

``xyxymatch``   pair up two coordinate lists
``imcentroid``  measure the shift between frames from the stars themselves
``imalign``     shift the frames so the pixels line up
``imshift``     the shift itself (images.imgeom in IRAF)

imcentroid, imshift and imalign are ports of IRAF's (core.imcentroid,
core.imshift); imalign is IRAF's CL script, which runs the other two.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import Field, field_validator

from ..core import advice
from ..core.registry import register
from ..core.task import Step, Task, TaskParams, TaskResult

WHY = {
    "read": "Two lists of positions of the same field, taken at different times "
            "or through different filters. Nothing about them says which entry "
            "is which star -- that is the whole problem.",
    "match:tolerance": "Each reference star takes the nearest input star within "
                       "the tolerance, once the input list has been put through "
                       "the transformation given (xin, xmag, xrotation, xref...) "
                       "or solved from the refpoints. Right when the frames are "
                       "already close, or that transformation is known.",
    "match:triangles": "Triangles of stars are described by quantities that "
                       "survive translation, rotation and scale -- the ratio of "
                       "the longest side to the shortest, and the angle between "
                       "them. So the same three stars give the same two numbers "
                       "in any frame however it moved, and matching triangles "
                       "vote for the pairs of stars that made them.",
    "fit": "When either list has more stars than nmatch, the triangles only "
           "matched the first nmatch; a linear transformation x' = a + bx + cy, "
           "y' = d + ex + fy fitted to those pairs carries every input star "
           "onto the reference, and the whole lists are then matched by "
           "tolerance.",
    "write": "The pairs as IRAF writes them -- reference x y, input x y and the "
             "line numbers in both files -- ready for geomap or imalign.",
    "coarse": "Without initial shifts the first star in the list is found with "
              "the big box in every image, and where it lands gives the shift "
              "every other star is looked for with. Put a bright, isolated star "
              "first.",
    "centre": "IRAF's centring: in a box around the nearest pixel, the pixels "
              "above the box's mean, less that mean, are summed into x and y "
              "marginals, and the centre is their first moment. Repeated from "
              "the new nearest pixel until it stops moving.",
    "compare": "The shift of an image is the mean over the stars centred in both "
               "it and the reference of reference minus image -- the shift that "
               "moves it onto the reference. Its error propagates the centring "
               "errors; the internal error is the scatter of the stars about the "
               "mean, and a large one means a wrong star or a field that rotated.",
    "measure": "The shifts are measured from the stars themselves rather than "
               "assumed, because guiding errors, flexure and refraction all move "
               "the field by amounts nobody wrote down.",
    "shift": "Resampling onto the reference grid. A whole-pixel shift moves "
             "values without touching them; a fractional one interpolates, which "
             "mixes neighbouring pixels and correlates the noise -- the sky then "
             "looks quieter than it is and any error estimated from it "
             "afterwards is too small.",
    "trim": "Cut to the region every frame has in common, so no pixel of the "
            "result is missing from some of them -- pulled in by one or two pixels "
            "more for poly3, poly5 and spline3, which reach across the edge.",
    "written": "Each frame records the shift in its HISTORY, and its WCS moves "
               "with the pixels (CRPIX and LTV), as IRAF's does.",
}


class XyxyMatchParams(TaskParams):
    input: Path = Field(..., description="Coordinate list to match.")
    reference: Path = Field(..., description="Coordinate list to match it against.")
    output: Path = Field(..., description="Matched pairs, as IRAF writes them.")
    tolerance: float = Field(3.0, gt=0, description="Matching tolerance in pixels.")
    refpoints: Path | None = Field(
        None, description="Tie points: a line of up to three reference x y pairs, "
                          "then a line of the same input stars. They give the "
                          "initial transformation, and override the one below.")
    xin: float | None = Field(None, description="X origin of the input list. With "
                              "xmag..xref, the initial transformation; empty is 0.")
    yin: float | None = Field(None, description="Y origin of the input list.")
    xmag: float | None = Field(None, description="X magnification from input to "
                               "reference; empty is 1.")
    ymag: float | None = Field(None, description="Y magnification; empty is 1.")
    xrotation: float | None = Field(None, description="X rotation, degrees; empty is 0.")
    yrotation: float | None = Field(None, description="Y rotation, degrees; empty is 0.")
    xref: float | None = Field(None, description="X origin of the reference list; "
                               "empty is 0. A pure shift is xref - xin.")
    yref: float | None = Field(None, description="Y origin of the reference list.")
    xcolumn: int = Field(1, ge=1, description="Input list column with x.")
    ycolumn: int = Field(2, ge=1, description="Input list column with y.")
    xrcolumn: int = Field(1, ge=1, description="Reference list column with x.")
    yrcolumn: int = Field(2, ge=1, description="Reference list column with y.")
    separation: float = Field(9.0, ge=0, description="Points closer together than "
                              "this, in pixels, are reduced to the first of them.")
    matching: Literal["tolerance", "triangles"] = Field(
        "triangles", description="triangles finds the transformation from the "
                                 "pattern of the stars and needs no prior; tolerance "
                                 "needs the initial transformation to be right "
                                 "already.")
    nmatch: int = Field(30, ge=3, description="Most points of each list the triangles "
                        "are made from (the first ones after sorting in y).")
    ratio: float = Field(10.0, ge=5.0, le=10.0, description="Longest to shortest side "
                         "of the triangles used.")
    nreject: int = Field(10, ge=0, description="Most rejection passes over the "
                         "matched triangles.")
    xformat: str = Field("%13.3f", description="Format of the x columns.")
    yformat: str = Field("%13.3f", description="Format of the y columns.")


@register
class XyxyMatch(Task[XyxyMatchParams]):
    """IRAF's xyxymatch: pair the stars of two coordinate lists, by tolerance
    after a known transformation, or by Groth's triangles with none. The
    output file is IRAF's, line for line."""

    name = "xyxymatch"
    package = "immatch"
    description = "Match two coordinate lists, by tolerance or by triangles."
    Params = XyxyMatchParams

    def plan(self, params: XyxyMatchParams) -> list[Step]:
        how = (f"triangles of up to {params.nmatch} stars, within {params.tolerance:g} px"
               if params.matching == "triangles"
               else f"nearest within {params.tolerance:g} px")
        return [
            Step("read", f"{params.input} against {params.reference}", why=WHY["read"]),
            Step("match", how, why=WHY[f"match:{params.matching}"]),
            Step("write", str(params.output), why=WHY["write"]),
        ]

    def execute(self, params: XyxyMatchParams) -> TaskResult:
        from ..core import xyxymatch as X

        xr, yr, rl = X.read_list(params.reference,
                                 *_named_columns(params.reference, params.xrcolumn,
                                                 params.yrcolumn))
        xi, yi, il = X.read_list(params.input,
                                 *_named_columns(params.input, params.xcolumn,
                                                 params.ycolumn))
        coeff = X.make_coeff(
            *(0.0 if v is None else v for v in (params.xin, params.yin)),
            *(1.0 if v is None else v for v in (params.xmag, params.ymag)),
            *(0.0 if v is None else v for v in (params.xrotation, params.yrotation,
                                                 params.xref, params.yref)))
        ties = _ties(params.refpoints) if params.refpoints else None
        result = X.xyxymatch(xr, yr, xi, yi, tolerance=params.tolerance,
                             matching=params.matching, separation=params.separation,
                             nmatch=params.nmatch, ratio=params.ratio,
                             nreject=params.nreject, coeff=coeff, ties=ties)
        params.output.parent.mkdir(parents=True, exist_ok=True)
        params.output.write_text(X.format_output(
            result, xr, yr, rl, xi, yi, il, str(params.input), str(params.reference),
            params.xformat, params.yformat), encoding="utf-8")

        final = result.matched if result.matched is not None else result.initial
        if result.matched is None and len(result.pairs) >= 3:
            # With both lists within nmatch, IRAF takes the pairs straight from
            # the matched triangles and fits no transformation, and the file
            # says none -- but a summary reporting the initial one, a shift of
            # zero, would be wrong. So the one the pairs imply is reported.
            r = [pair[0] for pair in result.pairs]
            k = [pair[1] for pair in result.pairs]
            fitted = X.tie_coeff(xr[r], yr[r], xi[k], yi[k])
            final = fitted if fitted is not None else final
        xmag, ymag, xrot, yrot = X.geometry(final)
        steps = self.plan(params)
        by_action = {step.action: step for step in steps}
        by_action["read"].measured = {"reference": len(xr) - 1, "input": len(xi) - 1}
        by_action["match"].measured = {"pairs": len(result.pairs)}
        warnings = []
        if not result.pairs:
            warnings.append(advice.warn(
                f"no star matched{': ' + result.message if result.message else ''}",
                "check that the two lists are of the same field; with matching "
                "triangles try a larger tolerance or nmatch, with tolerance give "
                "the transformation (xin, xref...) or refpoints"))
        return TaskResult(
            outputs=[params.output], steps=steps, warnings=warnings,
            data={"pairs": len(result.pairs), "xshift": round(float(final[3]), 4),
                  "yshift": round(float(final[6]), 4), "xmag": round(xmag, 6),
                  "ymag": round(ymag, 6), "xrotation": round(xrot, 4),
                  "yrotation": round(yrot, 4)},
            history=[f"xyxymatch: {len(result.pairs)} pairs by {params.matching}"])


def _ties(path: Path):
    """rg_getreftie: the first line's reference points, the second's input."""
    text = path.read_text(encoding="utf-8")
    lines = [line.split() for line in text.splitlines() if line.split()]
    if len(lines) < 2:
        advice.fail(f"{path} needs two lines: the reference points, then the input ones",
                    "write 'x1 y1 [x2 y2 [x3 y3]]' for the reference stars on the first "
                    "line and the same stars of the input list on the second")

    def points(words):
        values = []
        for word in words[:6]:
            try:
                values.append(float(word))
            except ValueError:
                break
        values = values[:len(values) // 2 * 2]
        return values[0::2], values[1::2]

    return points(lines[0]), points(lines[1])


def _named_columns(path, xcolumn: int, ycolumn: int) -> tuple[int, int]:
    """Which columns hold x and y. IRAF counts them, 1 and 2 unless told; a
    catalogue harwood wrote (ECSV) names them, and daofind's puts its id
    first -- so counting there matched the ids against the x's, quietly. Left
    at 1 and 2, a named file's own x and y are used."""
    from ..core import columns

    if (xcolumn, ycolumn) != (1, 2):
        return xcolumn, ycolumn
    try:
        text = Path(path).read_text(encoding="utf-8", errors="replace")
    except OSError:
        return xcolumn, ycolumn
    shape = columns.layout(text, str(path))
    names = [name.lower() for name in shape.names]
    if shape.format != "ecsv":
        return xcolumn, ycolumn
    for x, y in (("x", "y"), ("xcenter", "ycenter"), ("xcentroid", "ycentroid")):
        if x in names and y in names:
            return names.index(x) + 1, names.index(y) + 1
    return xcolumn, ycolumn


class _CentringParams(TaskParams):
    """imcentroid's parameters, which imalign passes through."""

    shifts: Path | None = Field(
        None,
        description="Initial shifts, one 'x y' line per input image, as reference "
                    "minus image. Without them, the first star in coords is "
                    "centred with the big box in each image to find one.",
    )
    boxsize: int = Field(
        7, ge=1, description="Size of the fine centring box, in pixels; an even "
                             "size is made odd, as IRAF does.")
    bigbox: int = Field(
        11, ge=1, description="Size of the coarse box the first star is found with "
                              "when no shifts are given.")
    negative: bool = Field(False, description="Are the features negative?")
    background: float | None = Field(
        None, description="Background level. Left out, the mean of each box.")
    lower: float | None = Field(None, description="Lower threshold for the data.")
    upper: float | None = Field(None, description="Upper threshold for the data.")
    niterate: int = Field(3, ge=2, description="Maximum number of centring passes.")
    tolerance: int = Field(
        0, ge=0,
        description="Whole pixels the centre may still move between passes and "
                    "count as converged.",
    )
    maxshift: float | None = Field(
        None, gt=0,
        description="Reject a star that lands further than this from where it "
                    "was looked for. Left out, no limit.",
    )
    verbose: bool = Field(True, description="Print every star's centre.")

    def settings(self):
        from ..core.imcentroid import Settings

        return Settings(boxsize=self.boxsize, bigbox=self.bigbox, negative=self.negative,
                        background=self.background, lower=self.lower, upper=self.upper,
                        niterate=self.niterate, tolerance=self.tolerance,
                        maxshift=self.maxshift)


class ImCentroidParams(_CentringParams):
    input: list[str] = Field(..., description="Images to measure. Accepts globs and @lists.")
    reference: str = Field(
        "", description="Image the shifts are measured against. Left out, only "
                        "the centres are measured.")
    coords: Path = Field(
        ..., description="Positions of stars in the reference image, 1-based (x,y).")


def _centre(params, specs):
    """Run imcentroid's arithmetic over the images a task names."""
    from ..core import catalog, images
    from ..core import imcentroid as ic

    table = catalog.read(params.coords)
    xs, ys = catalog.coordinates(table, params.coords)
    coords = list(zip((float(v) for v in xs), (float(v) for v in ys), strict=True))
    shifts = None
    if params.shifts is not None:
        shifts = []
        for line in Path(params.shifts).read_text(encoding="utf-8").splitlines():
            parts = line.split()
            if len(parts) >= 2 and not line.lstrip().startswith("#"):
                shifts.append((float(parts[0]), float(parts[1])))
    frames = [(str(spec), images.read(spec)[0]) for spec in specs]
    reference = None
    if params.reference:
        reference = (str(params.reference), images.read(params.reference)[0])
    return ic.measure(frames, reference, coords, params.settings(), shifts)


def _centring_advice(result) -> list:
    out = []
    for image in [*result.images, *([result.reference] if result.reference else [])]:
        failed = [(j, star.reason) for j, star in enumerate(image.stars, 1)
                  if star.rejected and star.reason]
        if failed:
            listed = ", ".join(f"{j} ({why})" for j, why in failed[:6])
            out.append(advice.note(
                f"{image.name}: {len(failed)} star(s) not used -- {listed}"
                + (" ..." if len(failed) > 6 else ""),
                "a star on blank sky, in a pair or by the edge does not centre; "
                "drop it from the coordinate list or widen --boxsize"))
        for message in image.warnings:
            out.append(advice.warn(message, "check the coordinate list against "
                                            "this image"))
    if result.vignette is not None:
        out.append(advice.warn(
            "#Vignette_Section = [{}:{},{}:{}]: the images are smaller than the "
            "trim section".format(*result.vignette),
            "imalign will not trim them; trim by hand to the vignetted section"))
    return out


@register
class ImCentroid(Task[ImCentroidParams]):
    name = "imcentroid"
    package = "immatch"
    description = "Measure the shift between images from the centroids of stars."
    Params = ImCentroidParams

    def plan(self, params: ImCentroidParams) -> list[Step]:
        from ..core import images

        specs = images.expand(params.input)
        steps = [Step("read", f"{params.coords} and {len(specs)} images", why=WHY["read"])]
        if params.shifts is None:
            steps.append(Step("coarse", f"first star, box {params.bigbox} px",
                              why=WHY["coarse"]))
        steps.append(Step("centre", f"box {params.boxsize} px, up to {params.niterate} "
                                    "passes", why=WHY["centre"]))
        if params.reference:
            steps.append(Step("compare", f"against {params.reference}", why=WHY["compare"]))
        return steps

    def execute(self, params: ImCentroidParams) -> TaskResult:
        from ..core import images
        from ..core import imcentroid as ic

        specs = images.expand(params.input)
        if not specs:
            raise ValueError("No images specified")
        result = _centre(params, specs)
        rows = [{"image": image.name, "xshift": image.xshift, "xerr": image.xerr,
                 "yshift": image.yshift, "yerr": image.yerr, "nstars": image.nstars,
                 "xinternal": image.xinternal, "yinternal": image.yinternal}
                for image in result.images if image.nstars]
        data: dict = {"text": ic.report(result, params.verbose), "shifts": rows}
        if result.trim is not None:
            data["trim"] = "[{}:{},{}:{}]".format(*result.trim)
        return TaskResult(
            data=data, steps=self.plan(params), warnings=_centring_advice(result),
            history=[f"imcentroid: {len(rows)} shifts against {params.reference} "
                     f"from {params.coords}" if params.reference
                     else f"imcentroid: centres from {params.coords}"],
        )


INTERP = Literal["nearest", "linear", "poly3", "poly5", "spline3"]
BOUNDARY = Literal["nearest", "constant", "reflect", "wrap"]


def _shift_header(header, xshift: float, yshift: float) -> None:
    """mw_shift: the logical system moves with the pixels."""
    import numpy as np

    for axis, value in ((1, xshift), (2, yshift)):
        value = float(np.float32(value))
        if f"CRPIX{axis}" in header:
            header[f"CRPIX{axis}"] = header[f"CRPIX{axis}"] + value
        header[f"LTV{axis}"] = header.get(f"LTV{axis}", 0.0) + value


def _cut_header(header, x1: int, y1: int) -> None:
    """What reading a section does to the WCS: the origin moves to (x1, y1)."""
    for axis, first in ((1, x1), (2, y1)):
        if f"CRPIX{axis}" in header:
            header[f"CRPIX{axis}"] = header[f"CRPIX{axis}"] - (first - 1)
        header[f"LTV{axis}"] = header.get(f"LTV{axis}", 0.0) - (first - 1)


class ImShiftParams(TaskParams):
    input: list[str] = Field(..., description="Images to shift. Accepts globs and @lists.")
    output: str = Field(
        "", description="Output images: a name, a list, a %old%new% template or a "
                        "directory. Left out, this run's own directory.")
    xshift: float | None = Field(None, description="Shift in x, pixels; positive "
                                                   "moves the image to +x.")
    yshift: float | None = Field(None, description="Shift in y, pixels.")
    shifts_file: Path | None = Field(
        None, description="One 'x y' line per input image, instead of xshift/yshift.")
    interp_type: str = Field(
        "linear", description="Interpolant for a fractional shift: nearest, linear, "
                              "poly3, poly5, spline3, sinc (31 points; sinc21 for 21), "
                              "lsinc, or drizzle (drizzle[0.5] for a pixel fraction). A "
                              "whole-pixel shift copies the pixels whatever this is.")

    @field_validator("interp_type")
    @classmethod
    def _known_interpolant(cls, value: str) -> str:
        from ..core import imshift

        if imshift.parse_interp(value)[0] not in imshift.INTERPOLANTS:
            raise ValueError(f"'{value}' is not an interpolant: use one of "
                             + ", ".join(imshift.INTERPOLANTS)
                             + ", or sinc21, lsinc21, drizzle[0.5]")
        return value
    boundary_type: BOUNDARY = Field(
        "nearest", description="How the image is extended past its edges.")
    constant: float = Field(0.0, description="Value for boundary_type=constant.")


@register
class ImShift(Task[ImShiftParams]):
    name = "imshift"
    package = "imgeom"
    description = "Shift images by a fraction of a pixel or more."
    Params = ImShiftParams

    def plan(self, params: ImShiftParams) -> list[Step]:
        source = (f"from {params.shifts_file}" if params.shifts_file
                  else f"{params.xshift} {params.yshift}")
        return [Step("shift", f"{source}, {params.interp_type}, boundary "
                              f"{params.boundary_type}", why=WHY["shift"]),
                Step("write", params.output or "this run's directory", why=WHY["written"])]

    def execute(self, params: ImShiftParams) -> TaskResult:
        from ..core import images

        specs = images.expand(params.input)
        if params.shifts_file is not None:
            pairs = []
            for line in params.shifts_file.read_text(encoding="utf-8").splitlines():
                parts = line.split()
                try:
                    pairs.append((float(parts[0]), float(parts[1])))
                except (IndexError, ValueError):
                    continue
            pairs = pairs[:len(specs)]
            if len(pairs) != len(specs):
                raise ValueError("The number of input images and shifts are not the same.")
        elif params.xshift is None or params.yshift is None:
            raise ValueError("give --xshift and --yshift, or --shifts-file")
        else:
            pairs = [(params.xshift, params.yshift)] * len(specs)
        destinations = images.outputs_for(specs, params.output)
        outputs, history = _shift_all(specs, destinations, pairs, params.interp_type,
                                      params.boundary_type, params.constant, "imshift")
        return TaskResult(outputs=outputs, history=history, steps=self.plan(params),
                          warnings=_fractional_warning(pairs, params.interp_type))


def _shift_all(specs, destinations, pairs, interp, boundary, constant, task,
               section=None):
    import numpy as np
    from astropy.io import fits

    from ..core import images, provenance
    from ..core.imshift import shift

    outputs, history = [], []
    for spec, destination, (dx, dy) in zip(specs, destinations, pairs, strict=True):
        data, header = images.read(spec)
        moved = shift(data, dx, dy, interp, boundary, constant)
        _shift_header(header, dx, dy)
        line = f"{task}: shifted by {dx:+.3f} {dy:+.3f} px, {interp}"
        if section is not None:
            x1, x2, y1, y2 = section
            moved = moved[y1 - 1:y2, x1 - 1:x2]
            _cut_header(header, x1, y1)
            line += f", trimmed to [{x1}:{x2},{y1}:{y2}]"
        provenance.stamp_header(header, [line])
        destination = Path(destination)
        destination.parent.mkdir(parents=True, exist_ok=True)
        fits.PrimaryHDU(np.asarray(moved, dtype=np.float32), header).writeto(
            destination, overwrite=True)
        outputs.append(destination)
        history.append(line)
    return outputs, history


def _fractional_warning(pairs, interp) -> list:
    fractional = [p for p in pairs if p[0] != int(p[0]) or p[1] != int(p[1])]
    if not fractional or interp == "nearest":
        return []
    return [advice.warn(
        f"{len(fractional)} of {len(pairs)} shifts were fractional, so {interp} "
        "interpolation mixed neighbouring pixels. That correlates the noise: the "
        "sky looks quieter than it is, and any error estimated from it afterwards "
        "will be too small.",
        "measure photometry on the unaligned frames and shift the coordinates "
        "instead, or use --interp-type nearest if the alignment only has to be "
        "good to a pixel.")]


class ImAlignParams(_CentringParams):
    input: list[str] = Field(..., description="Images to align. Accepts globs and @lists.")
    reference: str = Field(..., description="Image everything is aligned to.")
    coords: Path = Field(
        ..., description="Positions of stars in the reference image, 1-based (x,y).")
    output: str = Field(
        "", description="Output images: a list, a %old%new% template or a "
                        "directory. Left out, this run's own directory.")
    shiftimages: bool = Field(True, description="Shift the images? Without it only "
                                                "the shifts are measured.")
    interp_type: INTERP = Field("linear", description="Interpolant for the shift.")
    boundary_type: BOUNDARY = Field("nearest", description="Boundary extension.")
    constant: float = Field(0.0, description="Value for boundary_type=constant.")
    trimimages: bool = Field(
        True, description="Trim the shifted images to the section all of them cover.")


@register
class ImAlign(Task[ImAlignParams]):
    name = "imalign"
    package = "immatch"
    description = "Register images by the centroids of stars, shift and trim them."
    Params = ImAlignParams

    def plan(self, params: ImAlignParams) -> list[Step]:
        from ..core import images

        specs = images.expand(params.input)
        steps = [Step("measure", f"imcentroid of {len(specs)} images against "
                                 f"{params.reference}, box {params.boxsize} px",
                      why=WHY["measure"])]
        if params.shiftimages:
            steps.append(Step("shift", f"imshift, {params.interp_type}, boundary "
                                       f"{params.boundary_type}", why=WHY["shift"]))
            if params.trimimages:
                steps.append(Step("trim", "to #Trim_Section", why=WHY["trim"]))
            steps.append(Step("write", params.output or "this run's directory",
                              why=WHY["written"]))
        return steps

    def execute(self, params: ImAlignParams) -> TaskResult:
        from ..core import images
        from ..core import imcentroid as ic

        specs = images.expand(params.input)
        destinations = images.outputs_for(specs, params.output)
        result = _centre(params, specs)
        warnings = _centring_advice(result)
        data: dict = {"text": ic.report(result, params.verbose)}
        steps = self.plan(params)
        if not params.shiftimages:
            return TaskResult(data=data, steps=steps, warnings=warnings)

        missing = [image.name for image in result.images if not image.nstars]
        if all(image.nstars == 0 for image in result.images):
            raise RuntimeError("No shifts were calculated.")
        if missing:
            raise RuntimeError(
                f"no star was centred in both {', '.join(missing)} and the reference, "
                "so there is no shift for them; imalign would stop here too")
        # imalign reads the shifts back from imcentroid's printed table, so
        # what it applies is rounded to a thousandth of a pixel.
        pairs = [(round(image.xshift, 3), round(image.yshift, 3))
                 for image in result.images]

        section = None
        if params.trimimages:
            section, why = _trim_section(result, params.interp_type)
            if section is None:
                warnings.append(advice.note(why, "trim the outputs by hand with "
                                                 "imcopy and a section"))
            else:
                data["trim"] = "[{}:{},{}:{}]".format(*section)
        outputs, history = _shift_all(specs, destinations, pairs, params.interp_type,
                                      params.boundary_type, params.constant, "imalign",
                                      section)
        warnings += _fractional_warning(pairs, params.interp_type)
        data["shifts"] = [{"image": str(spec), "xshift": x, "yshift": y}
                          for spec, (x, y) in zip(specs, pairs, strict=True)]
        return TaskResult(outputs=outputs, history=history, steps=steps,
                          warnings=warnings, data=data)


def _trim_section(result, interp):
    """imalign.cl: the trim section, pulled in where the interpolant reached
    across the boundary."""
    if result.vignette is not None:
        return None, "Images not trimmed ! Vignetting is present."
    if result.trim is None:
        return None, "Images not trimmed ! Trim section is undefined."
    x1, x2, y1, y2 = result.trim
    grow = {"poly3": 1, "poly5": 2, "spline3": 2}.get(interp, 0)
    x1, x2, y1, y2 = x1 + grow, x2 - grow, y1 + grow, y2 - grow
    if 1 <= x1 <= x2 and 1 <= y1 <= y2:
        return (x1, x2, y1, y2), ""
    return None, "Images not trimmed !  No overlap region."
