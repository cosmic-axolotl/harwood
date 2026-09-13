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
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import Field

from ..core import advice
from ..core.registry import register
from ..core.task import Step, Task, TaskParams, TaskResult

WHY = {
    "read": "Two lists of positions of the same field, taken at different times "
            "or through different filters. Nothing about them says which entry "
            "is which star -- that is the whole problem.",
    "match:tolerance": "Nearest neighbour within a radius, after estimating the "
                       "shift from the histogram of all pairwise offsets. Fast, "
                       "and right when the frames are already close; it cannot "
                       "cope with rotation.",
    "match:triangles": "Triangles of stars are described by quantities that "
                       "survive translation, rotation and scale -- the ratio of "
                       "the longest side to the shortest, and the angle between "
                       "them. So the same three stars give the same two numbers "
                       "in any frame however it moved, and matching triangles "
                       "vote for the pairs of stars that made them.",
    "fit": "A linear transform: x' = a + bx + cy, y' = d + ex + fy. Six "
           "coefficients hold translation, rotation, scale and shear, and the "
           "shift, rotation and scale are read back out of them because "
           "'rotated by 0.03 degrees' says more than six numbers. Outliers are "
           "rejected against a robust sigma: least squares has no defence "
           "against one wrong pair, which will bend the whole solution towards "
           "itself while still returning an answer.",
    "write": "The matched pairs, so a later fit can use them without matching "
             "again, with the transform recorded in the metadata.",
    "centre": "Each star is centred by its flux centroid in a small box. A "
              "coarse box first, in case the frame moved further than the fine "
              "one covers; then the fine one, which is what sets the precision.",
    "compare": "The shift is the median of what the stars say, not the mean: one "
               "mismatched star would move a mean and cannot move a median. The "
               "scatter of the individual answers is reported beside it, because "
               "that is what tells you whether to trust the median.",
    "report": "A shift good to a tenth of a pixel is a real measurement; one "
              "whose stars disagree by a pixel means something else is going on "
              "-- rotation, a wrong star, or a field that moved more than the "
              "box.",
    "measure": "The shifts are measured from the stars themselves rather than "
               "assumed, because guiding errors, flexure and refraction all move "
               "the field by amounts nobody wrote down.",
    "shift": "Resampling onto the reference grid. A whole-pixel shift moves "
             "values without touching them; a fractional one interpolates, which "
             "mixes neighbouring pixels and correlates the noise -- the sky then "
             "looks quieter than it is and any error estimated from it "
             "afterwards is too small.",
    "trim": "Cut to the region every frame has in common, so no pixel of the "
            "result is missing from some of them.",
}


class XyxyMatchParams(TaskParams):
    input: Path = Field(..., description="Coordinate list to match.")
    reference: Path = Field(..., description="Coordinate list to match it against.")
    output: Path = Field(..., description="Matched pairs to write.")

    tolerance: float = Field(
        2.0, gt=0,
        description="How close two points must land to be called the same star, "
                    "in pixels, after the transformation is applied.",
    )
    matching: Literal["tolerance", "triangles"] = Field(
        "triangles",
        description="'triangles' finds the transformation from the pattern of the "
                    "stars themselves and needs no prior; 'tolerance' just pairs "
                    "nearest neighbours and needs the frames to be close already.",
    )
    nmatch: int = Field(
        30, ge=3,
        description="How many of the brightest stars take part in the triangle "
                    "search. More is slower and rarely better.",
    )
    xin: float | None = Field(
        None, description="Initial shift in x, if you already know it.")
    yin: float | None = Field(
        None, description="Initial shift in y, if you already know it.")
    ratio_tolerance: float = Field(
        0.02, gt=0,
        description="How closely two triangles must agree in shape to vote for "
                    "the same stars.",
    )


@register
class XyxyMatch(Task[XyxyMatchParams]):
    name = "xyxymatch"
    package = "immatch"
    description = "Match two coordinate lists and solve for the transformation."
    Params = XyxyMatchParams

    def plan(self, params: XyxyMatchParams) -> list[Step]:
        return [
            Step("read", f"{params.input} against {params.reference}",
                 why=WHY["read"]),
            Step("match", f"{params.matching}"
                 + (f", {params.nmatch} brightest" if params.matching == "triangles"
                    else f", within {params.tolerance} px"),
                 why=WHY[f"match:{params.matching}"]),
            Step("fit", "linear transform: shift, scale, rotation", why=WHY["fit"]),
            Step("write", str(params.output), why=WHY["write"]),
        ]

    def execute(self, params: XyxyMatchParams) -> TaskResult:
        from astropy.table import Table

        from ..core import catalog, match, provenance

        reference = catalog.read(params.reference)
        moving = catalog.read(params.input)
        rx, ry = catalog.coordinates(reference, params.reference)
        ix, iy = catalog.coordinates(moving, params.input)

        if params.matching == "triangles":
            found = match.by_triangles(
                rx, ry, ix, iy, nmatch=params.nmatch, tolerance=params.tolerance,
                ratio_tolerance=params.ratio_tolerance,
                rflux=_flux(reference), iflux=_flux(moving),
            )
        else:
            found = match.by_tolerance(rx, ry, ix, iy, tolerance=params.tolerance,
                                       xin=params.xin, yin=params.yin)
        if not found:
            raise RuntimeError(
                f"no usable match: {found.message or 'nothing paired up'}. "
                f"{len(reference)} reference and {len(moving)} input positions. "
                "Try --matching triangles, a larger --tolerance, or check that the "
                "two lists are of the same field."
            )

        transform = match.fit_transform(rx[found.reference], ry[found.reference],
                                        ix[found.input], iy[found.input],
                                        niterate=3, reject=2.5)
        pairs = Table({
            "xref": rx[found.reference], "yref": ry[found.reference],
            "xin": ix[found.input], "yin": iy[found.input],
            "ref_id": found.reference + 1, "in_id": found.input + 1,
        })
        catalog.write(pairs, params.output,
                      harwood_reference=str(params.reference),
                      harwood_input=str(params.input),
                      harwood_transform=list(transform["coefficients"]),
                      harwood_call=provenance.current_stamp() or "")

        steps = self.plan(params)
        by_action = {step.action: step for step in steps}
        by_action["read"].measured = {"reference": len(reference),
                                      "input": len(moving)}
        by_action["match"].measured = {"pairs": len(found), "method": found.method}
        by_action["fit"].measured = {
            "shift": f"{transform['xshift']:+.3f}, {transform['yshift']:+.3f} px",
            "rotation": f"{transform['rotation']:+.4f} deg",
            "scale": f"{transform['xscale']:.6f}",
            "rms": f"{transform['rms']:.3f} px"
                   + ("  -- what centroids are good for"
                      if transform["rms"] < 0.4 else "  -- looser than centroids"),
            "rejected": transform["nrejected"],
        }

        return TaskResult(
            outputs=[params.output],
            steps=steps,
            data={"pairs": len(found), "xshift": round(transform["xshift"], 4),
                  "yshift": round(transform["yshift"], 4),
                  "rotation_deg": round(transform["rotation"], 5),
                  "scale": round(transform["xscale"], 6),
                  "rms": round(transform["rms"], 4),
                  "rejected": transform["nrejected"]},
            history=[f"xyxymatch: {len(found)} pairs by {found.method}, "
                     f"shift {transform['xshift']:+.3f} {transform['yshift']:+.3f}, "
                     f"rotation {transform['rotation']:+.4f} deg, "
                     f"rms {transform['rms']:.3f}"],
        )


def _flux(table) -> object:
    """Brightness column, if the catalogue has one.

    The brightest stars are the ones both frames agree on, so they are the
    ones the triangle search should use.
    """
    import numpy as np

    for name in ("flux", "peak", "amplitude"):
        if name in table.colnames:
            return np.asarray(table[name], dtype=float)
    for name in ("mag", "magnitude"):
        if name in table.colnames:
            return -np.asarray(table[name], dtype=float)   # brighter is smaller
    return None


class ImCentroidParams(TaskParams):
    input: list[str] = Field(..., description="Images to measure. Accepts globs and @lists.")
    reference: str = Field(..., description="Image the shifts are measured against.")
    coords: Path = Field(
        ..., description="Positions of stars in the reference image, 1-based (x,y)."
    )
    boxsize: int = Field(
        7, ge=3, description="Size of the centring box, in pixels. About twice the FWHM."
    )
    bigbox: int = Field(
        11, ge=3,
        description="Size of the coarse search box used first, for frames that moved "
                    "further than the centring box.",
    )
    niterate: int = Field(3, ge=1, description="Centring passes per star.")
    tolerance: float = Field(
        0.05, gt=0, description="Stop centring when the position moves less than this."
    )
    maxshift: float = Field(
        20.0, gt=0,
        description="Reject a star whose measured shift differs from the frame's "
                    "median by more than this. Catches mismatched stars.",
    )
    shifts: Path | None = Field(
        None, description="Optional file to write the shifts to, one line per image."
    )


@register
class ImCentroid(Task[ImCentroidParams]):
    name = "imcentroid"
    package = "immatch"
    description = "Measure the shift between images from the centroids of stars."
    Params = ImCentroidParams

    def plan(self, params: ImCentroidParams) -> list[Step]:
        from ..core import images

        specs = images.expand(params.input)
        return [
            Step("read", f"{params.coords} and {len(specs)} images", why=WHY["read"]),
            Step("centre", f"box {params.boxsize} px, {params.niterate} passes, "
                           f"coarse search {params.bigbox} px", why=WHY["centre"]),
            Step("compare", f"against {params.reference}", why=WHY["compare"]),
            Step("report", "shift per image, with its scatter", why=WHY["report"]),
        ]

    def execute(self, params: ImCentroidParams) -> TaskResult:
        import numpy as np

        from ..core import catalog, images
        from ..viewer import analysis

        positions = catalog.read(params.coords)
        x0, y0 = catalog.coordinates(positions, params.coords)

        reference_data, _ = images.read(params.reference)
        rx, ry = _centroids(reference_data, x0, y0, params, analysis)

        rows, shifts = [], []
        for spec in images.expand(params.input):
            data, _ = images.read(spec)
            cx, cy = _centroids(data, x0, y0, params, analysis)
            good = np.isfinite(cx) & np.isfinite(rx)
            dx, dy = cx[good] - rx[good], cy[good] - ry[good]
            if dx.size == 0:
                raise RuntimeError(f"no star could be centred in {spec}")

            # A star that lands far from what every other star says is a
            # mismatch, not a measurement; the median is not fooled but the
            # scatter would be, and the scatter is what tells you to trust it.
            keep = (np.abs(dx - np.median(dx)) < params.maxshift) & \
                   (np.abs(dy - np.median(dy)) < params.maxshift)
            rows.append({
                "image": str(spec),
                "xshift": round(float(np.median(dx[keep])), 4),
                "yshift": round(float(np.median(dy[keep])), 4),
                "xrms": round(float(np.std(dx[keep])), 4),
                "yrms": round(float(np.std(dy[keep])), 4),
                "nstars": int(keep.sum()),
            })
            shifts.append(rows[-1])

        if params.shifts:
            params.shifts.parent.mkdir(parents=True, exist_ok=True)
            params.shifts.write_text(
                "".join(f"{r['xshift']:.4f} {r['yshift']:.4f}\n" for r in shifts),
                encoding="utf-8")

        return TaskResult(
            outputs=[params.shifts] if params.shifts else [],
            data={"rows": rows},
            history=[f"imcentroid: shifts against {params.reference} from "
                     f"{len(x0)} stars"],
            steps=self.plan(params),
        )


def _centroids(data, x0, y0, params, analysis):
    """Centre every star, coarse box first, then the fine one."""
    import numpy as np

    xs, ys = [], []
    for x, y in zip(x0, y0, strict=False):
        cx, cy = float(x), float(y)
        for size in (params.bigbox, params.boxsize):
            for _ in range(params.niterate):
                nx, ny = analysis.centroid(data, cx, cy, radius=size / 2)
                moved = np.hypot(nx - cx, ny - cy)
                cx, cy = nx, ny
                if moved < params.tolerance:
                    break
        inside = (1 <= cx <= data.shape[1]) and (1 <= cy <= data.shape[0])
        xs.append(cx if inside else np.nan)
        ys.append(cy if inside else np.nan)
    return np.array(xs), np.array(ys)


class ImAlignParams(TaskParams):
    input: list[str] = Field(..., description="Images to align. Accepts globs and @lists.")
    reference: str = Field(..., description="Image everything is aligned to.")
    output: Path = Field(..., description="Directory for the aligned images.")
    coords: Path | None = Field(
        None, description="Star positions used to measure the shifts. Without them, "
                          "--shifts must be given."
    )
    shifts: list[float] | None = Field(
        None, description="Shifts to apply directly, as x y pairs, instead of "
                          "measuring them."
    )
    boxsize: int = Field(7, ge=3, description="Centring box, in pixels.")
    interp: Literal["nearest", "linear", "spline3"] = Field(
        "linear",
        description="How pixels are resampled for a fractional shift. 'nearest' "
                    "moves whole pixels only and does not touch the values, which "
                    "matters when you intend to do photometry afterwards.",
    )
    trim: bool = Field(
        False, description="Trim every output to the region all frames have in common."
    )


@register
class ImAlign(Task[ImAlignParams]):
    name = "imalign"
    package = "immatch"
    description = "Shift images onto a common pixel grid."
    Params = ImAlignParams

    def plan(self, params: ImAlignParams) -> list[Step]:
        from ..core import images

        specs = images.expand(params.input)
        steps = [Step("read", f"{len(specs)} images, reference {params.reference}",
                      why=WHY["read"])]
        if params.shifts:
            steps.append(Step("shifts", "given on the command line",
                              why="Taken as stated rather than measured."))
        else:
            steps.append(Step("measure", f"centroids from {params.coords}, "
                                         f"box {params.boxsize} px",
                              why=WHY["measure"]))
        steps.append(Step("shift", f"{params.interp} interpolation", why=WHY["shift"]))
        if params.trim:
            steps.append(Step("trim", "to the common region", why=WHY["trim"]))
        steps.append(Step("write", str(params.output),
                          why="Each frame records the shift applied in XSHIFT and "
                              "YSHIFT, so the alignment can be undone or checked."))
        return steps

    def execute(self, params: ImAlignParams) -> TaskResult:
        import numpy as np
        from astropy.io import fits

        from ..core import images, provenance

        try:
            from scipy.ndimage import shift as ndshift
        except ImportError as error:
            raise RuntimeError(
                "imalign resamples with scipy, which is part of harwood[phot]. "
                "Run: pip install harwood[phot]"
            ) from error
        from ..core.registry import get_task

        specs = images.expand(params.input)
        if params.shifts:
            if len(params.shifts) != 2 * len(specs):
                raise ValueError(
                    f"--shifts needs two numbers per image: {len(specs)} images "
                    f"but {len(params.shifts)} numbers"
                )
            pairs = [(params.shifts[2 * i], params.shifts[2 * i + 1])
                     for i in range(len(specs))]
        else:
            if params.coords is None:
                raise ValueError("give --coords to measure the shifts, or --shifts "
                                 "to state them")
            measured = get_task("imcentroid").run(
                input=[str(s) for s in specs], reference=params.reference,
                coords=params.coords, boxsize=params.boxsize)
            pairs = [(row["xshift"], row["yshift"]) for row in measured.data["rows"]]

        order = {"nearest": 0, "linear": 1, "spline3": 3}[params.interp]
        params.output.mkdir(parents=True, exist_ok=True)
        outputs, history = [], []
        for spec, (dx, dy) in zip(specs, pairs, strict=False):
            data, header = images.read(spec)
            # A shift of +dx means the frame sits dx to the right of the
            # reference, so the pixels move back by dx to line up.
            moved = ndshift(data, (-dy, -dx), order=order, mode="constant", cval=np.nan)
            line = (f"imalign: shifted by {-dx:+.3f} {-dy:+.3f} px onto "
                    f"{Path(params.reference).name}, {params.interp}")
            provenance.stamp_header(header, [line])
            header["XSHIFT"] = (round(-dx, 4), "shift applied by harwood, pixels")
            header["YSHIFT"] = (round(-dy, 4), "shift applied by harwood, pixels")
            destination = params.output / Path(str(spec.path)).name
            fits.PrimaryHDU(np.nan_to_num(moved, nan=0.0).astype(np.float32),
                            header).writeto(destination, overwrite=True)
            outputs.append(destination)
            history.append(line)

        warnings = []
        fractional = [p for p in pairs if abs(p[0] % 1) > 0.01 or abs(p[1] % 1) > 0.01]
        if fractional and params.interp != "nearest":
            warnings.append(advice.warn(
                f"{len(fractional)} of {len(pairs)} shifts were fractional, so "
                f"{params.interp} interpolation mixed neighbouring pixels. That "
                "correlates the noise: the sky looks quieter than it is, and "
                "any error estimated from it afterwards will be too small.",
                "measure photometry on the unaligned frames and shift the "
                "coordinates instead, or use --interp nearest if the alignment "
                "only has to be good to a pixel."))
        if params.trim:
            warnings.append(advice.note(
                "--trim is not built yet; the frames were written whole.",
                "cut them yourself with a section, as in 'harwood imcopy "
                "out.fits[20:1000,20:1000] cropped.fits'."))

        return TaskResult(outputs=outputs, history=history, steps=self.plan(params),
                          warnings=warnings,
                          data={"images": len(outputs),
                                "max_shift": round(max(abs(v) for pair in pairs
                                                       for v in pair), 3)})
