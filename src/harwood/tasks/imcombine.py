"""imcombine -- combine images pixel by pixel.

One engine, reused. Section 3.6 of the inventory shows IRAF wiring `combine`,
`zerocombine`, `darkcombine`, `flatcombine`, `crcombine`, `scombine`,
`odcombine` and `lscombine` to the same code with different defaults. Here the
engine is written once and those names are **presets** of it (section 3.2), so
there is one implementation to get right and one set of parameter names to
learn.

Parameter names, values and meanings are IRAF's: `combine`, `reject`, `lsigma`,
`nlow`, `rdnoise`, `scale`. Typing `--reject ccdclip` gets IRAF's behaviour or
a clear error -- never a different behaviour under the same name.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

from pydantic import Field, model_validator

from ..core.registry import register
from ..core.task import Step, Task, TaskParams, TaskResult, format_field, uncertainty_field

#: What each step does to the data, and why. Shown by --explain, written here
#: rather than in a docstring because the person who needs it is running the
#: task, not reading the source.
WHY = {
    "read": "Combining needs frames of one shape taken of one thing. What the "
            "combination is *for* decides everything after: a bias measures the "
            "electronic pattern, a flat the pixel response, a stack of exposures "
            "the sky -- and each wants a different rejection.",
    "threshold": "Pixels outside these limits are dropped before anything else is "
                 "measured. It is the blunt instrument: saturated pixels have no "
                 "information left in them, and a saturated star in one frame "
                 "would otherwise pull the rejection statistics around.",
    "scale": "Frames taken through cloud, or with different exposure times, sit "
             "at different levels. Rejection compares each pixel against the "
             "others at that position, so the frames have to be brought to a "
             "common level first or the whole of a fainter frame looks deviant.",
    "zero": "An additive offset, where the difference between frames is a "
            "pedestal rather than a scale -- a changing sky, typically, which "
            "adds the same counts everywhere rather than multiplying them.",
    "reject:none": "Nothing is rejected: every pixel goes into the combination. "
                   "Right when the frames are already clean, and wrong the moment "
                   "a cosmic ray lands anywhere.",
    "reject:minmax": "The lowest and highest values at each position are simply "
                     "dropped, without asking whether they are deviant. Crude, "
                     "predictable, and the right choice for a handful of bias "
                     "frames where there is no noise model worth trusting and "
                     "nothing to protect but the odd cosmic ray.",
    "reject:sigclip": "Each value is compared against the spread of the others at "
                      "that pixel. That is its weakness as well as its strength: "
                      "the threshold comes from the data, so with few frames two "
                      "cosmic rays on the same pixel inflate the spread enough to "
                      "hide each other, and neither is rejected.",
    "reject:ccdclip": "The threshold comes from what the detector is known to do, "
                      "not from the spread of the stack: sigma^2 = rdnoise^2/gain^2 "
                      "+ level/gain + (snoise*level)^2 -- read noise, photon noise, "
                      "and flat-field error. Because it does not depend on the "
                      "stack, a second deviant value cannot hide the first.",
    "reject:avsigclip": "Like sigclip, but the spread is not measured at each pixel "
                        "from a handful of values: it is pooled along the whole "
                        "image line and scaled by the square root of the level, "
                        "the way photon noise scales. With few frames that is a far "
                        "steadier sigma than any single pixel can give, which is "
                        "why IRAF's flatcombine uses it.",
    "reject:pclip": "The spread is read off the stack itself: the value pclip "
                    "places away from the median stands for one sigma. Robust "
                    "when nothing is known about the noise.",
    "grow": "A cosmic ray's bright core is rejected, and its fainter edge would "
            "not be; rejecting its neighbours within grow pixels, in the same "
            "image, takes the edge too. The combine is then redone without them.",
    "offsets": "Dithered frames do not start at the same pixel. Each goes where its offset "
               "puts it in an output large enough for all of them, and at each output pixel "
               "only the images that reach it are combined.",
    "mask": "Each image's mask leaves out the pixels it marks before anything is measured "
            "or combined, the scaling statistics included.",
    "project": "The images are the planes of one cube, combined along its last axis.",
    "weight": "Each image counts in the average in proportion to its level or its "
              "exposure -- IRAF's weighting, by the signal itself.",
    "weight:ivar": "Each pixel of each image counts by the inverse of its variance, which "
                   "gives the lowest noise any average of them can have: a pixel in a poor "
                   "frame or under a bright star counts for less. It is not one of IRAF's, "
                   "whose weights go by the image and by the level, whatever the noise is.",
    "combine:lmedian": "The median, except that of two values the lower is taken "
                       "rather than their average.",
    "combine:average": "The mean of what survived. Lowest noise when the "
                       "survivors really are just noise, which is what the "
                       "rejection was for.",
    "combine:median": "The middle value of what survived. Noisier than the mean by "
                      "about 25 per cent, and almost unmoved by anything the "
                      "rejection missed -- which is why flats, where a few bad "
                      "pixels matter more than a little extra noise, usually take "
                      "the median.",
    "combine:sum": "Added rather than averaged, so the result is in the units of "
                   "the total exposure. Only sensible when the frames are already "
                   "on a common scale.",
    "combine:quadrature": "The square root of the sum of the squares, over n: how to "
                          "combine images of errors or sigmas.",
    "combine:nmodel": "The noise the CCD model gives each value, from rdnoise, gain and "
                      "snoise, combined in quadrature: an estimate of the sigma of the "
                      "average, with no need of a sigma image from the data.",
    "write": "The combined frame carries IMCMB cards naming every input, so what "
             "went into it survives being copied somewhere else.",
}

Combine = Literal["average", "median", "lmedian", "sum", "quadrature", "nmodel"]
Reject = Literal["none", "minmax", "sigclip", "avsigclip", "ccdclip", "crreject", "pclip"]
SCALES = ("none", "mode", "median", "mean", "exposure")
ZEROS = ("none", "mode", "median", "mean")


class ImCombineParams(TaskParams):
    """Parameters of imcombine. Names follow IRAF; see `harwood help imcombine`."""

    input: list[str] = Field(
        ...,
        description="Images to combine. Accepts a shell glob or an @list file.",
    )
    output: Path = Field(..., description="Combined image to write.")

    combine: Combine = Field(
        "average",
        description="How surviving pixels are combined once rejection is done. "
                    "lmedian takes the lower of two values where median "
                    "would average them; quadrature and nmodel combine errors.",
    )
    reject: Reject = Field(
        "none",
        description=(
            "Rejection algorithm. 'sigclip' judges each pixel against the spread "
            "of the stack; 'ccdclip' and 'crreject' judge it against the CCD noise "
            "model (gain, rdnoise, snoise), which is what you want when few images "
            "go in; 'avsigclip' uses a sigma pooled over each image line, scaled "
            "by the Poisson expectation -- IRAF's flatcombine default. 'pclip' "
            "measures the spread from the percentile pclip away from the median. "
            "'minmax' simply drops the nlow lowest and nhigh highest."
        ),
    )

    lsigma: float = Field(
        3.0, ge=0, description="Rejection threshold below the centre, in sigma."
    )
    hsigma: float = Field(
        3.0, ge=0, description="Rejection threshold above the centre, in sigma."
    )
    mclip: bool = Field(
        True,
        description="Judge deviations against the median rather than the mean. "
                    "Rarely turn off.",
    )
    nlow: int = Field(1, ge=0, description="For minmax: how many low pixels to drop.")
    nhigh: int = Field(1, ge=0, description="For minmax: how many high pixels to drop.")
    nkeep: int = Field(
        1, description="Minimum number of pixels to keep at each position; "
                       "negative, the most that may be rejected -- as in IRAF."
    )
    pclip: float = Field(
        -0.5, description="For pclip: which value, counted from the median, measures "
                          "the spread. Below 1 in size, a fraction of half the "
                          "images; negative counts down from the median."
    )
    grow: float = Field(
        0.0, ge=0, description="Also reject the neighbours of a rejected pixel within "
                               "this radius, in the same image -- a cosmic ray's "
                               "fainter edges with it."
    )
    sigscale: float = Field(
        0.1, ge=0, description="When the scales differ from 1 by more than this, the "
                               "sigma clipping corrects each image's sigma for its "
                               "scale, as IRAF does."
    )

    gain: float | str = Field(
        1.0, description="Detector gain in electrons per ADU, or the keyword that has it. "
                         "Used by ccdclip, crreject and nmodel."
    )
    rdnoise: float | str = Field(
        0.0, description="Read noise in electrons, or the keyword that has it. Used by "
                         "ccdclip, crreject and nmodel."
    )
    snoise: float | str = Field(
        0.0, description="Sensitivity noise as a fraction, for flat-field error, or the "
                         "keyword that has it. Used by ccdclip, crreject and nmodel.",
    )

    scale: str = Field(
        "none",
        description="Scale each image before combining: none, mode, median, mean, "
                    "exposure, @file of numbers or !keyword (factors to multiply by).",
    )
    zero: str = Field(
        "none", description="Subtract an additive offset from each image before combining: "
                            "none, mode, median, mean, @file or !keyword (offsets to add)."
    )
    weight: str = Field(
        "none", description="Weight each image in the average by its level or its "
                            "exposure time, as IRAF does (by the level itself, not "
                            "by its inverse variance), or by @file or !keyword; or ivar, "
                            "not IRAF's: each pixel by the inverse of its variance."
    )
    statsec: str | None = Field(
        None,
        description="Section used to measure scale and zero, e.g. '[100:400,100:400]'. "
                    "Default is the whole image. Before it, input means in each image, "
                    "output in the output's pixels, and overlap alone the part all share.",
    )

    project: bool = Field(
        False, description="The first image is a cube: combine its planes, along the last "
                           "axis, into an image of one dimension less."
    )
    offsets: str = Field(
        "none", description="Where each image goes in the output: none, wcs or world, "
                            "physical, grid nx dx ny dy, or a file with x y for each image "
                            "(a line '# absolute' makes them absolute)."
    )
    outlimits: str = Field(
        "", description="Keep part of the output: x1 x2 y1 y2, in pixels of the output "
                        "before them, INDEF for an open end."
    )

    lthreshold: float | None = Field(
        None, description="Ignore pixels below this value entirely, before rejection."
    )
    hthreshold: float | None = Field(
        None, description="Ignore pixels above this value entirely, before rejection."
    )
    blank: float = Field(
        0.0, description="Value written where every input pixel was rejected."
    )

    exposure_key: str = Field(
        "EXPTIME",
        description="Header keyword holding the exposure time, for scale='exposure'.",
    )
    ccdtype: str = Field(
        "",
        description="Only combine images of this type: zero, dark, flat, object. "
                    "Reads the instrument translation, so a whole night can be "
                    "given and the right frames picked out.",
    )
    subsets: bool = Field(
        False,
        description="Combine each subset separately, appending its short name to "
                    "the output: Flat becomes FlatB and FlatV. The names come from "
                    "the `subsets` file, which you edit to say B and V rather than "
                    "whatever the filter wheel reported.",
    )
    sigmas: Path | None = Field(
        None,
        description="Optional output image holding the pixel-by-pixel standard deviation.",
    )
    nrejmasks: Path | None = Field(
        None, description="Optional output image counting rejections at each pixel."
    )
    bpmasks: Path | None = Field(
        None, description="Optional output mask: 0 where something was combined, 1 where "
                          "nothing was, 2 where only values the masks call bad were."
    )
    rejmasks: Path | None = Field(
        None, description="Optional output mask with a plane for each image: 1 where it "
                          "was rejected, masked or does not reach."
    )
    expmasks: Path | None = Field(
        None, description="Optional output mask of the exposure time that went into each "
                          "pixel, from --exposure-key; MASKSCAL and MASKZERO scale it."
    )
    masktype: str = Field(
        "dq", description="dq leaves out the pixels each image's DQ flags -- saturated, "
                          "cosmic rays, bad columns -- unless every image flags it; an "
                          "image without a DQ flags nothing. none uses every pixel, as "
                          "IRAF does by default. IRAF's: goodvalue, badvalue, goodbits, "
                          "badbits or novalue, for the mask named by BPM, or !KEYWORD and "
                          "one of them for the mask another keyword names.")
    maskvalue: str = Field(
        "0", description="The value for --masktype; with < or > before it, good pixels "
                         "are those below or above it."
    )
    uncertainty: Literal["VAR", "ERR", "UNCERT"] = uncertainty_field()
    format: Literal["harwood", "iraf"] = format_field()

    @model_validator(mode="after")
    def _check(self) -> ImCombineParams:
        for name, value, allowed in (("scale", self.scale, SCALES), ("zero", self.zero, ZEROS),
                                     ("weight", self.weight, (*SCALES, "ivar"))):
            if value not in allowed and value[:1] not in ("@", "!"):
                raise ValueError(f"{name}='{value}': it is one of {', '.join(allowed)}, "
                                 "@file or !keyword")
        if self.reject in ("ccdclip", "crreject") and self.rdnoise == 0.0 and self.gain == 1.0:
            raise ValueError(
                f"reject='{self.reject}' needs the detector's gain and rdnoise; "
                "without them its threshold is meaningless. Set --gain and --rdnoise, "
                "or use --reject sigclip."
            )
        return self


@register
class ImCombine(Task[ImCombineParams]):
    name = "imcombine"
    package = "immatch"
    description = "Combine images pixel by pixel, with rejection."
    Params = ImCombineParams
    #: Which IRAF's scaling: images.imcombine's, or ccdred's combine (presets).
    flavor = "imcombine"

    presets: dict[str, dict[str, Any]] = {
        # IRAF's own, from ccdred's zerocombine.cl, darkcombine.cl, flatcombine.cl.
        "zero": {"combine": "average", "reject": "minmax", "nlow": 0, "nhigh": 1},
        "dark": {"combine": "average", "reject": "minmax", "nlow": 0, "nhigh": 1,
                 "scale": "exposure"},
        "flat": {"combine": "average", "reject": "avsigclip", "scale": "mode",
                 "blank": 1.0},
        "cosmic": {"combine": "average", "reject": "crreject"},
    }

    def plan(self, params: ImCombineParams) -> list[Step]:
        from ..core import images

        specs = images.expand(params.input)
        steps = [Step("read", f"{len(specs)} images, first is {specs[0]}",
                      why=WHY["read"])]
        if params.project:
            steps.append(Step("project", f"the planes of {specs[0]}", why=WHY["project"]))
        if params.offsets.strip() not in ("", "none") or params.outlimits.strip():
            steps.append(Step("offsets", params.offsets
                              + (f", output {params.outlimits}" if params.outlimits else ""),
                              why=WHY["offsets"]))
        if params.masktype not in ("none", "dq", ""):
            steps.append(Step("mask", f"{params.masktype}, value {params.maskvalue}",
                              why=WHY["mask"]))
        if params.lthreshold is not None or params.hthreshold is not None:
            steps.append(Step("threshold",
                              f"keep {params.lthreshold} .. {params.hthreshold}",
                              why=WHY["threshold"]))
        if params.scale != "none":
            steps.append(Step("scale", f"by {params.scale}"
                              + (f" over {params.statsec}" if params.statsec else ""),
                              why=WHY["scale"]))
        if params.zero != "none":
            steps.append(Step("zero", f"subtract {params.zero}", why=WHY["zero"]))

        if params.reject == "none":
            steps.append(Step("reject", "nothing", why=WHY["reject:none"]))
        elif params.reject == "minmax":
            steps.append(Step("reject", f"minmax, dropping {params.nlow} low "
                                        f"and {params.nhigh} high",
                              why=WHY["reject:minmax"]))
        elif params.reject in ("avsigclip", "pclip"):
            steps.append(Step("reject", f"{params.reject} at -{params.lsigma}/"
                                        f"+{params.hsigma} sigma"
                                        + (f", pclip {params.pclip:g}"
                                           if params.reject == "pclip" else ""),
                              why=WHY[f"reject:{params.reject}"]))
        elif params.reject in ("ccdclip", "crreject"):
            steps.append(Step("reject", f"{params.reject} against the noise model "
                                        f"(gain={params.gain}, rdnoise={params.rdnoise}, "
                                        f"snoise={params.snoise}) at "
                                        f"-{params.lsigma}/+{params.hsigma} sigma",
                              why=WHY["reject:ccdclip"]))
        else:
            steps.append(Step("reject", f"sigclip at -{params.lsigma}/+{params.hsigma} "
                                        f"sigma, centred on the "
                                        f"{'median' if params.mclip else 'mean'}",
                              why=WHY["reject:sigclip"]))
        if params.grow > 0 and params.reject != "none":
            steps.append(Step("grow", f"radius {params.grow:g} px", why=WHY["grow"]))
        if params.weight == "ivar":
            steps.append(Step("weight", "each pixel by the inverse of its variance",
                              why=WHY["weight:ivar"]))
        elif params.weight != "none":
            steps.append(Step("weight", f"by {params.weight}", why=WHY["weight"]))
        steps.append(Step("combine", params.combine,
                          why=WHY[f"combine:{params.combine}"]))
        steps.append(Step("write", str(params.output), why=WHY["write"]))
        for extra, what in ((params.sigmas, "sigma image"),
                            (params.nrejmasks, "rejection counts"),
                            (params.bpmasks, "pixels with no data"),
                            (params.rejmasks, "pixels left out of each image"),
                            (params.expmasks, "exposure in each pixel")):
            if extra:
                steps.append(Step("write", f"{what} -> {extra}"))
        return steps

    def execute(self, params: ImCombineParams) -> TaskResult:
        from ..core import images

        specs = images.expand(params.input)
        groups = _group(specs, params)
        if len(groups) == 1 and not params.subsets:
            return self._combine(list(groups.values())[0], params, params.output)

        outputs, data, history, steps = [], {}, [], []
        for name, members in sorted(groups.items()):
            destination = _with_subset(params.output, name)
            result = self._combine(members, params, destination)
            outputs.extend(result.outputs)
            history.extend(result.history)
            steps.extend(result.steps)
            data[name or "all"] = f"{result.data['images']} images -> {destination.name}"
        return TaskResult(outputs=outputs, data=data, history=history, steps=steps,
                          backend=None)

    def _combine(self, specs, params: ImCombineParams, output) -> TaskResult:
        import numpy as np

        from ..core import mef, provenance

        output = Path(output)
        frames, headers = _frames(specs, params)
        nimages = len(frames)

        # minmax drops a fixed number of values whether or not they deserve it,
        # so with too few frames it drops them all -- and every pixel then gets
        # the blank value, which is a frame full of zeros that looks like a
        # frame. Refuse rather than write it.
        if params.reject == "minmax":
            dropped = params.nlow + params.nhigh
            if dropped >= nimages:
                raise ValueError(
                    f"minmax would drop {params.nlow} low and {params.nhigh} high "
                    f"from {nimages} image(s), leaving nothing. Lower --nlow and "
                    f"--nhigh, give more frames, or use --reject none for a single "
                    f"frame."
                )
            if nimages - dropped < params.nkeep:
                raise ValueError(
                    f"minmax would leave {nimages - dropped} of {nimages} "
                    f"images, fewer than the {params.nkeep} --nkeep requires."
                )

        from ..core import combine as engine

        if self.flavor == "ccdred" and self.name != "combine" and (
                params.grow > 0 or params.weight != "none"):
            from ..core import advice

            advice.fail(f"{self.name} has no grow or weight in IRAF -- ccdred's "
                        "combine is called with grow=0 and weight=no",
                        "use imcombine for those, with this preset's other values")

        from ..core import icsetout

        if params.offsets.split()[:1] == ["physical"]:
            frames, headers = _flipped(specs, frames, headers, params)
        geometry = icsetout.setout([frame.shape for frame in frames], headers, params.offsets,
                                   params.outlimits, params.project)
        out_shape = tuple(geometry.lengths[::-1])
        stack = np.zeros((nimages, *out_shape), dtype=np.float64)
        covered = np.zeros(stack.shape, dtype=bool)
        where = []
        for index, frame in enumerate(frames):
            spot = icsetout.place(frame, geometry.offsets[index], geometry.lengths)
            where.append(spot)
            if spot is not None:
                stack[index][spot[0]] = frame[spot[1]]
                covered[index][spot[0]] = True

        # A mask is in each image's own pixels, for the statistics, and in the output's.
        kind = None if params.masktype in ("dq", "") else icsetout.masktype(params.masktype,
                                                                            params.maskvalue)
        masks = [np.zeros(frame.shape, dtype=bool) for frame in frames]
        fallback = None
        marked = np.zeros(out_shape, dtype=bool)
        if kind is not None and kind.kind != "none":
            if kind.kind == "novalue":
                fallback = np.zeros(stack.shape, dtype=bool)
            raws = _raw_masks(kind, frames, headers, specs, params)
            for index, raw in enumerate(raws):
                bad, gone = icsetout.bad_pixels(kind, raw, frames[index].shape)
                masks[index] = bad
                if where[index] is None or raw is None:
                    continue
                target, source = where[index]
                marked_here = np.zeros(out_shape, dtype=bool)
                marked_here[target] = raw[source] != 0
                marked |= marked_here
                if fallback is not None:
                    fallback[index][target] = (bad & ~gone)[source]
                covered[index][target] &= ~bad[source]

        valid = covered.copy()
        over = np.zeros(stack.shape, dtype=bool)
        if params.lthreshold is not None:
            over |= stack < params.lthreshold
        if params.hthreshold is not None:
            over |= stack > params.hthreshold
        if fallback is not None:
            fallback |= covered & over
            for index, spot in enumerate(where):
                if spot is None:
                    fallback[index] = False
        valid &= ~over
        variances, qualities, own = _planes(specs, frames, where, stack.shape, params.project)
        flagged_everywhere = None
        if params.masktype == "dq":
            good = qualities == 0
            masks = [plane != 0 for plane in own]
            # A pixel every image flags is still combined -- the DQ of the
            # result keeps what they all said -- rather than left empty.
            flagged_everywhere = ~(good & valid).any(axis=0)
            valid &= good | flagged_everywhere[None]

        sections = _sections(params.statsec, frames, geometry, out_shape)
        uses_exposure = "exposure" in (params.scale, params.weight) or bool(params.expmasks)
        if uses_exposure:
            missing = [str(s) for s, h in zip(specs, headers, strict=False)
                       if params.exposure_key not in h]
            if missing and not params.expmasks:
                raise ValueError(
                    f"scale or weight 'exposure' needs {params.exposure_key} in every "
                    f"header; {', '.join(missing[:3])} has none. Name the keyword "
                    "with --exposure-key.")
        ivar = params.weight == "ivar"
        if ivar and (params.combine != "average" or any(v is None for v in variances)):
            advice_needs = ("every image has to carry a variance: ccdproc writes one when it "
                            "knows the gain" if params.combine == "average"
                            else "use --combine average")
            raise ValueError(f"--weight ivar: {advice_needs}.")
        scaling = engine.ic_scale(
            frames, headers, scale=params.scale, zero=params.zero,
            weight="none" if ivar else params.weight,
            exposure_key=params.exposure_key if uses_exposure else "",
            section=sections, lthresh=params.lthreshold, hthresh=params.hthreshold,
            sigscale=params.sigscale, flavor=self.flavor, masks=masks)
        lines = stack.reshape(nimages, -1, out_shape[-1])
        combined, sigma, used, rejected = engine.combine(
            lines, valid.reshape(lines.shape), scaling, combine=params.combine,
            reject=params.reject,
            mclip=params.mclip, nlow=params.nlow, nhigh=params.nhigh,
            nkeep=params.nkeep, lsigma=params.lsigma, hsigma=params.hsigma,
            pclip=params.pclip, grow=params.grow, blank=params.blank,
            rdnoise=_noise_values(params.rdnoise, headers),
            gain=_noise_values(params.gain, headers),
            snoise=_noise_values(params.snoise, headers),
            flavor=self.flavor, marked=marked.reshape(lines.shape[1], -1).any(axis=1),
            fallback=None if fallback is None else fallback.reshape(lines.shape))
        combined, sigma = combined.reshape(out_shape), sigma.reshape(out_shape)
        used, rejected = used.reshape(out_shape), rejected.reshape(stack.shape)
        nrej = (stack.shape[0] - used).astype(np.int32)
        factors, offsets = scaling.scales, scaling.zeros
        keep = valid & ~rejected
        if not np.array_equal(keep.sum(axis=0), np.maximum(used, 0)):
            # nkeep can hand rejected values back; the count is the engine's,
            # and the variance follows the count where the two disagree.
            keep = _restore(keep, valid, used, stack, scaling)
        var, quality, notes = _uncertainty(variances, qualities, keep, scaling, params,
                                           flagged_everywhere, specs)
        if ivar:
            combined, var = _inverse_variance(stack, np.array(variances), keep, scaling,
                                              params.blank)
        along = (slice(None),) + (None,) * len(out_shape)
        working = np.where(covered, stack / factors[along] - offsets[along], np.nan)
        backend_label = engine.backend()

        steps = self.plan(params)
        _measure_steps(steps, np.where(covered, stack, np.nan), working, combined, nrej,
                       factors, offsets, backend_label, params)

        header = headers[0].copy()
        icsetout.update_wcs(header, geometry, params.project, len(out_shape))
        provenance.record_inputs(header, [s.path for s in specs])
        # As ic_scale writes them: how many exposures went in, and -- when the
        # inputs carry one -- the CCDMEAN a flat is later divided by.
        ncombine = [int(h.get("NCOMBINE", 1)) for h in headers]
        header["NCOMBINE"] = (int(sum(ncombine)), "number of images combined")
        means = [(w, float(h["CCDMEAN"]) / s) for w, h, s in
                 zip(scaling.weights, headers, scaling.scales, strict=True)
                 if "CCDMEAN" in h]
        if "CCDMEAN" in header and means:
            # imcombine divides by the weights of the images that had one;
            # ccdred's combine by nothing, its weights already summing to one.
            total = 1.0 if self.flavor == "ccdred" else sum(w for w, _ in means)
            if total > 0:
                header["CCDMEAN"] = sum(w * m for w, m in means) / total
                header.remove("CCDMEANT", ignore_missing=True)
        if self.flavor == "ccdred":
            times = [float(h.get(params.exposure_key, 0.0)) for h in headers]
            darks = [float(h.get("DARKTIME", t)) for h, t in zip(headers, times,
                                                                   strict=True)]
            weights, scales = scaling.weights, scaling.scales
            header[params.exposure_key] = float(sum(weights * np.array(times) / scales))
            header["DARKTIME"] = float(sum(weights * np.array(darks) / scales))
        history = [
            f"imcombine: combine={params.combine} reject={params.reject} "
            f"lsigma={params.lsigma} hsigma={params.hsigma}",
        ]
        if params.scale != "none" or params.zero != "none":
            history.append(f"imcombine: scale={params.scale} zero={params.zero}")
        if backend_label:
            history.append(provenance.backend_note(backend_label))
        provenance.stamp_header(header, history)

        output.parent.mkdir(parents=True, exist_ok=True)
        written, lost = mef.save(output, combined, header, var, quality,
                                 params.uncertainty, params.format)
        notes.extend(lost)
        outputs = list(written)
        for path, array in ((params.sigmas, sigma), (params.nrejmasks, nrej)):
            if path:
                _write_plane(path, array, header)
                outputs.append(path)
        if params.bpmasks:
            bad = np.where(used > 0, 0, np.where(used == 0, 1, 2))
            _write_plane(params.bpmasks, bad, header)
            outputs.append(params.bpmasks)
        if params.rejmasks:
            _write_plane(params.rejmasks, np.where(used == nimages, False, ~keep), header)
            outputs.append(params.rejmasks)
        if params.expmasks:
            exposure, scaled = _exposure(keep, headers, scaling.weights, params.exposure_key)
            mask_header = header.copy()
            if scaled != 1.0:
                mask_header["MASKZERO"] = 0.0
                mask_header["MASKSCAL"] = scaled
            _write_plane(params.expmasks, exposure, mask_header)
            outputs.append(params.expmasks)

        return TaskResult(
            outputs=outputs,
            data={
                "images": nimages,
                "rejected": int(np.asarray(nrej).sum()),
                "mean_level": float(np.nanmean(combined)),
            },
            history=history,
            steps=steps,
            warnings=notes,
            backend=backend_label,
        )


# --------------------------------------------------------------- helpers


def _inverse_variance(stack, variances, keep, scaling, blank):
    """The average of what the rejection left, each pixel by the inverse of its variance,
    and the variance of that, 1 / sum(1 / var)."""
    import numpy as np

    along = (slice(None),) + (None,) * (stack.ndim - 1)
    scaled = stack / scaling.scales[along] - scaling.zeros[along]
    var = variances / scaling.scales[along] ** 2
    weight = np.where(keep & (var > 0), 1.0 / np.where(var > 0, var, 1.0), 0.0)
    total = weight.sum(axis=0)
    none = total <= 0
    with np.errstate(invalid="ignore", divide="ignore"):
        combined = np.where(none, blank, (weight * scaled).sum(axis=0) / total)
        variance = np.where(none, 0.0, 1.0 / total)
    return combined, variance


def _frames(specs, params):
    """The images, each with its own shape, and their headers; or the planes of a cube,
    all with its one header."""
    from ..core import advice, images

    if params.project:
        cube, header = images.read(specs[0])
        if cube.ndim < 2:
            advice.fail("can't project one dimensional images", "--project is for a cube")
        return [plane for plane in cube], [header] * cube.shape[0]
    pairs = [images.read(spec) for spec in specs]
    if not pairs:
        raise ValueError("no input images")
    return [pair[0] for pair in pairs], [pair[1] for pair in pairs]


def _flipped(specs, frames, headers, params):
    """offsets physical reads an image against the first's sense with -* on the axis."""
    from ..core import icsetout, images

    if params.project:
        return frames, headers
    frames, headers = list(frames), list(headers)
    for index, section in enumerate(icsetout.flips(headers, frames[0].ndim)):
        if section:
            frames[index], headers[index] = images.read(f"{specs[index]}{section}")
    return frames, headers


def _raw_masks(kind, frames, headers, specs, params):
    """Each image's mask, by the keyword; for a cube the planes of its one mask."""
    import numpy as np

    from ..core import icsetout

    if params.project:
        shape = (len(frames), *frames[0].shape)
        cube = icsetout.read_mask(kind, headers[0], shape, specs[0].path)
        return [None] * len(frames) if cube is None else list(np.asarray(cube))
    return [icsetout.read_mask(kind, header, frame.shape, spec.path)
            for header, frame, spec in zip(headers, frames, specs, strict=True)]


def _sections(statsec, frames, geometry, out_shape):
    """ic_scale's reading of statsec: in each image's pixels, or the output's."""
    from ..core import combine as engine

    words = (statsec or "").split()
    word = words[0] if words else ""
    place = next((name for name in ("input", "output", "overlap")
                  if word and name.startswith(word)), "")
    text = (words[1] if len(words) > 1 else "") if place else word
    out = []
    if place == "overlap":
        first = [max(int(off[j]) + 1 for off in geometry.offsets)
                 for j in range(len(out_shape))]
        last = [min(int(off[j]) + frame.shape[::-1][j]
                    for off, frame in zip(geometry.offsets, frames, strict=True))
                for j in range(len(out_shape))]
        text = "[" + ",".join(f"{a}:{b}" for a, b in zip(first, last, strict=True)) + "]"
    for index, frame in enumerate(frames):
        if place in ("output", "overlap"):
            moved = tuple(int(value) for value in geometry.offsets[index][::-1])
            out.append(engine.stat_section(text, frame.shape, moved, out_shape))
        else:
            out.append(engine.stat_section(text, frame.shape))
    return out


def _noise_values(value, headers):
    """A number for all the images, or the keyword each has its own in."""
    import numpy as np

    from ..core import advice

    try:
        return float(value)
    except ValueError:
        pass
    found = []
    for header in headers:
        if value not in header:
            advice.fail(f"an image has no keyword {value}",
                        "give the number, or a keyword every image has")
        found.append(float(header[value]))
    return np.array(found)


def _write_plane(path, array, header) -> None:
    """A mask by the name asked for: IRAF's .pl, or else a FITS image."""
    import numpy as np
    from astropy.io import fits

    from ..core import plio

    path = Path(path)
    if path.suffix == ".pl":
        plio.write(path, np.asarray(array, dtype=np.int32))
        return
    fits.PrimaryHDU(data=np.asarray(array, dtype=np.float32),
                    header=header).writeto(path, overwrite=True)


def _exposure(keep, headers, weights, key):
    """ic_einit and ic_emask: the exposure of what went into each pixel, as whole units
    of a scale that fits; (values, the scale). Missing keyword: an exposure of 1."""
    import numpy as np

    from ..core.icsetout import nint

    real = np.float32
    times = np.array([max(real(0.0), real(float(h.get(key, 1.0)))) for h in headers], real)
    emax = real(0.0)
    emin = real(3.4028235e38)
    efrac = real(0.0)
    for exp in times:
        emax = real(emax + exp)
        if exp > 0:
            emin = min(exp, emin)
        efrac = max(real(abs(exp - nint(float(exp)))), efrac)
    escale = real(1.0)
    if emin < 1.0:
        escale = emin
        emin = real(emin / escale)
        emax = real(emax / escale)
    elif emin == real(3.4028235e38):
        emin = real(0.0)
    if efrac > 0.001 and emax - emin < 1000.0:
        escale = real(escale / 1000.0)
        emin = real(emin * 1000.0)
        emax = real(emax * 1000.0)
    while emax > 2 ** 27 - 1:
        escale = real(escale * 10.0)
        emin = real(emin / 10.0)
        emax = real(emax / 10.0)
    use = times * (weights > 0).astype(real)
    total = np.zeros(keep.shape[1:], real)
    for index in range(keep.shape[0]):
        total = (total + np.where(keep[index], use[index], real(0.0))).astype(real)
    ratio = (total / escale).astype(np.float64)
    return np.where(ratio >= 0, np.floor(ratio + 0.5), -np.floor(-ratio + 0.5)), float(escale)


def _planes(specs, frames, where, shape, project=False):
    """Each input's variance (None where it has none) and DQ, in the output's pixels, and
    the DQ in its own."""
    import numpy as np

    from ..core import dq, mef

    qualities = np.zeros(shape, dtype=dq.DTYPE)
    own = [np.zeros(frame.shape, dtype=dq.DTYPE) for frame in frames]
    if project:
        return [None] * len(frames), qualities, own
    variances = []
    for index, spec in enumerate(specs):
        _, _, var, quality = mef.load(spec)
        spot = where[index]
        if var is None:
            variances.append(None)
        else:
            plane = np.zeros(shape[1:])
            if spot is not None:
                plane[spot[0]] = var[spot[1]]
            variances.append(plane)
        if quality.shape == frames[index].shape:
            own[index] = quality
            if spot is not None:
                qualities[index][spot[0]] = quality[spot[1]]
    return variances, qualities, own


def _restore(keep, valid, used, stack, scaling):
    """Where the engine kept more values than survived rejection (nkeep), the
    ones it handed back are the valid values nearest the pixel's median."""
    import numpy as np

    short = keep.sum(axis=0) < used
    if not short.any():
        return keep
    along = (slice(None),) + (None,) * (stack.ndim - 1)
    scaled = stack / scaling.scales[along] - scaling.zeros[along]
    centre = np.nanmedian(np.where(valid, scaled, np.nan), axis=0)
    distance = np.where(valid, np.abs(scaled - centre), np.inf)
    ranks = np.argsort(np.argsort(distance, axis=0), axis=0)
    return np.where(short[None], valid & (ranks < used[None]), keep)


def _uncertainty(variances, qualities, keep, scaling, params, flagged_everywhere, specs):
    """The combined image's variance and DQ, and what to say about them."""
    import numpy as np

    from ..core import advice, dq
    from ..core import combine as engine

    notes = []
    var = None
    missing = [Path(str(specs[min(i, len(specs) - 1)])).name
               for i, v in enumerate(variances) if v is None]
    if not missing and params.combine not in ("quadrature", "nmodel"):
        var = engine.propagate(np.array(variances), keep, scaling, params.combine)
    elif missing and len(missing) < len(variances):
        notes.append(advice.warn(
            f"{len(missing)} of the {len(variances)} images carry no variance "
            f"({', '.join(missing[:3])}{'...' if len(missing) > 3 else ''}), so the "
            "combined one has none either.",
            "give them one: ccdproc writes it when it knows the gain."))
    quality = np.zeros(keep.shape[1:], dtype=dq.DTYPE)
    if flagged_everywhere is not None and flagged_everywhere.any():
        common = np.bitwise_and.reduce(np.where(keep, qualities, 0xFFFF).astype(dq.DTYPE),
                                       axis=0)
        quality = np.where(flagged_everywhere & keep.any(axis=0), common, 0).astype(dq.DTYPE)
    quality[keep.sum(axis=0) == 0] |= dq.NO_DATA
    return var, quality, notes


def _level(frame, how: str):
    import numpy as np

    if how == "median":
        return float(np.median(frame))
    if how == "mean":
        return float(np.mean(frame))
    if how == "mode":
        # 2.5*median - 1.5*mean, the estimator IRAF uses for sky-like data.
        return float(2.5 * np.median(frame) - 1.5 * np.mean(frame))
    raise ValueError(f"cannot measure level by '{how}'")


def _scale_factors(stack, headers, params, stat):
    import numpy as np

    if params.scale == "none":
        return np.ones(stack.shape[0])
    if params.scale == "exposure":
        times = np.array([float(h.get(params.exposure_key, 1.0)) for h in headers])
        if np.any(times <= 0):
            raise ValueError(f"scale='exposure' needs a positive {params.exposure_key} "
                             "in every input header")
        return times / times[0]
    levels = np.array([_level(frame[stat], params.scale) for frame in stack])
    if np.any(levels == 0):
        raise ValueError(f"scale='{params.scale}' measured a level of zero in one image")
    return levels / levels[0]


def _zero_offsets(stack, params, stat):
    import numpy as np

    if params.zero == "none":
        return np.zeros(stack.shape[0])
    if params.zero == "exposure":
        raise ValueError("zero='exposure' is not meaningful; use scale='exposure'")
    levels = np.array([_level(frame[stat], params.zero) for frame in stack])
    return levels - levels[0]


def _minmax(stack, nlow, nhigh, mask):
    import numpy as np

    order = np.argsort(np.where(mask, np.inf, stack), axis=0)
    keep = ~mask
    n = stack.shape[0]
    for index in range(nlow):
        if index < n:
            np.put_along_axis(keep, order[index:index + 1], False, axis=0)
    for index in range(nhigh):
        position = n - 1 - index
        if position >= 0:
            np.put_along_axis(keep, order[position:position + 1], False, axis=0)
    count = keep.sum(axis=0)
    mean = np.divide(np.where(keep, stack, 0.0).sum(axis=0), count,
                     out=np.zeros(stack.shape[1:]), where=count > 0)
    return mean, (n - count).astype(np.int32), keep


def _finish(stack, mean, nrej, keep_mask, params, mask):
    """Apply `combine`, compute the sigma image, and fill empty pixels."""
    import numpy as np

    n = stack.shape[0]
    if keep_mask is None:
        # The clipping kernels return the mean; recover which values survived
        # so that median and sum see the same set of pixels.
        deviation = np.abs(stack - mean)
        order = np.argsort(np.where(mask, np.inf, deviation), axis=0)
        keep_mask = ~mask
        ranks = np.argsort(order, axis=0)
        keep_mask &= ranks < (n - nrej)

    if params.combine == "median":
        values = np.where(keep_mask, stack, np.nan)
        with np.errstate(invalid="ignore"):
            combined = np.nanmedian(values, axis=0)
    elif params.combine == "sum":
        combined = np.where(keep_mask, stack, 0.0).sum(axis=0)
    else:
        combined = mean

    count = keep_mask.sum(axis=0)
    with np.errstate(invalid="ignore"):
        sigma = np.sqrt(
            np.divide(np.where(keep_mask, (stack - mean) ** 2, 0.0).sum(axis=0),
                      np.maximum(count - 1, 1),
                      out=np.zeros(stack.shape[1:]), where=count > 1)
        )
    combined = np.where(count >= params.nkeep, combined, params.blank)
    return np.nan_to_num(combined, nan=params.blank), sigma


def _make_preset(name: str, doc: str, defaults: dict[str, Any]):
    """Register one of IRAF's *combine names as a preset of the engine.

    Section 3.2, "packages as presets": the user types the name they know and
    there is still one implementation. The preset's values are a separate
    origin, so `harwood lpar zerocombine` shows `reject minmax (preset:
    zerocombine)` rather than passing it off as a default you chose.
    """
    return register(type(
        name.title().replace("_", ""),
        (ImCombine,),
        {"name": name, "package": "ccdred", "description": doc, "iraf": name,
         "Params": ImCombineParams, "presets": {}, "defaults": defaults,
         "flavor": "ccdred"},
    ))


ZeroCombine = _make_preset(
    "zerocombine",
    "Combine bias frames. Preset of imcombine.",
    {"combine": "average", "reject": "minmax", "nlow": 0, "nhigh": 1,
     "ccdtype": "zero"},
)
DarkCombine = _make_preset(
    "darkcombine",
    "Combine dark frames, scaled by exposure time. Preset of imcombine.",
    {"combine": "average", "reject": "minmax", "nlow": 0, "nhigh": 1,
     "scale": "exposure", "ccdtype": "dark"},
)
FlatCombine = _make_preset(
    "flatcombine",
    "Combine flat fields, one per filter, scaled to a common level. "
    "Preset of imcombine.",
    {"combine": "average", "reject": "avsigclip", "scale": "mode", "blank": 1.0,
     "ccdtype": "flat", "subsets": True},
)


CcdCombine = _make_preset(
    "combine",
    "ccdred's combine: images of a type, by subset if asked, with ccdred's scaling. "
    "Preset of imcombine.",
    {"combine": "average", "reject": "none", "ccdtype": "", "subsets": False},
)


def _measure_steps(steps, stack, working, combined, nrej, factors, offsets,
                   backend, params) -> None:
    """Fill in what each planned step actually came to.

    The plan is built before anything runs, so the numbers have to be put back
    afterwards. Matching by action name rather than by position, because the
    plan's shape depends on which corrections were asked for.
    """
    import warnings

    import numpy as np

    from ..core.task import Task

    found = {step.action: step for step in steps}

    if "read" in found:
        levels = [float(np.nanmedian(frame)) for frame in stack]
        found["read"].measured = {
            "frames": stack.shape[0],
            "shape": "x".join(str(n) for n in reversed(stack.shape[1:])),
            "levels": f"{min(levels):.4g} to {max(levels):.4g}",
        }
    if "scale" in found:
        found["scale"].measured = {
            "factors": ", ".join(f"{value:.4f}" for value in factors[:6])
                       + (" ..." if factors.size > 6 else ""),
            "spread": f"{100 * (factors.max() / factors.min() - 1):.2f}%",
        }
    if "zero" in found:
        found["zero"].measured = {
            "offsets": ", ".join(f"{value:+.3f}" for value in offsets[:6])
                       + (" ..." if offsets.size > 6 else ""),
        }
    if "reject" in found and params.reject != "none":
        total = int(np.asarray(nrej).sum())
        pixels = int(np.prod(stack.shape))
        touched = int((np.asarray(nrej) > 0).sum())
        found["reject"].measured = {
            "values dropped": f"{total} of {pixels}, {100 * total / pixels:.2f}%",
            "pixels affected": f"{touched} of {int(np.prod(stack.shape[1:]))}",
            "worst pixel": f"{int(np.asarray(nrej).max())} of {stack.shape[0]} frames",
        }
        if backend:
            found["reject"].measured["backend"] = backend

    if "combine" in found:
        # The honest comparison is against the frame-to-frame scatter at each
        # pixel -- the actual noise -- not against the spread within a frame,
        # which is mostly structure and varies between frames. Reporting the
        # latter makes a nine-frame average look better than 1/sqrt(N), which
        # is impossible and would have gone unnoticed.
        count = working.shape[0]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            spread = np.nanstd(working, axis=0, ddof=1)
        noise = float(np.nanmedian(spread)) if count > 1 and np.isfinite(spread).any() else 0.0
        expected = noise / np.sqrt(count) if noise else 0.0
        actual = float(np.std(combined))
        structure = float(np.sqrt(max(actual ** 2 - expected ** 2, 0.0)))

        found["combine"].measured = {
            "result": f"median {float(np.median(combined)):.4g}",
            "noise per frame": f"{noise:.4g}",
            f"expected over {count}": f"{expected:.4g}  (noise / sqrt {count})",
            "result spread": f"{actual:.4g}",
            "structure left": f"{structure:.4g}"
                              + (" -- the pattern averaging cannot remove, which "
                                 "is what this frame is for"
                                 if structure > 0.3 * actual else ""),
        }
        found["combine"].before = Task.statistics(working[0][np.isfinite(working[0])])
        found["combine"].after = Task.statistics(combined)


def _group(specs, params) -> dict:
    """Split the inputs by type and, when asked, by subset.

    IRAF's `flatcombine m92*.fits Flat` on a whole night produces FlatB and
    FlatV: it picks the flats out by their image type and combines each filter
    on its own, because combining two filters together produces a flat for
    neither.
    """
    from ..core import instrument as instruments
    from ..core import subsets as subset_file

    wanted = params.ccdtype.strip().lower()
    if not wanted and not params.subsets:
        return {"": list(specs)}

    inst = instruments.from_project()
    groups: dict[str, list] = {}
    seen = []
    for spec in specs:
        from .ccdred import _peek

        header = _peek(spec)
        if wanted and inst.image_type(header) != wanted:
            continue
        subset = inst.subset(header)
        seen.append(subset)
        key = subset if params.subsets else ""
        groups.setdefault(key, []).append(spec)

    if not groups:
        raise RuntimeError(
            f"none of the {len(specs)} images is of type '{wanted}'. "
            "harwood ccdlist shows what the instrument translation makes of each."
        )
    if params.subsets:
        known = subset_file.record(sorted(set(seen)))
        groups = {known.get(key, key): members for key, members in groups.items()}
    return groups


def _with_subset(output, subset: str) -> Path:
    """`Flat` and `V` give `FlatV.fits`, keeping whatever extension was asked for."""
    path = Path(output)
    if not subset:
        return path
    suffix = path.suffix or ".fits"
    stem = path.name[: -len(path.suffix)] if path.suffix else path.name
    return path.parent / f"{stem}{subset}{suffix}"
