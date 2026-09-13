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
from ..core.task import Step, Task, TaskParams, TaskResult

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
    "write": "The combined frame carries IMCMB cards naming every input, so what "
             "went into it survives being copied somewhere else.",
}

Combine = Literal["average", "median", "sum"]
Reject = Literal["none", "minmax", "sigclip", "ccdclip", "crreject"]
Scale = Literal["none", "mode", "median", "mean", "exposure"]


class ImCombineParams(TaskParams):
    """Parameters of imcombine. Names follow IRAF; see `harwood help imcombine`."""

    input: list[str] = Field(
        ...,
        description="Images to combine. Accepts a shell glob or an @list file.",
    )
    output: Path = Field(..., description="Combined image to write.")

    combine: Combine = Field(
        "average",
        description="How surviving pixels are combined once rejection is done.",
    )
    reject: Reject = Field(
        "none",
        description=(
            "Rejection algorithm. 'sigclip' judges each pixel against the spread "
            "of the stack; 'ccdclip' and 'crreject' judge it against the CCD noise "
            "model (gain, rdnoise, snoise), which is what you want when few images "
            "go in. 'minmax' simply drops the nlow lowest and nhigh highest."
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
        1, ge=1, description="Minimum number of pixels to keep at each position."
    )

    gain: float = Field(
        1.0, gt=0, description="Detector gain in electrons per ADU. Used by ccdclip."
    )
    rdnoise: float = Field(0.0, ge=0, description="Read noise in electrons. Used by ccdclip.")
    snoise: float = Field(
        0.0, ge=0,
        description="Sensitivity noise as a fraction, for flat-field error. Used by ccdclip.",
    )

    scale: Scale = Field(
        "none",
        description="Scale each image before combining, by its level or its exposure time.",
    )
    zero: Scale = Field(
        "none", description="Subtract an additive offset from each image before combining."
    )
    statsec: str | None = Field(
        None,
        description="Section used to measure scale and zero, e.g. '[100:400,100:400]'. "
                    "Default is the whole image.",
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
    sigmas: Path | None = Field(
        None,
        description="Optional output image holding the pixel-by-pixel standard deviation.",
    )
    nrejmasks: Path | None = Field(
        None, description="Optional output image counting rejections at each pixel."
    )

    @model_validator(mode="after")
    def _check(self) -> ImCombineParams:
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

    presets: dict[str, dict[str, Any]] = {
        "zero": {"combine": "average", "reject": "minmax"},
        "dark": {"combine": "average", "reject": "minmax", "scale": "exposure"},
        "flat": {"combine": "median", "reject": "ccdclip", "scale": "mode"},
        "cosmic": {"combine": "average", "reject": "crreject"},
    }

    def plan(self, params: ImCombineParams) -> list[Step]:
        from ..core import images

        specs = images.expand(params.input)
        steps = [Step("read", f"{len(specs)} images, first is {specs[0]}",
                      why=WHY["read"])]
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
        steps.append(Step("combine", params.combine,
                          why=WHY[f"combine:{params.combine}"]))
        steps.append(Step("write", str(params.output), why=WHY["write"]))
        for extra, what in ((params.sigmas, "sigma image"),
                            (params.nrejmasks, "rejection counts")):
            if extra:
                steps.append(Step("write", f"{what} -> {extra}"))
        return steps

    def execute(self, params: ImCombineParams) -> TaskResult:
        import numpy as np
        from astropy.io import fits

        from .. import kernels
        from ..core import images, provenance

        specs = images.expand(params.input)
        stack, headers = images.read_stack(specs)

        mask = np.zeros(stack.shape, dtype=bool)
        if params.lthreshold is not None:
            mask |= stack < params.lthreshold
        if params.hthreshold is not None:
            mask |= stack > params.hthreshold

        stat = _stat_slice(params.statsec, stack.shape[1:])
        factors = _scale_factors(stack, headers, params, stat)
        offsets = _zero_offsets(stack, params, stat)
        working = (stack - offsets[:, None, None]) / factors[:, None, None]

        backend_label = None
        if params.reject in ("sigclip",):
            kernel = kernels.get("sigclip_mean")
            backend_label = kernel.label
            mean, nrej = kernel(working, params.lsigma, params.hsigma, mclip=params.mclip,
                                mask=mask)
            keep_mask = None
        elif params.reject in ("ccdclip", "crreject"):
            kernel = kernels.get("ccdclip_mean")
            backend_label = kernel.label
            mean, nrej = kernel(working, gain=params.gain, rdnoise=params.rdnoise,
                                snoise=params.snoise, lsigma=params.lsigma,
                                hsigma=params.hsigma, mclip=params.mclip, mask=mask)
            keep_mask = None
        elif params.reject == "minmax":
            mean, nrej, keep_mask = _minmax(working, params.nlow, params.nhigh, mask)
        else:
            keep = ~mask
            count = keep.sum(axis=0)
            mean = np.divide(np.where(keep, working, 0.0).sum(axis=0), count,
                             out=np.zeros(working.shape[1:]), where=count > 0)
            nrej = (working.shape[0] - count).astype(np.int32)
            keep_mask = keep

        combined, sigma = _finish(working, mean, nrej, keep_mask, params, mask)

        steps = self.plan(params)
        _measure_steps(steps, stack, working, combined, nrej, factors, offsets,
                       backend_label, params)

        header = headers[0].copy()
        provenance.record_inputs(header, [s.path for s in specs])
        history = [
            f"imcombine: combine={params.combine} reject={params.reject} "
            f"lsigma={params.lsigma} hsigma={params.hsigma}",
        ]
        if params.scale != "none" or params.zero != "none":
            history.append(f"imcombine: scale={params.scale} zero={params.zero}")
        if backend_label:
            history.append(provenance.backend_note(backend_label))
        provenance.stamp_header(header, history)

        params.output.parent.mkdir(parents=True, exist_ok=True)
        fits.PrimaryHDU(data=combined.astype(np.float32), header=header).writeto(
            params.output, overwrite=True
        )
        outputs = [params.output]
        for path, array in ((params.sigmas, sigma), (params.nrejmasks, nrej)):
            if path:
                fits.PrimaryHDU(data=np.asarray(array, dtype=np.float32),
                                header=header).writeto(path, overwrite=True)
                outputs.append(path)

        return TaskResult(
            outputs=outputs,
            data={
                "images": len(specs),
                "rejected": int(np.asarray(nrej).sum()),
                "mean_level": float(np.nanmean(combined)),
            },
            history=history,
            steps=steps,
            backend=backend_label,
        )


# --------------------------------------------------------------- helpers


def _stat_slice(statsec, shape):
    from ..core import pixels

    if not statsec:
        return tuple(slice(None) for _ in shape)
    return pixels.parse_section(statsec, shape)


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
         "Params": ImCombineParams, "presets": {}, "defaults": defaults},
    ))


ZeroCombine = _make_preset(
    "zerocombine",
    "Combine bias frames. Preset of imcombine.",
    {"combine": "average", "reject": "minmax"},
)
DarkCombine = _make_preset(
    "darkcombine",
    "Combine dark frames, scaled by exposure time. Preset of imcombine.",
    {"combine": "average", "reject": "minmax", "scale": "exposure"},
)
FlatCombine = _make_preset(
    "flatcombine",
    "Combine flat fields, scaled to a common level. Preset of imcombine.",
    {"combine": "median", "reject": "ccdclip", "scale": "mode"},
)


def _measure_steps(steps, stack, working, combined, nrej, factors, offsets,
                   backend, params) -> None:
    """Fill in what each planned step actually came to.

    The plan is built before anything runs, so the numbers have to be put back
    afterwards. Matching by action name rather than by position, because the
    plan's shape depends on which corrections were asked for.
    """
    import numpy as np

    from ..core.task import Task

    found = {step.action: step for step in steps}

    if "read" in found:
        found["read"].measured = {
            "frames": stack.shape[0],
            "shape": "x".join(str(n) for n in reversed(stack.shape[1:])),
            "levels": f"{float(np.median(stack, axis=(1, 2)).min()):.4g} to "
                      f"{float(np.median(stack, axis=(1, 2)).max()):.4g}",
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
        noise = float(np.median(np.std(working, axis=0, ddof=1))) if count > 1 else 0.0
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
        found["combine"].before = Task.statistics(working[0])
        found["combine"].after = Task.statistics(combined)
