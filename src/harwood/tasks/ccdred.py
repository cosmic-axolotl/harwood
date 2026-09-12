"""ccdred -- CCD reduction.

`ccdproc` is the spine of any imaging reduction and the place where IRAF's
design still shows its quality. Two behaviours matter as much as the arithmetic
and are usually the first thing a hand-rolled script loses:

* **the pipeline is ordered and named** -- fixpix, overscan, trim, zero, dark,
  flat -- and you can print it before running it (`--plan`, IRAF's `noproc`);
* **the header remembers** -- once a frame is bias-subtracted it carries
  `ZEROCOR`, and running ccdproc again skips that step instead of subtracting
  the bias twice. Doing that to a night of data is silent and unrecoverable.

The algorithms come from `ccdproc` the Python package; what is written here is
the ordering, the header bookkeeping and the provenance.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import Field

from ..core.registry import register
from ..core.task import Step, Task, TaskParams, TaskResult

#: Keyword written once a step has been applied, so it is never applied twice.
MARKS = {
    "overscan": ("OVERSCAN", "overscan corrected by harwood"),
    "trim": ("TRIM", "trimmed by harwood"),
    "zerocor": ("ZEROCOR", "bias subtracted by harwood"),
    "darkcor": ("DARKCOR", "dark subtracted by harwood"),
    "flatcor": ("FLATCOR", "flat fielded by harwood"),
    "fixpix": ("FIXPIX", "bad pixels replaced by harwood"),
}


class CcdProcParams(TaskParams):
    input: list[str] = Field(..., description="Images to process. Accepts globs and @lists.")
    output: Path = Field(
        ...,
        description="Output image, or output directory when several images are processed.",
    )

    fixpix: bool = Field(False, description="Replace bad pixels listed in the mask.")
    overscan: bool = Field(False, description="Fit and subtract the overscan level.")
    trim: bool = Field(False, description="Trim the image to its useful section.")
    zerocor: bool = Field(False, description="Subtract the master bias.")
    darkcor: bool = Field(False, description="Subtract the master dark, scaled by exposure time.")
    flatcor: bool = Field(False, description="Divide by the master flat.")

    biassec: str | None = Field(
        None,
        description="Overscan section, 1-based and (x,y), e.g. '[2049:2080,*]'. "
                    "Read from the header keyword BIASSEC when not given.",
    )
    trimsec: str | None = Field(
        None, description="Useful section to keep, e.g. '[1:2048,1:2048]'. "
                          "Read from TRIMSEC when not given."
    )
    zero: Path | None = Field(None, description="Master bias image.")
    dark: Path | None = Field(None, description="Master dark image.")
    flat: Path | None = Field(None, description="Master flat image.")
    bpm: Path | None = Field(None, description="Bad pixel mask, non-zero where a pixel is bad.")

    function: Literal["legendre", "chebyshev", "spline3", "mean"] = Field(
        "legendre", description="Function fitted to the overscan along the readout axis."
    )
    order: int = Field(1, ge=1, description="Order of the overscan fit.")
    readaxis: Literal["line", "column"] = Field(
        "column", description="Axis the overscan is averaged along."
    )

    exposure_key: str = Field("EXPTIME", description="Header keyword holding the exposure time.")
    minreplace: float = Field(
        1.0, description="Flat values below this are replaced, to avoid dividing by noise."
    )


@register
class CcdProc(Task[CcdProcParams]):
    name = "ccdproc"
    package = "ccdred"
    description = "Process CCD images: overscan, trim, bias, dark, flat."
    Params = CcdProcParams

    def plan(self, params: CcdProcParams) -> list[Step]:
        """The steps, against the *actual* headers of the actual inputs.

        A plan that shows a step the run will skip is worse than no plan: it
        is the header bookkeeping made invisible again. So this reads the first
        input's header and reports what it finds.
        """
        from ..core import images

        specs = images.expand(params.input)
        header = _peek(specs[0])
        steps = [Step("read", f"{len(specs)} image(s), first is {specs[0]}")]

        def add(flag: str, action: str, detail: str) -> None:
            if _already(header, flag):
                keyword = MARKS[flag][0]
                steps.append(Step(action, skipped=True,
                                  reason=f"{keyword} is already in the header "
                                         f"({header[keyword]})"))
            else:
                steps.append(Step(action, detail))

        if params.fixpix:
            add("fixpix", "fixpix", str(params.bpm) if params.bpm
                else "no mask given -- nothing to fix")
        if params.overscan:
            add("overscan", "overscan",
                f"{params.function} order {params.order} over "
                f"{params.biassec or header.get('BIASSEC') or 'BIASSEC from the header'}")
        if params.trim:
            add("trim", "trim",
                params.trimsec or header.get("TRIMSEC") or "TRIMSEC from the header")
        for flag, key, image in (("zerocor", "zero", params.zero),
                                 ("darkcor", "dark", params.dark),
                                 ("flatcor", "flat", params.flat)):
            if getattr(params, flag):
                add(flag, key, str(image) if image
                    else f"no --{key} image given -- this would fail")
        steps.append(Step("write", str(params.output)))
        return steps

    def execute(self, params: CcdProcParams) -> TaskResult:
        import numpy as np
        from astropy.io import fits

        from ..core import images, pixels, provenance

        specs = images.expand(params.input)
        many = len(specs) > 1
        if many:
            params.output.mkdir(parents=True, exist_ok=True)

        outputs, all_steps, history = [], [], []
        for spec in specs:
            data, header = images.read(spec)
            steps: list[Step] = []

            if params.fixpix and params.bpm is not None:
                data, step = _fixpix(data, params.bpm, header)
                steps.append(step)
            if params.overscan:
                data, step = _overscan(data, header, params)
                steps.append(step)
            if params.trim:
                data, step = _trim(data, header, params)
                steps.append(step)
            for flag, key, image in (("zerocor", "zero", params.zero),
                                     ("darkcor", "dark", params.dark),
                                     ("flatcor", "flat", params.flat)):
                if getattr(params, flag):
                    data, step = _calibrate(data, header, key, image, params)
                    steps.append(step)

            destination = params.output / Path(str(spec.path)).name if many else params.output
            lines = [f"ccdproc: {step.render()}" for step in steps if not step.skipped]
            provenance.stamp_header(header, lines)
            destination.parent.mkdir(parents=True, exist_ok=True)
            fits.PrimaryHDU(data=data.astype(np.float32), header=header).writeto(
                destination, overwrite=True
            )
            outputs.append(destination)
            all_steps.extend(steps)
            history.extend(lines)

        skipped = sum(1 for step in all_steps if step.skipped)
        return TaskResult(outputs=outputs, history=history, steps=all_steps,
                          data={"images": len(specs), "steps_skipped": skipped})


def _peek(spec):
    """Read one header, without the pixels."""
    from ..core import images

    hdul, hdu = images.open_hdu(spec)
    try:
        return hdu.header.copy()
    finally:
        hdul.close()


def _already(header, flag: str) -> bool:
    keyword = MARKS[flag][0]
    return keyword in header


def _mark(header, flag: str) -> None:
    keyword, comment = MARKS[flag]
    from datetime import datetime, timezone

    header[keyword] = (datetime.now(timezone.utc).isoformat(timespec="seconds"), comment)


def _fixpix(data, bpm_path, header):
    import numpy as np

    from ..core import images

    if _already(header, "fixpix"):
        return data, Step("fixpix", skipped=True, reason="FIXPIX already in the header")
    mask, _ = images.read(bpm_path)
    bad = mask != 0
    if not bad.any():
        return data, Step("fixpix", "mask marks no bad pixels")
    fixed = data.copy()
    fixed[bad] = np.nan
    # Linear interpolation along x, the way IRAF's fixpix does by default.
    for row in np.where(bad.any(axis=1))[0]:
        line = fixed[row]
        holes = np.isnan(line)
        line[holes] = np.interp(np.flatnonzero(holes), np.flatnonzero(~holes), line[~holes])
    _mark(header, "fixpix")
    return fixed, Step("fixpix", f"{int(bad.sum())} pixels replaced")


def _overscan(data, header, params):
    import numpy as np

    from ..core import pixels

    if _already(header, "overscan"):
        return data, Step("overscan", skipped=True, reason="OVERSCAN already in the header")
    section = params.biassec or header.get("BIASSEC")
    if not section:
        raise ValueError(
            "overscan was asked for but no section was given and the header has no BIASSEC. "
            "Pass --biassec '[2049:2080,*]'."
        )
    strip = data[pixels.parse_section(section, data.shape)]
    axis = 1 if params.readaxis == "column" else 0
    profile = strip.mean(axis=axis)

    if params.function == "mean":
        fit = np.full(profile.shape, profile.mean())
    else:
        x = np.arange(profile.size, dtype=float)
        series = {"legendre": np.polynomial.Legendre,
                  "chebyshev": np.polynomial.Chebyshev,
                  "spline3": np.polynomial.Polynomial}[params.function]
        fit = series.fit(x, profile, deg=params.order)(x)

    corrected = data - (fit[:, None] if axis == 1 else fit[None, :])
    _mark(header, "overscan")
    header["BIASSEC"] = section
    return corrected, Step("overscan",
                           f"{params.function} order {params.order} over {section}, "
                           f"mean level {float(fit.mean()):.2f}")


def _trim(data, header, params):
    from ..core import pixels

    if _already(header, "trim"):
        return data, Step("trim", skipped=True, reason="TRIM already in the header")
    section = params.trimsec or header.get("TRIMSEC")
    if not section:
        raise ValueError(
            "trim was asked for but no section was given and the header has no TRIMSEC. "
            "Pass --trimsec '[1:2048,1:2048]'."
        )
    trimmed = data[pixels.parse_section(section, data.shape)]
    _mark(header, "trim")
    header["TRIMSEC"] = section
    return trimmed, Step("trim", f"{section} -> "
                                 f"{trimmed.shape[1]}x{trimmed.shape[0]}")


def _calibrate(data, header, key, image, params):
    import numpy as np

    from ..core import images

    flag = {"zero": "zerocor", "dark": "darkcor", "flat": "flatcor"}[key]
    if _already(header, flag):
        return data, Step(key, skipped=True,
                          reason=f"{MARKS[flag][0]} already in the header")
    if image is None:
        raise ValueError(f"{flag} was asked for but no --{key} image was given")

    calibration, cal_header = images.read(image)
    if calibration.shape != data.shape:
        raise ValueError(
            f"{key} image {image} is {calibration.shape} but the frame is {data.shape}. "
            "Was one of them trimmed and the other not?"
        )

    if key == "zero":
        out, detail = data - calibration, f"subtracted {Path(image).name}"
    elif key == "dark":
        frame_time = float(header.get(params.exposure_key, 1.0))
        dark_time = float(cal_header.get(params.exposure_key, 1.0))
        factor = frame_time / dark_time if dark_time else 1.0
        out = data - calibration * factor
        detail = f"subtracted {Path(image).name} scaled by {factor:.3f}"
    else:
        safe = np.where(calibration < params.minreplace, params.minreplace, calibration)
        normalised = safe / np.median(safe)
        out = data / normalised
        detail = f"divided by {Path(image).name}, normalised to its median"

    _mark(header, flag)
    return out, Step(key, detail)
