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

from dataclasses import dataclass
from datetime import UTC
from pathlib import Path
from typing import Literal

from pydantic import Field

from ..core import advice
from ..core.registry import register
from ..core.task import Step, Task, TaskParams, TaskResult, Template

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
    output: Template = Field(
        "",
        description="Where to write. One image: a file name. Several: a directory, "
                    "or a substitution template such as '%m%tr%'. Left out, the run "
                    "writes to its own directory under output/ and the frames keep "
                    "their names.",
    )
    inplace: bool = Field(
        False,
        description="Correct the input images themselves, as IRAF does. The header "
                    "marks make it safe to repeat -- a step already applied is never "
                    "applied twice -- but the originals are gone.",
    )
    ccdtype: str = Field(
        "",
        description="Only correct images of this type: object, zero, dark, flat, "
                    "comp. Calibration frames still get processed when a correction "
                    "needs them. Empty means every image given.",
    )
    instrument: str | None = Field(
        None,
        description="Header translation to use, such as 'ccddb$kpno/direct.dat'. "
                    "Taken from the project file when not given.",
    )

    fixpix: bool = Field(False, description="Replace bad pixels listed in the mask.")
    overscan: bool = Field(False, description="Fit and subtract the overscan level.")
    trim: bool = Field(False, description="Trim the image to its useful section.")
    zerocor: bool = Field(False, description="Subtract the master bias.")
    darkcor: bool = Field(
        False, description="Subtract the master dark, scaled by exposure time."
    )
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
    bpm: Path | None = Field(
        None, description="Bad pixel mask, non-zero where a pixel is bad."
    )

    # The overscan fit is an icfit, with icfit's parameter names -- the same
    # ones identify, aptrace, continuum and sensfunc use. Learn them once.
    function: Literal["legendre", "chebyshev", "spline1", "spline3"] = Field(
        "legendre", description="Function fitted to the overscan along the readout axis."
    )
    order: int = Field(
        1, ge=1,
        description="Number of terms in the fit, as in IRAF: 1 is a constant, "
                    "2 a straight line. For a spline, the number of pieces.",
    )
    sample: str = Field(
        "*", description="Which part of the overscan to fit, in pixels along the "
                         "readout axis, e.g. '20:500'. * uses all of it.",
    )
    naverage: int = Field(
        1, description="Average this many points before fitting; negative takes the "
                       "median instead, which ignores spikes.",
    )
    niterate: int = Field(
        1, ge=0, description="Rejection passes over the overscan fit; 1, as in IRAF."
    )
    low_reject: float = Field(
        3.0, ge=0, description="Rejection threshold below the fit, in sigma."
    )
    high_reject: float = Field(
        3.0, ge=0, description="Rejection threshold above the fit, in sigma."
    )
    grow: float = Field(
        0.0, ge=0, description="Also reject points within this distance of a rejected one."
    )
    readaxis: Literal["line", "column"] = Field(
        "column", description="Axis the overscan is averaged along."
    )
    interactive: bool = Field(
        False,
        description="Fit the overscan by hand, in the icfit panel: same keys as "
                    "identify, aptrace and continuum. The parameters you settle on "
                    "are written to the log, so the run can be repeated in batch.",
    )

    exposure_key: str = Field(
        "EXPTIME", description="Header keyword holding the exposure time."
    )
    logfile: str = Field(
        "harwood.log",
        description="Plain text log of what was done to each frame, appended as "
                    "the run goes so you can watch it in another window. Empty "
                    "keeps none; the JSON log and the headers are kept regardless.",
    )
    minreplace: float = Field(
        1.0, description="When a flat is processed, values below this are raised to "
                         "it, so dividing by the flat never divides by noise -- as "
                         "IRAF does."
    )


@register
class CcdProc(Task[CcdProcParams]):
    name = "ccdproc"
    package = "ccdred"
    description = "Process CCD images: overscan, trim, bias, dark, flat."
    Params = CcdProcParams

    def plan(self, params: CcdProcParams) -> list[Step]:
        """The steps, against the actual headers of the actual inputs.

        A plan that shows a step the run will skip is worse than no plan: it
        is the header bookkeeping made invisible again. So this reads the
        headers, classifies every frame through the instrument translation,
        and says which calibration it would reach for.
        """
        from ..core import images

        specs = images.expand(params.input)
        survey = _survey(specs, params)
        targets = survey["targets"]
        header = _peek(targets[0] if targets else specs[0])

        steps = [Step(
            "read", f"{len(specs)} image(s): " + ", ".join(
                f"{count} {kind}" for kind, count in survey["counts"].items()),
            why="Every frame is classified by its IMAGETYP, translated through "
                f"the '{survey['instrument'].name}' instrument file. That is what "
                "lets the calibrations be found rather than named.",
            measured={"instrument": survey["instrument"].name,
                      "to correct": f"{len(targets)}"
                      + (f" of type {params.ccdtype}" if params.ccdtype else "")},
        )]

        for flag, key, chosen in (("zerocor", "zero", survey["zero"]),
                                  ("darkcor", "dark", survey["dark"])):
            if getattr(params, flag) and chosen and not getattr(params, key):
                steps.append(Step(
                    f"find {key}", str(chosen),
                    why=f"No --{key} was given, so the {key} frame was taken from "
                        "the input list by its image type. It is corrected first, "
                        "then used.",
                ))
        if params.flatcor and not params.flat and survey["flats"]:
            steps.append(Step(
                "find flat",
                ", ".join(f"{subset or 'no subset'} -> {path.name}"
                          for subset, path in survey["flats"].items()),
                why="Flats are matched to frames by subset -- the filter, or "
                    "whatever else separates one set of flats from another. A "
                    "V frame gets the V flat without anyone saying so.",
            ))

        def add(flag: str, action: str, detail: str, why: str) -> None:
            if _already(header, flag):
                keyword = MARKS[flag][0]
                steps.append(Step(action, skipped=True,
                                  reason=f"{keyword} is already in the header "
                                         f"({header[keyword]})"))
            else:
                steps.append(Step(action, detail, why=why))

        if params.fixpix:
            add("fixpix", "fixpix", str(params.bpm) if params.bpm
                else "no mask given -- nothing to fix", WHY["fixpix"])
        if params.overscan:
            add("overscan", "overscan",
                f"{params.function} order {params.order}"
                + (f", sample {params.sample}" if params.sample != "*" else "")
                + (f", {params.niterate} rejection passes" if params.niterate else "")
                + " over "
                + f"{params.biassec or header.get('BIASSEC') or 'BIASSEC from the header'}",
                WHY["overscan"])
        if params.trim:
            add("trim", "trim",
                params.trimsec or header.get("TRIMSEC") or "TRIMSEC from the header",
                WHY["trim"])
        for flag, key in (("zerocor", "zero"), ("darkcor", "dark"),
                          ("flatcor", "flat")):
            if getattr(params, flag):
                given = getattr(params, key)
                found = survey["zero"] if key == "zero" else (
                    survey["dark"] if key == "dark"
                    else (next(iter(survey["flats"].values()), None)))
                add(flag, key,
                    str(given or found or f"no --{key} image and none found"),
                    WHY[key])
        steps.append(Step(
            "write",
            "in place, over the inputs" if params.inplace
            else (params.output or "this run's own directory under output/"),
            why=("Writing in place is safe here only because every step applied "
                 "is marked in the header, so running again corrects nothing "
                 "twice. The originals are gone either way."
                 if params.inplace else
                 "Nowhere was named, so the run gets its own dated directory and "
                 "the frames keep their names. What made them is written beside "
                 "them, in run.yaml.")))
        return steps

    def execute(self, params: CcdProcParams) -> TaskResult:
        import time

        import numpy as np
        from astropy.io import fits

        from ..core import images, logfile, provenance, subsets

        specs = images.expand(params.input)
        survey = _survey(specs, params)
        targets = survey["targets"]
        subsets.record(sorted(set(survey["subsets"].values())))
        if not targets:
            raise RuntimeError(
                f"none of the {len(specs)} images is of type '{params.ccdtype}'. "
                + ", ".join(f"{count} {kind}" for kind, count
                            in survey["counts"].items())
                + f". The '{survey['instrument'].name}' translation decides this; "
                "harwood ccdlist shows what it makes of each frame."
            )

        if params.inplace:
            if params.output:
                raise ValueError(
                    "--inplace overwrites the inputs, so --output has nothing to "
                    "mean. Give one or the other."
                )
            destinations = [spec.path for spec in targets]
        else:
            destinations = images.outputs_for(targets, params.output)

        # The calibrations have to go through the same geometry first, or a
        # 352x512 bias meets a 318x509 object and the subtraction fails -- or
        # worse, would succeed on a frame that happened to match. IRAF does
        # this automatically and so does harwood; the header marks make it
        # safe to run over a calibration that is already corrected.
        prepared = _prepare_calibrations(survey, params)

        outputs, all_steps, history = [], [], []
        for spec, destination in zip(targets, destinations, strict=True):
            data, header = images.read(spec)
            steps: list[Step] = []
            calibration = _calibration_for(header, survey, params)
            calibration = {key: prepared.get(str(value), value)
                           for key, value in calibration.items()}

            if params.fixpix and params.bpm is not None:
                data, step = _fixpix(data, params.bpm, header)
                steps.append(step)
            if params.overscan:
                data, step = _overscan(data, header, params)
                steps.append(step)
            if params.trim:
                data, step = _trim(data, header, params)
                steps.append(step)
            for flag, key in (("zerocor", "zero"), ("darkcor", "dark"),
                              ("flatcor", "flat")):
                if getattr(params, flag):
                    data, step = _calibrate(data, header, key,
                                            calibration[key], params)
                    steps.append(step)

            # A flat is left with nothing below minreplace, as IRAF leaves it,
            # so that dividing by it later can never divide by noise or zero.
            if survey["kinds"].get(spec) == "flat" and steps:
                low = data < params.minreplace
                if low.any():
                    data = np.where(low, params.minreplace, data)
                    steps.append(Step("minreplace", f"{int(low.sum())} pixels below "
                                      f"{params.minreplace:g} set to it",
                                      why=WHY["minreplace"],
                                      measured={"replaced": int(low.sum())}))
            # CCDMEAN, as IRAF writes it on everything ccdproc touches: it is
            # the scale a flat is divided by, and IRAF reads it from the file.
            if steps and any(not step.skipped for step in steps):
                header["CCDMEAN"] = (float(np.mean(data)), "mean of the image after ccdproc")
                header["CCDMEANT"] = (int(time.time()), "when CCDMEAN was computed")

            lines = [f"ccdproc: {step.render()}" for step in steps if not step.skipped]
            provenance.stamp_header(header, lines)
            logfile.write(params.logfile, "ccdproc", str(spec),
                          [step.render() for step in steps])
            destination.parent.mkdir(parents=True, exist_ok=True)
            fits.PrimaryHDU(data=data.astype(np.float32), header=header).writeto(
                destination, overwrite=True
            )
            outputs.append(destination)
            all_steps.extend(steps)
            history.extend(lines)

        skipped = sum(1 for step in all_steps if step.skipped)
        all_steps = prepared.pop("_steps", []) + all_steps
        logfile.note(params.logfile,
                     f"ccdproc finished: {len(outputs)} image(s), "
                     f"{skipped} step(s) already done")
        warnings = []
        if params.inplace:
            warnings.append(advice.warn(
                f"{len(outputs)} image(s) were corrected in place. The header "
                "marks mean running again corrects nothing twice, but the "
                "originals are gone.",
                "give --output next time so the raw frames survive. To get "
                "them back now you would have to copy them from the archive "
                "again."))
        return TaskResult(outputs=outputs, history=history, steps=all_steps,
                          warnings=warnings,
                          data={"images": len(targets), "steps_skipped": skipped,
                                "instrument": survey["instrument"].name})


#: What each step does to the data, and why that is the right thing. Shown by
#: --explain. Written here rather than in the docstring because the person who
#: needs it is running the task, not reading the source.
WHY = {
    "fixpix": "Bad columns and hot pixels carry no information, and a bad column "
              "running through a star ruins its photometry silently. Each bad "
              "region is replaced by interpolating across its narrowest side, "
              "as IRAF does -- a bad column along the rows, a bad row along the "
              "columns -- which makes the image look right without making the "
              "data real, so the mask travels with the frame.",
    "overscan": "The overscan is a strip of pixels the detector reads out without "
                "exposing them, so it carries the electronic offset the amplifier "
                "adds to every pixel and nothing else. That offset drifts during a "
                "readout, so it is fitted as a function of row rather than "
                "averaged to one number, and the fit is subtracted from the whole "
                "frame.",
    "trim": "The edges of a CCD are not illuminated: the overscan strip, and "
            "usually a row or two at each border that the shutter or the mask "
            "clips. Keeping them means every later statistic is computed over "
            "pixels that never saw sky.",
    "zero": "Even with the overscan removed, the bias has structure that varies "
            "from pixel to pixel and not from row to row -- fixed pattern in the "
            "amplifier and the clocking. A bias frame is a zero-second exposure, "
            "so it holds that pattern and nothing else; subtracting it removes "
            "what the overscan fit could not see.",
    "dark": "Thermal electrons accumulate in each pixel whether or not light "
            "falls on it, at a rate that depends on the temperature and varies "
            "across the chip. The dark frame measures that rate, and is scaled to "
            "this exposure's time before subtraction -- twice the exposure, twice "
            "the dark.",
    "minreplace": "A flat is about to be divided into every frame of the night, so "
                  "a pixel near zero in it -- a dead column, a vignetted corner -- "
                  "would turn into an enormous value in each of them. IRAF raises "
                  "anything below minreplace to it when it processes a flat, and so "
                  "does harwood.",
    "flat": "Pixels do not respond equally: quantum efficiency varies across the "
            "chip, dust on the window shadows it, and the optics vignette the "
            "corners. A flat field is a uniformly lit exposure, so it measures "
            "the response of every pixel relative to the rest. Dividing by it, "
            "normalised to one, turns counts into a quantity proportional to the "
            "light that arrived.",
}


def _survey(specs, params) -> dict:
    """Classify every input, and work out which calibration goes with which."""
    from collections import Counter

    from ..core import instrument as instruments

    inst = (instruments.load(params.instrument) if params.instrument
            else instruments.from_project())

    kinds, subsets = {}, {}
    for spec in specs:
        header = _peek(spec)
        kinds[spec] = inst.image_type(header)
        subsets[spec] = inst.subset(header)

    wanted = params.ccdtype.strip().lower()
    targets = [spec for spec in specs
               if not wanted or kinds[spec] == wanted]

    zero = params.zero or next(
        (s.path for s in specs if kinds[s] == "zero"), None)
    dark = params.dark or next(
        (s.path for s in specs if kinds[s] == "dark"), None)
    flats: dict[str, object] = {}
    if not params.flat:
        for spec in specs:
            if kinds[spec] == "flat":
                flats.setdefault(subsets[spec], spec.path)

    return {"instrument": inst, "kinds": kinds, "subsets": subsets,
            "targets": targets, "zero": zero, "dark": dark, "flats": flats,
            "counts": dict(Counter(kinds.values()))}


def _prepare_calibrations(survey, params) -> dict:
    """Put the calibration frames through the same geometry as the science.

    Only the steps that change the shape or the level of a calibration: a bias
    needs the overscan and the trim, a flat needs those and the bias. A flat
    does not get flat-fielded by itself, which is the mistake this exists to
    avoid making by accident.
    """
    from ..core import images

    done: dict = {"_steps": []}
    candidates = []
    if params.zerocor and not params.zero and survey["zero"]:
        candidates.append(("zero", survey["zero"]))
    if params.darkcor and not params.dark and survey["dark"]:
        candidates.append(("dark", survey["dark"]))
    if params.flatcor and not params.flat:
        candidates.extend(("flat", path) for path in survey["flats"].values())

    for kind, path in candidates:
        data, header = images.read(path)
        steps: list[Step] = []
        if params.overscan:
            data, step = _overscan(data, header, params)
            steps.append(step)
        if params.trim:
            data, step = _trim(data, header, params)
            steps.append(step)
        if kind in ("dark", "flat") and params.zerocor:
            zero = done.get(str(survey["zero"]), survey["zero"])
            if zero is not None:
                data, step = _calibrate(data, header, "zero", zero, params)
                steps.append(step)

        for step in steps:
            if not step.skipped:
                step.action = f"{kind} {Path(path).name}: {step.action}"
        done["_steps"].extend(steps)
        done[str(path)] = _Prepared(data, header, Path(path))

    return done


@dataclass
class _Prepared:
    """A calibration frame already put through the geometry, held in memory."""

    data: object
    header: object
    path: Path

    def __str__(self) -> str:
        return str(self.path)


def _calibration_for(header, survey, params) -> dict:
    """The zero, dark and flat that apply to one frame."""
    subset = survey["instrument"].subset(header)
    flat = params.flat
    if flat is None and survey["flats"]:
        flat = survey["flats"].get(subset)
        if flat is None and len(survey["flats"]) == 1:
            flat = next(iter(survey["flats"].values()))
        elif flat is None:
            raise ValueError(
                f"no flat field for subset '{subset}'. Flats were found for "
                + ", ".join(repr(s) for s in survey["flats"])
                + ". Give --flat, or check the instrument translation."
            )
    return {"zero": params.zero or survey["zero"],
            "dark": params.dark or survey["dark"],
            "flat": flat}


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
    from datetime import datetime

    header[keyword] = (datetime.now(UTC).isoformat(timespec="seconds"), comment)


def _fixpix(data, bpm_path, header):
    from ..core import images

    if _already(header, "fixpix"):
        return data, Step("fixpix", skipped=True, reason="FIXPIX already in the header")
    mask, _ = images.read(bpm_path)
    bad = mask != 0
    if not bad.any():
        return data, Step("fixpix", "mask marks no bad pixels")
    # IRAF's fixpix, the same code the fixpix task runs: across the narrowest
    # side of each bad region. (This used to interpolate along x always,
    # which is right for a bad column and wrong for a bad row.)
    from ..core import fixpix

    fixed, summary = fixpix.fix(data, mask)
    _mark(header, "fixpix")
    return fixed, Step(
        "fixpix", f"{int(bad.sum())} pixels replaced", why=WHY["fixpix"],
        measured={"replaced": f"{int(bad.sum())} pixels, "
                              f"{100 * bad.mean():.3f}%",
                  "along lines": summary["along lines"],
                  "along columns": summary["along columns"]},
        before=Task.statistics(data), after=Task.statistics(fixed))


def _overscan(data, header, params):
    import numpy as np

    from ..core import icfit, pixels

    if _already(header, "overscan"):
        return data, Step("overscan", skipped=True, reason="OVERSCAN already in the header")
    section = params.biassec or header.get("BIASSEC")
    if not section:
        raise ValueError(
            "overscan was asked for but no section was given and the header has no BIASSEC. "
            "Pass --biassec '[2049:2080,*]'."
        )
    slices = pixels.parse_section(section, data.shape)
    strip = data[slices]
    axis = 1 if params.readaxis == "column" else 0
    profile = strip.mean(axis=axis)

    # The positions are the *image's* row (or column) numbers, 1-based -- not
    # positions within the strip. A bias section like [335:350,2:510] covers
    # rows 2 to 510, so the profile describes those rows and nothing else, and
    # --sample given in image coordinates has to mean the same thing here as it
    # does everywhere else on the image.
    along = slices[0] if axis == 1 else slices[1]
    indices = np.arange(data.shape[0 if axis == 1 else 1])[along]
    position = indices.astype(float) + 1.0

    spec = icfit.FitSpec(
        function=params.function, order=params.order, sample=params.sample,
        naverage=params.naverage, niterate=params.niterate,
        low_reject=params.low_reject, high_reject=params.high_reject,
        grow=params.grow,
    )
    if params.interactive:
        from ..viewer.icfit_panel import run as run_icfit

        result = run_icfit(position, profile, spec,
                           title=f"ccdproc · overscan {section}",
                           xlabel="line (pixels)", ylabel="overscan level (ADU)")
        spec = result.spec
    else:
        result = icfit.fit(position, profile, spec)
    if result.model is None:
        raise ValueError(f"the overscan fit failed: {result.message}")

    # Evaluate the fit over *every* row of the image, not only the rows the
    # bias section sampled. A section that stops short of the edges -- which is
    # the normal case, because the first and last rows are usually bad -- still
    # has to correct the whole frame, and extrapolating a low-order fit a few
    # rows past its sample is exactly what it is for.
    full = np.arange(1.0, data.shape[0 if axis == 1 else 1] + 1.0)
    level = np.asarray(result(full), dtype=float)
    corrected = data - (level[:, None] if axis == 1 else level[None, :])
    _mark(header, "overscan")
    header["BIASSEC"] = section
    detail = (f"{spec.function} order {spec.order} over {section}, "
              f"mean level {float(level.mean()):.2f}, rms {result.rms:.2f}")
    if spec.sample != "*":
        detail += f", sample {spec.sample}"
    if int(result.deleted.sum()):
        detail += f", {int(result.deleted.sum())} points deleted by hand"
    if result.nrejected:
        detail += f", {result.nrejected} points rejected"
    return corrected, Step(
        "overscan", detail, why=WHY["overscan"],
        measured={"level": round(float(level.mean()), 2),
                  "drift": f"{float(level.min()):.2f} to {float(level.max()):.2f}",
                  "fit rms": round(float(result.rms), 3),
                  "rejected": f"{result.nrejected} of {int(result.mask.size)}"},
        before=Task.statistics(data), after=Task.statistics(corrected))


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
    lost = data.size - trimmed.size
    return trimmed, Step(
        "trim", f"{section} -> {trimmed.shape[1]}x{trimmed.shape[0]}",
        why=WHY["trim"],
        measured={"kept": f"{trimmed.shape[1]}x{trimmed.shape[0]}",
                  "dropped": f"{lost} pixels, {100 * lost / data.size:.1f}%"},
        before=Task.statistics(data), after=Task.statistics(trimmed))


def _calibrate(data, header, key, image, params):
    import numpy as np

    from ..core import images

    flag = {"zero": "zerocor", "dark": "darkcor", "flat": "flatcor"}[key]
    if _already(header, flag):
        return data, Step(key, skipped=True,
                          reason=f"{MARKS[flag][0]} already in the header")
    if image is None:
        raise ValueError(f"{flag} was asked for but no --{key} image was given")

    if isinstance(image, _Prepared):
        calibration, cal_header = image.data, image.header
    else:
        calibration, cal_header = images.read(image)
    if calibration.shape != data.shape:
        raise ValueError(
            f"{key} image {image} is {calibration.shape} but the frame is "
            f"{data.shape}. Was one of them trimmed and the other not?"
        )

    measured = {"image": Path(str(image)).name}
    if key == "zero":
        out = data - calibration
        detail = f"subtracted {Path(str(image)).name}"
        measured["level removed"] = round(float(np.median(calibration)), 3)
    elif key == "dark":
        frame_time = float(header.get(params.exposure_key, 1.0))
        dark_time = float(cal_header.get(params.exposure_key, 1.0))
        factor = frame_time / dark_time if dark_time else 1.0
        out = data - calibration * factor
        detail = f"subtracted {Path(str(image)).name} scaled by {factor:.3f}"
        measured["scale"] = f"{frame_time:g}s / {dark_time:g}s = {factor:.3f}"
    else:
        # IRAF: out = data * scale / flat, the scale being the flat's CCDMEAN --
        # the mean, not the median, of the flat as ccdproc left it. A flat
        # without CCDMEAN is one IRAF would process as a flat before using it,
        # which replaces values below minreplace and then takes the mean; that
        # is done here. (harwood used to normalise by the median: 0.11% off on
        # the 1987 M92 flat, a constant 1.2 mmag in every magnitude.)
        safe = np.where(calibration < params.minreplace, params.minreplace, calibration)
        if "CCDMEAN" in cal_header:
            level = float(cal_header["CCDMEAN"])
            source = "its CCDMEAN"
        else:
            level = float(np.mean(safe))
            source = "its mean"
        normalised = safe / level
        out = data / normalised
        detail = f"divided by {Path(str(image)).name}, normalised to {source}"
        measured["flat level"] = round(level, 3)
        measured["response range"] = (f"{float(normalised.min()):.3f} to "
                                      f"{float(normalised.max()):.3f}")
        replaced = int((calibration < params.minreplace).sum())
        if replaced:
            measured["clamped"] = f"{replaced} pixels below {params.minreplace}"

    _mark(header, flag)
    return out, Step(key, detail, why=WHY[key], measured=measured,
                     before=Task.statistics(data), after=Task.statistics(out))


class SetInstrumentParams(TaskParams):
    instrument: str | None = Field(
        None,
        description="Translation file to use: a name like 'kpno/direct', an IRAF "
                    "path like 'ccddb$kpno/direct.dat', or a file of your own. "
                    "Leave it out to see what is installed.",
        json_schema_extra={"positional": True},
    )
    list_available: bool = Field(
        False, description="Show what translations are installed and stop."
    )
    workdir: Path = Field(
        Path("."), description="Reduction directory whose project file to set."
    )


@register
class SetInstrument(Task[SetInstrumentParams]):
    name = "setinstrument"
    package = "ccdred"
    description = "Choose the header translation this reduction uses."
    Params = SetInstrumentParams

    def plan(self, params: SetInstrumentParams) -> list[Step]:
        if params.list_available:
            return [Step("list", "installed translations")]
        return [
            Step("read", str(params.instrument),
                 why="A translation file says which keyword holds the filter, and "
                     "what each IMAGETYP value means. Without one, harwood reads "
                     "the common keywords and reports anything else as unknown."),
            Step("write", f"instrument into {params.workdir}/harwood.yaml",
                 why="It goes in the project file rather than a setting on the "
                     "machine, so the reduction carries its own conventions and "
                     "someone else can repeat it."),
        ]

    def execute(self, params: SetInstrumentParams) -> TaskResult:
        from ..core import instrument as instruments

        if params.list_available or not params.instrument:
            found = instruments.installed()
            rows = [{"observatory": place, "translations": ", ".join(names)}
                    for place, names in sorted(found.items())]
            if not rows:
                raise RuntimeError(
                    "no instrument translations found. harwood does not ship them "
                    "yet; install iraf-noao, or write a file of your own -- two "
                    "columns, 'parameter keyword' or 'HEADER-VALUE imagetype'."
                )
            if not params.instrument:
                return TaskResult(data={"rows": rows},
                                  warnings=["Name one to set it, as in "
                                            "`harwood setinstrument kpno/direct`."],
                                  steps=self.plan(params))
            return TaskResult(data={"rows": rows}, steps=self.plan(params))

        chosen = instruments.load(params.instrument)

        import yaml

        from ..core import params as store

        project = store.find_project(params.workdir) or \
            (params.workdir / store.PROJECT_FILE)
        document = {}
        if project.exists():
            document = yaml.safe_load(project.read_text(encoding="utf-8")) or {}
        document["instrument"] = str(params.instrument)
        header = _project_header(project)
        project.write_text(header + yaml.safe_dump(document, sort_keys=False,
                                                   allow_unicode=False),
                           encoding="utf-8")

        return TaskResult(
            outputs=[project],
            data={"instrument": chosen.name, "source": chosen.source,
                  "keywords": len(chosen.keywords), "types": len(chosen.types)},
            history=[f"setinstrument: {params.instrument}"],
            steps=self.plan(params),
        )


def _project_header(project: Path) -> str:
    if not project.exists():
        return ""
    kept = []
    for line in project.read_text(encoding="utf-8").splitlines():
        if line.startswith("#") or not line.strip():
            kept.append(line)
        else:
            break
    return ("\n".join(kept) + "\n") if kept else ""


class CcdListParams(TaskParams):
    input: list[str] = Field(..., description="Images to list. Accepts globs and @lists.")
    ccdtype: str = Field(
        "", description="Show only images of this type: object, zero, dark, flat, comp."
    )
    long: bool = Field(
        False, description="Also show which corrections have already been applied."
    )
    instrument: str | None = Field(
        None, description="Translation to use. Taken from the project file when not given."
    )


@register
class CcdList(Task[CcdListParams]):
    name = "ccdlist"
    package = "ccdred"
    description = "List images as the reduction sees them: type, subset, what is done."
    Params = CcdListParams

    def plan(self, params: CcdListParams) -> list[Step]:
        return [Step("classify", f"{len(params.input)} pattern(s) through the "
                                 f"instrument translation",
                     why="This is the check to run before ccdproc: if a frame comes "
                         "back 'unknown', ccdproc will not know what to do with it "
                         "either, and the translation file is what to fix.")]

    def execute(self, params: CcdListParams) -> TaskResult:
        from ..core import images
        from ..core import instrument as instruments

        inst = (instruments.load(params.instrument) if params.instrument
                else instruments.from_project())
        wanted = params.ccdtype.strip().lower()

        from ..core import subsets as subset_file

        rows, unknown, seen = [], 0, []
        for spec in images.expand(params.input):
            hdul, hdu = images.open_hdu(spec)
            try:
                header = hdu.header.copy()
                shape = ("x".join(str(n) for n in reversed(hdu.data.shape))
                         if getattr(hdu, "data", None) is not None else "-")
            finally:
                hdul.close()

            kind = inst.image_type(header)
            if wanted and kind != wanted:
                continue
            if kind == "unknown":
                unknown += 1

            subset = inst.subset(header)
            seen.append(subset)
            row = {"image": str(spec), "type": kind,
                   "subset": subset_file.name_for(subset, create=False) or "-",
                   "raw subset": subset or "-", "shape": shape,
                   "exptime": inst.value(header, "exptime", "-"),
                   "object": header.get("OBJECT", "")}
            if params.long:
                done = [name for name, (keyword, _) in MARKS.items()
                        if keyword in header]
                row["done"] = ", ".join(done) if done else "-"
            rows.append(row)

        known = subset_file.record(sorted(set(seen)))
        for row in rows:
            raw = row["raw subset"]
            row["subset"] = known.get(raw, row["subset"]) if raw != "-" else "-"

        warnings = []
        if unknown:
            warnings.append(advice.warn(
                f"{unknown} image(s) came back 'unknown'. ccdproc will refuse "
                f"them too, so they would drop out of the reduction silently.",
                f"add the missing IMAGETYP value to the '{inst.name}' "
                "translation, or pick another with 'harwood setinstrument'. "
                "'harwood imheader <image> IMAGETYP' shows what the frames "
                "actually say."))
        return TaskResult(data={"rows": rows, "instrument": inst.name},
                          warnings=warnings, steps=self.plan(params))
