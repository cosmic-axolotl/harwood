"""ccdred -- CCD reduction.

`ccdproc` is the spine of any imaging reduction and the place where IRAF's
design still shows its quality. Two behaviours matter as much as the arithmetic
and are usually the first thing a hand-rolled script loses:

* **the pipeline is ordered and named** -- fixpix, overscan, trim, zero, dark,
  flat -- and you can print it before running it (`--plan`, IRAF's `noproc`);
* **the header remembers** -- once a frame is bias-subtracted it carries
  `ZEROCOR`, and running ccdproc again skips that step instead of subtracting
  the bias twice. Doing that to a night of data is silent and unrecoverable.

The arithmetic is IRAF's ccdproc, ported (the overscan fit is icfit, as in
IRAF) and checked against IRAF's own output on the M92 frames; the community
`ccdproc` package is not used.

What IRAF's ccdproc never did is keep the uncertainty (design contract,
section 12). Here the variance is born once the overscan and the bias are
off -- what is left counts the electrons the pixel collected, so its
variance is data/gain + (readnoise/gain)^2 in ADU^2 -- and is carried through
the dark and the flat with the calibrations' own. Saturated pixels, the
bad pixels fixpix papered over and the dead pixels of the flat go into the
DQ. With no gain anywhere there is no variance, and the run says so: one is
never made up.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC
from pathlib import Path
from typing import Literal

from pydantic import Field

from ..core import advice
from ..core.registry import register
from ..core.task import (
    Step,
    Task,
    TaskParams,
    TaskResult,
    Template,
    format_field,
    uncertainty_field,
)

#: Keyword written once a step has been applied, so it is never applied twice.
MARKS = {
    "overscan": ("OVERSCAN", "overscan corrected by harwood"),
    "trim": ("TRIM", "trimmed by harwood"),
    "zerocor": ("ZEROCOR", "bias subtracted by harwood"),
    "darkcor": ("DARKCOR", "dark subtracted by harwood"),
    "flatcor": ("FLATCOR", "flat fielded by harwood"),
    "fixpix": ("FIXPIX", "bad pixels replaced by harwood"),
    "illumcor": ("ILLUMCOR", "illumination corrected by harwood"),
    "fringecor": ("FRINGCOR", "fringes subtracted by harwood"),
}

#: The steps of each image type, as t_ccdproc switches on it. What is not named here
#: runs them all, and takes a CCDMEAN.
ALL_STEPS = ("fixpix", "overscan", "trim", "zerocor", "darkcor", "flatcor", "illumcor",
             "fringecor")
BY_TYPE = {"zero": ALL_STEPS[:3], "dark": ALL_STEPS[:4], "flat": ALL_STEPS[:5],
           "illum": ALL_STEPS[:6]}


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
    illumcor: bool = Field(
        False, description="Divide by an illumination image, normalised by its mean. It has "
                           "to carry the MKILLUM keyword, as mkillumcor leaves it."
    )
    fringecor: bool = Field(
        False, description="Subtract a fringe image, scaled by the exposure times and its "
                           "FRINGSCL. It has to carry the MKFRINGE keyword."
    )
    readcor: bool = Field(
        False, description="A zero becomes its average across the readout axis, one line or "
                           "one column that goes for the whole frame."
    )
    scancor: bool = Field(
        False, description="The zero, dark and flat become what a scanned detector saw: a "
                           "moving average for a shortscan, one line for a longscan."
    )

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
    illum: Path | None = Field(None, description="Illumination image.")
    fringe: Path | None = Field(None, description="Fringe image.")
    scantype: Literal["shortscan", "longscan"] = Field(
        "shortscan", description="For --scancor: moved during the exposure and read at the "
                                 "end, or read continuously."
    )
    nscan: int = Field(
        1, ge=1, description="For --scancor with a shortscan: the steps of the scan, for an "
                             "object with no NSCANROW keyword."
    )
    bpm: Path | None = Field(
        None, description="Bad pixel mask, non-zero where a pixel is bad."
    )

    # The overscan fit is an icfit, with icfit's parameter names -- the same
    # ones identify, aptrace, continuum and sensfunc use. Learn them once.
    function: Literal["legendre", "chebyshev", "spline1", "spline3", "mean", "median",
                      "minmax"] = Field(
        "legendre", description="Function fitted to the overscan along the readout axis; "
                                "or with mean, median or minmax no fit, a value for each "
                                "line (readaxis line only). minmax leaves out the highest "
                                "and the lowest."
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
        "line", description="The readout axis, as in IRAF: line takes the overscan as a "
                            "function of the line, its strip at the end of each."
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

    gain: float | str | None = Field(
        None, description="Electrons per ADU, or the header keyword holding it. Left "
                          "out, the instrument's gain keyword (GAIN). Without one the "
                          "variance cannot be known and none is written.")
    readnoise: float | str | None = Field(
        None, description="Read noise in electrons, or the header keyword holding it. "
                          "Left out, the instrument's (RDNOISE), else 0.")
    saturation: float | None = Field(
        None, description="Raw level, in ADU, at which a pixel is flagged saturated in "
                          "the DQ. Left out, none is.")
    uncertainty: Literal["VAR", "ERR", "UNCERT"] = uncertainty_field()
    format: Literal["harwood", "iraf"] = format_field()


@register
class CcdProc(Task[CcdProcParams]):
    name = "ccdproc"
    package = "ccdred"
    description = "Process CCD images: overscan, trim, bias, dark, flat, illumination, fringe."
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
                          ("flatcor", "flat"), ("illumcor", "illum"),
                          ("fringecor", "fringe")):
            if getattr(params, flag):
                given = getattr(params, key)
                found = survey["zero"] if key == "zero" else (
                    survey["dark"] if key == "dark"
                    else (next(iter(survey[f"{key}s"].values()), None)))
                add(flag, key,
                    str(given or found or f"no --{key} image and none found"),
                    WHY[key])
        if params.readcor:
            steps.append(Step("readcor", f"the zero, across the {params.readaxis}s",
                              why=WHY["readcor"]))
        if params.scancor:
            steps.append(Step("scancor", params.scantype
                              + (f", {params.nscan} steps" if params.scantype == "shortscan"
                                 else ""), why=WHY["scancor"]))
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
        from ..core import images, logfile, mef, provenance, subsets

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

        outputs, all_steps, history, warnings = [], [], [], []
        no_gain = []
        for spec, destination in zip(targets, destinations, strict=True):
            frame = _Frame.load(spec)
            header = frame.header
            steps: list[Step] = []
            calibration = _calibration_for(header, survey, params)
            calibration = {key: prepared.get(str(value), value)
                           for key, value in calibration.items()}

            kind = survey["kinds"].get(spec, "object")
            steps.extend(_process(frame, params, survey["instrument"], calibration, kind=kind,
                                  warnings=warnings))
            if frame.var is None:
                no_gain.append(Path(str(spec)).name)
            _finish(frame, kind, steps, params)
            data = frame.data
            header = frame.header

            lines = [f"ccdproc: {step.render()}" for step in steps if not step.skipped]
            provenance.stamp_header(header, lines)
            logfile.write(params.logfile, "ccdproc", str(spec),
                          [step.render() for step in steps])
            destination.parent.mkdir(parents=True, exist_ok=True)
            written, notes = mef.save(destination, data, header, frame.var, frame.dq,
                                      params.uncertainty, params.format)
            outputs.append(destination)
            outputs.extend(written[1:])
            if notes and not any("--format iraf" in str(w) for w in warnings):
                warnings.extend(notes)
            all_steps.extend(steps)
            history.extend(lines)

        skipped = sum(1 for step in all_steps if step.skipped)
        all_steps = prepared.pop("_steps", []) + all_steps
        warnings.extend(prepared.pop("_warnings", []))
        logfile.note(params.logfile,
                     f"ccdproc finished: {len(outputs)} image(s), "
                     f"{skipped} step(s) already done")
        if no_gain:
            warnings.append(advice.warn(
                f"no gain for {len(no_gain)} of {len(outputs)} image(s) "
                f"({', '.join(no_gain[:3])}{'...' if len(no_gain) > 3 else ''}), so "
                "no variance was written for them: it cannot be known without one.",
                "give --gain (electrons per ADU) and --readnoise, or the header "
                "keywords that hold them; the instrument's sheet or the observatory's "
                "web page has them."))
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
    "saturation": "Above its full well a pixel stops counting, and its value is a "
                  "floor, not a measurement. It is flagged in the DQ while the frame "
                  "is raw -- after the bias and the flat are off, the level it hit is "
                  "no longer the number the detector reported.",
    "variance": "With the overscan and the bias subtracted, what is left in a pixel "
                "counts the electrons it collected -- star, sky and dark current. "
                "Their number fluctuates as Poisson says, and the amplifier adds its "
                "read noise, so the variance is data/gain + (readnoise/gain)^2, in "
                "ADU^2. From here on it travels with the data: the dark and the flat "
                "add their own.",
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
    "illum": "A dome flat is lit differently from the sky, by a pattern that is smooth "
             "across the chip. An illumination image holds that pattern, usually from "
             "dark-sky flats, and dividing by it, normalised by its mean, flattens the sky.",
    "fringe": "In the red, night-sky lines interfere in the thin chip and print a pattern "
              "that adds to the sky and does not divide. A fringe image holds it for some "
              "exposure, and it is subtracted scaled by the exposure times.",
    "readcor": "A zero's structure can run only along the readout, the same for every line. "
               "Averaging across that axis leaves the structure and removes the noise of "
               "all those pixels.",
    "scancor": "A scanned detector moves the charge while it exposes, so each pixel of the "
               "image was seen by several real pixels, and the calibrations have to be made "
               "the same way: a moving average over the scan, or a single line for a scan "
               "as long as the exposure.",
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
    by_subset: dict[str, dict] = {"flat": {}, "illum": {}, "fringe": {}}
    for key, found in by_subset.items():
        if not getattr(params, key):
            for spec in specs:
                if kinds[spec] == key:
                    found.setdefault(subsets[spec], spec.path)

    return {"instrument": inst, "kinds": kinds, "subsets": subsets,
            "targets": targets, "zero": zero, "dark": dark, "flats": by_subset["flat"],
            "illums": by_subset["illum"], "fringes": by_subset["fringe"],
            "counts": dict(Counter(kinds.values()))}


def _prepare_calibrations(survey, params) -> dict:
    """Put the calibration frames through what they need first, as ccdcheck and ccdproc do
    inside set_zero, set_dark and set_flat: each by its own type, whatever its header says,
    and in memory: harwood does not rewrite a calibration given to it. The header marks
    leave what is already done alone. A flat does not get flat-fielded by itself, which is
    the mistake this exists to avoid making by accident.
    """

    done: dict = {"_steps": [], "_warnings": []}
    candidates = []
    if params.zerocor and survey["zero"]:
        candidates.append(("zero", survey["zero"]))
    if params.darkcor and survey["dark"]:
        candidates.append(("dark", survey["dark"]))
    if params.flatcor:
        if params.flat:
            candidates.append(("flat", params.flat))
        candidates.extend(("flat", path) for path in survey["flats"].values())

    for kind, path in candidates:
        if str(path) in done:
            continue
        frame = _Frame.load(path)
        calibration = {"zero": done.get(str(survey["zero"]), survey["zero"]),
                       "dark": done.get(str(survey["dark"]), survey["dark"]),
                       "flat": None, "illum": None, "fringe": None}
        steps = _process(frame, params, survey["instrument"], calibration, kind=kind,
                         warnings=done["_warnings"])
        _finish(frame, kind, steps, params)
        for step in steps:
            if not step.skipped:
                step.action = f"{kind} {Path(path).name}: {step.action}"
        done["_steps"].extend(steps)
        done[str(path)] = frame

    return done


@dataclass
class _Frame:
    """An image as ccdproc works on it: the pixels, the header, and the
    variance (None until there is a gain to make one) and DQ that travel
    with them. A calibration frame prepared in memory is one too."""

    data: object
    header: object
    var: object
    dq: object
    path: Path
    # What _calibrate used it as: the dark's scale, the flat's normalised
    # response, its level and the pixels raised to minreplace.
    factor: float = 1.0
    normalised: object = None
    level: float = 1.0
    clamped: object = None
    sections: object = None

    @classmethod
    def load(cls, spec) -> _Frame:
        from ..core import mef

        data, header, var, quality = mef.load(spec)
        return cls(data, header, var, quality.copy(), Path(str(getattr(spec, "path", spec))))

    def __str__(self) -> str:
        return str(self.path)


def _noise(header, params, inst) -> tuple[float | None, float]:
    """The gain (None when nothing says) and the read noise, in electrons:
    the parameter as a number or as a keyword, else the instrument's."""

    def value(given, parameter):
        if given is not None:
            try:
                return float(given)
            except ValueError:
                return float(header[given]) if given in header else None
        found = inst.value(header, parameter)
        try:
            return float(found) if found is not None else None
        except (TypeError, ValueError):
            return None

    gain = value(params.gain, "gain")
    readnoise = value(params.readnoise, "rdnoise")
    return (gain if gain and gain > 0 else None), (readnoise or 0.0)


def _process(frame: _Frame, params, inst, calibration: dict,
             only: tuple[str, ...] = ALL_STEPS, kind: str = "object",
             warnings: list | None = None) -> list[Step]:
    """ccdproc's steps, in IRAF's order, on the data with its variance and DQ, each
    in real as cor.gx writes it: `(out - overscan - zero - darkscale*dark) * flatscale /
    flat * illumscale / illum - frgscale*fringe`, one operation at a time. The overscan
    comes from the raw lines and goes over the trimmed ones. `kind` is the image type,
    which decides the steps; a zero frame has no photons in it, only read noise, and its
    variance is that alone."""
    import numpy as np

    from ..core import ccdsections, dq

    steps: list[Step] = []
    header = frame.header
    allowed = set(only) & set(BY_TYPE.get(kind, ALL_STEPS))
    raw = not any(MARKS[flag][0] in header for flag in MARKS)
    if params.saturation is not None:
        if raw:
            saturated = frame.data >= params.saturation
            frame.dq[saturated] |= dq.SATURATED
            steps.append(Step("saturation", f"{int(saturated.sum())} pixels at or above "
                                            f"{params.saturation:g} flagged in the DQ",
                              why=WHY["saturation"],
                              measured={"saturated": int(saturated.sum())}))
        else:
            steps.append(Step("saturation", skipped=True,
                              reason="the frame is already processed, and saturation "
                                     "is a raw level"))

    biassec = params.biassec or header.get("BIASSEC") or ""
    trimsec = params.trimsec or header.get("TRIMSEC") or ""
    data = np.asarray(frame.data, np.float32)
    source = data
    sec = ccdsections.sections(header, data.shape, biassec, trimsec)
    trimming = "trim" in allowed and params.trim
    if trimming and not _already(header, "trim"):
        if not trimsec:
            raise ValueError(
                "trim was asked for but no section was given and the header has no TRIMSEC. "
                "Pass --trimsec '[1:2048,1:2048]'.")
        ccdsections.trim(sec)
    changed = False

    # The overscan vector first: it is measured on the raw lines, and fixed pixels only
    # reach a per-line overscan, as those go through xt_fps with the rest of the line.
    vector = None
    if "overscan" in allowed and params.overscan:
        vector, step = _overscan(data, header, params, sec, biassec)
        if not step.skipped:
            changed = True
    else:
        step = None

    shape = data.shape
    if sec.trimmed:
        cut = (slice(sec.trim_l1 - 1, sec.trim_l2), slice(sec.trim_c1 - 1, sec.trim_c2))
        data = data[cut].copy()
        frame.dq = frame.dq[cut]
        if frame.var is not None:
            frame.var = frame.var[cut]
    else:
        cut = None
        data = data.copy()
    region = (slice(sec.out_l1 - 1, sec.out_l2), slice(sec.out_c1 - 1, sec.out_c2))

    if "fixpix" in allowed and params.fixpix and params.bpm is not None:
        fixed, fix_step, bad = _fixpix(source, data, params.bpm, header, sec, region)
        if not fix_step.skipped:
            data = fixed
            frame.dq[region][bad] |= dq.BAD_PIXEL
            changed = True
        steps.append(fix_step)

    if step is not None:
        if not step.skipped:
            before = Task.statistics(data)
            data[region] = (data[region] - vector).astype(np.float32)
            step.before, step.after = before, Task.statistics(data)
        steps.append(step)
    if trimming:
        if _already(header, "trim"):
            steps.append(Step("trim", skipped=True, reason="TRIM already in the header"))
        else:
            lost = shape[0] * shape[1] - data.size
            _mark(header, "trim")
            changed = True
            steps.append(Step(
                "trim", f"{trimsec} -> {data.shape[1]}x{data.shape[0]}", why=WHY["trim"],
                measured={"kept": f"{data.shape[1]}x{data.shape[0]}",
                          "dropped": f"{lost} pixels, "
                                     f"{100 * lost / (shape[0] * shape[1]):.1f}%"},
                before=Task.statistics(source), after=Task.statistics(data)))

    frame.data = data
    frame.sections = sec

    if "zerocor" in allowed and params.zerocor:
        cal = _calibration_frame("zero", calibration["zero"], frame, params, inst, kind)
        frame.data, cor, used = _calibrate(frame.data, header, "zero", cal, params, sec,
                                           region)
        steps.append(cor)
        changed = changed or not cor.skipped
        _carry(frame, used, cor, 1.0, params, inst, "zero", sec)

    # The variance is born here: with the overscan and the bias off, the data
    # counts the electrons each pixel collected, dark current and all.
    if frame.var is None:
        gain, readnoise = _noise(header, params, inst)
        offset_off = any(key in header for key in (MARKS["overscan"][0],
                                                   MARKS["zerocor"][0]))
        if gain is not None and offset_off:
            photons = 0.0 if kind == "zero" else np.maximum(frame.data, 0.0) / gain
            noise = (readnoise / gain) ** 2
            frame.var = photons + np.zeros_like(frame.data, dtype=float) + noise
            model = ("read noise only, a zero frame: " if kind == "zero" else "data/"
                     f"{gain:g} + ") + f"({readnoise:g}/{gain:g})^2, in ADU^2"
            steps.append(Step("variance", model, why=WHY["variance"],
                              measured={"gain": f"{gain:g} e-/ADU",
                                        "read noise": f"{readnoise:g} e-",
                                        "median sigma":
                                            f"{float(np.median(np.sqrt(frame.var))):.3g}"
                                            " ADU"}))
        elif gain is not None:
            steps.append(Step("variance", skipped=True,
                              reason="the bias level is still in the data, so its counts "
                                     "are not electrons yet: run --overscan or --zerocor"))

    if "darkcor" in allowed and params.darkcor:
        cal = _calibration_frame("dark", calibration["dark"], frame, params, inst, kind)
        flat_cal = None
        if "flatcor" in allowed and params.flatcor and calibration.get("flat") is not None:
            flat_cal = _calibration_frame("flat", calibration["flat"], frame, params, inst,
                                          kind)
        frame.data, cor, used = _calibrate(frame.data, header, "dark", cal, params, sec,
                                           region, inst=inst, flat=flat_cal)
        steps.append(cor)
        changed = changed or not cor.skipped
        if used is not None:
            _carry(frame, used, cor, used.factor, params, inst, "dark", sec)
    if "flatcor" in allowed and params.flatcor:
        before = frame.data
        cal = _calibration_frame("flat", calibration["flat"], frame, params, inst, kind)
        frame.data, cor, used = _calibrate(frame.data, header, "flat", cal, params, sec,
                                           region)
        steps.append(cor)
        changed = changed or not cor.skipped
        if used is not None:
            n = used.normalised
            if frame.var is not None:
                inner = np.ones_like(frame.var)
                inner[region] = n
                full = _cal_variance(used, params, inst, cor, header, "flat")
                frame.var = frame.var / inner ** 2
                piece = _in_region(full, used, sec) / used.level ** 2
                frame.var[region] = frame.var[region] + (
                    (before[region] / n) ** 2 * piece / n ** 2)
            frame.dq[region] |= _in_region(used.dq, used, sec)
            clamped = np.broadcast_to(used.clamped, frame.dq[region].shape)
            frame.dq[region][clamped] |= dq.BAD_PIXEL
    for flag, key in (("illumcor", "illum"), ("fringecor", "fringe")):
        if flag in allowed and getattr(params, flag):
            cor = _extra(frame, key, calibration.get(key), params, inst, sec, region, warnings)
            steps.append(cor)
            changed = changed or not cor.skipped
    if changed:
        ccdsections.update_header(header, sec, frame.data.shape)
    return steps


def _finish(frame: _Frame, kind: str, steps: list[Step], params) -> None:
    """What is left after the lines: a flat raises what is below minreplace, so that
    dividing by it can never divide by noise or zero; it and what has no known type take
    a CCDMEAN. A flat missing one takes it from the file (ccdmean). A zero with --readcor
    becomes one line or one column."""
    import time

    import numpy as np

    from ..core import ccdsections

    done = any(not step.skipped for step in steps)
    if kind == "flat" and done:
        low = frame.data < params.minreplace
        region = getattr(frame.sections, "out_c1", None)
        if region is not None:
            sec = frame.sections
            inside = np.zeros_like(low)
            inside[sec.out_l1 - 1:sec.out_l2, sec.out_c1 - 1:sec.out_c2] = True
            low &= inside
        if low.any():
            frame.data = np.where(low, np.float32(params.minreplace), frame.data).astype(
                np.float32)
            steps.append(Step("minreplace", f"{int(low.sum())} pixels below "
                              f"{params.minreplace:g} set to it",
                              why=WHY["minreplace"],
                              measured={"replaced": int(low.sum())}))
    if done and kind not in ("zero", "dark", "illum", "object", "comp"):
        sec = frame.sections
        inner = frame.data[sec.out_l1 - 1:sec.out_l2, sec.out_c1 - 1:sec.out_c2]
        total = np.float32(0.0)
        for line in inner:
            total = np.float32(total + ccdsections.asum(line))
        mean = np.float32(np.float32(total / np.float32(inner.shape[1]))
                          / np.float32(inner.shape[0]))
        frame.header["CCDMEAN"] = (_real_card(mean), "mean of the image after ccdproc")
        frame.header["CCDMEANT"] = (int(time.time()), "when CCDMEAN was computed")
    if kind == "zero" and params.readcor and "READCOR" not in frame.header:
        before = Task.statistics(frame.data)
        frame.data = ccdsections.readcor(frame.data, frame.header, params.readaxis == "line")
        _mark_as(frame.header, "READCOR", "converted to readout format by harwood")
        quality, variance = frame.dq, frame.var
        frame.dq = _one_d(quality, params.readaxis == "line", "or")
        frame.var = None if variance is None else _one_d(variance / max(
            quality.shape[0 if params.readaxis == "line" else 1], 1),
            params.readaxis == "line", "mean")
        steps.append(Step("readcor", f"to {frame.data.shape[1]}x{frame.data.shape[0]}",
                          why=WHY["readcor"], before=before,
                          after=Task.statistics(frame.data)))


def _one_d(plane, line_axis: bool, how: str):
    import numpy as np

    axis = 0 if line_axis else 1
    if how == "or":
        return np.bitwise_or.reduce(plane, axis=axis, keepdims=True)
    return plane.mean(axis=axis, keepdims=True)


def _mark_as(header, keyword: str, comment: str) -> None:
    from datetime import datetime

    header[keyword] = (datetime.now(UTC).isoformat(timespec="seconds"), comment)


def _real_card(value) -> float:
    """A real as hdmputr leaves it in a card, seven digits, which is what is read back
    from it: a flat is divided by that, not by the mean it worked out."""
    return float(f"{float(value):.7g}")


def _in_region(plane, cal, sec):
    """A calibration's plane, for this frame's lines."""
    from ..core import ccdsections

    cut = ccdsections.cal_region(plane.shape, cal.header, sec, str(cal))
    return plane[cut]


def _extra(frame, key, image, params, inst, sec, region, warnings) -> Step:
    """set_illum and set_fringe: what stops them only warns, and the frame goes on."""
    import numpy as np

    from ..core import ccdsections, dq

    flag = {"illum": "illumcor", "fringe": "fringecor"}[key]
    header = frame.header
    if _already(header, flag):
        return Step(key, skipped=True, reason=f"{MARKS[flag][0]} already in the header")
    if image is None:
        raise ValueError(f"{flag} was asked for but no --{key} image was given")
    cal = image if isinstance(image, _Frame) else _Frame.load(image)
    needed = {"illum": "MKILLUM", "fringe": "MKFRINGE"}[key]
    if needed not in cal.header:
        message = advice.warn(
            f"{needed} flag missing from the {key} image {Path(str(image)).name}, so it was "
            "not used.", f"mark it with `hedit {Path(str(image)).name} {needed} yes` if it "
            f"is what mk{key}cor makes.")
        if warnings is not None:
            warnings.append(message)
        return Step(key, skipped=True, reason=f"{needed} is not in {Path(str(image)).name}")
    cut = ccdsections.cal_region(cal.data.shape, cal.header, sec, Path(str(image)).name)
    plane = np.asarray(cal.data[cut], np.float32)
    data = frame.data
    before_stats = Task.statistics(data)
    if key == "illum":
        if "CCDMEAN" in cal.header:
            scale = _hdm_real(cal.header, "CCDMEAN")
        else:
            scale = np.float32(_real_card(ccdsections.mean_of(cal.data)))
        inner = (data[region] * scale).astype(np.float32)
        data[region] = (inner / plane).astype(np.float32)
        if frame.var is not None:
            frame.var[region] = frame.var[region] * (scale / plane.astype(float)) ** 2
        detail = f"divided by {Path(str(image)).name} with scale {float(scale):g}"
    else:
        time1 = _hdm_real(header, "EXPTIME", params.exposure_key)
        time2 = _hdm_real(cal.header, "EXPTIME", params.exposure_key)
        each = np.float32(_hdm_real(cal.header, "FRINGSCL")) if "FRINGSCL" in cal.header \
            else np.float32(1.0)
        scale = np.float32(np.float32(time1 / time2) * each)
        data[region] = (data[region] - (scale * plane).astype(np.float32)).astype(np.float32)
        if frame.var is not None and cal.var is not None:
            frame.var[region] = frame.var[region] + float(scale) ** 2 * cal.var[cut]
        detail = f"subtracted {Path(str(image)).name} with scale {float(scale):g}"
    frame.dq[region] |= cal.dq[cut].astype(dq.DTYPE)
    _mark(header, flag)
    return Step(key, detail, why=WHY[key], measured={"image": Path(str(image)).name,
                                                     "scale": f"{float(scale):g}"},
                before=before_stats, after=Task.statistics(data))


def _hdm_real(header, key: str, fallback: str = ""):
    """hdmgetr: a card's text through IRAF's reading of numbers, as a real."""
    import numpy as np

    from ..core import mwcs

    name = key
    if fallback and key == "EXPTIME":
        name = fallback
    return np.float32(mwcs._number(header, name, 0.0))


def _calibration_frame(role, image, frame, params, inst, kind):
    """The calibration for this frame: with --scancor, what a frame with its scan saw.
    ccdnscan: the frame's NSCANROW, or for an object the parameter, and with a calibration
    or no scan, the image itself. cal_scan names the one made from it, as `Flat.3.fits`
    or `Flat.1d.fits`, and it is used if it is there."""
    if not params.scancor or image is None:
        return image
    from ..core import ccdsections, mef

    calibrating = kind in ("zero", "dark", "flat", "illum", "fringe")
    header = frame.header
    if "NSCANROW" in header:
        nscan = int(header["NSCANROW"])
    elif calibrating:
        nscan = 1
    elif params.scantype == "shortscan":
        nscan = params.nscan
    else:
        nscan = None
    if nscan == 1:
        return image
    base = image if isinstance(image, _Frame) else _Frame.load(image)
    path = Path(base.path)
    tag = "1d" if nscan is None else str(nscan)
    made = path.with_name(f"{path.stem}.{tag}{path.suffix}")
    if made.exists():
        return made
    head = base.header.copy()
    line_axis = params.readaxis == "line"
    floor = params.minreplace if role == "flat" else None
    if nscan is None:
        data, mean = ccdsections.longscan(base.data, head, line_axis)
    else:
        data, mean = ccdsections.shortscan(base.data, head, nscan, floor, line_axis)
        head["NSCANROW"] = nscan
    head["CCDMEAN"] = (_real_card(mean), "mean of the image after scancor")
    _mark_as(head, "SCANCOR", "converted to a scan by harwood")
    mef.save(made, data, head, None, None, "VAR", "harwood")
    return made


def _cal_variance(cal, params, inst, step, frame_header, key):
    """A calibration's variance: its own VAR when it has one. Otherwise, with
    a gain, the noise of the NCOMBINE frames it was combined from -- read
    noise and photons, averaged -- which is what an IRAF-made Zero.fits holds;
    without a gain, none, and the step says it was left out. The gain is the
    calibration's if its header has one, else the frame's: it is the same
    detector, and a master bias or flat seldom carries GAIN."""
    import numpy as np

    if cal.var is not None:
        return cal.var
    gain, readnoise = _noise(cal.header, params, inst)
    if gain is None:
        gain, readnoise = _noise(frame_header, params, inst)
    if gain is None:
        step.measured["calibration variance"] = "unknown (no gain): left out"
        return np.zeros_like(cal.data)
    count = int(cal.header.get("NCOMBINE", 1) or 1)
    step.measured["calibration variance"] = (
        f"estimated from the gain and read noise, NCOMBINE={count}")
    photons = 0.0 if key == "zero" else np.maximum(cal.data, 0.0) / gain
    return (photons + (readnoise / gain) ** 2) / count


def _carry(frame, cal, step, factor, params, inst, key, sec) -> None:
    """A subtracted calibration: its variance adds (scaled by the square of
    what it was multiplied by), its DQ joins the frame's."""
    if cal is None:
        return
    region = (slice(sec.out_l1 - 1, sec.out_l2), slice(sec.out_c1 - 1, sec.out_c2))
    if frame.var is not None:
        full = _cal_variance(cal, params, inst, step, frame.header, key)
        frame.var[region] = frame.var[region] + factor ** 2 * _in_region(full, cal, sec)
    frame.dq[region] |= _in_region(cal.dq, cal, sec)


def _calibration_for(header, survey, params) -> dict:
    """The zero, dark, flat, illumination and fringe that apply to one frame."""
    subset = survey["instrument"].subset(header)

    def by_subset(key, label):
        given = getattr(params, key)
        found = survey[f"{key}s"]
        if given is not None or not found:
            return given
        chosen = found.get(subset)
        if chosen is None and len(found) == 1:
            return next(iter(found.values()))
        if chosen is None:
            raise ValueError(
                f"no {label} for subset '{subset}'. They were found for "
                + ", ".join(repr(s) for s in found)
                + f". Give --{key}, or check the instrument translation."
            )
        return chosen

    return {"zero": params.zero or survey["zero"],
            "dark": params.dark or survey["dark"],
            "flat": by_subset("flat", "flat field"),
            "illum": by_subset("illum", "illumination image"),
            "fringe": by_subset("fringe", "fringe image")}


def _peek(spec):
    """Read one header, without the pixels."""
    from ..core import images

    return images.read_header(spec)


def _already(header, flag: str) -> bool:
    keyword = MARKS[flag][0]
    return keyword in header


def _mark(header, flag: str) -> None:
    keyword, comment = MARKS[flag]
    from datetime import datetime

    header[keyword] = (datetime.now(UTC).isoformat(timespec="seconds"), comment)


def _fixpix(raw, data, bpm_path, header, sec, region):
    """xt_fps over the lines and columns of the data section, not past them, and the mask
    goes pixel by pixel from the raw image's first, or from the trimmed frame's if that
    is its size."""
    import numpy as np

    from ..core import images

    if _already(header, "fixpix"):
        return data, Step("fixpix", skipped=True, reason="FIXPIX already in the header"), None
    mask, _ = images.read(bpm_path)
    if mask.shape == raw.shape:
        mask = mask[sec.in_l1 - 1:sec.in_l2, sec.in_c1 - 1:sec.in_c2]
    else:
        mask = mask[region]
    bad = mask != 0
    if not bad.any():
        return data, Step("fixpix", "mask marks no bad pixels"), bad
    # IRAF's fixpix, the same code the fixpix task runs: across the narrowest
    # side of each bad region. (This used to interpolate along x always,
    # which is right for a bad column and wrong for a bad row.)
    from ..core import fixpix

    inner, summary = fixpix.fix(data[region], mask)
    out = data.copy()
    out[region] = np.asarray(inner, np.float32)
    _mark(header, "fixpix")
    return out, Step(
        "fixpix", f"{int(bad.sum())} pixels replaced", why=WHY["fixpix"],
        measured={"replaced": f"{int(bad.sum())} pixels, "
                              f"{100 * bad.mean():.3f}%",
                  "along lines": summary["along lines"],
                  "along columns": summary["along columns"]},
        before=Task.statistics(data), after=Task.statistics(out)), bad


def _overscan(data, header, params, sec, section):
    """setoverscan: (what to subtract from the frame's lines, in the shape that goes, and
    the step). The fit is IRAF's in real, to the last digit, or the panel's by hand."""
    import numpy as np

    from ..core import ccdsections, icfit, icfitbits

    if _already(header, "overscan"):
        return None, Step("overscan", skipped=True, reason="OVERSCAN already in the header")
    if not section:
        raise ValueError(
            "overscan was asked for but no section was given and the header has no BIASSEC. "
            "Pass --biassec '[2049:2080,*]'."
        )
    if sec.bias_c1 < 1 or sec.bias_c2 > sec.nc or sec.bias_l1 < 1 or sec.bias_l2 > sec.nl:
        raise ValueError(f"error in the bias section: the image is {sec.nc} by {sec.nl}, "
                         f"the section {section}.")
    if (sec.bias_c1, sec.bias_c2, sec.bias_l1, sec.bias_l2) == (1, sec.nc, 1, sec.nl):
        raise ValueError("bias section not specified or given as full image.")
    line_axis = params.readaxis == "line"
    if params.function in ("mean", "median", "minmax"):
        if not line_axis:
            raise ValueError(f"overscan function {params.function} is not allowed with "
                             "readaxis column.")
        strip = data[sec.in_l1 - 1:sec.in_l2, sec.bias_c1 - 1:sec.bias_c2]
        level = ccdsections.find_overscan(strip, params.function)
        _mark(header, "overscan")
        return level[:, None], Step(
            "overscan", f"{params.function} over {section}", why=WHY["overscan"],
            measured={"level": round(float(level.mean()), 2),
                      "drift": f"{float(level.min()):.2f} to {float(level.max()):.2f}"})

    x, values = ccdsections.overscan_vector(data, sec, line_axis)
    spec = icfit.FitSpec(
        function=params.function, order=params.order, sample=params.sample,
        naverage=params.naverage, niterate=params.niterate,
        low_reject=params.low_reject, high_reject=params.high_reject,
        grow=params.grow,
    )
    deleted = 0
    level = None
    if params.interactive:
        from ..viewer.icfit_panel import run as run_icfit

        result = run_icfit(x.astype(float), values.astype(float), spec,
                           title=f"ccdproc · overscan {section}",
                           xlabel="pixel", ylabel="overscan level (ADU)")
        spec = result.spec
        deleted = int(result.deleted.sum())
        if result.model is None:
            raise ValueError(f"the overscan fit failed: {result.message}")
        if deleted:
            level = np.asarray(result(x.astype(float)), np.float32)
    ic = icfitbits.Ic(function=spec.function, order=spec.order, sample=spec.sample,
                      naverage=spec.naverage, niterate=spec.niterate,
                      low_reject=spec.low_reject, high_reject=spec.high_reject, grow=spec.grow)
    weights = np.ones(x.size, np.float32)
    if level is None:
        curve = icfitbits.fit(ic, x, values, weights)
        level = np.asarray(curve.vector_at(x), np.float32)
    rms = float(np.sqrt(np.mean((values.astype(float) - level.astype(float)) ** 2)))
    nrejected = int(getattr(ic, "nreject", 0) or 0)
    _mark(header, "overscan")
    detail = (f"{spec.function} order {spec.order} over {section}, "
              f"mean level {float(level.mean()):.2f}, rms {rms:.2f}")
    if spec.sample != "*":
        detail += f", sample {spec.sample}"
    if deleted:
        detail += f", {deleted} points deleted by hand"
    if nrejected:
        detail += f", {nrejected} points rejected"
    return (level[:, None] if line_axis else level[None, :]), Step(
        "overscan", detail, why=WHY["overscan"],
        measured={"level": round(float(level.mean()), 2),
                  "drift": f"{float(level.min()):.2f} to {float(level.max()):.2f}",
                  "fit rms": round(rms, 3),
                  "rejected": f"{nrejected} of {x.size}"})


def _calibrate(data, header, key, image, params, sec, region, inst=None, flat=None):
    """set_zero, set_dark and set_flat, and each one's part of cor.gx. The scale of a dark
    and a flat in real, from the header's cards as IRAF reads them."""
    import numpy as np

    from ..core import ccdsections

    flag = {"zero": "zerocor", "dark": "darkcor", "flat": "flatcor"}[key]
    if _already(header, flag):
        return data, Step(key, skipped=True,
                          reason=f"{MARKS[flag][0]} already in the header"), None
    if image is None:
        raise ValueError(f"{flag} was asked for but no --{key} image was given")

    cal = image if isinstance(image, _Frame) else _Frame.load(image)
    name = Path(str(image)).name
    cut = ccdsections.cal_region(cal.data.shape, cal.header, sec, name)
    calibration = np.asarray(cal.data[cut], np.float32)
    cal_header = cal.header
    in_line = cal.data.shape[0] == 1
    in_column = cal.data.shape[1] == 1

    measured = {"image": name}
    out = data.copy()
    if key == "zero":
        out[region] = (data[region] - calibration).astype(np.float32)
        detail = f"subtracted {name}"
        measured["level removed"] = round(float(np.median(calibration)), 3)
    elif key == "dark":
        frame_time = _time(header, params, inst)
        dark_time = _time(cal_header, params, inst)
        if dark_time <= 0:
            raise ValueError(f"the dark {name} has a time of {float(dark_time):g}.")
        factor = np.float32(frame_time / dark_time)
        if in_line and params.readaxis == "line":
            # proc1's own: a dark of one line is scaled by the flat's scale, or by a
            # zero left in the structure with no flat.
            factor = _flat_scale(flat, params) if flat is not None else np.float32(0.0)
        out[region] = (data[region] - (factor * calibration).astype(np.float32)).astype(
            np.float32)
        cal.factor = float(factor)
        detail = f"subtracted {name} scaled by {float(factor):.3f}"
        measured["scale"] = (f"{float(frame_time):g}s / {float(dark_time):g}s = "
                             f"{float(factor):.3f}")
    else:
        # IRAF: out = data * scale / flat, the scale being the flat's CCDMEAN --
        # the mean, not the median, of the flat as ccdproc left it. A flat
        # without CCDMEAN is one IRAF would process as a flat before using it,
        # which replaces values below minreplace and then takes the mean; that
        # is done here. (harwood used to normalise by the median: 0.11% off on
        # the 1987 M92 flat, a constant 1.2 mmag in every magnitude.)
        floor = np.float32(params.minreplace)
        safe = np.where(calibration < floor, floor, calibration).astype(np.float32)
        source = "its CCDMEAN" if "CCDMEAN" in cal_header else "its mean"
        level = _flat_scale(cal, params)
        if in_column and params.readaxis == "column":
            # flatval = flatscale / flat[line]
            out[region] = (data[region] * (level / safe).astype(np.float32)).astype(np.float32)
        else:
            out[region] = ((data[region] * level).astype(np.float32) / safe).astype(np.float32)
        normalised = safe.astype(float) / float(level)
        cal.normalised, cal.level = normalised, float(level)
        cal.clamped = calibration < floor
        detail = f"divided by {name}, normalised to {source}"
        measured["flat level"] = round(float(level), 3)
        measured["response range"] = (f"{float(normalised.min()):.3f} to "
                                      f"{float(normalised.max()):.3f}")
        replaced = int((calibration < floor).sum())
        if replaced:
            measured["clamped"] = f"{replaced} pixels below {params.minreplace}"

    _mark(header, flag)
    return out, Step(key, detail, why=WHY[key], measured=measured,
                     before=Task.statistics(data), after=Task.statistics(out)), cal


def _flat_scale(cal, params):
    """FLATSCALE: the flat's CCDMEAN as read from its card, or the mean of what ccdproc
    would leave of it."""
    import numpy as np

    from ..core import ccdsections

    cal = cal if isinstance(cal, _Frame) else _Frame.load(cal)
    if "CCDMEAN" in cal.header:
        return _hdm_real(cal.header, "CCDMEAN")
    floor = np.float32(params.minreplace)
    safe = np.where(np.asarray(cal.data, np.float32) < floor, floor,
                    np.asarray(cal.data, np.float32)).astype(np.float32)
    return np.float32(_real_card(ccdsections.mean_of(safe)))


def _time(header, params, inst):
    """set_dark: the dark time if there is one, or the exposure time."""
    if "DARKTIME" in header:
        return _hdm_real(header, "DARKTIME")
    return _hdm_real(header, params.exposure_key)


class SetInstrumentParams(TaskParams):
    instrument: str | None = Field(
        None,
        description="Translation file to use: a name like 'kpno/direct', an IRAF "
                    "path like 'ccddb$kpno/direct.dat', or a file of your own. "
                    "Leave it out to see what is installed.",
        json_schema_extra={"positional": True},
    )
    site: str = Field("kpno", description="Observatory directory of ccddb$ an "
                                          "instrument name without one is looked for in.")
    review: bool = Field(
        True, description="Show every setting made, as IRAF opens ccdred and ccdproc "
                          "for review.")
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
                    "no instrument translations found, not even harwood's own "
                    "copy of IRAF's (data/iraf/ccddb): the installation is "
                    "incomplete. Reinstall harwood, or write a file of your own -- "
                    "two columns, 'parameter keyword' or 'HEADER-VALUE imagetype'."
                )
            if not params.instrument:
                return TaskResult(data={"rows": rows},
                                  warnings=["Name one to set it, as in "
                                            "`harwood setinstrument kpno/direct`."],
                                  steps=self.plan(params))
            return TaskResult(data={"rows": rows}, steps=self.plan(params))

        name = str(params.instrument)
        if "/" not in name and "$" not in name and params.site and \
                not Path(name).exists():
            name = f"{params.site}/{name}"
        chosen = instruments.load(name)

        import yaml

        from ..core import params as store

        project = store.find_project(params.workdir) or \
            (params.workdir / store.PROJECT_FILE)
        document = {}
        if project.exists():
            document = yaml.safe_load(project.read_text(encoding="utf-8")) or {}
        document["instrument"] = name
        applied, skipped = _apply_settings(instruments.settings(name), document)
        header = _project_header(project)
        project.write_text(header + yaml.safe_dump(document, sort_keys=False,
                                                   allow_unicode=False),
                           encoding="utf-8")

        data: dict = {"instrument": chosen.name, "source": chosen.source,
                      "keywords": len(chosen.keywords), "types": len(chosen.types),
                      "settings": len(applied)}
        if params.review and (applied or skipped):
            data["rows"] = applied + skipped
        warnings = []
        if store.in_home(project):
            message, fix = store.HOME_NOTE
            warnings.append(advice.note(message, fix))
        if skipped:
            warnings.append(advice.note(
                f"{len(skipped)} of the instrument's settings have no harwood "
                "counterpart and were not applied; the table says why",
                "nothing, usually: they are IRAF's package and display settings"))
        return TaskResult(
            outputs=[project],
            data=data, warnings=warnings,
            history=[f"setinstrument: {name}"],
            steps=self.plan(params),
        )


#: ccdred's own tasks under harwood's names.
_TASKS = {"combine": "imcombine"}
#: ccdred package settings harwood keeps elsewhere, and where.
_PACKAGE = {
    "pixeltype": "harwood computes in double and writes the input's type",
    "verbose": "harwood always reports what it did",
    "logfile": "harwood.log.jsonl is the log",
    "plotfile": "plots are saved from the window",
    "backup": "harwood writes new files and keeps the input",
    "ssfile": "subsets are read from the instrument translation",
    "graphics": "the window is harwood's own",
    "cursor": "the cursor is the window's",
}


def _apply_settings(found, document) -> tuple[list[dict], list[dict]]:
    """Each 'task.param = value' of the .cl file into the project's params,
    when harwood has that task and parameter and the value is one it takes."""
    from pydantic import TypeAdapter, ValidationError

    from ..core.registry import exists, get_class

    params = document.setdefault("params", {}) or {}
    document["params"] = params
    applied, skipped = [], []
    for task, parameter, text in found:
        if task == "ccdred" and parameter == "instrument":
            continue
        row = {"setting": f"{task}.{parameter}", "value": text}
        if task == "ccdred":
            skipped.append({**row, "applied": "no -- " + _PACKAGE.get(
                parameter, "a package setting harwood does not have")})
            continue
        target = _TASKS.get(task, task)
        if not exists(target):
            skipped.append({**row, "applied": f"no -- harwood has no {task}"})
            continue
        field = get_class(target).Params.model_fields.get(parameter)
        if field is None:
            skipped.append({**row, "applied": f"no -- {target} has no {parameter} "
                                              "(not ported)"})
            continue
        if parameter in ("biassec", "trimsec") and text in ("image", ""):
            # IRAF's "image": from the header -- harwood's own default.
            params.get(target, {}).pop(parameter, None)
            applied.append({**row, "applied": "yes -- read from the header, "
                                              "harwood's default"})
            continue
        value = _cl_value(text)
        try:
            value = TypeAdapter(field.annotation).validate_python(value)
        except ValidationError:
            skipped.append({**row, "applied": f"no -- not a value {target} --{parameter} "
                                              "takes (a header keyword, in IRAF)"})
            continue
        params.setdefault(target, {})[parameter] = value
        applied.append({**row, "applied": f"yes -- {target}"})
    return applied, skipped


def _cl_value(text: str):
    if text in ("yes", "no"):
        return text == "yes"
    for kind in (int, float):
        try:
            return kind(text)
        except ValueError:
            pass
    return text


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
                header = images.inherited(hdul, hdu)
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


#: ccdred's image types (ccdtypes.h), in IRAF's order.
CCDTYPES = ("object", "zero", "dark", "flat", "illum", "fringe", "other", "comp")


def _ccdtype_name(inst, header) -> str:
    """ccdtypes: the image type as ccdred names it -- 'none' when the header
    has no type at all, 'unknown' when the translation does not know it."""
    raw = inst.value(header, "imagetyp")
    if raw in (None, "") or not str(raw).strip():
        return "none"
    text = str(raw).strip()
    name = inst.types.get(text.upper(), text)
    return name if name in CCDTYPES else "unknown"


def _separation(ra, dec, ra0, dec0):
    """position_group's distance in arcseconds, in IRAF's single precision:
    RA in hours (wrapped at 12), declination in degrees."""
    import math

    import numpy as np

    f32 = np.float32
    dra, ddec = f32(ra - ra0), f32(dec - dec0)
    if dra > 12:
        dra = f32(dra - 24)
    if dra < -12:
        dra = f32(dra + 24)
    dra = f32(f32(dra * f32(math.cos(math.radians(float(dec))))) * 15)
    return f32(math.sqrt(float(f32(dra * dra + ddec * ddec))) * 3600)


def _sexagesimal(value) -> float:
    """A header number IRAF's way: '17:17:07.5' is 17.2854..., as ctod reads it."""
    if isinstance(value, int | float):
        return float(value)
    text = str(value).strip()
    sign = -1.0 if text.startswith("-") else 1.0
    parts = [float(p) for p in text.lstrip("+-").replace(" ", ":").split(":") if p]
    if not parts:
        raise ValueError(f"'{value}' is not a number")
    total = 0.0
    for scale, part in zip((1.0, 60.0, 3600.0), parts, strict=False):
        total += part / scale
    return sign * total


class CcdGroupsParams(TaskParams):
    images: list[str] = Field(..., description="CCD images to group. Globs and @lists.")
    output: str = Field(..., description="Root of the group files: the images of group "
                        "1 go to <output>1, and so on, or <output>object, "
                        "<output>flat... when grouping by ccdtype.")
    group: Literal["position", "title", "date", "ccdtype", "subset"] = Field(
        "ccdtype", description="What a group shares: the pointing (within radius), "
                               "the title (OBJECT), the date (DATE-OBS without the "
                               "time), the image type, or the subset (filter).")
    radius: float = Field(60.0, gt=0, description="For group=position: images closer "
                          "than this, in arcseconds, to a group's first image join it.")
    ccdtype: str = Field("", description="Only images of this type: object, zero, "
                         "dark, flat, illum, fringe, other, comp, none, unknown.")
    instrument: str | None = Field(
        None, description="Translation to use. Taken from the project file when not given.")


@register
class CcdGroups(Task[CcdGroupsParams]):
    """Sort a night into lists, as IRAF's ccdgroups does: one file per group,
    each holding the names of its images, ready for @list input.

    A group is numbered in the order its first image is met. Group files are
    appended to, as in IRAF: run it twice and every name is listed twice --
    delete the files to start over (harwood warns when it finds them)."""

    name = "ccdgroups"
    package = "ccdred"
    description = "Group CCD images into lists by position, title, date, type or subset."
    Params = CcdGroupsParams

    def plan(self, params: CcdGroupsParams) -> list[Step]:
        what = {"position": f"pointing within {params.radius:g} arcsec",
                "title": "title (OBJECT)", "date": "date of observation",
                "ccdtype": "image type", "subset": "subset"}[params.group]
        select = f", {params.ccdtype} images only" if params.ccdtype.strip() else ""
        return [Step("group", f"by {what}{select}",
                     why="A night is easier to reduce as lists: the flats of one "
                         "filter, the frames of one field, one date at a time."),
                Step("write", f"{params.output}<group>",
                     why="Each file lists one group's images, one per line, for "
                         "any task's @list input.")]

    def execute(self, params: CcdGroupsParams) -> TaskResult:
        import numpy as np

        from ..core import images
        from ..core import instrument as instruments
        from ..core import subsets as subset_file

        f32 = np.float32
        inst = (instruments.load(params.instrument) if params.instrument
                else instruments.from_project())
        wanted = params.ccdtype.strip()
        groups: list = []
        assignments: list[tuple[str, str]] = []
        warnings = []
        existing = set()
        for spec in images.expand(params.images):
            name = str(spec)
            header = images.read_header(spec)
            if wanted and _ccdtype_name(inst, header) != wanted:
                continue
            try:
                if params.group == "position":
                    ra = f32(_sexagesimal(header[inst.keyword("ra")]))
                    dec = f32(_sexagesimal(header[inst.keyword("dec")]))
                    number = next((n for n, (ra0, dec0) in enumerate(groups, 1)
                                   if _separation(ra, dec, ra0, dec0) < params.radius),
                                  None)
                    if number is None:
                        groups.append((ra, dec))
                        number = len(groups)
                    suffix = str(number)
                elif params.group in ("title", "date"):
                    if params.group == "title":
                        key = str(header.get("OBJECT", "")).strip()
                    else:
                        key = str(header[inst.keyword("date-obs")]).strip().split("T")[0]
                    if key not in groups:
                        groups.append(key)
                    suffix = str(groups.index(key) + 1)
                elif params.group == "ccdtype":
                    suffix = _ccdtype_name(inst, header)
                else:
                    suffix = subset_file.name_for(inst.subset(header))
            except (KeyError, ValueError) as error:
                warnings.append(advice.warn(
                    f"{name}: {error} -- not grouped",
                    "check the header with 'harwood imheader', or the keyword "
                    "names in the instrument translation"))
                continue
            target = Path(f"{params.output}{suffix}")
            if target.exists() and target not in existing:
                warnings.append(advice.warn(
                    f"{target} already exists; the names are added to it, as IRAF "
                    "does", f"delete {target} first to start the list afresh"))
            existing.add(target)
            with target.open("a", encoding="utf-8") as handle:
                handle.write(name + "\n")
            assignments.append((name, str(target)))
        rows = [{"image": image, "group": target} for image, target in assignments]
        outputs = sorted({Path(target) for _, target in assignments})
        return TaskResult(outputs=outputs, data={"rows": rows}, warnings=warnings,
                          steps=self.plan(params))


# -------------------------------------------------------------- badpiximage

class BadPixImageParams(TaskParams):
    fixfile: str = Field(..., description="The bad pixel file: column line, or c1 c2 l1 l2.",
                         json_schema_extra={"positional": True})
    template: str = Field(..., description="The image that gives the size and the header.",
                          json_schema_extra={"positional": True})
    image: str = Field(..., description="The image to make; .pl makes a mask file.",
                       json_schema_extra={"positional": True})
    goodvalue: int = Field(1, description="The value of a good pixel.")
    badvalue: int = Field(0, description="The value of a bad pixel.")


@register
class BadPixImage(Task[BadPixImageParams]):
    """IRAF's badpiximage: an image of the template's size from a bad pixel file, with
    its sense of good and bad: 1 is good, unlike a mask. A comment with the word
    untrimmed says the coordinates are of the frame before its trim."""

    name = "badpiximage"
    package = "ccdred"
    description = "An image of the bad pixels from a bad pixel file."
    Params = BadPixImageParams

    def plan(self, params: BadPixImageParams) -> list[Step]:
        return [Step("write", f"{params.fixfile} -> {params.image}",
                     why="To see the bad pixels against the data, blink one against the "
                         "other, or to carry them through arithmetic.")]

    def execute(self, params: BadPixImageParams) -> TaskResult:
        import re

        import numpy as np
        from astropy.io import fits

        from ..core import images, plio, provenance
        from .proto import _region, scan_ints

        file = Path(params.fixfile)
        if not file.is_file():
            advice.fail(f"Cannot open file ({params.fixfile})",
                        "check the bad pixel file's name")
        header = images.read_header(params.template)
        nc, nl = int(header["NAXIS1"]), int(header["NAXIS2"])
        pixels = np.full((nl, nc), params.goodvalue, dtype=np.int16)
        x1, xstep, y1, ystep = 1, 1, 1, 1
        for line in file.read_text(errors="replace").splitlines():
            words = line.split()
            if words and words[0].startswith("#"):
                if "untrimmed" in line.lower() and "TRIM" in header:
                    found = re.search(r"\[(.*?)\]", str(header["TRIM"]))
                    if found:
                        x1, xstep, y1, ystep = _section(found[1], nc, nl)
                continue
            region = _region(scan_ints(line))
            if region is None:
                continue
            c1 = max(1, _idiv(region[0] - x1 + xstep - 1, xstep) + 1)
            c2 = min(nc, _idiv(region[1] - x1, xstep) + 1)
            l1 = max(1, _idiv(region[2] - y1 + ystep - 1, ystep) + 1)
            l2 = min(nl, _idiv(region[3] - y1, ystep) + 1)
            if c1 > c2 or l1 > l2:
                continue
            pixels[l1 - 1:l2, c1 - 1:c2] = params.badvalue
        out = images.forget_limits(header.copy())
        out["OBJECT"] = f"Bad pixel image from bad pixel file {params.fixfile}"
        target = images.parse(params.image).path
        if target.suffix == ".pl":
            written = plio.write(target, pixels.astype(np.int64), header=out)
        else:
            provenance.stamp_header(out, [f"badpiximage: from {params.fixfile}"])
            fits.PrimaryHDU(pixels, out).writeto(target, overwrite=True)
            written = target
        return TaskResult(data={"bad": int((pixels == params.badvalue).sum())},
                          outputs=[written], steps=self.plan(params))


def _idiv(a: int, b: int) -> int:
    """Fortran's integer division, toward zero."""
    return int(a / b)


def _section(text: str, nc: int, nl: int) -> tuple[int, int, int, int]:
    """ccd_section, of what it takes here: first and step along x and y."""
    out = []
    for part in text.split(",")[:2]:
        pieces = part.strip().split(":")
        first = 1 if pieces[0] in ("", "*") else int(pieces[0])
        step = int(pieces[2]) if len(pieces) > 2 else 1
        out += [first, step]
    while len(out) < 4:
        out += [1, 1]
    return out[0], out[1], out[2], out[3]


# ------------------------------------------------------------------ ccdmask

class CcdMaskParams(TaskParams):
    image: str = Field(..., description="A flat field, or better the ratio of two at "
                                        "different exposure levels.",
                       json_schema_extra={"positional": True})
    mask: str = Field(..., description="The mask to make; .pl is added if not there.",
                      json_schema_extra={"positional": True})
    ncmed: int = Field(7, ge=1, description="Columns of the moving median.")
    nlmed: int = Field(7, ge=1, description="Lines of the moving median.")
    ncsig: int = Field(15, ge=10, description="Columns of the box the sigma is measured in.")
    nlsig: int = Field(15, ge=10, description="Lines of the box the sigma is measured in.")
    lsigma: float = Field(6.0, description="How many sigma below the median is bad.")
    hsigma: float = Field(6.0, description="How many sigma above the median is bad.")
    ngood: int = Field(5, ge=1, description="A good stretch of a column shorter than this "
                                            "is bad too.")
    linterp: int = Field(2, ge=1, description="Value where the run is narrower along lines.")
    cinterp: int = Field(3, ge=1, description="Value where it is narrower along columns.")
    eqinterp: int = Field(2, ge=1, description="Value where they are equal.")


@register
class CcdMask(Task[CcdMaskParams]):
    """IRAF's ccdmask: a mask of the pixels that deviate from a local median, with a
    search along columns for traps too weak to show pixel by pixel."""

    name = "ccdmask"
    package = "ccdred"
    description = "A bad pixel mask from a flat field or a ratio of flats."
    Params = CcdMaskParams

    def plan(self, params: CcdMaskParams) -> list[Step]:
        return [Step("measure", f"{params.image}: median {params.ncmed}x{params.nlmed}, "
                                f"sigma in {params.ncsig}x{params.nlsig}",
                     why="A ratio of two flats at different levels leaves only what does "
                         "not flat field, and what stands out from it is really bad."),
                Step("write", params.mask, why="A mask, to give to fixpix or ccdproc.")]

    def execute(self, params: CcdMaskParams) -> TaskResult:
        import numpy as np

        from ..core import images, plio
        from ..core.ccdmask import ccdmask

        data, header = images.read(params.image)
        if data.ndim != 2:
            advice.fail(f"{params.image} is not two-dimensional",
                        "give a frame, or a section of one")
        if data.shape[1] <= params.ncsig or data.shape[0] < params.nlsig:
            advice.fail(f"{params.image} is smaller than the box for the sigma, "
                        f"{params.ncsig}x{params.nlsig}",
                        "use a larger frame, or a smaller --ncsig and --nlsig")
        mask = ccdmask(data, params.ncmed, params.nlmed, params.ncsig, params.nlsig,
                       params.lsigma, params.hsigma, params.ngood, params.linterp,
                       params.cinterp, params.eqinterp)
        written = plio.write(params.mask, mask.astype(np.int64), header=header)
        return TaskResult(data={"bad": int((mask != 0).sum()), "pixels": int(mask.size)},
                          outputs=[written], steps=self.plan(params))


class CcdInstrumentParams(TaskParams):
    images: list[str] = Field(
        ..., description="The images to try the translations on.",
        json_schema_extra={"positional": True},
    )
    instrument: str = Field(
        "", description="The translation file, or the one in harwood.yaml if not given."
    )
    ssfile: str = Field("subsets", description="The record of subset identifiers, added to.")
    edit: bool = Field(True, description="Take commands; without it, only the table.")
    parameters: Literal["basic", "common", "all"] = Field(
        "basic", description="How many of the parameters the table has."
    )


@register
class CcdInstrument(Task[CcdInstrumentParams]):
    """IRAF's ccdinstrument: what an instrument file makes of an image's header, a table
    of parameter, keyword, default and values, and the commands to change it."""

    name = "ccdinstrument"
    package = "ccdred"
    description = "Review and edit instrument translation files."
    Params = CcdInstrumentParams

    def plan(self, params: CcdInstrumentParams) -> list[Step]:
        return [
            Step(
                "show",
                f"the translations of {params.instrument or 'harwood.yaml'} for "
                f"{', '.join(params.images)}",
                why="A wrong keyword here is a frame processed as what it is not.",
            )
        ]

    def execute(self, params: CcdInstrumentParams) -> TaskResult:
        import sys

        from ..core import ccdinstsession, images
        from ..core import params as store

        name = params.instrument
        if not name:
            document, _ = store.load_project(None)
            name = str(document.get("instrument") or "")
        table: list[str] = []

        def read():
            line = sys.stdin.readline()
            return line.rstrip("\n") if line else None

        def live(text):
            sys.stdout.write(text)
            sys.stdout.flush()

        def err(text):
            sys.stderr.write(text)
            sys.stderr.flush()

        write = live if params.edit else table.append
        session = ccdinstsession.Session(
            name, params.ssfile, params.parameters, read, write, err
        )
        for spec in images.expand(params.images):
            label = str(spec)
            header = ccdinstsession.open_image(label)
            if header is None:
                err(f"Warning: Cannot open image ({label})\n")
                continue
            session.image, session.header = label, header
            if params.edit:
                session.edit(label, header)
            else:
                session.show()
            if session.ncmd == "quit":
                break
        session.finish()
        stdout = "".join(table).split("\n")[:-1] if table else []
        return TaskResult(data={"instrument": name}, stdout=stdout, steps=self.plan(params))
