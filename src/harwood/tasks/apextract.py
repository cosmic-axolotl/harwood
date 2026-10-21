"""apextract -- turning a two-dimensional frame into one-dimensional spectra.

`apall` is the task people actually run, and it is four things in a trench
coat: find the spectra across the slit, size the apertures, follow each one
along the dispersion, and add it up. IRAF splits them into `apfind`,
`apresize`, `aptrace` and `apsum` as well, and keeps the result in
`database/ap<image>` so the next frame can reuse it.

harwood writes the extracted spectra as one FITS file with a table per
aperture rather than as multispec: every number is in the file, and
`obj.fits[SPEC,2]` addresses the second aperture with the syntax harwood
already understands. See core/spectrum.py.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import Field

from ..core import advice
from ..core.registry import register
from ..core.task import Step, Task, TaskParams, TaskResult

WHY = {
    "read": "A long-slit or fibre frame holds the spectrum running one way and "
            "the slit the other. Which axis is which is the first thing to get "
            "right: everything downstream is transposed if it is wrong, and the "
            "symptom is a trace that cannot be followed.",
    "profile": "Collapsing along the dispersion is what makes a faint spectrum "
               "findable -- a thousand rows of a few counts each become a peak "
               "you can centroid. It only works while the spectrum is straight "
               "enough that adding the whole frame does not smear it, which is "
               "what nsum limits.",
    "reference": "An arc has no star in it to find, and a faint exposure may not "
                 "show its star well enough to trace. Both are extracted with the "
                 "apertures of a frame that does -- the same centre, limits and "
                 "trace -- so the arc's lines are measured on exactly the pixels "
                 "the star's light was summed from.",
    "recenter": "The reference's apertures, moved to where this frame's profile "
                "peaks: by default all by the median of their moves, so a small "
                "drift of the star on the slit is followed and the apertures keep "
                "their spacing.",
    "find": "The strongest peaks in the spatial profile, none closer than minsep "
            "to a stronger one, each centred with center1d. One for a star on a "
            "long slit, many for fibres or an echelle.",
    "edit": "The finder is right most of the time, and the times it is not are "
            "the ones that matter: two stars it merged, a galaxy reaching past "
            "the threshold, a sky band landing on a neighbour. You have to look "
            "at the profile, so this shows it.",
    "resize": "The aperture is set where the profile, above a straight background "
              "between the minima either side, falls to a fraction of its peak -- "
              "a fibre and a long slit want very different apertures and the "
              "profile knows which it is looking at.",
    "trace": "The spectrum does not sit on one column. The camera has "
             "distortion and the spectrograph flexes as the telescope tracks, "
             "so the centre wanders by a pixel or several from one end of the "
             "dispersion to the other. Following it and fitting a low-order "
             "curve is what keeps the aperture on the light.",
    "background": "Sky is measured in the sample regions either side of the "
                  "aperture, fitted across the slit at every line, and the fit "
                  "subtracted under the aperture. The gap between aperture and "
                  "sample matters: the wings of the profile reach further than "
                  "the aperture does, and sky measured on them includes the star.",
    "extract:none": "A plain sum of the pixels inside the aperture, the end pixels "
                    "by the fraction of their light inside the edge. Every pixel "
                    "counts the same, which is right when they all have similar "
                    "signal and wrong when the edges are mostly noise. There is no "
                    "sigma: IRAF only has one with variance weights.",
    "extract:variance": "Optimal extraction (Horne 1986): the profile is fitted "
                        "along the dispersion, each pixel weighted by how much of "
                        "it it holds and divided by its variance, so the noisy "
                        "edges stop diluting the bright middle; then the whole "
                        "spectrum is scaled to the unweighted total, so only the "
                        "noise changes. Cleaning replaces pixels the profile "
                        "says are cosmic rays.",
    "write": "One FITS file, one binary table per aperture, with the pixel, the "
             "flux, the uncertainty and the sky in it. Nothing has to be "
             "recomputed to read it, and `wspectext` turns it into text without "
             "knowing anything the file does not say.",
}


FUNCTIONS = Literal["legendre", "chebyshev", "spline1", "spline3"]


class ApAllParams(TaskParams):
    input: str = Field(..., description="Two-dimensional frame to extract from.")
    output: Path | None = Field(
        None,
        description="Where to write the spectra. Left out, it goes to this run's "
                    "own directory under output/, named after the frame.",
        json_schema_extra={"positional": True},
    )
    dispaxis: Literal[1, 2] | None = Field(
        None,
        description="1 if the dispersion runs along the image lines (x), 2 if "
                    "along the columns (y). Left out, the DISPAXIS keyword, else 2.",
    )

    references: str = Field(
        "", description="Frame whose apertures (database/ap<frame>) are used instead "
                        "of finding them -- as for an arc, extracted with its star's. "
                        "Empty: this frame's own, if it has any. OLD: only a frame "
                        "that has its own; NEW<frame>: only one that does not.")
    find: bool = Field(True, description="Find the apertures, when there are none?")
    recenter: bool = Field(
        True, description="Recenter apertures taken from the database on this frame?")
    aprecenter: str = Field(
        "", description="Apertures used to work out the recentering (a range list; "
                        "empty is all).")
    npeaks: float | None = Field(
        None, ge=0, description="Recenter on only the brightest this many (below 1, "
                                "this fraction); None is all.")
    shift: bool = Field(
        True, description="Move every aperture by the median recentering rather "
                          "than each by its own.")
    resize: bool = Field(True, description="Resize the apertures to the profile?")
    edit: bool = Field(
        False,
        description="Adjust the apertures by hand before extracting, on the "
                    "profile they came from (IRAF's apedit).",
    )
    trace: bool = Field(True, description="Trace the apertures along the dispersion?")
    fittrace: bool = Field(False, description="Fit the traced points by hand?")
    extract: bool = Field(True, description="Extract the spectra?")

    line: int | None = Field(
        None, ge=1, description="Dispersion line the apertures are found, sized and "
                                "traced from. Left out, the middle.")
    nsum: int = Field(
        10, description="Dispersion lines summed for the profile; negative, their "
                        "median.")
    lower: float = Field(-5.0, description="Lower aperture limit relative to the centre.")
    upper: float = Field(5.0, description="Upper aperture limit relative to the centre.")

    b_function: FUNCTIONS = Field("chebyshev", description="Background function.")
    b_order: int = Field(1, ge=1, description="Background function order.")
    b_sample: str = Field(
        "-10:-6,6:10", description="Background regions, relative to the aperture "
                                   "centre.")
    b_naverage: int = Field(
        -3, description="Background points averaged (negative: median) before fitting.")
    b_niterate: int = Field(0, ge=0, description="Background rejection iterations.")
    b_low_reject: float = Field(3.0, ge=0, description="Background lower rejection sigma.")
    b_high_reject: float = Field(3.0, ge=0, description="Background upper rejection sigma.")
    b_grow: float = Field(0.0, ge=0, description="Background rejection growing radius.")

    width: float = Field(5.0, ge=0, description="Profile centring width.")
    radius: float = Field(10.0, description="Profile centring radius.")
    threshold: float = Field(
        0.0, ge=0, description="Least range of data a profile must span to be centred.")

    nfind: int = Field(
        1, description="How many spectra to find; negative divides the slit evenly.")
    minsep: float = Field(5.0, ge=1, description="Least separation between spectra.")
    order: Literal["increasing", "decreasing"] = Field(
        "increasing", description="Order the apertures are numbered in.")

    llimit: float | None = Field(None, description="Lowest limit resizing may set.")
    ulimit: float | None = Field(None, description="Highest limit resizing may set.")
    ylevel: float | None = Field(
        0.1, description="Fraction of the peak (or level) the aperture is resized to. "
                         "None sets llimit and ulimit.")
    peak: bool = Field(True, description="Is ylevel a fraction of the peak?")
    bkg: bool = Field(True, description="Take off a background before resizing?")
    r_grow: float = Field(0.0, description="Grow the limits by this fraction.")
    avglimits: bool = Field(False, description="Give every aperture the average limits?")

    t_nsum: int = Field(10, ge=1, description="Dispersion lines summed at each trace step.")
    t_step: int = Field(10, ge=1, description="Tracing step.")
    t_nlost: int = Field(3, ge=1, description="Steps lost in a row before stopping.")
    t_function: FUNCTIONS = Field("legendre", description="Trace fitting function.")
    t_order: int = Field(2, ge=1, description="Trace fitting function order.")
    t_sample: str = Field("*", description="Trace sample regions.")
    t_naverage: int = Field(1, description="Trace points averaged (negative: median).")
    t_niterate: int = Field(0, ge=0, description="Trace rejection iterations.")
    t_low_reject: float = Field(3.0, ge=0, description="Trace lower rejection sigma.")
    t_high_reject: float = Field(3.0, ge=0, description="Trace upper rejection sigma.")
    t_grow: float = Field(0.0, ge=0, description="Trace rejection growing radius.")

    background: Literal["none", "average", "median", "minimum", "fit"] = Field(
        "none", description="Background to subtract.")
    skybox: int = Field(1, ge=1, description="Box-car smoothing length for the sky.")
    weights: Literal["none", "variance"] = Field(
        "none", description="Extraction weights. 'variance' is optimal extraction, and "
                            "the only one that gives a sigma.")
    pfit: Literal["fit1d"] = Field(
        "fit1d", description="Profile fitting for variance weights (fit2d is not ported).")
    clean: bool = Field(False, description="Detect and replace bad pixels? Implies "
                                           "variance weights.")
    saturation: float | None = Field(None, description="Saturation level.")
    readnoise: float | str = Field(0.0, description="Read noise in electrons, or the header "
                                             "keyword holding it.")
    gain: float | str = Field(1.0, description="Gain in electrons per ADU, or the header "
                                        "keyword holding it.")
    lsigma: float = Field(4.0, ge=0, description="Lower rejection threshold for cleaning.")
    usigma: float = Field(4.0, ge=0, description="Upper rejection threshold for cleaning.")
    nclean: float = Field(0.5, description="Most pixels cleaned per line; below 1, a "
                                           "fraction of the aperture.")
    niterate: int = Field(5, ge=0, description="Profile fitting iterations when cleaning.")

    database: Path = Field(Path("."), description="Reduction directory holding database/.")


@register
class ApAll(Task[ApAllParams]):
    name = "apall"
    package = "apextract"
    description = "Find, trace and extract spectra from a 2-D frame."
    Params = ApAllParams

    def plan(self, params: ApAllParams) -> list[Step]:
        steps = [
            Step("read", f"{params.input}, DISPAXIS "
                         f"{params.dispaxis or 'from the header'}", why=WHY["read"]),
            Step("profile", f"{'sum' if params.nsum > 0 else 'median'} of "
                            f"{abs(params.nsum)} lines", why=WHY["profile"]),
        ]
        stored = _stored_name(params)
        if stored:
            steps.append(Step("reference", f"the apertures of database/ap{stored}",
                              why=WHY["reference"]))
            if params.recenter:
                steps.append(Step("recenter", "all by the median move" if params.shift
                                  else "each on its own peak", why=WHY["recenter"]))
        elif params.find:
            steps.append(Step("find", f"up to {params.nfind}, at least {params.minsep:g} "
                                      f"px apart, centred over {params.width:g} px",
                              why=WHY["find"]))
        if params.resize:
            steps.append(Step("resize", f"to {params.ylevel} of the "
                                        f"{'peak' if params.peak else 'level'}"
                                        if params.ylevel is not None
                                        else f"{params.llimit} to {params.ulimit}",
                              why=WHY["resize"]))
        if params.edit:
            steps.append(Step("edit", "by hand, on the spatial profile", why=WHY["edit"]))
        if params.trace:
            steps.append(Step("trace", f"every {params.t_step} lines, {params.t_function} "
                                       f"order {params.t_order}"
                                       + (", by hand" if params.fittrace else ""),
                              why=WHY["trace"]))
        if params.background != "none":
            steps.append(Step("background", f"{params.background}, sample "
                                            f"{params.b_sample}", why=WHY["background"]))
        if params.extract:
            steps.append(Step("extract", "variance" if params.clean else params.weights,
                              why=WHY["extract:variance" if params.clean
                                      else f"extract:{params.weights}"]))
            steps.append(Step("write", str(params.output or "this run's directory"),
                              why=WHY["write"]))
        return steps

    def execute(self, params: ApAllParams) -> TaskResult:
        import numpy as np

        from ..core import apextract, database, icfit, images, output, spectrum

        data, header = images.read(params.input)
        if data.ndim != 2:
            advice.fail(
                f"'{params.input}' is {data.ndim}-dimensional; apall extracts "
                "spectra from a 2-D frame.",
                "give a single frame, or a section of one, as in "
                "'name.fits[*,*,1]'.")
        dispaxis = params.dispaxis or int(header.get("DISPAXIS", 2))
        if dispaxis not in (1, 2):
            advice.fail(f"DISPAXIS is {dispaxis}; it must be 1 or 2.",
                        "give --dispaxis 1 or 2")
        frame = apextract.frame(data, dispaxis)
        steps = self.plan(params)
        by_action = {step.action: step for step in steps}

        profile, line = apextract.getdata(frame, params.line, params.nsum)
        by_action["profile"].measured = {"line": line, "across": f"{profile.size} px"}
        background = icfit.FitSpec(
            function=params.b_function, order=params.b_order, sample=params.b_sample,
            naverage=params.b_naverage, niterate=params.b_niterate,
            low_reject=params.b_low_reject, high_reject=params.b_high_reject,
            grow=params.b_grow)

        # t_apall: the reference's apertures, or this frame's own; found only
        # when there are none.
        apertures = _apertures_from_database(params, frame.shape[0])
        recentred = None
        if apertures:
            if params.recenter:
                moved, delta = apextract.recenter(
                    apertures, profile, line, params.width, params.radius,
                    params.threshold, params.aprecenter, params.npeaks, params.shift)
                # ap_recenter's log line, as IRAF words it.
                recentred = (f"RECENTER  - {moved} apertures shifted by {delta:.2f} for "
                             f"{params.input}." if delta is not None else
                             f"RECENTER - {moved} apertures recentered for {params.input}")
                if "recenter" in by_action:
                    by_action["recenter"].measured = {"moved": recentred}
            if "reference" in by_action:
                by_action["reference"].measured = {
                    "apertures": len(apertures),
                    "centres": ", ".join(f"{a.centre:.2f}" for a in apertures[:6])}
        elif params.find:
            centres = apextract.find(profile, params.nfind, params.minsep, params.width,
                                     params.radius, params.threshold, params.order)
            if "find" in by_action:
                by_action["find"].measured = {
                    "found": len(centres),
                    "at": ", ".join(f"{c:.2f}" for c in centres[:6])}
            apertures = [apextract.Aperture(number, centre, params.lower, params.upper,
                                            float(line), background=background)
                         for number, centre in enumerate(centres, start=1)]
        if not apertures and not params.edit:
            advice.fail(
                f"no spectrum found in {params.input}.",
                "check --dispaxis: with the axes the wrong way round the profile is "
                "the spectrum rather than the slit; or set --edit and mark them; "
                "or give --references a frame whose apertures to use.",
                RuntimeError)
        if params.resize and apertures:
            apextract.resize(apertures, profile, line, params.llimit, params.ulimit,
                             params.ylevel, params.peak, params.bkg, params.r_grow,
                             params.avglimits)
            by_action["resize"].measured = {
                "limits": ", ".join(f"{a.low:+.2f} {a.high:+.2f}" for a in apertures[:4])}

        if params.edit:
            apertures, background = self._edit(params, profile, apertures, background, line)

        traced = []
        if params.trace:
            spec = icfit.FitSpec(function=params.t_function, order=params.t_order,
                                 sample=params.t_sample, naverage=params.t_naverage,
                                 niterate=params.t_niterate,
                                 low_reject=params.t_low_reject,
                                 high_reject=params.t_high_reject, grow=params.t_grow)
            traced = apextract.trace(data, dispaxis, apertures, line, params.t_step,
                                     params.t_nsum, params.t_nlost, params.width,
                                     params.radius, params.threshold, spec)
            if params.fittrace:
                self._fit_by_hand(params, apertures, traced, frame.shape[0], spec)
            short = [a.number for a, t in zip(apertures, traced, strict=True)
                     if t.positions.size < 2]
            if short:
                advice.fail(f"not enough points traced for aperture(s) "
                            f"{', '.join(map(str, short))}.",
                            "raise --t-nsum so each step sees more signal, or lower "
                            "--t-nlost's patience with --t-step.", RuntimeError)
            fit = apertures[0].trace.fit
            ends = apertures[0].trace(np.array([1.0, float(frame.shape[0])]))
            by_action["trace"].measured = {
                "points": f"{traced[0].positions.size}",
                "rms": f"{fit.rms:.3f} px" if fit is not None else "-",
                "drift": f"{ends[1] - ends[0]:+.2f} px end to end"}

        outputs = []
        if params.extract:
            gain = _header_value(params.gain, header, "gain")
            readnoise = _header_value(params.readnoise, header, "readnoise")
            extracted = apextract.extract(
                frame, apertures, background=params.background, skybox=params.skybox,
                weights=params.weights, pfit=params.pfit, clean=params.clean,
                saturation=params.saturation, readnoise=readnoise, gain=gain,
                lsigma=params.lsigma, usigma=params.usigma, nclean=params.nclean,
                niterate=params.niterate)
            pixel = np.arange(1.0, frame.shape[0] + 1.0)
            written = []
            for aperture, result in zip(apertures, extracted, strict=True):
                written.append(spectrum.Aperture(
                    number=aperture.number, centre=aperture.centre, low=aperture.low,
                    high=aperture.high, pixel=pixel,
                    flux=np.asarray(result.spec, dtype=float),
                    error=(None if result.sigma is None
                           else np.asarray(result.sigma, dtype=float)),
                    sky=None if result.sky is None else np.asarray(result.sky, dtype=float),
                    raw=None if result.raw is None else np.asarray(result.raw, dtype=float),
                    trace=aperture.centre_at(pixel),
                    title=str(header.get("OBJECT", "")),
                    meta={"EXPTIME": float(header.get("EXPTIME", 0.0)),
                          "EPADU": gain, "RDNOISE": readnoise,
                          "WEIGHTS": "variance" if params.clean else params.weights,
                          "BACKGRND": params.background, "CLEAN": params.clean}))
            destination = params.output or output.named_for(params.input, ".ms.fits")
            spectrum.write(destination, written, header=header, source=str(params.input))
            outputs.append(destination)
            first = extracted[0]
            measured = {"median flux": f"{float(np.median(first.spec)):.1f}"}
            if first.sigma is not None:
                good = first.sigma > 0
                if good.any():
                    measured["median S/N"] = (
                        f"{float(np.median(first.spec[good] / first.sigma[good])):.1f}")
            for message in (m for r in extracted for m in r.log):
                if "WARNING" in message:
                    measured.setdefault("warning", message)
            by_action["extract"].measured = measured
            if "background" in by_action and first.sky is not None:
                by_action["background"].measured = {
                    "sky": f"{float(np.median(first.sky)):.1f} counts in the aperture"}

        stored = database.write(_solution(params, apertures), params.database, prefix="ap")
        outputs.append(stored)
        data = {"apertures": len(apertures),
                "centres": ", ".join(f"{a.centre:.2f}" for a in apertures),
                "limits": ", ".join(f"{a.low:+.2f}:{a.high:+.2f}" for a in apertures)}
        if recentred is not None:
            data["recenter"] = recentred
        return TaskResult(
            outputs=outputs, steps=steps, data=data,
            history=[f"apall: {len(apertures)} aperture(s), background="
                     f"{params.background}, weights="
                     f"{'variance' if params.clean else params.weights}, trace "
                     f"{params.t_function} order {params.t_order}"],
        )

    def _edit(self, params, profile, apertures, background, line):
        """apedit: the panel works on dicts and a band outside each edge;
        a band set there becomes the background sample around each aperture."""
        from dataclasses import replace

        from ..core import apextract
        from ..viewer.aperture_panel import run as run_apedit

        found = [{"centre": a.centre, "low": a.low, "high": a.high} for a in apertures]
        edited, band = run_apedit(profile, found, None,
                                  params.ylevel if params.ylevel is not None else 0.1,
                                  title=f"apedit · {params.input}")
        if not edited:
            advice.fail("every aperture was deleted, so there is nothing to extract.",
                        "run it again and keep at least one aperture, or drop --edit "
                        "to take the ones apfind proposes.", RuntimeError)
        out = []
        for number, chosen in enumerate(edited, start=1):
            spec = background
            if band:
                inner, outer = band
                low, high = chosen["low"], chosen["high"]
                spec = replace(background, sample=f"{low - outer:g}:{low - inner:g},"
                                                  f"{high + inner:g}:{high + outer:g}")
            out.append(apextract.Aperture(number, chosen["centre"], chosen["low"],
                                          chosen["high"], float(line), background=spec))
        return out, (out[0].background if out else background)

    def _fit_by_hand(self, params, apertures, traced, length, spec):
        from ..core import apextract, icfit
        from ..viewer.icfit_panel import run as run_icfit

        for aperture, points in zip(apertures, traced, strict=True):
            if points.positions.size < 2:
                continue
            fit = run_icfit(points.positions, points.centres - aperture.centre, spec,
                            title=f"aptrace · aperture {aperture.number} of {params.input}",
                            xlabel="position along the dispersion (pixels)",
                            ylabel="offset from the aperture centre (pixels)")
            aperture.trace = apextract.Trace(
                fit, *icfit.cv_range(points.positions, fit.spec, (1.0, float(length))))


def _header_value(text: float | str, header, what: str) -> float:
    """apgimr: a number, or the name of the header keyword that holds it."""
    try:
        return float(text)
    except ValueError:
        pass
    if text in header:
        return float(header[text])
    advice.fail(f"--{what} '{text}' is neither a number nor a keyword in the header.",
                f"give the {what} as a number, or the keyword that holds it")


def _solution(params, apertures):
    """The apertures, for database/ap<image>, so the next frame can reuse them."""
    import numpy as np

    from ..core import apextract, database

    centres = np.array([a.centre for a in apertures])
    rms = [a.trace.fit.rms for a in apertures if a.trace.fit is not None]
    return database.Solution(
        image=str(params.input),
        pixel=centres,
        wavelength=np.array([a.high - a.low for a in apertures]),
        function=params.t_function, order=params.t_order,
        rms=float(rms[0]) if rms else 0.0,
        units="pixels",
        used=np.ones(centres.size, dtype=bool),
        fitted=centres,
        extra={"task": "apall",
               "apertures": [apextract.to_record(a) for a in apertures]},
    )


def _stored_name(params) -> str | None:
    """Whose database/ap record t_apall starts from: the reference's, or
    this frame's own when it has one -- None when it finds afresh."""
    from ..core import database

    reference = params.references.strip()
    own = database._root(params.input)
    if reference.upper().startswith("NEW"):
        reference = reference[3:]
    if reference in ("", own) or reference.upper() == "OLD":
        path = database.path_for(params.input, params.database, "ap")
        return own if path.exists() else None
    return database._root(reference)


def _apertures_from_database(params, length: int) -> list:
    """ap_dbread of the reference (or of this frame), as t_apall chooses."""
    from ..core import apextract, database

    reference = params.references.strip()
    own = database.path_for(params.input, params.database, "ap")
    if reference.upper() == "OLD" and not own.exists():
        advice.fail(f"{params.input} has no apertures of its own (database/"
                    f"{own.name}), and --references OLD only takes frames that do.",
                    "drop --references to find them, or name the frame to take them "
                    "from.", FileNotFoundError)
    if reference.upper().startswith("NEW"):
        if own.exists():
            advice.fail(f"{params.input} already has apertures (database/{own.name}), "
                        "and --references NEW only takes frames that do not.",
                        "give the reference without NEW to replace them.",
                        FileExistsError)
        reference = reference[3:]
    name = database._root(reference) if reference and reference.upper() != "OLD" else None
    if name is None or name == database._root(params.input):
        if not own.exists():
            return []
        record = database.read(params.input, params.database, prefix="ap")
    else:
        path = database.path_for(reference, params.database, "ap")
        if not path.exists():
            advice.fail(f"Reference image ({reference}) apertures not found: no "
                        f"database/{path.name}.",
                        f"run apall on {reference} first, so it has apertures to "
                        "lend.", FileNotFoundError)
        record = database.read(reference, params.database, prefix="ap")
    return [apextract.from_record(r, length)
            for r in record.extra.get("apertures", [])]


class WSpecTextParams(TaskParams):
    input: str = Field(..., description="Extracted spectrum file, from apall.")
    output: Path | None = Field(
        None,
        description="Text file to write. Left out, it goes to this run's own "
                    "directory, named after the input.",
        json_schema_extra={"positional": True},
    )
    aperture: int = Field(
        1, ge=1, description="Which aperture to write. One file holds several."
    )
    format: Literal["csv", "tsv", "fixed"] = Field(
        "csv", description="csv for a spreadsheet, tsv for cut and awk, fixed for "
                           "reading on a screen."
    )
    header: bool = Field(
        True, description="Write the column names and what they mean, as comments."
    )
    columns: str | None = Field(
        None, description="Which columns, comma separated. All of them by default."
    )
    lines: bool = Field(
        True,
        description="When the spectrum has wavelengths, list the catalogued "
                    "lines that fall in its range as comments at the top.",
    )


def _lines_in_range(aperture, params: WSpecTextParams) -> list[str]:
    """The catalogued lines the spectrum covers, as comments.

    A column of wavelengths is a column of numbers. Saying that 6562.8 of them
    is hydrogen costs four lines of text and turns the file into something
    somebody can read without a second program open beside it.
    """
    import numpy as np

    from ..core import species

    if not params.lines or aperture.wavelength is None:
        return []
    wave = np.asarray(aperture.wavelength, dtype=float)
    if not np.any(wave):
        return []

    found = species.within(float(wave.min()), float(wave.max()), minimum=2)
    if not found:
        return ["#", "# No catalogued line falls in this wavelength range."]

    out = ["#", f"# Lines catalogued between {wave.min():.2f} and "
                f"{wave.max():.2f} A -- what is *in range*, not what was",
           "# measured. Run lineid to find out which are actually present."]
    for line in found:
        where = "/".join(line.origin)
        out.append(f"#   {line.wavelength:9.2f}  {line.species:<10s} "
                   f"{line.label or '':<16s} {where}")
    return out


@register
class WSpecText(Task[WSpecTextParams]):
    name = "wspectext"
    package = "onedspec"
    description = "Write an extracted spectrum out as text."
    Params = WSpecTextParams

    def plan(self, params: WSpecTextParams) -> list[Step]:
        return [
            Step("read", f"{params.input}, aperture {params.aperture}",
                 why="A FITS table is self-describing but not readable without "
                     "something that opens FITS. A file whose contents can only "
                     "be seen through the program that wrote it is a smaller "
                     "black box, not an open one."),
            Step("write", f"{params.output or 'this run''s directory'}, "
                          f"{params.format}",
                 why="Every column the spectrum has, as numbers, with a comment "
                     "saying what each one is. Nothing is recomputed and nothing "
                     "is left out."),
        ]

    def execute(self, params: WSpecTextParams) -> TaskResult:
        import numpy as np

        from ..core import output, spectrum

        found = spectrum.read(params.input, number=params.aperture)
        if not found:
            total = spectrum.count(params.input)
            advice.fail(
                f"'{params.input}' has no aperture {params.aperture}; it holds "
                f"{total}.",
                "use --aperture with a number in that range.")
        aperture = found[0]
        table = aperture.columns()
        if params.columns:
            wanted = [name.strip() for name in params.columns.split(",")
                      if name.strip()]
            missing = [name for name in wanted if name not in table]
            if missing:
                advice.fail(
                    f"no column called {', '.join(missing)}.",
                    f"this spectrum has {', '.join(table)}; name one of those, "
                    "or leave --columns out to write them all.")
            table = {name: table[name] for name in wanted}

        separator = {"csv": ",", "tsv": "\t", "fixed": "  "}[params.format]
        lines = []
        if params.header:
            lines.append(f"# {aperture.describe()}"
                         + (f" -- {aperture.title}" if aperture.title else ""))
            lines.append(f"# from {params.input}")
            for name in table:
                lines.append(f"# {name}: {spectrum.MEANING.get(name, '')}")
            lines += _lines_in_range(aperture, params)
        if params.format == "fixed":
            lines.append("  ".join(f"{name:>14}" for name in table))
        else:
            lines.append(separator.join(table))

        rows = np.column_stack([np.asarray(values, dtype=float)
                                for values in table.values()])
        for row in rows:
            if params.format == "fixed":
                lines.append("  ".join(f"{value:14.6g}" for value in row))
            else:
                lines.append(separator.join(f"{value:.6g}" for value in row))

        destination = params.output or output.named_for(
            params.input, f".{params.format if params.format != 'fixed' else 'txt'}")
        Path(destination).parent.mkdir(parents=True, exist_ok=True)
        Path(destination).write_text("\n".join(lines) + "\n", encoding="utf-8")

        return TaskResult(
            outputs=[Path(destination)],
            steps=self.plan(params),
            data={"aperture": aperture.number, "points": rows.shape[0],
                  "columns": ", ".join(table)},
        )
