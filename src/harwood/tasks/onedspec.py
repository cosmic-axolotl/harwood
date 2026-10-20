"""onedspec -- wavelength calibration.

`identify` is where a spectrum stops being counts against pixels and becomes
counts against wavelength. It is also the clearest case in IRAF for why an
interactive task exists at all: a thorium lamp puts three thousand catalogue
lines into a few hundred Angstroms, and no automatic search can tell which of
them a given peak is without being told roughly where the grating was set. IRAF
asks you to mark five lines by hand, fits, and then places the rest.

harwood does the same, and adds the automatic path for when you *can* say where
the spectrum sits -- `--crval` and `--cdelt`, as IRAF's `autoidentify` takes
them. The fit itself is `icfit`, so `:function chebyshev`, `:order 3` and
`:niterate 2` mean here exactly what they mean in the overscan.

The solution goes to `database/id<image>`, the directory everything downstream
reads.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import Field

from ..core import advice
from ..core.registry import register
from ..core.task import Step, Task, TaskParams, TaskResult

WHY = {
    "read": "A two-dimensional arc is collapsed to one dimension by adding "
            "several rows or columns across the slit. The lamp fills the slit "
            "evenly, so adding across it costs nothing and buys signal -- which "
            "is why identify does this itself rather than requiring an "
            "extraction first.",
    "auto": "IRAF's autoidentify: the spacing ratios of every three neighbouring "
            "lines in the spectrum are matched against those in the list, "
            "patterns of up to five lines vote for which peak is which line, "
            "straight dispersions are drawn through the best-voted pairs, and "
            "each is tried -- fitted, more lines located from the list, and "
            "scored on its rms and how much of both lists it accounts for.",
    "find": "An arc line is an image of the slit, so its width is the "
            "instrumental resolution and not the star's seeing. Peaks are "
            "centred by their flux centroid, which is good to about a tenth of "
            "a pixel -- and that number is the yardstick for the whole "
            "solution: a fit that scatters much more than that has matched "
            "lines to the wrong wavelengths.",
    "catalogue": "The line list is laboratory wavelengths of a known lamp, in "
                 "air. Which lamp matters and so does how dense the list is: "
                 "thorium has three thousand lines, which is why a blind search "
                 "for the dispersion is ambiguous and why IRAF asks you to mark "
                 "a few by hand.",
    "mark": "Four or five lines spread across the spectrum pin down a low-order "
            "solution; the line list then places the rest. Spread matters more "
            "than number -- five lines bunched in the middle constrain the ends "
            "not at all.",
    "match": "Each peak is assigned the catalogue line the approximate "
             "dispersion puts nearest to it. The approximation has to be good "
             "to about one part in a hundred, or a peak lands nearer a "
             "neighbour of the right line than the right line itself.",
    "search": "With no approximate dispersion given, one is searched for. "
              "Against a dense line list several quite different dispersions "
              "score nearly the same and a cubic fits any of them to a tenth of "
              "a pixel, so the result is checked before it is believed.",
    "fit": "Wavelength against pixel is smooth but not linear: a grating "
           "disperses as a sine, and the camera adds its own distortion. IRAF's "
           "default, a cubic spline of one piece, is a cubic; more freedom "
           "starts following the errors in the line centres instead.",
    "write": "The solution goes to database/id<image>, where reidentify, "
             "refspectra and dispcor all look for it. Rejected and deleted "
             "features are kept in the file: knowing which lines were thrown "
             "out is part of knowing the solution.",
}


FUNCTIONS = Literal["legendre", "chebyshev", "spline1", "spline3"]


class _ArcParams(TaskParams):
    """What identify and autoidentify share."""

    image: str = Field(..., description="Arc spectrum: 1-D, or a 2-D frame and a section.")
    coordlist: str = Field(
        "linelists$idhenear.dat",
        description="Line list of known coordinates. IRAF's linelists$ prefix works "
                    "when IRAF is installed; a path works anywhere.")
    units: str = Field("", description="Coordinate units; left out, the list's.")
    section: str = Field(
        "middle line",
        description="Which image line or column of a 2-D frame: 'middle line' is the "
                    "middle line, a spectrum along x; 'middle column' one along y; "
                    "or 'line 120', 'first column', ...")
    ftype: Literal["emission", "absorption"] = Field("emission", description="Feature type.")
    fwidth: float = Field(4.0, gt=0, description="Feature width in pixels.")
    cradius: float = Field(5.0, ge=0, description="Centring radius in pixels.")
    threshold: float = Field(
        0.0, ge=0, description="Least range of data a feature must span to be centred.")
    minsep: float = Field(2.0, ge=0, description="Least separation between features, px.")
    match: float = Field(
        -3.0, description="How close a line list entry must be to be matched: in "
                          "coordinate units, or in pixels if negative.")
    function: FUNCTIONS = Field("spline3", description="Coordinate function.")
    order: int = Field(1, ge=1, description="Order of the coordinate function.")
    sample: str = Field("*", description="Pixel regions fitted.")
    grow: float = Field(0.0, ge=0, description="Rejection growing radius.")
    database: Path = Field(Path("."), description="Reduction directory holding database/.")

    # aidpars, for the automatic identification
    crpix: float | None = Field(None, description="Pixel crval refers to; the middle.")
    crquad: float | None = Field(None, description="Quadratic distortion at crpix.")
    cddir: Literal["sign", "increasing", "decreasing", "unknown"] = Field(
        "sign", description="Dispersion direction: as cdelt's sign, or as stated.")
    crsearch: float | None = Field(
        None, description="Search radius for crval; negative, a fraction of the range.")
    cdsearch: float | None = Field(
        None, description="Search radius for cdelt; negative, a fraction of cdelt.")
    ntarget: int = Field(100, ge=1, description="Number of target features.")
    npattern: int = Field(5, ge=3, le=10, description="Lines in a pattern.")
    nneighbors: int = Field(10, ge=2, description="Nearest neighbours in patterns.")
    nbins: int = Field(6, ge=1, description="Most search bins.")
    ndmax: int = Field(500, ge=1, description="Most dispersions to evaluate.")
    aidord: int = Field(3, ge=2, description="Order of the search fits.")
    maxnl: float = Field(0.02, ge=0, description="Most non-linearity allowed.")
    nfound: int = Field(6, ge=3, description="Fewest lines in a solution.")
    aidsigma: float = Field(0.05, ge=0, description="Line centring sigma, pixels "
                                                    "(aidpars.sigma).")
    minratio: float = Field(0.1, ge=0, le=1, description="Least spacing ratio used.")
    rms: float = Field(0.1, ge=0, description="RMS goal, in fwidths.")
    fmatch: float = Field(0.2, ge=0, le=1, description="Matching goal (fraction unmatched).")

    def aidpars(self):
        from ..core.identify import AidPars

        return AidPars(crpix=self.crpix, crquad=self.crquad, cddir=self.cddir,
                       crsearch=self.crsearch, cdsearch=self.cdsearch,
                       ntarget=self.ntarget, npattern=self.npattern,
                       nneighbors=self.nneighbors, nbins=self.nbins, ndmax=self.ndmax,
                       aidord=self.aidord, maxnl=self.maxnl, nfound=self.nfound,
                       sigma=self.aidsigma, minratio=self.minratio, rms=self.rms,
                       fmatch=self.fmatch)


class IdentifyParams(_ArcParams):
    nsum: int = Field(10, ge=1, description="Lines or columns summed in a 2-D frame.")
    maxfeatures: int = Field(50, ge=1, description="Most features for automatic "
                                                   "identification ('l').")
    niterate: int = Field(0, ge=0, description="Rejection iterations.")
    low_reject: float = Field(3.0, ge=0, description="Lower rejection sigma.")
    high_reject: float = Field(3.0, ge=0, description="Upper rejection sigma.")
    crval: float | None = Field(
        None, description="Approximate coordinate at crpix. With cdelt, the lines are "
                          "identified automatically, as identify's 'b' key does.")
    cdelt: float | None = Field(None, description="Approximate dispersion per pixel.")
    interactive: bool = Field(
        False, description="Mark lines and fit by hand, in the identify panel.")


class AutoIdentifyParams(_ArcParams):
    crval: float | None = Field(..., description="Approximate coordinate at crpix "
                                                 "(None to search).")
    cdelt: float | None = Field(..., description="Approximate dispersion (None to "
                                                 "search).")
    nsum: int = Field(1, ge=1, description="Lines or columns summed in a 2-D frame.")
    niterate: int = Field(10, ge=0, description="Rejection iterations.")
    low_reject: float = Field(2.0, ge=0, description="Lower rejection sigma.")
    high_reject: float = Field(2.0, ge=0, description="Upper rejection sigma.")


def _solve(task, params, maxfeatures: int, automatic: bool):
    """Read the arc, identify its lines (automatically or by hand), fit, and
    write database/id<image>. Shared by identify and autoidentify."""
    import numpy as np

    from ..core import arc, database, icfit, images
    from ..core import identify as idn

    data, header = images.read(params.image)
    vector, where = idn.arc_vector(data, params.section, params.nsum)
    linelist = idn.read_linelist(arc.resolve_linelist(params.coordlist))
    units = params.units or linelist.units
    spec = icfit.FitSpec(function=params.function, order=params.order,
                         sample=params.sample, naverage=1, niterate=params.niterate,
                         low_reject=params.low_reject, high_reject=params.high_reject,
                         grow=params.grow)
    ident = idn.Identify(vector, linelist, spec, fwidth=params.fwidth,
                         ftype=params.ftype, cradius=params.cradius,
                         threshold=params.threshold, minsep=params.minsep,
                         match=params.match, maxfeatures=maxfeatures)
    steps = task.plan(params)
    by_action = {step.action: step for step in steps}
    by_action["read"].measured = {"pixels": vector.size,
                                  "section": where or "1-D", "lines in list": len(linelist)}
    warnings: list = []

    if getattr(params, "interactive", False):
        from ..viewer.identify_panel import run as run_identify

        peaks = arc.find_peaks(vector, fwidth=params.fwidth, threshold=3.0,
                               minsep=params.minsep)
        fit, pixel, wavelength = run_identify(
            vector, peaks, linelist.values, spec, title=f"identify · {params.image}",
            match=abs(params.match), units=units)
        ident.features = [idn.Feature(float(p), float(w), float(w))
                          for p, w in zip(pixel, wavelength, strict=True)]
        ident.dofit()
        ident.update_fitdata()
        ident.fit_features()
    else:
        _enough_lines(arc, vector, where, data, params)
        if task.name == "identify" and params.crval is None and params.cdelt is None:
            # IRAF's identify asks a person; with nothing to start from, the
            # automatic search covers every dispersion the list allows, and on
            # the 1992 thorium arc it ran past half an hour without an answer.
            # That is a search to ask for by name, not to fall into.
            advice.fail(
                f"{params.image}{where}: no approximate wavelength or dispersion "
                "to start from",
                "give --crval (the wavelength at the middle) and --cdelt (per "
                "pixel), good to about one per cent -- from the observing log or "
                "a previous night; or mark four or five lines with --interactive; "
                f"or `autoidentify {params.image} INDEF INDEF` for the blind "
                "search, which on a dense list takes tens of minutes",
                RuntimeError)
        result = idn.autoidentify(ident, params.crval, params.cdelt, params.aidpars())
        by_action["auto"].measured = {
            "merit": f"{result.best:.3f} (a solution is 1 or less)"
            if result.best < 1e30 else "no candidate",
            "rms": f"{result.rms:.3f} fwidths" if result.found else "-",
            "unmatched": f"{result.fmatch:.0%} of the list, {result.ftmatch:.0%} of "
                         "the lines" if result.found else "-"}
        if not result.found:
            advice.fail(
                f"{params.image}{where}: No solution found.",
                "check --crval and --cdelt against the lamp's range -- the "
                "OBJECT name of an arc is the grating setting, not necessarily "
                "the wavelengths it covers -- and that --coordlist is that lamp's "
                "list; or mark a few lines by hand with identify --interactive.",
                RuntimeError)

    used = [f for f in ident.features if f.user is not None and f.wts != 0]
    if len(used) < 2 or ident.fit is None:
        advice.fail("fewer than two features are identified, so there is no "
                    "solution.", "mark more lines, or loosen --match", RuntimeError)
    rms = ident.rms()
    scale = abs(float(ident.fitdata[-1] - ident.fitdata[0])) / (vector.size - 1)
    by_action["fit"].measured = {
        "features": f"{len(used)} used of {len(ident.features)}",
        "rms": f"{rms:.4f} {units} ({rms / scale:.3f} px)" if scale else f"{rms:.4f}",
        "dispersion": f"{scale:.5f} {units}/px",
    }
    if scale and rms / scale > 0.3:
        warnings.append(advice.warn(
            f"the solution scatters by {rms / scale:.2f} px, looser than an arc line "
            "centre is measured. Something is matched to the wrong list entry.",
            "look at the identifications with --interactive and delete the ones "
            "far off the curve"))

    pixel = np.array([f.pix for f in ident.features])
    solution = database.Solution(
        image=str(params.image), pixel=pixel,
        wavelength=np.array([np.nan if f.user is None else f.user for f in ident.features]),
        function=params.function, order=params.order, rms=float(rms), units=units,
        coordlist=params.coordlist,
        used=np.array([f.user is not None and f.wts != 0 for f in ident.features]),
        fitted=np.array([f.fit for f in ident.features]),
        extra={"section": where, "npts": int(vector.size),
               "xrange": [1.0, float(vector.size)], "niterate": params.niterate,
               "low_reject": params.low_reject, "high_reject": params.high_reject,
               "sample": params.sample, "grow": params.grow})
    destination = database.write(solution, params.database)
    first, last = float(ident.fitdata[0]), float(ident.fitdata[-1])
    return TaskResult(
        outputs=[destination], steps=steps, warnings=warnings,
        data={"features": len(ident.features), "used": len(used), "rms": round(rms, 5),
              "dispersion": round(scale, 5),
              "range": f"{min(first, last):.2f} to {max(first, last):.2f} {units}"},
        history=[f"{task.name}: {len(used)} features, {params.function} order "
                 f"{params.order}, rms {rms:.4f} {units}, from {params.coordlist}"])


@register
class Identify(Task[IdentifyParams]):
    name = "identify"
    package = "onedspec"
    description = "Identify arc lines and fit the dispersion solution."
    Params = IdentifyParams

    def plan(self, params: IdentifyParams) -> list[Step]:
        steps = [Step("read", f"{params.image}, {params.section}, nsum {params.nsum}",
                      why=WHY["read"]),
                 Step("catalogue", params.coordlist, why=WHY["catalogue"])]
        if params.interactive:
            steps.append(Step("mark", "by hand, in the identify panel", why=WHY["mark"]))
        else:
            steps.append(Step("auto", f"autoidentify, crval {params.crval}, cdelt "
                                      f"{params.cdelt}", why=WHY["auto"]))
        steps.append(Step("fit", f"{params.function} order {params.order}"
                                 + (f", {params.niterate} rejection passes"
                                    if params.niterate else ""), why=WHY["fit"]))
        steps.append(Step("write", str(_database_path(params)), why=WHY["write"]))
        return steps

    def execute(self, params: IdentifyParams) -> TaskResult:
        return _solve(self, params, params.maxfeatures, not params.interactive)


@register
class AutoIdentify(Task[AutoIdentifyParams]):
    name = "autoidentify"
    package = "onedspec"
    description = "Identify arc lines automatically from an approximate dispersion."
    Params = AutoIdentifyParams

    def plan(self, params: AutoIdentifyParams) -> list[Step]:
        return [Step("read", f"{params.image}, {params.section}, nsum {params.nsum}",
                     why=WHY["read"]),
                Step("catalogue", params.coordlist, why=WHY["catalogue"]),
                Step("auto", f"crval {params.crval}, cdelt {params.cdelt}"
                     + (" -- a blind search over every dispersion the list allows; "
                        "on a dense list it takes tens of minutes"
                        if params.crval is None and params.cdelt is None else ""),
                     why=WHY["auto"]),
                Step("fit", f"{params.function} order {params.order}, "
                            f"{params.niterate} rejection passes", why=WHY["fit"]),
                Step("write", str(_database_path(params)), why=WHY["write"])]

    def execute(self, params: AutoIdentifyParams) -> TaskResult:
        return _solve(self, params, params.ntarget, True)


def _enough_lines(arc, vector, where: str, data, params) -> None:
    """Refuse at once a section with too few lines to solve: the search over
    them finds nothing anyway, and takes minutes to do it. The usual cause is
    the section running across the slit rather than along the spectrum --
    'middle line' on a frame whose dispersion runs along the columns -- so
    that is what the refusal suggests."""
    peaks = arc.find_peaks(vector, fwidth=params.fwidth, threshold=3.0,
                           minsep=params.minsep)
    if len(peaks) >= params.nfound:
        return
    other = ""
    if getattr(data, "ndim", 1) == 2 and vector.size < max(data.shape):
        turned = "middle column" if "line" in params.section else "middle line"
        other = (f"; the frame is {max(data.shape)} pixels the other way, so if the "
                 f"spectrum runs along it, give --section '{turned}'")
    advice.fail(
        f"{params.image}{where} is {vector.size} pixels with {len(peaks)} line"
        f"{'s' if len(peaks) != 1 else ''} in it, fewer than the {params.nfound} "
        "a solution needs",
        "check that the section runs along the dispersion" + other, RuntimeError)


def _database_path(params) -> Path:
    from ..core import database

    return database.path_for(params.image, params.database)



DISPWHY = {
    "read": "An extracted spectrum is counts against pixel. Which pixel is "
            "which wavelength is not in it -- that lives in the dispersion "
            "solution the arc gave, which is why the arc was taken.",
    "solution": "The solution from identify: a function from pixel to "
                "wavelength, fitted to lines whose laboratory wavelengths are "
                "known. It belongs to the arc, and applies to the object frames "
                "taken near it in time -- the spectrograph flexes as the "
                "telescope moves, so an arc from the start of the night is not "
                "the arc for the end of it.",
    "apply": "Every pixel gets the wavelength the solution puts it at. The "
             "spacing is not uniform -- a grating disperses as a sine -- so the "
             "wavelengths come out unevenly spaced, and that is correct. "
             "Resampling onto an even grid is a separate choice with a cost: it "
             "interpolates, which correlates neighbouring pixels.",
    "resample": "Put onto an evenly spaced wavelength grid, which is what most "
                "analysis expects. The flux per pixel is conserved by weighting "
                "each output bin by how much of each input pixel falls in it, so "
                "a line's equivalent width survives -- but neighbouring pixels "
                "are now correlated and the errors know it less well than they "
                "did.",
    "write": "The same file shape as before, with the wavelength column filled "
             "in. Nothing is thrown away: the pixel column stays, so the "
             "calibration can be checked or redone.",
    "check": "The strongest features of the calibrated spectrum are looked up "
             "in the line catalogue. This checks the solution against physics "
             "rather than against itself: the arc's rms says how well the fit "
             "reproduces the lines it was fitted to, which a wrong solution "
             "can do beautifully. Real features landing on real wavelengths "
             "is a different claim, and an independent one.",
}


class DispCorParams(TaskParams):
    input: str = Field(..., description="Extracted spectrum, from apall.")
    output: Path | None = Field(
        None,
        description="Where to write. Left out, it goes to this run's own "
                    "directory, named after the input.",
        json_schema_extra={"positional": True},
    )
    reference: str = Field(
        "",
        description="Arc whose dispersion solution to use -- the name identify "
                    "was run on. Read from database/id<reference>.",
    )
    linearize: bool = Field(
        True,
        description="Resample onto an evenly spaced grid, as IRAF does by default. "
                    "It interpolates, which correlates neighbouring pixels; with "
                    "--no-linearize the wavelength column carries the solution and "
                    "the pixels are untouched.",
    )
    w1: float | None = Field(None, description="Starting wavelength of the grid.")
    w2: float | None = Field(None, description="Ending wavelength of the grid.")
    dw: float | None = Field(None, description="Wavelength interval per pixel.")
    nw: int | None = Field(None, ge=2, description="Number of output pixels.")
    log: bool = Field(False, description="Logarithmic wavelength grid (log10)?")
    flux: bool = Field(
        True, description="Conserve flux: each output pixel holds the counts that fall "
                          "in it, not their mean.")
    blank: float | None = Field(0.0, description="Value for output pixels outside the "
                                                 "input.")
    interp: Literal["nearest", "linear", "poly3", "poly5", "spline3"] = Field(
        "poly5", description="Interpolant the resampling integrates (onedspec.interp).")
    database: Path = Field(
        Path("."), description="Reduction directory holding database/."
    )
    check: bool = Field(
        True,
        description="Look the strongest features up in the line catalogue, as "
                    "a check that the wavelengths landed on something real.",
    )


@register
class DispCor(Task[DispCorParams]):
    name = "dispcor"
    package = "onedspec"
    description = "Put wavelengths on an extracted spectrum."
    Params = DispCorParams

    def plan(self, params: DispCorParams) -> list[Step]:
        steps = [
            Step("read", params.input, why=DISPWHY["read"]),
            Step("solution",
                 f"database/id{Path(params.reference).stem}"
                 if params.reference else "the arcs refspectra assigned",
                 why=DISPWHY["solution"]),
            Step("apply", "wavelength for every pixel", why=DISPWHY["apply"]),
        ]
        if params.linearize:
            steps.append(Step("resample", "onto an even wavelength grid",
                              why=DISPWHY["resample"]))
        steps.append(Step("write", str(params.output or "this run's directory"),
                          why=DISPWHY["write"]))
        if params.check:
            steps.append(Step("check", "the strongest features, by name",
                              why=DISPWHY["check"]))
        return steps

    def execute(self, params: DispCorParams) -> TaskResult:
        import math

        import numpy as np

        from ..core import database, dispcor, icfit, output, spectrum

        names, weights = _references(params)
        apertures = spectrum.read(params.input)

        solutions = [database.read(name, params.database) for name in names]
        curves, pixels = [], []
        for name, solution in zip(names, solutions, strict=True):
            used = solution.used if solution.used is not None else None
            pixel = solution.pixel if used is None else solution.pixel[used]
            pixels.append(pixel)
            saved = solution.extra.get("curve")
            if saved:
                # IRAF's own curve, coefficient for coefficient.
                curves.append(dispcor.Curve.from_saved(saved))
                continue
            # A harwood solution: the curve identify fitted, refitted from the
            # points it kept, over the range its spline pieces were laid on.
            wavelength = (solution.wavelength if used is None
                          else solution.wavelength[used])
            xrange = solution.extra.get("xrange")
            fitted = icfit.fit(pixel, wavelength,
                               icfit.FitSpec(function=solution.function,
                                             order=solution.order),
                               xrange=tuple(xrange) if xrange else None)
            if fitted.model is None:
                advice.fail(
                    f"the solution in database/id{Path(name).stem} could not be "
                    f"refitted: {fitted.message}",
                    "open the file and look at how many features it kept; a "
                    "solution with too few surviving lines for its order cannot "
                    "be rebuilt. Re-run identify on that arc.", RuntimeError)
            curves.append(lambda x, model=fitted: float(model(np.array([x]))[0]))

        solution = solutions[0]
        pixel = pixels[0]

        steps = self.plan(params)
        by_action = {step.action: step for step in steps}
        by_action["solution"].measured = {
            "arcs": " + ".join(
                f"{name} ({weight:.3f})" if len(names) > 1 else str(name)
                for name, weight in zip(names, weights, strict=True)),
            "features": " and ".join(f"{len(used)} of {len(whole)}"
                                     for used, whole in zip(pixels, solutions, strict=True)),
            "function": f"{solution.function} order {solution.order}",
            "rms": " and ".join(f"{item.rms:.4f} {item.units}"
                                for item in solutions),
        }

        warnings = []
        grid = None
        for aperture in apertures:
            npts = len(aperture.flux)
            disp = dispcor.Dispersion([(w, 0.0, c) for w, c in
                                       zip(weights, curves, strict=True)], npts)
            if not params.linearize:
                aperture.wavelength = disp(np.asarray(aperture.pixel, dtype=float))
                continue
            a, b = disp.eval(1.0), disp.eval(float(npts))
            if params.log:
                a, b = math.log10(a), math.log10(b)
            w1, w2, dw, nw = dispcor.defaults(a, b, npts, params.w1, params.w2,
                                              params.dw, params.nw)
            grid = (w1, w2, dw, nw)
            for column in ("flux", "error", "sky", "raw"):
                values = getattr(aperture, column, None)
                if values is None:
                    continue
                # Every band is resampled the same way, as IRAF does them.
                disp = dispcor.Dispersion([(w, 0.0, c) for w, c in
                                           zip(weights, curves, strict=True)], npts)
                setattr(aperture, column, dispcor.resample(
                    values, disp, w1, dw, nw, params.flux, params.blank, params.interp,
                    params.log).astype(float))
            wave = w1 + np.arange(nw) * dw
            aperture.wavelength = 10 ** wave if params.log else wave
            aperture.pixel = np.arange(1.0, nw + 1.0)

        first = apertures[0]
        low, high = float(first.wavelength.min()), float(first.wavelength.max())
        spacing = np.diff(first.wavelength)
        by_action["apply"].measured = {
            "range": f"{low:.2f} to {high:.2f} {solution.units}",
            "dispersion": f"{float(np.median(spacing)):.5f} {solution.units}/px",
            "spacing varies by":
                f"{100 * float(np.ptp(spacing) / abs(np.median(spacing))):.2f}%",
        }
        if params.linearize and grid is not None:
            by_action["resample"].measured = {
                "grid": f"w1 {grid[0]:.4f}, w2 {grid[1]:.4f}, dw {grid[2]:.6g}, "
                        f"nw {grid[3]}" + (" (log10)" if params.log else ""),
                "interpolant": params.interp,
                "flux": "conserved" if params.flux else "averaged",
            }
            warnings.append(advice.note(
                "linearizing interpolates, so neighbouring pixels are now "
                "correlated: a spectrum resampled twice is smoother than the "
                "data, and errors estimated from its scatter are too small.",
                "use --no-linearize when nothing downstream needs an even grid; "
                "the wavelength column carries the solution without it."))

        # The solution was fitted over the arc's pixels; using it far outside
        # that range is extrapolation, and it is the quiet kind of wrong.
        covered = (float(np.min(pixel)), float(np.max(pixel)))
        outside = int((np.asarray(first.pixel) < covered[0]).sum()
                      + (np.asarray(first.pixel) > covered[1]).sum())
        if outside:
            warnings.append(advice.warn(
                f"{outside} of {len(first.pixel)} pixels fall outside the range "
                f"the arc lines covered ({covered[0]:.0f} to {covered[1]:.0f}); "
                "their wavelengths are extrapolated, and a polynomial "
                "extrapolates badly.",
                "identify arc lines nearer the ends of the frame, or use only "
                f"pixels {covered[0]:.0f} to {covered[1]:.0f} of this "
                "spectrum."))

        if params.check:
            named, unnamed = _recognise(first)
            by_action["check"].measured = {
                "strongest features": len(named) + len(unnamed),
                "on a catalogued line": len(named),
                "landing on": ", ".join(named[:5]) or "nothing recognisable",
            }
            if named and len(named) < max(2, (len(named) + len(unnamed)) // 4):
                warnings.append(advice.warn(
                    f"only {len(named)} of the {len(named) + len(unnamed)} "
                    "strongest features land on a catalogued line. That is "
                    "weak confirmation: either the spectrum is of something "
                    "whose lines are not in the catalogue, or the solution is "
                    "in the wrong place.",
                    "run lineid on the result and look at what it names. If "
                    "nothing recognisable appears, redo identify -- the arc's "
                    "rms will not tell you this."))
            elif not named:
                warnings.append(advice.problem(
                    "none of the strongest features lands within an Angstrom "
                    "of a catalogued line. The arc's rms says the fit "
                    "reproduces the lines it was given -- a wrong solution can "
                    "do that too -- so the solution is very likely in the "
                    "wrong place.",
                    "redo identify with --interactive and mark lines by hand. "
                    "Check the wavelength range against what the grating was "
                    "set to before trusting anything downstream."))

        destination = params.output or output.named_for(params.input, ".w.fits")
        spectrum.write(destination, apertures, source=str(params.input))

        return TaskResult(
            outputs=[Path(destination)],
            steps=steps,
            warnings=warnings,
            data={"apertures": len(apertures),
                  "range": f"{low:.2f} to {high:.2f} {solution.units}",
                  "arcs": " + ".join(str(name) for name in names),
                  "extrapolated": outside},
            history=[f"dispcor: {solution.function} order {solution.order} from "
                     + " + ".join(
                         f"{name} ({weight:.3f})" if len(names) > 1 else str(name)
                         for name, weight in zip(names, weights, strict=True))
                     + f", rms {solution.rms:.4f} {solution.units}"],
        )


def _recognise(aperture, limit: int = 12) -> tuple[list[str], list[float]]:
    """Name the strongest features, as a check on where the solution landed.

    Deliberately small: the few strongest features, at a tolerance of an
    Angstrom, against the curated catalogue. It is a sanity check and not an
    analysis -- `lineid` is the task for the analysis.
    """
    import numpy as np

    from ..core import arc, species

    flux = np.asarray(aperture.flux, dtype=float)
    wave = np.asarray(aperture.wavelength, dtype=float)
    peaks = (arc.find_peaks(flux, threshold=8.0, absorption=True)
             + arc.find_peaks(flux, threshold=8.0))
    peaks.sort(key=lambda peak: -peak.flux)

    named, unnamed = [], []
    for peak in peaks[:limit]:
        measured = float(np.interp(peak.pixel, aperture.pixel, wave))
        match = species.at(measured, tolerance=1.0)
        if match.best is not None:
            named.append(match.best.line.name)
        else:
            unnamed.append(round(measured, 2))
    return list(dict.fromkeys(named)), unnamed


def _references(params: DispCorParams) -> tuple[list[str], list[float]]:
    """Which arcs to use: the one given, or the ones refspectra recorded.

    A spectrum that has been through refspectra carries its own answer, which
    is the point of the keywords: the file says which arc calibrates it without
    anyone having to remember.
    """
    from astropy.io import fits

    from ..core import refspec
    from ..core.database import listing

    if params.reference:
        return [params.reference], [1.0]

    try:
        names, weights = refspec.read_keywords(fits.getheader(params.input))
    except OSError:
        names, weights = [], []
    if names:
        total = sum(weights)
        return names, [weight / total for weight in weights]

    known = [path.name[2:] for path in listing(params.database)
             if path.name.startswith("id")]
    advice.fail(
        f"'{params.input}' does not say which arc calibrates it, and there is "
        "no way to guess which arc belongs to which exposure.",
        "name one with --reference, or run refspectra over the night to assign "
        "them by time. "
        + (f"This reduction has solutions for {', '.join(known)}."
           if known else "This reduction has none yet, so run identify on an "
                         "arc first."))


REFWHY = {
    "times": "Every frame is placed on one clock -- the middle of its exposure, "
             "not its start -- and the clock wraps at an hour nobody observes, "
             "so a night that crosses midnight still sorts in the order it "
             "happened. This is why setairmass runs first: it is what puts "
             "UTMIDDLE in the header.",
    "sort": "The arcs are put in time order. A dispersion solution is not a "
            "property of the spectrograph but of the spectrograph at a moment: "
            "the instrument flexes as the telescope tracks, and the wavelength "
            "landing on a given pixel moves with it. That is why a night has an "
            "arc at each end rather than one in the middle.",
    "assign": "Each exposure is given the arc or arcs that bracket it. With "
              "'interp' it gets both, weighted by how close each one is, which "
              "assumes the flexure between them is smooth -- the assumption the "
              "two arcs were taken to let you make. 'nearest' makes no such "
              "assumption and throws away half the information.",
    "write": "The assignment goes into the spectrum's own header as REFSPEC1 "
             "and REFSPEC2, the keywords IRAF uses, so dispcor needs no "
             "argument: the file says which arc calibrates it, and still says "
             "so next year.",
}


class RefSpectraParams(TaskParams):
    input: list[str] = Field(
        ..., description="Extracted object spectra. Globs and @lists expand.",
        json_schema_extra={"positional": True},
    )
    references: list[str] = Field(
        default_factory=list,
        description="Arc spectra. Left out, every solution in database/ is "
                    "offered.",
    )
    select: str = Field(
        "interp",
        description="How to choose: interp, nearest, preceding, following, "
                    "average, match.",
    )
    sort: str = Field(
        "utmiddle",
        description="Header keyword to sort on. The default is the "
                    "mid-exposure time setairmass writes.",
    )
    time: bool = Field(
        True, description="Is the sort key a time? Then it wraps at --timewrap."
    )
    timewrap: float = Field(
        17.0, ge=0, le=24,
        description="Hour before which a time belongs to the following day.",
    )
    database: Path = Field(
        Path("."), description="Reduction directory holding database/."
    )
    override: bool = Field(
        False, description="Replace an assignment already in the header."
    )
    assign: bool = Field(
        True, description="Write the assignment. Off, it only reports it."
    )


@register
class RefSpectra(Task[RefSpectraParams]):
    name = "refspectra"
    package = "onedspec"
    description = "Assign arc spectra to the exposures they calibrate."
    Params = RefSpectraParams

    def plan(self, params: RefSpectraParams) -> list[Step]:
        return [
            Step("times", f"{params.sort}, from each header", why=REFWHY["times"]),
            Step("sort", "arcs in time order", why=REFWHY["sort"]),
            Step("assign", f"select = {params.select}", why=REFWHY["assign"]),
            Step("write", "REFSPEC1 and REFSPEC2" if params.assign
                 else "nothing -- --no-assign was given", why=REFWHY["write"],
                 skipped=not params.assign,
                 reason="" if params.assign else "reporting only"),
        ]

    def execute(self, params: RefSpectraParams) -> TaskResult:
        from astropy.io import fits

        from ..core import images, refspec

        steps = self.plan(params)
        by_action = {step.action: step for step in steps}

        targets = [spec.path for spec in images.expand(params.input)]
        if not targets:
            advice.fail(f"nothing matched {', '.join(params.input)}.",
                        "check the path and the pattern with 'harwood sections "
                        f"{params.input[0]}', which shows what a template "
                        "expands to.")

        names = params.references or _solutions(params.database)
        if not names:
            advice.fail(
                "there are no arcs to assign: --references names them, and "
                f"database/ under {params.database} holds no solutions yet.",
                "run identify on an arc first, or name the arc frames directly "
                "with --references.")

        references, unreadable = [], []
        for name in names:
            when = _sort_key(name, params, targets)
            if when is None:
                unreadable.append(_bare(name))
            # The name that goes in the header is the bare one -- the name the
            # database indexes solutions by. A full path would not survive the
            # eighty columns of a FITS card, and would stop being true the
            # moment the reduction moved.
            references.append((_bare(name), when))

        entries = [(str(path), _sort_key(path, params, targets)) for path in targets]
        assignments = refspec.assign(entries, references, params.select)

        by_action["times"].measured = {
            "objects": len(targets),
            "arcs": len(references),
            "span": _span(entries + references),
        }
        ordered = sorted((item for item in references if item[1] is not None),
                         key=lambda item: item[1])
        by_action["sort"].measured = {
            "order": " -> ".join(f"{name} at {when % 24:.3f}h"
                                 for name, when in ordered)
        }

        warnings = []
        if unreadable:
            warnings.append(advice.warn(
                f"no time could be read for {', '.join(unreadable)}: the header "
                f"has neither {params.sort} nor UT and EXPTIME, so those arcs "
                "took no part in the assignment.",
                "run setairmass on the arcs to put UTMIDDLE in their headers, "
                "or sort on a keyword they do have with --sort."))

        rows, written = [], []
        for assignment in assignments:
            rows.append({
                "spectrum": Path(assignment.image).name,
                "arc": " + ".join(assignment.references) or "none",
                "weights": " ".join(f"{weight:.3f}"
                                    for weight in assignment.weights) or "-",
                "why": assignment.reason,
            })
            if not assignment.references:
                warnings.append(advice.warn(
                f"{Path(assignment.image).name}: no arc assigned "
                f"-- {assignment.reason}.",
                "check the frame has UT and EXPTIME in its header, and that "
                "--references names arcs from the same night."))
                continue
            if not params.assign:
                continue
            with fits.open(assignment.image, mode="update") as hdus:
                header = hdus[0].header
                existing, _ = refspec.read_keywords(header)
                if existing and not params.override:
                    warnings.append(advice.note(
                        f"{Path(assignment.image).name} already names "
                        f"{', '.join(existing)}, so it was left alone.",
                        "add --override to replace the assignment, or leave it "
                        "if the one already there is right."))
                    continue
                refspec.write_keywords(header, assignment)
                brief = refspec.Assignment(
                    _bare(assignment.image), assignment.time,
                    assignment.references, assignment.weights)
                header.add_history(
                    f"refspectra: {params.select}, " + brief.describe())
            written.append(Path(assignment.image))

        interpolated = sum(1 for item in assignments if len(item.references) > 1)
        by_action["assign"].measured = {
            "assigned": sum(1 for item in assignments if item.references),
            "from two arcs": interpolated,
            "from one": sum(1 for item in assignments
                            if len(item.references) == 1),
        }

        return TaskResult(
            outputs=written,
            steps=steps,
            warnings=warnings,
            data={"rows": rows},
            history=[],
        )


def _bare(name) -> str:
    """A frame's name without its directory or its extension."""
    text = Path(str(name)).name
    for suffix in (".ms.fits", ".fits", ".imh"):
        if text.endswith(suffix):
            return text[: -len(suffix)]
    return text


def _solutions(root: Path) -> list[str]:
    """Every arc the database has a solution for."""
    from ..core.database import listing

    return sorted(path.name[2:] for path in listing(root)
                  if path.name.startswith("id"))


def _sort_key(name, params: RefSpectraParams, near) -> float | None:
    """The time to sort a frame by, hunting for the file if given a bare name."""
    from astropy.io import fits

    from ..core import obstime

    path = _locate(name, near)
    if path is None:
        return None
    try:
        header = fits.getheader(path)
    except OSError:
        return None

    if not params.time:
        value = header.get(params.sort.upper())
        return None if value is None else float(value)

    value = obstime.middle(header, key=params.sort.upper())
    return obstime.sortable(value, params.timewrap)


def _locate(name, near):
    """Find the frame a reference names, which may be bare: 'sp0018'."""
    candidate = Path(str(name))
    if candidate.exists():
        return candidate

    stems = [candidate.name]
    for suffix in (".ms.fits", ".fits", ".ms", ""):
        if candidate.name.endswith(suffix) and suffix:
            stems.append(candidate.name[:-len(suffix)])
    folders = [candidate.parent] + [Path(other).parent for other in near]
    for folder in dict.fromkeys(folders):
        for stem in dict.fromkeys(stems):
            for suffix in (".ms.fits", ".fits", ""):
                guess = folder / f"{stem}{suffix}"
                if guess.is_file():
                    return guess
    return None


def _span(entries) -> str:
    times = [when for _, when in entries if when is not None]
    if not times:
        return "no times"
    return f"{min(times) % 24:.3f}h to {max(times) % 24:.3f}h UT"


LINEWHY = {
    "detect": "Features are found against a running-median continuum, so a "
              "sloping spectrum does not tilt the threshold. The threshold is "
              "in units of the local noise rather than counts, which means the "
              "same thing on a bright spectrum and a faint one. Absorption and "
              "emission are the same arithmetic on a flipped residual.",
    "catalogue": "The catalogue is curated, not complete, and that is "
                 "deliberate: against a complete atomic line list every "
                 "wavelength is within a hundredth of an Angstrom of something, "
                 "so a complete list answers everything confidently and is "
                 "useless. What is here are the lines that name themselves. A "
                 "feature it cannot name is reported as unidentified rather "
                 "than assigned to the nearest of a thousand possibilities.",
    "match": "Candidates are ranked by how close they are and by how likely "
             "that line is to be the one somebody is looking at. Where two "
             "species fall within the tolerance both are kept and the "
             "identification is called ambiguous -- deciding between them "
             "needs something the spectrum does not contain.",
    "velocity": "A radial velocity moves every line in the object by the same "
                "fraction, so identifications that are right agree on one "
                "shift and identifications that are wrong do not. The scatter "
                "is the useful half. Telluric bands are excluded because they "
                "do not move: they are in the air above the telescope, which "
                "makes them the check on whether the shift is real.",
}


class LineIdParams(TaskParams):
    input: str = Field(..., description="A wavelength-calibrated spectrum.")
    aperture: int = Field(1, ge=1, description="Which aperture to look at.")
    output: Path | None = Field(
        None,
        description="Write the identifications as a table. Left out, they go "
                    "to this run's own directory.",
        json_schema_extra={"positional": True},
    )
    features: Literal["absorption", "emission", "both"] = Field(
        "both", description="Which way the lines go."
    )
    threshold: float = Field(
        5.0, gt=0, description="Detection threshold, in units of the local noise."
    )
    fwidth: float = Field(
        4.0, gt=0, description="Expected full width of a feature, in pixels."
    )
    tolerance: float = Field(
        1.0, gt=0,
        description="How far from a catalogue wavelength a feature may be, in "
                    "Angstroms.",
    )
    velocity: float = Field(
        0.0,
        description="Radial velocity of the object in km/s. Features are moved "
                    "back to rest before the catalogue is consulted.",
    )
    origin: str = Field(
        "",
        description="Only consider lines of one origin: stellar, nebular, "
                    "interstellar, telluric.",
    )
    prominence: int = Field(
        1, ge=1, le=3,
        description="Ignore catalogue lines fainter than this, 1 to 3.",
    )
    plot: bool = Field(
        False,
        description="Open the spectrum with every line drawn where it falls, "
                    "coloured by what produces it.",
    )


@register
class LineId(Task[LineIdParams]):
    name = "lineid"
    package = "onedspec"
    description = "Name the atoms, ions and molecules behind the features."
    Params = LineIdParams

    def plan(self, params: LineIdParams) -> list[Step]:
        return [
            Step("detect", f"{params.features} at {params.threshold} sigma",
                 why=LINEWHY["detect"]),
            Step("catalogue", params.origin or "every origin",
                 why=LINEWHY["catalogue"]),
            Step("match", f"within {params.tolerance} A at "
                          f"{params.velocity:+.0f} km/s", why=LINEWHY["match"]),
            Step("velocity", "from the lines that were identified",
                 why=LINEWHY["velocity"]),
        ]

    def execute(self, params: LineIdParams) -> TaskResult:
        import numpy as np

        from ..core import arc, output, species, spectrum

        found = spectrum.read(params.input, number=params.aperture)
        if not found:
            advice.fail(
                f"'{params.input}' has no aperture {params.aperture}; it holds "
                f"{spectrum.count(params.input)}.",
                "use --aperture with a number in that range, or run wspectext "
                "to see what the file contains.")
        aperture = found[0]
        if aperture.wavelength is None or not np.any(aperture.wavelength):
            advice.fail(
                f"'{params.input}' has no wavelengths yet -- it is still in "
                "pixels, and naming a line needs to know what wavelength it is "
                "at.",
                "run dispcor on it first. That needs a dispersion solution, so "
                "identify on an arc comes before it.")

        if params.origin and params.origin not in species.ORIGINS:
            advice.fail(
                f"no origin called '{params.origin}'.",
                f"use one of {', '.join(species.ORIGINS)}, or leave --origin "
                "out to consider every kind of line.")

        steps = self.plan(params)
        by_action = {step.action: step for step in steps}
        wave = np.asarray(aperture.wavelength, dtype=float)
        flux = np.asarray(aperture.flux, dtype=float)

        peaks = []
        if params.features in ("absorption", "both"):
            peaks += [(peak, "absorption") for peak in arc.find_peaks(
                flux, params.fwidth, params.threshold, absorption=True)]
        if params.features in ("emission", "both"):
            peaks += [(peak, "emission") for peak in arc.find_peaks(
                flux, params.fwidth, params.threshold)]
        peaks.sort(key=lambda item: item[0].pixel)

        low, high = float(np.min(wave)), float(np.max(wave))
        available = species.within(low, high, params.prominence,
                                   params.origin or None)
        by_action["detect"].measured = {
            "features": len(peaks),
            "range": f"{low:.2f} to {high:.2f} A",
        }
        by_action["catalogue"].measured = {
            "lines in range": len(available),
            "species in range": len({line.species for line in available}),
        }

        rows, matches, warnings = [], [], []
        for peak, direction in peaks:
            # The peak is a 1-based pixel; the wavelength scale is per pixel,
            # so the position between samples is interpolated rather than
            # rounded -- a tenth of a pixel is a tenth of an Angstrom here.
            measured = float(np.interp(peak.pixel, aperture.pixel, wave))
            match = species.at(measured, params.tolerance, params.velocity,
                               params.origin or None)
            match.candidates = [item for item in match.candidates
                                if item.line.prominence >= params.prominence]
            matches.append(match)

            best = match.best
            rows.append({
                "wavelength": round(measured, 3),
                "kind": direction,
                "species": best.line.species if best else "-",
                "line": best.line.name if best else "unidentified",
                "offset": round(best.offset, 3) if best else None,
                "sure": ("yes" if match.confident else "ambiguous") if best else "-",
                "also": ", ".join(item.line.name for item in match.candidates[1:3]),
            })

        named = [match for match in matches if match.best is not None]
        by_action["match"].measured = {
            "identified": f"{len(named)} of {len(matches)}",
            "confident": sum(1 for match in named if match.confident),
            "species found": len({match.best.line.species for match in named}),
        }

        implied = species.velocity_from(named)
        if implied is None:
            by_action["velocity"].skipped = True
            by_action["velocity"].reason = (
                "fewer than three lines were identified confidently, which is "
                "not enough for a shift to mean anything")
        else:
            median, scatter = implied
            by_action["velocity"].measured = {
                "implied shift": f"{median:+.1f} km/s",
                "scatter": f"{scatter:.1f} km/s",
                "assumed": f"{params.velocity:+.1f} km/s",
            }
            # Lines that agree on a shift are a measurement; lines scattered
            # over hundreds of km/s are a set of misidentifications wearing a
            # median. Which of the two this is has to be said.
            if scatter > 3.0 * max(abs(median), 1.0):
                warnings.append(advice.problem(
                    f"the identified lines do not agree on a shift: "
                    f"{median:+.1f} km/s with a scatter of {scatter:.1f}. Lines "
                    "that are identified correctly agree on one velocity; this "
                    "scatter says several of them are wrong.",
                    "narrow --tolerance, or raise --prominence to 2 so only "
                    "the lines that name themselves are considered."))
            elif abs(median) > 2.0 * scatter and abs(median) > 20.0:
                warnings.append(advice.note(
                    f"the lines agree that everything is shifted by "
                    f"{median:+.1f} km/s (scatter {scatter:.1f}). The shift is "
                    "the object's radial velocity, give or take the wavelength "
                    "calibration.",
                    f"run it again with --velocity {median:.0f} to centre the "
                    "identifications."))

        unidentified = len(matches) - len(named)
        if unidentified:
            warnings.append(advice.note(
                f"{unidentified} of {len(matches)} features are not in the "
                "catalogue, which holds only the lines that name themselves. "
                "Unidentified here means unnamed, not unreal.",
                "nothing, usually. If you expected one of them to be named, "
                "widen --tolerance or lower --prominence, and check "
                "'harwood lines --range' for what is catalogued there."))

        ambiguous = [match for match in named if not match.confident]
        if ambiguous:
            example = ambiguous[0]
            warnings.append(advice.note(
                f"{len(ambiguous)} identifications are ambiguous -- at "
                f"{example.wavelength:.2f} A the catalogue offers "
                + " and ".join(item.line.name for item in example.candidates[:2])
                + ". Wavelength alone cannot choose between them.",
                "narrow it with --origin stellar, nebular, interstellar or "
                "telluric. What you pointed the telescope at is exactly the "
                "information the spectrum does not contain."))

        destination = params.output or output.named_for(params.input, ".lines.csv")
        _write_lines(destination, rows, params, aperture)

        if params.plot:
            from ..viewer.species_panel import run as show_lines

            show_lines(aperture, params.velocity, params.tolerance,
                       name=str(params.input))

        return TaskResult(
            outputs=[Path(destination)],
            steps=steps,
            warnings=warnings,
            data={"rows": rows},
            history=[f"lineid: {len(named)} of {len(matches)} features named"],
        )


def _write_lines(destination, rows, params: LineIdParams, aperture) -> None:
    """The identifications as a table anybody can open."""
    from ..core import species

    columns = ["wavelength", "kind", "species", "line", "offset", "sure", "also"]
    lines = [
        f"# line identifications for {params.input}, aperture {params.aperture}",
        f"# assumed radial velocity {params.velocity:+.1f} km/s, "
        f"tolerance {params.tolerance} A",
        "# wavelengths are air wavelengths in Angstroms",
        "# offset: measured minus the catalogue wavelength at that velocity",
        "# sure: 'ambiguous' means more than one species fits; see the 'also' column",
    ]
    if aperture.title:
        lines.insert(1, f"# {aperture.title}")
    lines.append(",".join(columns))
    for row in rows:
        lines.append(",".join(
            "" if row[name] is None else f'"{row[name]}"'
            if isinstance(row[name], str) and "," in row[name] else str(row[name])
            for name in columns))

    seen = {row["species"] for row in rows if row["species"] != "-"}
    if seen:
        lines.append("")
        for name in sorted(seen):
            entries = species.named(name)
            if entries and entries[0].about:
                lines.append(f"# {name}: {entries[0].about}")
    Path(destination).write_text("\n".join(lines) + "\n", encoding="utf-8")
