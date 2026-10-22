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

import contextlib
import math
from pathlib import Path
from typing import Literal

from pydantic import Field, create_model

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

    extracted = _extracted(params.image)
    aperture = None
    if extracted:
        try:
            vector, where, aperture = idn.spectrum_vector(extracted, params.section)
        except ValueError as error:
            advice.fail(f"{params.image}: {error}", "give --section 'line 2' or "
                        "'aperture 2'; 'middle line' is the middle aperture.")
        data = vector
    else:
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
    extra_ap = {}
    if aperture is not None:
        extra_ap = {"aplow": round(float(aperture.centre + aperture.low), 4),
                    "aphigh": round(float(aperture.centre + aperture.high), 4)}
    solution = database.Solution(
        image=str(params.image), pixel=pixel, section=where,
        aperture=aperture.number if aperture is not None else None,
        wavelength=np.array([np.nan if f.user is None else f.user for f in ident.features]),
        function=params.function, order=params.order, rms=float(rms), units=units,
        coordlist=params.coordlist,
        used=np.array([f.user is not None and f.wts != 0 for f in ident.features]),
        fitted=np.array([f.fit for f in ident.features]),
        extra={**extra_ap, "npts": int(vector.size),
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


def _extracted(path) -> list:
    """The apertures of a harwood spectrum file, or [] for an image."""
    from ..core import spectrum as spectra

    try:
        if spectra.count(str(path).split("[")[0]) == 0:
            return []
        return spectra.read(str(path).split("[")[0])
    except (OSError, ValueError):
        return []


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


class _DispCorParams(TaskParams):
    input: str = Field(..., description="Extracted spectra, from apall. Commas, globs "
                                        "and @lists expand.")
    output: Path | str | None = Field(
        None,
        description="Where to write, one name per input. Left out, each goes to "
                    "this run's own directory, named after its input.",
        json_schema_extra={"positional": True},
    )
    reference: str = Field(
        "",
        description="Arc whose dispersion solution to use for every input -- the "
                    "name identify was run on -- instead of the REFSPEC keywords "
                    "refspectra wrote. harwood's; IRAF reads the keywords only.",
    )
    linearize: bool = Field(
        True,
        description="Resample onto an evenly spaced grid, as IRAF does by default. "
                    "It interpolates, which correlates neighbouring pixels; with "
                    "--no-linearize the wavelength column carries the solution and "
                    "the pixels are untouched.",
    )
    table: str = Field(
        "", description="Grids per aperture: a file of 'aperture w1 w2 dw nw' lines "
                        "(INDEF for any to work out), or a linear spectrum whose "
                        "apertures' grids to copy. The others take w1, w2, dw, nw.")
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
    samedisp: bool = Field(False, description="One grid for every aperture of a "
                                              "spectrum: from the lowest to the highest "
                                              "wavelength at the finest dispersion.")
    ignoreaps: bool = Field(
        False, description="An aperture with no solution of its own takes the arc's "
                           "first one. Off, as in IRAF, that is an error.")
    listonly: bool = Field(False, description="Only say what the grids would be.")
    verbose: bool = Field(True, description="Say the grid of each aperture.")
    logfile: str = Field("", description="A file the same lines are appended to.")
    interp: Literal["nearest", "linear", "poly3", "poly5", "spline3", "sinc"] = Field(
        "poly5", description="Interpolant the resampling integrates (onedspec.interp).")
    database: Path = Field(
        Path("."), description="Reduction directory holding database/."
    )
    check: bool = Field(
        True,
        description="Look the strongest features up in the line catalogue, as "
                    "a check that the wavelengths landed on something real.",
    )


#: `global` is IRAF's name, and a word Python keeps for itself, so the field
#: is added under it this way.
DispCorParams = create_model(
    "DispCorParams", __base__=_DispCorParams,
    **{"global": (bool, Field(False, description=(
        "The same grid for an aperture in every spectrum of the list: each "
        "aperture's lowest to highest wavelength over all of them (with "
        "samedisp, one grid for all).")))})


@register
class DispCor(Task[DispCorParams]):
    """IRAF's dispcor, noao/onedspec/dispcor: t_dispcor.x and dcio.x."""

    name = "dispcor"
    package = "onedspec"
    description = "Put wavelengths on an extracted spectrum."
    Params = DispCorParams

    def plan(self, params) -> list[Step]:
        steps = [
            Step("read", params.input, why=DISPWHY["read"]),
            Step("solution",
                 f"database/id{Path(params.reference).stem}"
                 if params.reference else "the arcs refspectra assigned",
                 why=DISPWHY["solution"]),
            Step("apply", "wavelength for every pixel of every aperture",
                 why=DISPWHY["apply"]),
        ]
        if params.linearize:
            steps.append(Step("resample", "onto an even wavelength grid"
                              + (", the same for every aperture" if params.samedisp
                                 else "") +
                              (", the same in every spectrum" if getattr(params, "global")
                               else ""),
                              why=DISPWHY["resample"]))
        steps.append(Step("write", str(params.output or "this run's directory"),
                          why=DISPWHY["write"]))
        if params.check:
            steps.append(Step("check", "the strongest features, by name",
                              why=DISPWHY["check"]))
        return steps

    def execute(self, params) -> TaskResult:
        import numpy as np
        from astropy.io import fits

        from ..core import dispcor, images, output, spectrum

        # IRAF's image names leave the extension out: star3 is star3.fits.
        inputs = [name if Path(name).exists() or not Path(name + ".fits").exists()
                  else name + ".fits"
                  for name in (str(spec) for spec in images.expand(params.input))]
        if not inputs:
            advice.fail(f"'{params.input}' matches nothing.",
                        "check the path; 'harwood sections' shows what a template "
                        "expands to.", FileNotFoundError)
        outs = [part if Path(part).suffix else part + ".fits"
                for part in (p.strip() for p in images.split_list(str(params.output)))] \
            if params.output else []
        said: list[str] = []
        table = dispcor.Table(params.w1, params.w2, params.dw, params.nw)
        if params.table.strip():
            _read_table(table, params.table.strip())
        if params.ignoreaps:
            table.fill_from_first()
        cache: dict = {}
        if params.linearize and getattr(params, "global"):
            _global_grids(table, inputs, params, cache, params.samedisp)

        steps = self.plan(params)
        by_action = {step.action: step for step in steps}
        warnings: list = []
        written: list[Path] = []
        first_info = None
        for index, name in enumerate(inputs):
            apertures = spectrum.read(name)
            header = fits.getheader(name)
            solved = _dispersions(name, apertures, header, params, cache, said)
            target = outs[index] if index < len(outs) else \
                str(params.output or output.named_for(name, ".w.fits")) \
                if len(inputs) == 1 else str(output.named_for(name, ".w.fits"))
            label = Path(target).name.removesuffix(".fits")
            rows = []
            if params.linearize:
                if params.samedisp:
                    grid = dispcor.same_grid(table.default,
                                             [(s.w1, s.w2, s.dwc) for s in solved])
                    grids = [grid] * len(solved)
                    if grid[2] * (grid[1] - grid[0]) <= 0 or grid[3] < 1:
                        advice.fail(f"{name}: Error in wavelength scale",
                                    "check w1, w2, dw and nw against the solutions.")
                else:
                    grids = [dispcor.grid_for(table.entry(s.number), s.w1, s.w2, s.npts)
                             for s in solved]
                # The grid is worked out and said in wavelength; with log
                # the same ends and length are spaced in log10.
                rows = [(s.number, w1, w2, dw, nw)
                        for s, (w1, w2, dw, nw) in zip(solved, grids, strict=True)]
            else:
                rows = [(s.number, s.w1, s.w2, s.dwc, s.npts) for s in solved]
            lines = dispcor.log_lines(label, rows, params.log)
            said.extend(lines)
            if params.listonly:
                continue
            for aperture, s, row in zip(apertures, solved, rows, strict=True):
                if not params.linearize:
                    aperture.wavelength = s.dispersion(np.asarray(aperture.pixel, float))
                    continue
                _, w1, w2, dw, nw = row
                if params.log:
                    w1 = math.log10(w1)
                    dw = (math.log10(w2) - w1) / (nw - 1)
                # One inverse for the aperture's every band, as IRAF's
                # transformation keeps where its last inversion ended.
                inverse = s.fresh()
                for column in ("flux", "raw", "sky", "error"):      # apall's bands
                    values = getattr(aperture, column, None)
                    if values is None:
                        continue
                    setattr(aperture, column, dispcor.resample(
                        values, inverse, w1, dw, nw, params.flux, params.blank,
                        params.interp, params.log).astype(float))
                wave = w1 + np.arange(nw) * dw
                if getattr(aperture, "dq", None) is not None:
                    before = s.dispersion(np.asarray(aperture.pixel, float))
                    aperture.dq = _resampled_dq(aperture.dq, before,
                                                10 ** wave if params.log else wave)
                aperture.wavelength = 10 ** wave if params.log else wave
                aperture.pixel = np.arange(1.0, nw + 1.0)
            out_header = _dclog(header)
            spectrum.write(target, apertures, header=out_header, source=name)
            written.append(Path(target))
            if first_info is None:
                first_info = (apertures[0], solved[0])

        if params.logfile.strip():
            with open(params.logfile.strip(), "a", encoding="utf-8") as handle:
                handle.write("\n".join(said) + "\n")
        # The arcs and their weights, said rather than only used: a spectrum
        # between two arcs is calibrated by both, and the record should show it.
        arcs = list(dict.fromkeys(
            " + ".join(f"{name} ({weight:.3f})" if len(names) > 1 else str(name)
                       for name, weight in zip(names, weights, strict=True))
            for names, weights, _ in (value for key, value in cache.items()
                                      if key[0] == "solved")
            if names))
        by_action["solution"].measured = {
            "spectra": len(inputs),
            "arcs": "; ".join(arcs) or "the spectra's own wavelengths",
        }
        data = {"spectra": len(inputs), "text": said}
        if params.listonly or first_info is None:
            return TaskResult(outputs=written, steps=steps, warnings=warnings, data=data)

        first, solution = first_info
        low, high = float(np.min(first.wavelength)), float(np.max(first.wavelength))
        spacing = np.diff(first.wavelength)
        by_action["apply"].measured = {
            "range": f"{low:.2f} to {high:.2f} A",
            "dispersion": f"{float(np.median(spacing)):.5f} A/px",
        }
        if params.linearize:
            by_action["resample"].measured = {
                "grids": "; ".join(line.split(": ", 1)[1] for line in said
                                   if ", nw = " in line)[:300],
                "interpolant": params.interp,
                "flux": "conserved" if params.flux else "averaged",
            }
            warnings.append(advice.note(
                "linearizing interpolates, so neighbouring pixels are now "
                "correlated: a spectrum resampled twice is smoother than the "
                "data, and errors estimated from its scatter are too small.",
                "use --no-linearize when nothing downstream needs an even grid; "
                "the wavelength column carries the solution without it."))
        if solution.covered is not None:
            lo, hi = solution.covered
            outside = int(((np.arange(1, solution.npts + 1) < lo)
                           | (np.arange(1, solution.npts + 1) > hi)).sum())
            data["extrapolated"] = outside
            if outside:
                warnings.append(advice.warn(
                    f"{outside} of {solution.npts} pixels fall outside the range "
                    f"the arc lines covered ({lo:.0f} to {hi:.0f}); their "
                    "wavelengths are extrapolated, and a polynomial "
                    "extrapolates badly.",
                    "identify arc lines nearer the ends of the frame, or use only "
                    f"pixels {lo:.0f} to {hi:.0f} of this spectrum."))
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
        data.update({"apertures": len(first_info and apertures),
                     "range": f"{low:.2f} to {high:.2f} A"})
        return TaskResult(
            outputs=written, steps=steps, warnings=warnings, data=data,
            history=[f"dispcor: {len(inputs)} spectra; " + "; ".join(said[:6])])


class _Solved:
    """One aperture's dispersion function and the grid it implies (dcio's
    DC_W1, DC_W2, DC_DW, DC_NW)."""

    def __init__(self, number: int, npts: int, make, covered=None) -> None:
        self.number, self.npts, self.make, self.covered = number, npts, make, covered
        self.dispersion = make()
        self.w1 = self.dispersion.eval(1.0)
        self.w2 = self.dispersion.eval(float(npts))
        self.dwc = (self.w2 - self.w1) / (npts - 1)

    def fresh(self):
        """A dispersion function with the inverse's memory unused, as each
        dispcora call starts one."""
        return self.make()


def _curve(solution):
    """The solution's curve: IRAF's own coefficients, or harwood's fit
    refitted from the features it kept."""
    import numpy as np

    from ..core import dispcor, icfit

    saved = solution.extra.get("curve")
    if saved:
        return dispcor.Curve.from_saved(saved)
    used = solution.used if solution.used is not None else np.ones(len(solution), bool)
    xrange = solution.extra.get("xrange")
    fitted = icfit.fit(solution.pixel[used], solution.wavelength[used],
                       icfit.FitSpec(function=solution.function, order=solution.order),
                       xrange=tuple(xrange) if xrange else None)
    if fitted.model is None:
        advice.fail(
            f"the solution in database/id{solution.record} could not be refitted: "
            f"{fitted.message}",
            "open the file and look at how many features it kept; a solution with "
            "too few surviving lines for its order cannot be rebuilt. Re-run "
            "identify on that arc.", RuntimeError)
    return lambda x, model=fitted: float(model(np.array([x]))[0])


def _records(name: str, params, cache: dict) -> list:
    from ..core import database

    key = ("db", name)
    if key not in cache:
        cache[key] = database.read_all(name, params.database)
    return cache[key]


def _find(records, number: int, ignoreaps: bool, spec: str):
    """dcio's lookup: the aperture's record, else one that belongs to no
    aperture, else (ignoreaps) the first."""
    for record in records:
        if record.aperture == number:
            return record
    for record in records:
        if record.aperture is None:
            return record
    if ignoreaps:
        return records[0]
    advice.fail(f"{spec} - Missing reference for aperture {number}",
                "reidentify the arc so every aperture has a solution, or give "
                "--ignoreaps to use the arc's first one.", RuntimeError)


def _dispersions(name: str, apertures, header, params, cache: dict, said: list) -> list:
    """dc_gms for one spectrum: each aperture's dispersion function from
    its references, or its own wavelengths when it has none."""
    import numpy as np

    from ..core import database, dispcor, refspec

    spec = Path(name).name.removesuffix(".fits")
    if params.reference:
        names, weights = [params.reference], [1.0]
    else:
        names, weights = refspec.read_keywords(header)
    key = ("solved", name)
    if not names:
        if apertures[0].wavelength is None or not np.any(apertures[0].wavelength):
            known = [path.name[2:] for path in database.listing(params.database)
                     if path.name.startswith("id")]
            advice.fail(
                f"'{name}' does not say which arc calibrates it, and there is "
                "no way to guess which arc belongs to which exposure.",
                "name one with --reference, or run refspectra over the night to "
                "assign them by time. "
                + (f"This reduction has solutions for {', '.join(known)}."
                   if known else "This reduction has none yet, so run identify on an "
                                 "arc first."))
        if key not in cache:
            said.append(f"{spec}: Resampling using current coordinate system")
        solved = []
        for a in apertures:
            wave = np.asarray(a.wavelength, float)
            solved.append(_Solved(a.number, len(a.flux),
                                  lambda w=wave: dispcor.SpectrumDispersion(w)))
        cache[key] = ([], [], solved)
        return solved
    if key in cache:
        return cache[key][2]
    total = sum(weights)
    weights = [float(f"{w / total:.8g}") for w in weights]
    for i, (ref, weight) in enumerate(zip(names, weights, strict=True)):
        said.append(f"{spec}: REFSPEC{i + 1} = '{ref} {_g8(weight)}'")
    refs = [_records(ref, params, cache) for ref in names]
    shifts = [_refshift(header, f"REFSHFT{i + 1}", ref, records, params, cache, said, spec)
              for i, (ref, records) in enumerate(zip(names, refs, strict=True))]
    solved = []
    for aperture in apertures:
        number, npts = aperture.number, len(aperture.flux)
        terms, covered = [], None
        for weight, records, extra in zip(weights, refs, shifts, strict=True):
            record = _find(records, number, params.ignoreaps, spec)
            shift = float(record.extra.get("shift", 0.0)) + extra
            terms.append((weight, shift, _curve(record)))
            if covered is None and record.used is not None and np.any(record.used):
                pix = record.pixel[record.used]
                covered = (float(np.min(pix)), float(np.max(pix)))
        solved.append(_Solved(number, npts,
                              lambda t=terms, n=npts: dispcor.Dispersion(t, n), covered))
    cache[key] = (names, weights, solved)
    return solved


def _refshift(header, keyword, ref, records, params, cache, said, spec) -> float:
    """dc_refshft, the plain (mean) form: the mean shift of another arc's
    solutions added to every aperture."""
    import math

    value = str(header.get(keyword, "")).split()
    if not value or value[0] == ref:
        return 0.0
    other = _records(value[0], params, cache)
    shifts = [float(r.extra.get("shift", 0.0)) for r in other]
    mean = sum(shifts) / len(shifts)
    rms = math.sqrt(max(0.0, sum(s * s for s in shifts) / len(shifts) - mean ** 2))
    option = value[1] if len(value) > 1 else ""
    said.append(f"{spec}: {keyword} = '{value[0]} {option}', shift = {_g6(mean)}, "
                f"rms = {_g6(rms)}")
    return mean


def _g8(value: float) -> str:
    from ..core.iraffmt import printf

    return printf(float(value), "g", 8)


def _g6(value: float) -> str:
    from ..core.iraffmt import printf

    return printf(float(value), "g", 6)


def _read_table(table, name: str) -> None:
    """dc_table: a linear spectrum's grids, else a text table."""
    from ..core import spectrum

    with contextlib.suppress(OSError, ValueError):
        if spectrum.count(name):
            rows = []
            for a in spectrum.read(name):
                w = a.wavelength
                if w is None or len(w) < 2:
                    continue
                rows.append((a.number, float(w[0]), float(w[-1]),
                             float((w[-1] - w[0]) / (len(w) - 1)), len(w)))
            table.read(None, rows)
            return
    if not Path(name).exists():
        advice.fail("Can't access wavelength table",
                    f"'{name}' is neither a table of 'aperture w1 w2 dw nw' lines nor "
                    "a dispersion-corrected spectrum.", FileNotFoundError)
    table.read(name)


def _global_grids(table, inputs, params, cache, samedisp: bool) -> None:
    """dc_global and dc_global1: the grids from every spectrum of the list."""
    from astropy.io import fits

    from ..core import dispcor, spectrum

    wmin, wmax, dwmin = 3.4028234663852886e38, -3.4028234663852886e38, \
        3.4028234663852886e38
    quiet: list[str] = []
    for name in inputs:
        apertures = spectrum.read(name)
        solved = _dispersions(name, apertures, fits.getheader(name), params, cache, quiet)
        for s in solved:
            if samedisp:
                wmin = min(wmin, s.w1, s.w2)
                wmax = max(wmax, s.w1, s.w2)
                dwmin = min(dwmin, abs(s.dwc))
            else:
                grid = table.entry(s.number)
                grid.wmin = min(grid.wmin, s.w1, s.w2)
                grid.wmax = max(grid.wmax, s.w1, s.w2)
                grid.nwmax = max(grid.nwmax, s.npts)
    # The REFSPEC lines belong to the pass that writes; dcio prints them then.
    for key in [k for k in cache if k[0] == "solved"]:
        del cache[key]
    if samedisp:
        nwmax = int((wmax - wmin) / dwmin + 1.5)
        d = table.default
        d.w1, d.w2, d.dw, d.nw = dispcor.defaults(wmin, wmax, nwmax, d.w1, d.w2, d.dw,
                                                  d.nw)
        return
    for grid in [table.default, *table.entries.values()]:
        if grid.nwmax:
            grid.w1, grid.w2, grid.dw, grid.nw = dispcor.defaults(
                grid.wmin, grid.wmax, grid.nwmax, grid.w1, grid.w2, grid.dw, grid.nw)


def _resampled_dq(quality, before, after):
    """The DQ on the new grid: each output pixel takes the bits of every input
    pixel its wavelength interval overlaps, as its value was interpolated
    from them; one beyond the input's range is no data."""
    import numpy as np

    from ..core import dq

    def edges(centres):
        centres = np.asarray(centres, dtype=float)
        middle = (centres[1:] + centres[:-1]) / 2
        return np.concatenate(([2 * centres[0] - middle[0]], middle,
                               [2 * centres[-1] - middle[-1]]))

    quality = np.asarray(quality, dtype=np.int64)
    order = np.argsort(before)
    lo_in = np.minimum(edges(before)[:-1], edges(before)[1:])[order]
    hi_in = np.maximum(edges(before)[:-1], edges(before)[1:])[order]
    bits_in = quality[order]
    out_edges = edges(after)
    out = np.zeros(len(after), dtype=np.int64)
    for j in range(len(after)):
        lo, hi = sorted((out_edges[j], out_edges[j + 1]))
        first = np.searchsorted(hi_in, lo, side="right")
        last = np.searchsorted(lo_in, hi, side="left")
        if last <= first:
            out[j] = dq.NO_DATA
        else:
            out[j] = int(np.bitwise_or.reduce(bits_in[first:last]))
    return out.astype(np.int32)


def _dclog(header):
    """dc_refspec: the input's header, its REFSPEC and REFSHFT keywords kept
    as DCLOG1, DCLOG2..."""
    out = header.copy()
    n = 1
    while f"DCLOG{n}" in out:
        n += 1
    for keyword in ("REFSPEC1", "REFSPEC2", "REFSHFT1", "REFSHFT2"):
        if keyword in out:
            out[f"DCLOG{n}"] = f"{keyword} = {out[keyword]}"
            del out[keyword]
            n += 1
    for keyword in ("HRWDSPEC", "HRWDFROM"):
        out.remove(keyword, ignore_missing=True, remove_all=True)
    return out


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


REFWHY = {
    "times": "Every frame is placed by its sort keyword -- by default JD, the "
             "Julian date of mid-exposure setjd writes, worked out the same way "
             "when the header has none -- and grouped by its night, LJD, so an "
             "arc from another night is never used. With --time the keyword is "
             "an hour, wrapped at --timewrap so a night that crosses midnight "
             "sorts in the order it happened.",
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
                    "offered. As in IRAF, a reference is a spectrum whose REFSPEC1 "
                    "names itself -- or, here, one with a solution in database/.",
    )
    apertures: str = Field("", description="Apertures of the input to assign, as a "
                                           "range list: '1,3-5'. Empty is all.")
    refaps: str = Field("", description="Apertures of the references to use. Empty is all.")
    ignoreaps: bool = Field(
        True, description="Use a reference whatever its aperture. Off, each spectrum "
                          "takes only references of its own aperture (BEAM-NUM).")
    select: str = Field(
        "interp",
        description="How to choose: interp, nearest, preceding, following, "
                    "average, match.",
    )
    sort: str = Field(
        "jd", description="Header keyword to sort on. JD is the Julian date of "
                          "mid-exposure; when a header has none it is worked out as "
                          "setjd would.")
    group: str = Field(
        "ljd", description="Keyword whose value an input and its reference must share: "
                           "LJD, the night (worked out as setjd would when absent). "
                           "Empty or none groups nothing.")
    time: bool = Field(
        False, description="Is the sort key an hour of the day? Then it wraps at "
                           "--timewrap.")
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
    confirm: bool = Field(
        False, description="Ask before each assignment, as IRAF does by default: no "
                           "skips it, yes takes it, YES takes it and every one after.")
    assign: bool = Field(
        True, description="Write the assignment. Off, it only reports it."
    )
    logfiles: str = Field(
        "STDOUT", description="Where the assignments are logged: STDOUT, and files, "
                              "comma separated. IRAF's default adds a file called "
                              "logfile; harwood keeps its own log anyway.")
    verbose: bool = Field(False, description="Also log why a spectrum was passed over.")


@register
class RefSpectra(Task[RefSpectraParams]):
    """IRAF's refspectra (noao/onedspec/dispcor/ref*.x), ported: the same
    selections, groups and apertures, the same weights and lines. One
    difference is IRAF's own bug: its preceding never falls back on the
    following arc, though its manual says it does -- harwood does what the
    code does, and says how to get the other."""

    name = "refspectra"
    package = "onedspec"
    description = "Assign arc spectra to the exposures they calibrate."
    Params = RefSpectraParams

    def plan(self, params: RefSpectraParams) -> list[Step]:
        none = params.group.lower() in ("", "none")
        grouped = "" if none else f", grouped by {params.group}"
        return [
            Step("times", f"{params.sort}{grouped}", why=REFWHY["times"]),
            Step("sort", "arcs by the same keyword", why=REFWHY["sort"]),
            Step("assign", f"select = {params.select}", why=REFWHY["assign"]),
            Step("write", "REFSPEC1 and REFSPEC2" if params.assign
                 else "nothing -- --no-assign was given", why=REFWHY["write"],
                 skipped=not params.assign,
                 reason="" if params.assign else "reporting only"),
        ]

    def execute(self, params: RefSpectraParams) -> TaskResult:
        from astropy.io import fits

        from ..core import images, ranges, refspec

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
        aps = ranges.parse(params.apertures)
        raps = ranges.parse(params.refaps)
        log: list[str] = []
        verbose: list[str] = []
        notes: list[str] = []
        computed: set[str] = set()
        warnings = []

        reader = _Reader(params, targets, computed)
        solutions = set(_solutions(params.database))
        asking = params.confirm
        rows, written, kept, every = [], [], [], []
        for group, listed in _table_groups(params, targets, names, warnings):
            candidates = []
            for name in listed:
                frame, header = reader.frame(name)
                if header is None:
                    verbose.append(f"[{_bare(name)}] Reference spectrum not found")
                    continue
                own = refspec.read_keywords(header)[0]
                itself = bool(own and own[0] in (frame.name, _iraf_name(name)))
                candidates.append((frame, itself or frame.name in solutions))
            marked = [frame for frame, qualified in candidates if qualified]
            if not marked and candidates:
                # None is marked as a reference (REFSPEC1 naming itself, or a
                # solution in database/): they were named as the arcs, so they are.
                marked = [frame for frame, _ in candidates]
            else:
                verbose += [f"[{frame.name}] Not a reference spectrum"
                            for frame, qualified in candidates if not qualified]
            references = []
            for frame in marked:
                if frame.ap not in raps:
                    verbose.append(f"[{frame.name}] Wrong reference aperture: {frame.ap}")
                    continue
                references.append(frame)

            entries, paths = [], {}
            for path in group:
                frame, header = reader.frame(path)
                label = _iraf_name(path)
                existing, _ = refspec.read_keywords(header) if header is not None else ([], [])
                if existing:
                    both = " ".join(str(header.get(k, "")) for k in ("REFSPEC1", "REFSPEC2"))
                    if not params.override:
                        verbose.append(f"[{label}] Reference spectra already defined: {both}")
                        kept.append(Path(path).name)
                        continue
                    verbose.append(f"[{label}] Overriding previous reference spectra: {both}")
                if frame.ap not in aps:
                    verbose.append(f"[{label}] Wrong aperture: {frame.ap}")
                    continue
                frame.name = label
                entries.append(frame)
                paths[label] = path

            assignments = refspec.assign(entries, references, params.select,
                                         ignoreaps=params.ignoreaps, group=params.group,
                                         notes=notes)
            every += assignments
            for assignment in assignments:
                row = {"spectrum": Path(paths[assignment.image]).name,
                       "arc": " + ".join(assignment.references) or "none",
                       "weights": " ".join(f"{w:.3f}" for w in assignment.weights) or "-",
                       "why": assignment.reason}
                rows.append(row)
                if not assignment.references:
                    lines = assignment.messages or [f"[{assignment.image}] No reference "
                                                    "spectrum found"]
                    warnings.append(advice.warn(
                        "; ".join(lines) + (f" ({assignment.reason})"
                                            if assignment.reason else ""),
                        "check the arcs are from the same night (the group keyword) and, "
                        "with --no-ignoreaps, of the same aperture"))
                    continue
                lines = refspec.log_lines(assignment)
                if asking:
                    answer = _confirm(lines)
                    if answer == "no":
                        row["why"] = "not accepted"
                        continue
                    if answer == "YES":
                        asking = False
                if not asking or not params.confirm:
                    log += lines
                if not params.assign:
                    continue
                with fits.open(paths[assignment.image], mode="update") as hdus:
                    header = hdus[0].header
                    refspec.write_keywords(header, assignment)
                    header.add_history(f"refspectra: {params.select}, "
                                       + refspec.Assignment(_bare(assignment.image), None,
                                                            assignment.references,
                                                            assignment.weights).describe())
                written.append(Path(paths[assignment.image]))

        if params.select == "preceding" and any(not a.references for a in every):
            warnings.append(advice.note(
                "some spectra have no arc before them and were left unassigned: IRAF's "
                "preceding does not fall back on the following arc, though its manual "
                "says it does, and harwood does what IRAF does",
                "assign those with --select following, or use interp"))
        if kept:
            warnings.append(advice.note(
                f"{', '.join(kept)} already name{'s' if len(kept) == 1 else ''} "
                "reference spectra, so they were left alone",
                "add --override to replace the assignment, or leave it if the one "
                "already there is right"))
        if computed:
            warnings.append(advice.note(
                f"{', '.join(sorted(computed))} not in every header, so worked out as "
                "setjd would, from DATE-OBS, UT and EXPTIME",
                "run setjd first to keep them in the headers"))
        for note in notes:
            warnings.append(advice.note(note, "name the arcs with --references, in the "
                                              "order they should be used"))

        shown = _logs(params.logfiles, log, verbose if params.verbose else [],
                      params.confirm)
        by_action["times"].measured = {"objects": len(targets), "arcs": len(names)}
        by_action["assign"].measured = {
            "assigned": sum(1 for a in every if a.references),
            "from two arcs": sum(1 for a in every if len(a.references) > 1)}
        data: dict = {"rows": rows}
        if shown:
            data = {"text": shown, "rows": rows}
        return TaskResult(outputs=written, steps=steps, warnings=warnings, data=data,
                          history=[])


def _table_groups(params, targets, names, warnings):
    """The (inputs, references) groups to assign: all the inputs against all
    the references -- or, when the one reference named is a text file,
    IRAF's reference table: lines of 'inputs references', a line without
    references taking the previous line's."""
    from ..core import images

    table = Path(str(names[0])) if len(names) == 1 else None
    if table is None or not table.is_file() or table.suffix.lower() in (".fits", ".fit") \
            or table.read_bytes()[:6] == b"SIMPLE":
        return [(targets, names)]
    lists: dict[str, str] = {}
    previous = ""
    for line in table.read_text().splitlines():
        words = line.split()
        if not words or words[0].startswith("#"):
            continue
        previous = words[1] if len(words) > 1 else previous
        for spec in images.expand([words[0]]):
            lists[_iraf_name(spec.path)] = previous
    groups: dict[str, list] = {}
    for path in targets:
        refs = lists.get(_iraf_name(path))
        if refs is None:
            warnings.append(advice.warn(f"[{_iraf_name(path)}] No reference spectrum found",
                                        f"add a line for it to {table}"))
            continue
        groups.setdefault(refs, []).append(path)
    return [(group, [str(spec.path) for spec in images.expand(refs.split(","))])
            for refs, group in groups.items()]


class _Reader:
    """Each frame's sort value, aperture and group, read once."""

    def __init__(self, params, near, computed: set) -> None:
        self.params, self.near, self.computed = params, near, computed
        self.cache: dict = {}

    def frame(self, name):
        from astropy.io import fits

        from ..core import refspec

        path = _locate(name, self.near)
        key = str(path)
        if key in self.cache:
            return self.cache[key]
        if path is None:
            self.cache[key] = (refspec.Frame(_bare(name)), None)
            return self.cache[key]
        header = fits.getheader(path)
        p = self.params
        try:
            ap = int(header.get("BEAM-NUM", 1))
        except (TypeError, ValueError):
            ap = 1
        value = None
        if p.select not in ("match", "average") and p.sort.lower() not in ("", "none"):
            value = self._value(header, p.sort)
            if value is not None and p.time:
                value = (value + 24.0 - p.timewrap) % 24.0
        group = ""
        if p.select not in ("match", "average") and p.group.lower() not in ("", "none"):
            group = self._group(header, p.group)
        self.cache[key] = (refspec.Frame(_bare(name), value, ap, group), header)
        return self.cache[key]

    def _value(self, header, keyword: str):
        from ..core import obstime
        from ..core.iraffmt import ctod
        from .imutil import image_parameter

        raw = image_parameter(header, "", keyword)
        if raw is not None:
            # imgetd: the card's own text, read by IRAF's ctod -- which is
            # what decides which of two arcs equally far away is nearer
            value = ctod(raw)
            return value if value is not None else obstime.hours(raw)
        if keyword.lower() == "jd":
            found = _setjd(header)
            if found is not None:
                self.computed.add("JD")
                return found[0]
        if keyword.lower() == "utmiddle":
            return obstime.middle(header, key="UTMIDDLE")
        return None

    def _group(self, header, keyword: str) -> str:
        raw = header.get(keyword.upper())
        if raw is None and keyword.lower() == "ljd":
            found = _setjd(header)
            if found is not None:
                self.computed.add("LJD")
                raw = found[1]
        if raw is None:
            return f"<no {keyword}>"
        try:                                # 2448647 and 2448647. are one night
            number = float(raw)
            return str(int(number)) if number == int(number) else repr(number)
        except (TypeError, ValueError):
            return str(raw).strip()


def _setjd(header):
    """The JD of mid-exposure and the LJD, as setjd computes them."""
    from ..core import asttools, observatory, obstime
    from .astutil import SetJDParams, _read_for_jd

    here = observatory.from_header(header) or observatory.default()
    zone = here.timezone if here is not None else 0.0
    try:
        ujd, ljd, _ = _read_for_jd(header, SetJDParams(images=["x"]), zone, asttools,
                                   obstime)
    except ValueError:
        return None
    return ujd, int(ljd)


def _confirm(lines: list[str]) -> str:
    import sys

    for line in lines[:-1]:
        sys.stderr.write(line + "\n")
    sys.stderr.write(lines[-1] + "   Accept assignment (no|yes|YES)? ")
    sys.stderr.flush()
    answer = sys.stdin.readline().strip()
    if answer in ("YES", "yes", "no"):
        return answer
    return "yes" if answer in ("", "y") else "no"


def _logs(logfiles: str, log: list[str], verbose: list[str], confirmed: bool) -> list[str]:
    """IRAF's logfiles: STDOUT is printed (not when confirming, which has
    printed them already), anything else is appended to."""
    shown = []
    for name in (n.strip() for n in logfiles.split(",") if n.strip()):
        if name == "STDOUT":
            if not confirmed:
                shown += verbose + log
            continue
        with open(name, "a", encoding="utf-8") as out:
            out.writelines(line + "\n" for line in verbose + log)
    return shown


def _iraf_name(name) -> str:
    """refnoextn: the name as given, without .fits."""
    text = str(name)
    for suffix in (".fits", ".fit", ".imh"):
        if text.endswith(suffix):
            return text[: -len(suffix)]
    return text


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


REIDWHY = {
    "reference": "The reference's solutions: the one identify made, and any it "
                 "already has for its other apertures or lines.",
    "carry": "Each reference feature is put where the reference solution says its "
             "wavelength now falls, plus the shift, and centred on the new "
             "spectrum; the ones that cannot be centred are dropped. Apertures "
             "share an arc's lines but not its dispersion, so the solution is "
             "refitted from the features found rather than shifted.",
    "write": "One record per aperture or line in database/id<image>, where "
             "dispcor finds the aperture's own.",
}


class ReidentifyParams(TaskParams):
    reference: str = Field(..., description="Arc already identified (identify or "
                                            "autoidentify).")
    images: str = Field(..., description="Spectra to reidentify: the reference itself "
                                         "for its other apertures or lines, and the "
                                         "night's other arcs. Commas, globs, @lists.")
    interactive: Literal["no", "yes", "NO", "YES"] = Field(
        "no", description="IRAF's stop at each fit; not ported yet, so only no "
                          "runs.")
    section: str = Field("middle line", description="The reference's line, column or "
                                                    "aperture to start from.")
    newaps: bool = Field(True, description="Reidentify an aperture the reference has "
                                           "no solution for, from the last one used.")
    override: bool = Field(False, description="Redo spectra that already have a "
                                              "solution.")
    refit: bool = Field(True, description="Refit the solution from the features "
                                          "found; off, only shift it.")
    trace: bool = Field(False, description="Along a long slit, carry each line's "
                                           "solution to the next rather than the "
                                           "reference's to every line.")
    step: int = Field(10, ge=0, description="Lines between the solutions traced along "
                                            "a long slit (0: only the reference's).")
    nsum: int = Field(10, ge=1, description="Lines summed around each of them.")
    shift: str = Field("0.", description="Shift added to the reference features, in "
                                         "wavelength; INDEF to find it from the "
                                         "spectrum.")
    search: str = Field("0.", description="With shift INDEF: INDEF finds the shift "
                                          "from the spectrum's peaks, 0 uses none. "
                                          "(A radius, IRAF's pattern search, is not "
                                          "ported.)")
    nlost: int = Field(0, ge=0, description="Features that may be lost before a "
                                            "solution along the reference is refused.")
    ftype: Literal["emission", "absorption"] = Field("emission", description="Feature "
                                                                             "type.")
    fwidth: float = Field(4.0, gt=0, description="Feature width in pixels, where the "
                                                 "reference does not say.")
    cradius: float = Field(5.0, ge=0, description="Centring radius in pixels.")
    threshold: float = Field(0.0, ge=0, description="Least range of data a feature "
                                                    "must span to be centred.")
    addfeatures: bool = Field(False, description="Add lines of the list the solution "
                                                 "finds on each spectrum.")
    coordlist: str = Field("linelists$idhenear.dat", description="Line list, for "
                                                                 "addfeatures.")
    match: float = Field(-3.0, description="Matching limit for added lines.")
    maxfeatures: int = Field(50, ge=1, description="Most features to add.")
    minsep: float = Field(2.0, ge=0, description="Least separation between features, "
                                                 "px.")
    database: Path = Field(Path("."), description="Reduction directory holding "
                                                  "database/.")
    logfiles: str = Field("logfile", description="Files the log is appended to "
                                                 "(comma separated; empty for none).")
    verbose: bool = Field(False, description="Say also what is skipped and what "
                                             "becomes a new reference.")


@register
class Reidentify(Task[ReidentifyParams]):
    """IRAF's reidentify, noao/onedspec/identify/t_reidentify.x:
    core/reidentify.py."""

    name = "reidentify"
    package = "onedspec"
    description = "Carry an arc's solution to its other apertures and to other arcs."
    Params = ReidentifyParams

    def plan(self, params: ReidentifyParams) -> list[Step]:
        return [Step("reference", f"{params.reference}, {params.section}",
                     why=REIDWHY["reference"]),
                Step("carry", f"to {params.images}, shift {params.shift}"
                     + (", refitted" if params.refit else ", shifted"),
                     why=REIDWHY["carry"]),
                Step("write", "database/id<image>", why=REIDWHY["write"])]

    def execute(self, params: ReidentifyParams) -> TaskResult:
        from ..core import arc, images
        from ..core import identify as idn
        from ..core.iraffmt import INDEF, ctod

        if params.interactive.lower() == "yes":
            advice.fail("reidentify cannot stop at each fit yet: the identify panel "
                        "is not wired to it.",
                        "run it with --interactive no and read its log: a spectrum "
                        "whose RMS stands out or that lost features can be "
                        "identified again by hand with 'harwood identify "
                        "--interactive' (which starts from the lines, not from "
                        "reidentify's solution).",
                        NotImplementedError)
        value = ctod(params.shift)
        shift = None if value is None or abs(value) >= INDEF else float(value)
        value = ctod(params.search)
        search = None if value is None or abs(value) >= INDEF else float(value)
        linelist = None
        if params.addfeatures:
            linelist = idn.read_linelist(arc.resolve_linelist(params.coordlist))
        log: list[str] = []
        said: list[str] = []
        state = _ReidState(params, linelist, log, said)
        state.search = search
        reference = _bare(params.reference)
        state.reference(params.reference, reference, shift)
        for name in (str(spec) for spec in images.expand(params.images)):
            if _bare(name) != reference:
                state.image(params.reference, name, shift)
        for target in [t.strip() for t in params.logfiles.split(",") if t.strip()]:
            with open(Path(params.database) / target, "a", encoding="utf-8") as handle:
                handle.write("\n".join(["", "REIDENTIFY: harwood"] + log) + "\n")
        steps = self.plan(params)
        steps[1].measured = {"spectra reidentified": state.done,
                             "refused (too many features lost)": state.lost}
        warnings = []
        if state.lost:
            warnings.append(advice.warn(
                f"{state.lost} spectra along the reference lost more features than "
                f"nlost ({params.nlost}) allows, and have no solution.",
                "raise --nlost, or check the shift: features that cannot be "
                "centred are usually ones the shift put more than cradius away."))
        return TaskResult(outputs=state.written, steps=steps, warnings=warnings,
                          data={"text": said + log, "spectra": state.done},
                          history=[f"reidentify: {state.done} spectra from "
                                   f"{params.reference}"])


def _bare(name: str) -> str:
    text = Path(str(name)).name
    return text[:-5] if text.endswith(".fits") else text


class _ReidState:
    """ri_reference and ri_image: the references kept by aperture or line,
    and the identify state that carries from one spectrum to the next."""

    def __init__(self, params, linelist, log, said) -> None:
        self.p, self.linelist, self.log, self.said = params, linelist, log, said
        self.refs: dict = {}
        self.order: list = []
        self.current = None
        self.restored = False
        self.written: list[Path] = []
        self.done = self.lost = 0

    # -- the spectra of a file, IRAF's lines
    def lines(self, name: str):
        """[(key, section, vector, aperture)] in line order, and whether the
        file is an extracted (multispec) one."""
        import numpy as np

        from ..core import identify as idn
        from ..core import images

        path = name if Path(name).exists() or not Path(name + ".fits").exists() \
            else name + ".fits"
        extracted = _extracted(path)
        if extracted:
            return [((a.number, 1), f" - Ap {a.number}", np.asarray(a.flux, np.float32), a)
                    for a in extracted], True, path
        data, _ = images.read(path)
        if np.ndim(data) == 1:
            return [((1, 1), "", np.asarray(data, np.float32), None)], False, path
        kind, _ = idn.parse_section(self.p.section, data.shape)
        n = data.shape[0] if kind == "line" else data.shape[1]
        out = []
        for line in range(1, n + 1):
            vector, where = idn.arc_vector(data, f"{kind} {line}", self.p.nsum)
            out.append(((line, 1), where, vector, None))
        return out, False, path

    def _records(self, name):
        from ..core import database

        try:
            return database.read_all(name, self.p.database)
        except FileNotFoundError:
            return []

    def _has(self, name, section) -> bool:
        return any(r.section == section for r in self._records(name))

    def _save(self, key, ident, record=None):
        ref = self._reference_from_ident(key, ident, record)
        if key not in self.refs:
            self.order.append(key)
        self.refs[key] = ref

    def _reference_from_ident(self, key, ident, record):
        import copy

        from ..core import reidentify as ri

        return ri.Reference(key, copy.deepcopy(ident.features), ident.fit, ident.shift,
                            ident.spec, ident.fwidth, record, ident.newfeatures)

    def reference(self, given: str, bare: str, shift) -> None:
        """ri_reference."""
        from ..core import reidentify as ri

        lines, multispec, path = self.lines(given)
        records = self._records(bare)
        if not records:
            advice.fail(f"no solution for '{given}' in database/id{bare}",
                        "run identify or autoidentify on it first.", FileNotFoundError)
        start = self._start_index(lines, multispec)
        key0, section0, vector0, aperture0 = lines[start]
        record = next((r for r in records if r.section == section0), None)
        if record is None:
            advice.fail(f"Entry not found: identify {bare}{section0}",
                        "identify that aperture or line of the reference, or give "
                        "--section for the one identified.", FileNotFoundError)
        ref = ri.reference_from(record, key0, self.p.fwidth)
        self.refs[key0] = ref
        self.order.append(key0)
        self.current = ref
        nreid = max(1, len(ref.features) - self.p.nlost)
        step = 1 if multispec else (self.p.step or len(lines))
        if not self.p.override:
            for index in list(range(start - step, -1, -step)) + \
                    list(range(start + step, len(lines), step)):
                key, section, _, _ = lines[index]
                other = next((r for r in records if r.section == section), None)
                if other is not None:
                    self.refs[key] = ri.reference_from(other, key, self.p.fwidth)
                    if key not in self.order:
                        self.order.append(key)
        loghdr = True
        for direction in (-1, 1):
            # Going down the reference is the one read from the database;
            # going up it has been restored by id_getap (see ri.start).
            self.current, self.restored = ref, direction > 0
            # ri_reidentify's fit_shift is in and out: without trace each
            # line starts from the shift measured on the one before, plus
            # the given shift per step (minus, going down).
            fit_shift = None if shift is None else 0.0
            first = start + direction * step
            indices = range(first, -1, -step) if direction < 0 else \
                range(first, len(lines), step)
            for index in indices:
                key, section, vector, aperture = lines[index]
                if not self.p.override and self._has(bare, section):
                    continue
                if self.p.trace:
                    base, restored = self.current, self.restored
                else:
                    base, restored = ref, True
                ident = ri.start(base, vector, self.linelist, self.p, restored)
                if loghdr:
                    self.log.extend(ri.header(bare, bare, self.p.refit))
                    loghdr = False
                if shift is None:
                    fit_shift = None
                elif not self.p.trace:
                    fit_shift = fit_shift + direction * shift
                else:
                    fit_shift = direction * shift
                result = ri.reidentify(ident, fit_shift, self.search, self.p.refit,
                                       self.p.addfeatures, self.p.maxfeatures)
                fit_shift = result.fit_shift
                self.log.append(ri.log_line(bare + section, result))
                if not result.monotonic:
                    self.log.append("Non-monotonic dispersion function")
                if len(result.features) < nreid:
                    self.log.append("    ** Too many features lost **")
                    self.lost += 1
                    if self.p.trace:
                        break
                    continue
                if result.features:
                    self._write(bare, section, aperture, ident, len(vector))
                    self._save(key, ident)
                    self.current, self.restored = self.refs[key], False

    def _start_index(self, lines, multispec: bool) -> int:
        from ..core import identify as idn

        if multispec:
            apertures = [line[3] for line in lines]
            _, where, aperture = idn.spectrum_vector(apertures, self.p.section)
            return next(i for i, line in enumerate(lines) if line[3] is aperture)
        if len(lines) == 1:
            return 0
        words = self.p.section.lower().split()
        n = len(lines)
        where = next((w for w in words if w not in ("line", "column", "lines",
                                                    "columns")), "middle")
        index = {"middle": (n + 1) // 2, "first": 1, "last": n}.get(where)
        return (int(where) if index is None else index) - 1

    def image(self, reference: str, given: str, shift) -> None:
        """ri_image."""
        from ..core import reidentify as ri

        lines, multispec, path = self.lines(given)
        bare = _bare(given)
        loghdr = True
        if multispec:
            for key, section, vector, aperture in lines:
                ref = self.refs.get(key)
                if ref is None:
                    if not self.p.newaps:
                        if self.p.verbose:
                            self.said.append(f"{bare}: Reference for aperture {key[0]} "
                                             "not found")
                        continue
                    # A new aperture starts from whatever is in memory: the
                    # aperture done last.
                    base, restored = self.current, self.restored
                else:
                    base, restored = ref, True
                    self.current, self.restored = ref, True
                if not self.p.override and self._has(bare, section):
                    continue
                if base is None:
                    raise RuntimeError("No features in reference")
                ident = ri.start(base, vector, self.linelist, self.p, restored)
                if loghdr:
                    self.log.extend(ri.header(_bare(reference), bare, self.p.refit))
                    loghdr = False
                result = ri.reidentify(ident, shift, self.search, self.p.refit,
                                       self.p.addfeatures, self.p.maxfeatures)
                self.log.append(ri.log_line(bare + section, result))
                if not result.monotonic:
                    self.log.append("Non-monotonic dispersion function")
                self.current = self._reference_from_ident(key, ident, None)
                self.restored = False
                if result.features:
                    self._write(bare, section, aperture, ident, len(vector))
                    if ref is None and self.p.newaps:
                        self._save(key, ident)
                        if self.p.verbose:
                            self.said.append(f"{bare}: New reference for aperture "
                                             f"{key[0]}")
            return
        by_key = {line[0]: line for line in lines}
        for key in list(self.order):
            if key not in by_key:
                continue
            _, section, vector, aperture = by_key[key]
            if not self.p.override and self._has(bare, section):
                continue
            ident = ri.start(self.refs[key], vector, self.linelist, self.p, True)
            if loghdr:
                self.log.extend(ri.header(_bare(reference), bare, self.p.refit))
                loghdr = False
            result = ri.reidentify(ident, shift, self.search, self.p.refit,
                                   self.p.addfeatures, self.p.maxfeatures)
            self.log.append(ri.log_line(bare + section, result))
            if result.features:
                self._write(bare, section, aperture, ident, len(vector))

    def _write(self, bare, section, aperture, ident, npts) -> None:
        import numpy as np

        from ..core import database
        from ..core import reidentify as ri

        spec = ident.spec
        extra = {"npts": int(npts), "xrange": [1.0, float(npts)],
                 "niterate": spec.niterate, "low_reject": spec.low_reject,
                 "high_reject": spec.high_reject, "sample": spec.sample,
                 "grow": spec.grow, "naverage": spec.naverage, "fwidth": ident.fwidth}
        if ident.shift:
            extra["shift"] = float(ident.shift)
        if isinstance(ident.fit, ri.SavedFit) and not self.p.refit:
            extra["curve"] = list(ident.fit.curve_values)
        if aperture is not None:
            extra["aplow"] = round(float(aperture.centre + aperture.low), 4)
            extra["aphigh"] = round(float(aperture.centre + aperture.high), 4)
        features = ident.features
        solution = database.Solution(
            image=bare, section=section,
            aperture=aperture.number if aperture is not None else None,
            pixel=np.array([f.pix for f in features]),
            wavelength=np.array([np.nan if f.user is None else f.user for f in features]),
            fitted=np.array([f.fit for f in features]),
            used=np.array([f.user is not None and f.wts != 0 for f in features]),
            function=spec.function, order=spec.order, rms=float(ident.rms()),
            units=getattr(self.linelist, "units", "Angstroms") or "Angstroms",
            coordlist=self.p.coordlist, extra=extra)
        destination = database.write(solution, self.p.database)
        ident.newfeatures = False                       # id_dbwrite
        if destination not in self.written:
            self.written.append(destination)
        self.done += 1


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
            # median. Which of the two this is has to be said. A pixel is the
            # least a line centre can scatter by: a star at rest, its lines a
            # pixel apart, is not a set of misidentifications.
            pixel_kms = 299792.458 * float(np.median(np.abs(np.diff(wave)))
                                           / np.median(wave))
            if scatter > 3.0 * max(abs(median), pixel_kms, 1.0):
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
