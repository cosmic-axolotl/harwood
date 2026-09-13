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

from ..core.registry import register
from ..core.task import Step, Task, TaskParams, TaskResult

WHY = {
    "read": "A two-dimensional arc is collapsed to one dimension by adding "
            "several rows or columns across the slit. The lamp fills the slit "
            "evenly, so adding across it costs nothing and buys signal -- which "
            "is why identify does this itself rather than requiring an "
            "extraction first.",
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
           "disperses as a sine, and the camera adds its own distortion. Three "
           "or four terms of a Chebyshev polynomial follow that; more terms "
           "start following the errors in the line centres instead.",
    "write": "The solution goes to database/id<image>, where reidentify, "
             "refspectra and dispcor all look for it. Rejected and deleted "
             "features are kept in the file: knowing which lines were thrown "
             "out is part of knowing the solution.",
}


class IdentifyParams(TaskParams):
    image: str = Field(..., description="Arc spectrum. A 2-D frame is collapsed first.")

    coordlist: str = Field(
        "linelists$thorium.dat",
        description="Line list of known wavelengths. IRAF's linelists$ prefix works "
                    "when IRAF is installed; a path works anywhere.",
    )
    section: Literal["middle line", "middle column"] = Field(
        "middle line",
        description="Which way the dispersion runs, for a 2-D frame. 'middle line' "
                    "means the dispersion is along the rows.",
    )
    nsum: int = Field(
        10, ge=1,
        description="How many lines or columns to add together before finding peaks.",
    )

    fwidth: float = Field(
        4.0, gt=0, description="Expected full width of an arc line, in pixels."
    )
    threshold: float = Field(
        10.0, gt=0, description="Peak detection threshold, in units of the local noise."
    )
    minsep: float = Field(
        2.0, gt=0, description="Two maxima closer than this are one blended line."
    )
    match: float = Field(
        1.5, gt=0,
        description="How close a peak must fall to a catalogue line to be called that "
                    "line, in pixels.",
    )
    maxfeatures: int = Field(
        100, ge=2, description="Most features to keep in the solution."
    )

    crval: float | None = Field(
        None,
        description="Approximate wavelength at the middle of the spectrum. Give this "
                    "and --cdelt and the lines are matched automatically.",
    )
    cdelt: float | None = Field(
        None,
        description="Approximate dispersion, in wavelength units per pixel. Negative "
                    "if wavelength decreases with pixel number.",
    )
    crval_search: float = Field(
        0.0, ge=0, description="Widen the search around --crval by this much."
    )
    cdelt_search: float = Field(
        0.0, ge=0, description="Widen the search around --cdelt by this fraction."
    )

    function: Literal["legendre", "chebyshev", "spline1", "spline3"] = Field(
        "chebyshev", description="Function fitted to wavelength against pixel."
    )
    order: int = Field(3, ge=1, description="Number of terms in the fit, as in IRAF.")
    sample: str = Field("*", description="Pixel ranges to fit over.")
    niterate: int = Field(0, ge=0, description="Rejection passes over the solution.")
    low_reject: float = Field(
        3.0, ge=0, description="Rejection threshold below the fit, in sigma."
    )
    high_reject: float = Field(
        3.0, ge=0, description="Rejection threshold above the fit, in sigma."
    )
    grow: float = Field(
        0.0, ge=0,
        description="Also reject features within this distance of a rejected one.",
    )

    interactive: bool = Field(
        False,
        description="Mark lines and fit by hand, with the icfit keys. The way to do "
                    "this when no approximate dispersion is known.",
    )
    database: Path = Field(
        Path("."), description="Reduction directory holding database/."
    )
    units: str = Field("Angstroms", description="Wavelength units, for the record.")
    maxrms: float = Field(
        0.5, gt=0,
        description="Refuse to write a solution scattering by more than this many "
                    "pixels. An arc line centre is good to about a tenth of a pixel, "
                    "so anything near one means the lines are misidentified.",
    )


@register
class Identify(Task[IdentifyParams]):
    name = "identify"
    package = "onedspec"
    description = "Identify arc lines and fit the dispersion solution."
    Params = IdentifyParams

    def plan(self, params: IdentifyParams) -> list[Step]:
        steps = [
            Step("read", f"{params.image}, {params.section}, nsum {params.nsum}",
                 why=WHY["read"]),
            Step("find", f"peaks wider than {params.minsep} px, above "
                         f"{params.threshold} sigma, fwidth {params.fwidth}",
                 why=WHY["find"]),
            Step("catalogue", params.coordlist, why=WHY["catalogue"]),
        ]
        if params.interactive:
            steps.append(Step("mark", "by hand, in the identify panel",
                              why=WHY["mark"]))
        elif params.crval is not None and params.cdelt is not None:
            steps.append(Step("match", f"around crval {params.crval}, "
                                       f"cdelt {params.cdelt}"
                              + (f" +/- {params.cdelt_search:.0%}"
                                 if params.cdelt_search else ""),
                              why=WHY["match"]))
        else:
            steps.append(Step("match", "no crval/cdelt given -- searching blind, "
                                       "which a dense line list may defeat",
                              why=WHY["search"]))
        steps.append(Step("fit", f"{params.function} order {params.order}"
                                 + (f", {params.niterate} rejection passes"
                                    if params.niterate else ""),
                          why=WHY["fit"]))
        steps.append(Step("write", str(_database_path(params)), why=WHY["write"]))
        return steps

    def execute(self, params: IdentifyParams) -> TaskResult:
        import numpy as np

        from ..core import arc, database, icfit, images

        data, header = images.read(params.image)
        dispaxis = 2 if params.section == "middle line" else 1
        flux = (arc.collapse(data, dispaxis=dispaxis, nsum=params.nsum)
                if data.ndim > 1 else data)
        npixels = flux.size

        peaks = arc.find_peaks(flux, fwidth=params.fwidth,
                               threshold=params.threshold, minsep=params.minsep)
        if len(peaks) < 4:
            raise RuntimeError(
                f"only {len(peaks)} arc lines found. Lower --threshold (now "
                f"{params.threshold}) or check --fwidth (now {params.fwidth})."
            )
        catalogue, _ = arc.read_linelist(params.coordlist)

        spec = icfit.FitSpec(function=params.function, order=params.order,
                             sample=params.sample, niterate=params.niterate,
                             low_reject=params.low_reject,
                             high_reject=params.high_reject, grow=params.grow)
        warnings: list[str] = []

        if params.interactive:
            from ..viewer.identify_panel import run as run_identify

            fit, pixel, wavelength = run_identify(
                flux, peaks, catalogue, spec, title=f"identify · {params.image}",
                match=params.match, units=params.units)
        else:
            found = self._match(params, peaks, catalogue, npixels, warnings)
            if not found:
                raise RuntimeError(
                    "no dispersion solution could be found automatically. Give "
                    "--crval and --cdelt if you know roughly where the spectrum "
                    "sits, or use --interactive and mark a few lines by hand -- "
                    "which is what IRAF's identify asks for, and for this reason."
                )
            pixel, wavelength = found.pixel, found.wavelength
            fit = icfit.fit(pixel, wavelength, spec)
            # IRAF's `l`: once a solution exists, place the rest of the list.
            if fit.model is not None:
                more_pixel, more_wave = arc.add_from_list(
                    peaks, fit, catalogue, tolerance=params.match)
                if more_pixel.size > pixel.size:
                    pixel, wavelength = more_pixel, more_wave
                    fit = icfit.fit(pixel, wavelength, spec)

        if fit.model is None:
            raise RuntimeError(f"the dispersion fit failed: {fit.message}")

        used = fit.mask & ~fit.deleted
        scale = float(np.median(np.abs(np.gradient(fit(np.arange(1.0, npixels + 1.0))))))
        rms_pixels = fit.rms / scale if scale else float("nan")

        # A solution this loose is not a solution. An arc line centre is good to
        # about a tenth of a pixel, so scatter near a whole pixel means the peaks
        # were matched to the wrong catalogue entries -- and a cubic will fit
        # wrong entries beautifully. Writing it to the database would hand
        # dispcor a wavelength scale that is quietly wrong for every spectrum
        # taken that night, which is the failure this project exists to prevent.
        if rms_pixels > params.maxrms and not params.interactive:
            raise RuntimeError(
                f"the solution scatters by {rms_pixels:.2f} px ({fit.rms:.4f} "
                f"{params.units}), well beyond the {params.maxrms} px a line centre "
                f"is uncertain by, so the {int(used.sum())} lines are probably "
                "matched to the wrong catalogue entries. Nothing was written. "
                "Sharpen --crval and --cdelt, or use --interactive and judge it "
                "yourself; --maxrms lifts this if you have a reason."
            )
        if rms_pixels > 0.3:
            warnings.append(
                f"the solution scatters by {rms_pixels:.2f} px, which is looser than "
                "an arc line centre. Check the identified lines before relying on it."
            )

        steps = self.plan(params)
        by_action = {step.action: step for step in steps}
        by_action["read"].measured = {"pixels": npixels}
        by_action["find"].measured = {
            "peaks": len(peaks),
            "typical width": f"{float(np.median([p.width for p in peaks])):.2f} px",
        }
        by_action["catalogue"].measured = {"lines in list": catalogue.size}
        if "match" in by_action:
            by_action["match"].measured = {"identified": len(pixel)}
        by_action["fit"].measured = {
            "used": f"{int(used.sum())} of {len(pixel)}",
            "rms": f"{fit.rms:.4f} {params.units}",
            "in pixels": f"{rms_pixels:.3f} px"
                         + ("  -- about a line centre's precision"
                            if rms_pixels < 0.2 else "  -- looser than a line centre"),
            "dispersion": f"{scale:.5f} {params.units}/px",
        }

        solution = database.Solution(
            image=str(params.image), pixel=pixel, wavelength=wavelength,
            function=params.function, order=params.order, rms=float(fit.rms),
            units=params.units, coordlist=params.coordlist,
            used=used, fitted=fit.values,
        )
        destination = database.write(solution, params.database)

        low, high = float(fit(np.array([1.0]))[0]), float(fit(np.array([float(npixels)]))[0])
        return TaskResult(
            outputs=[destination],
            data={"peaks_found": len(peaks), "features": len(pixel),
                  "used": int(used.sum()), "rms": round(float(fit.rms), 5),
                  "rms_pixels": round(rms_pixels, 3),
                  "dispersion": round(scale, 5),
                  "range": f"{min(low, high):.2f} to {max(low, high):.2f} {params.units}"},
            history=[f"identify: {int(used.sum())} features, {params.function} "
                     f"order {params.order}, rms {fit.rms:.4f} {params.units}, "
                     f"from {params.coordlist}"],
            steps=steps,
            warnings=warnings,
        )

    def _match(self, params, peaks, catalogue, npixels, warnings):

        from ..core import arc

        if params.crval is not None and params.cdelt is not None:
            return arc.autoidentify(
                peaks, catalogue, crval=params.crval, cdelt=params.cdelt,
                npixels=npixels, crval_search=params.crval_search,
                cdelt_search=params.cdelt_search, tolerance=params.match)

        warnings.append(
            "no --crval or --cdelt given, so the dispersion was searched for blind. "
            "Against a dense line list that is not reliable: check the reported "
            "range and rms before trusting the solution."
        )
        return arc.search_dispersion(
            peaks, catalogue,
            window=(float(catalogue.min()), float(catalogue.max())),
            cdelt_range=(0.02, 5.0), npixels=npixels)


def _database_path(params: IdentifyParams) -> Path:
    from ..core import database

    return database.path_for(params.image, params.database)

