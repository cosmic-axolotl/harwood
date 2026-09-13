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
    "find": "The peaks in the spatial profile are the spectra. One for a star on "
            "a long slit, many for fibres or an echelle.",
    "edit": "The finder is right most of the time, and the times it is not are "
            "the ones that matter: two stars it merged, a galaxy reaching past "
            "the threshold, a sky band landing on a neighbour. You have to look "
            "at the profile, so this shows it.",
    "resize": "The aperture is set where the profile falls to a fraction of its "
              "peak, rather than to a fixed width, because a fibre and a long "
              "slit want very different apertures and the profile knows which "
              "it is looking at. Too narrow loses light as the seeing changes; "
              "too wide adds sky noise for nothing.",
    "trace": "The spectrum does not sit on one column. The camera has "
             "distortion and the spectrograph flexes as the telescope tracks, "
             "so the centre wanders by a pixel or several from one end of the "
             "dispersion to the other. Following it and fitting a low-order "
             "curve is what keeps the aperture on the light.",
    "background": "Sky is measured in bands either side of the aperture and "
                  "subtracted. The gap matters: the wings of the profile reach "
                  "further than the aperture does, and sky measured on them is "
                  "sky that includes the star.",
    "extract:none": "A plain sum of the pixels inside the aperture. Every pixel "
                    "counts the same, which is right when they all have similar "
                    "signal and wrong when the edges are mostly noise.",
    "extract:variance": "Optimal extraction (Horne 1986): each pixel is weighted "
                        "by how much of the profile it holds and divided by its "
                        "variance, so the noisy edges of the aperture stop "
                        "diluting the bright middle. It recovers signal-to-noise "
                        "a plain sum throws away -- and it must not change the "
                        "flux, only the noise, which is worth checking.",
    "write": "One FITS file, one binary table per aperture, with the pixel, the "
             "flux, the uncertainty and the sky in it. Nothing has to be "
             "recomputed to read it, and `wspectext` turns it into text without "
             "knowing anything the file does not say.",
}


class ApAllParams(TaskParams):
    input: str = Field(..., description="Two-dimensional frame to extract from.")
    output: Path | None = Field(
        None,
        description="Where to write the spectra. Left out, it goes to this run's "
                    "own directory under output/, named after the frame.",
        json_schema_extra={"positional": True},
    )

    dispaxis: Literal[1, 2] = Field(
        2,
        description="Which way the dispersion runs: 2 along the rows (the long "
                    "axis of most long-slit frames), 1 along the columns.",
    )
    nfind: int = Field(1, ge=1, description="How many spectra to find.")
    minsep: float = Field(
        5.0, gt=0, description="Closest two apertures may be, across the slit."
    )
    threshold: float = Field(
        5.0, ge=0, description="How far above the background a peak must be, in "
                               "units of the profile's own scatter."
    )
    nsum: int = Field(
        20, ge=1,
        description="How much of the dispersion to add when finding and when "
                    "re-centring during the trace.",
    )

    lower: float | None = Field(
        None, description="Aperture's lower edge, relative to its centre. "
                          "Measured from the profile when not given."
    )
    upper: float | None = Field(
        None, description="Aperture's upper edge, relative to its centre."
    )
    ylevel: float = Field(
        0.05, gt=0, lt=1,
        description="Fraction of the peak at which the aperture edge is set when "
                    "--lower and --upper are not given.",
    )

    edit: bool = Field(
        False,
        description="Adjust the apertures by hand before extracting, on the "
                    "profile they came from. The times the finder is wrong are "
                    "the ones that matter, and none of them is a parameter you "
                    "can set in advance.",
    )
    trace: bool = Field(
        True, description="Follow the spectrum along the dispersion. Turning this "
                          "off assumes it sits on one column, which it does not."
    )
    t_step: int = Field(
        10, ge=1, description="How far apart the trace's sample points are."
    )
    t_function: Literal["legendre", "chebyshev", "spline1", "spline3"] = Field(
        "legendre", description="Function fitted to the trace."
    )
    t_order: int = Field(3, ge=1, description="Number of terms in the trace fit.")
    t_niterate: int = Field(2, ge=0, description="Rejection passes over the trace.")
    t_interactive: bool = Field(
        False, description="Fit the trace by hand, in the icfit panel."
    )

    background: bool = Field(True, description="Subtract a background.")
    b_inner: float = Field(
        3.0, ge=0, description="Gap between the aperture edge and the background "
                               "band, in pixels."
    )
    b_outer: float = Field(
        15.0, gt=0, description="Outer edge of the background band, from the "
                                "aperture edge."
    )

    weights: Literal["none", "variance"] = Field(
        "none",
        description="'variance' is optimal extraction: better signal-to-noise on "
                    "a faint spectrum, at the cost of assuming the profile.",
    )
    gain: float = Field(1.0, gt=0, description="Detector gain in electrons per ADU.")
    rdnoise: float = Field(0.0, ge=0, description="Read noise in electrons.")

    database: Path = Field(
        Path("."), description="Reduction directory holding database/."
    )


@register
class ApAll(Task[ApAllParams]):
    name = "apall"
    package = "apextract"
    description = "Find, trace and extract spectra from a 2-D frame."
    Params = ApAllParams

    def plan(self, params: ApAllParams) -> list[Step]:
        steps = [
            Step("read", f"{params.input}, dispersion along "
                         f"{'rows' if params.dispaxis == 2 else 'columns'}",
                 why=WHY["read"]),
            Step("profile", f"collapse {params.nsum} of the dispersion",
                 why=WHY["profile"]),
            Step("find", f"up to {params.nfind} peak(s), at least {params.minsep} "
                         f"px apart, above {params.threshold} sigma",
                 why=WHY["find"]),
            Step("resize",
                 f"{params.lower:+g} to {params.upper:+g}"
                 if params.lower is not None and params.upper is not None
                 else f"edges where the profile falls to {params.ylevel:.0%}",
                 why=WHY["resize"]),
        ]
        if params.edit:
            steps.append(Step("edit", "by hand, on the spatial profile",
                              why=WHY["edit"]))
        if params.trace:
            steps.append(Step("trace", f"every {params.t_step} px, "
                                       f"{params.t_function} order {params.t_order}"
                                       + (", by hand" if params.t_interactive else ""),
                              why=WHY["trace"]))
        if params.background:
            steps.append(Step("background",
                              f"bands {params.b_inner} to {params.b_outer} px "
                              f"outside each edge", why=WHY["background"]))
        steps.append(Step("extract", params.weights,
                          why=WHY[f"extract:{params.weights}"]))
        steps.append(Step("write", str(params.output or "this run's directory"),
                          why=WHY["write"]))
        return steps

    def execute(self, params: ApAllParams) -> TaskResult:
        import numpy as np

        from ..core import apextract, database, icfit, images, output, spectrum
        from ..core.task import Task as _Task

        data, header = images.read(params.input)
        if data.ndim != 2:
            raise ValueError(f"'{params.input}' is {data.ndim}-dimensional; apall "
                             "extracts spectra from a 2-D frame")

        steps = self.plan(params)
        by_action = {step.action: step for step in steps}

        spatial = apextract.profile(data, dispaxis=params.dispaxis,
                                    nsum=params.nsum)
        by_action["profile"].measured = {
            "across the slit": f"{spatial.size} px",
            "along the dispersion": f"{data.shape[0 if params.dispaxis == 2 else 1]} px",
        }

        peaks = apextract.find(spatial, nfind=params.nfind, minsep=params.minsep,
                               threshold=params.threshold)
        if not peaks:
            raise RuntimeError(
                f"no spectrum found in {params.input}. Lower --threshold (now "
                f"{params.threshold}), or check --dispaxis: with the axes the "
                "wrong way round the profile is the spectrum, not the slit."
            )
        by_action["find"].measured = {
            "found": len(peaks),
            "at": ", ".join(f"{peak['centre']:.2f}" for peak in peaks[:6]),
        }

        found = [{"centre": peak["centre"],
                  **self._limits(spatial, peak["centre"], params)}
                 for peak in peaks]
        band = [params.b_inner, params.b_outer] if params.background else None
        if params.edit:
            from ..viewer.aperture_panel import run as run_apedit

            before = len(found)
            found, band = run_apedit(spatial, found, band, params.ylevel,
                                     title=f"apedit · {params.input}")
            if not found:
                raise RuntimeError(
                    "every aperture was deleted, so there is nothing to extract."
                )
            by_action["edit"].measured = {
                "kept": len(found),
                "changed": "yes" if len(found) != before else "edges or centres",
            }

        spec = icfit.FitSpec(function=params.t_function, order=params.t_order,
                             niterate=params.t_niterate)
        apertures, traces = [], []
        for number, chosen in enumerate(found, start=1):
            centre = chosen["centre"]
            low, high = chosen["low"], chosen["high"]

            if params.trace:
                fit = self._trace(data, centre, params, spec)
                if fit.model is None:
                    raise RuntimeError(f"aperture {number}: {fit.message}")
                traces.append(fit)
                centre_at = fit
            else:
                traces.append(None)

                def centre_at(position, value=centre):
                    """No trace: the aperture sits on one column, which it does
                    not really, and --no-trace is how you say you know that."""
                    return np.full(np.shape(position), value)

            measured = apextract.extract(
                data, centre_at, low, high, dispaxis=params.dispaxis,
                background=tuple(band) if band else None,
                gain=params.gain, rdnoise=params.rdnoise, weights=params.weights)

            apertures.append(spectrum.Aperture(
                number=number, centre=centre, low=low, high=high,
                pixel=measured["pixel"], flux=measured["flux"],
                error=measured["error"], sky=measured["sky"], raw=measured["raw"],
                title=str(header.get("OBJECT", "")),
                meta={"EXPTIME": float(header.get("EXPTIME", 0.0)),
                      "EPADU": params.gain, "RDNOISE": params.rdnoise,
                      "WEIGHTS": params.weights},
            ))

        widths = [aperture.width for aperture in apertures]
        by_action["resize"].measured = {
            "widths": ", ".join(f"{width:.1f}" for width in widths[:6]) + " px",
        }
        if params.trace and traces[0] is not None:
            fit = traces[0]
            length = data.shape[0 if params.dispaxis == 2 else 1]
            ends = fit(np.array([1.0, float(length)]))
            by_action["trace"].measured = {
                "points": f"{int((fit.mask & ~fit.deleted).sum())} of {fit.x.size}",
                "rms": f"{fit.rms:.3f} px",
                "drift": f"{ends[1] - ends[0]:+.2f} px end to end",
            }
        if params.background and "background" in by_action:
            by_action["background"].measured = {
                "sky": f"{float(np.median(apertures[0].sky)):.1f} counts per pixel",
                "band": f"{band[0]:g} to {band[1]:g} px outside each edge"
                        if band else "none",
            }

        first = apertures[0]
        good = np.isfinite(first.flux) & (first.error > 0)
        by_action["extract"].measured = {
            "median flux": f"{float(np.median(first.flux[good])):.1f}",
            "median S/N": f"{float(np.median(first.flux[good] / first.error[good])):.1f}",
        }
        by_action["extract"].before = _Task.statistics(data)
        by_action["extract"].after = _Task.statistics(first.flux)

        destination = params.output or output.named_for(params.input, ".ms.fits")
        spectrum.write(destination, apertures, header=header,
                       source=str(params.input))
        stored = database.write(
            _solution(params, apertures, traces), params.database)

        return TaskResult(
            outputs=[destination, stored],
            steps=steps,
            data={"apertures": len(apertures),
                  "centres": ", ".join(f"{a.centre:.2f}" for a in apertures),
                  "widths": ", ".join(f"{a.width:.1f}" for a in apertures)},
            history=[f"apall: {len(apertures)} aperture(s), weights="
                     f"{params.weights}, trace {params.t_function} order "
                     f"{params.t_order}"],
        )

    def _limits(self, spatial, centre, params) -> dict:
        """Where this aperture's edges go, before anyone edits them."""
        from ..core import apextract

        if params.lower is not None and params.upper is not None:
            return {"low": params.lower, "high": params.upper}
        low, high = apextract.edges(spatial, centre, params.ylevel)
        return {"low": low, "high": high}

    def _trace(self, data, centre, params, spec):
        from ..core import apextract

        fit = apextract.trace(data, centre, dispaxis=params.dispaxis,
                              step=params.t_step, nsum=params.nsum, spec=spec)
        if params.t_interactive and fit.model is not None:
            from ..viewer.icfit_panel import run as run_icfit

            fit = run_icfit(fit.x, fit.y, spec,
                            title=f"aptrace · {params.input}",
                            xlabel="position along the dispersion (pixels)",
                            ylabel="aperture centre (pixels)")
        return fit


def _solution(params, apertures, traces):
    """The apertures, for database/ap<image>, so the next frame can reuse them."""
    import numpy as np

    from ..core import database

    centres = np.array([aperture.centre for aperture in apertures])
    return database.Solution(
        image=str(params.input),
        pixel=centres,
        wavelength=np.array([aperture.width for aperture in apertures]),
        function=params.t_function, order=params.t_order,
        rms=float(traces[0].rms) if traces and traces[0] is not None else 0.0,
        units="pixels",
        used=np.ones(centres.size, dtype=bool),
        fitted=centres,
        extra={"task": "apall",
               "apertures": [{"number": a.number, "centre": round(a.centre, 4),
                              "low": round(a.low, 4), "high": round(a.high, 4)}
                             for a in apertures]},
    )


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
            raise ValueError(
                f"'{params.input}' has no aperture {params.aperture}; it holds "
                f"{total}."
            )
        aperture = found[0]
        table = aperture.columns()
        if params.columns:
            wanted = [name.strip() for name in params.columns.split(",")
                      if name.strip()]
            missing = [name for name in wanted if name not in table]
            if missing:
                raise ValueError(
                    f"no column called {', '.join(missing)}. This spectrum has "
                    f"{', '.join(table)}."
                )
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
