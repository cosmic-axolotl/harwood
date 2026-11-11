"""splot -- onedspec's interactive spectrum plot and its measurements.

A spectrum (an IRAF onedspec or multispec, a line of a long slit, or harwood's own
spectrum file) and the keys that measure it: the cursor position, equivalent
widths, line fits and deblending, signal to noise, smoothing and arithmetic. The
measurements are core/splot.py, a port of IRAF's; they go to the terminal and to
the save_file, splot.log, as IRAF's do.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import Field

from ..core import advice
from ..core.registry import register
from ..core.task import Step, Task, TaskParams, TaskResult

WHY = {
    "open": "The spectrum as onedspec opens it: a 1-D image's world system, a line of a "
            "long slit summed over nsum, a multispec aperture and its dispersion.",
    "keys": "Each key's measurement, in real as IRAF computes it: the cursor position, "
            "e, k, d, h, m and the rest.",
    "log": "The measurements appended to the save_file, with the date and the "
           "spectrum's name, as IRAF writes them.",
}


class SplotParams(TaskParams):
    images: list[str] = Field(..., description="Spectra to plot, one after the other.",
                              json_schema_extra={"positional": True})
    line: int | None = Field(None, description="Line or aperture to plot. Left out, asked "
                                               "for when the image has several.")
    band: int | None = Field(None, description="Band to plot. Left out, asked for when "
                                               "the image has several.")
    units: str = Field("", description="Coordinate units: angstroms, nm, km/s 6563 "
                                       "angstroms... Empty: the image's own.")
    options: str = Field("auto wreset", description="auto, zero, xydraw, histogram, "
                                                    "nosysid, wreset, flip, overplot.")
    xmin: float | None = Field(None, description="Left end of the first graph.")
    xmax: float | None = Field(None, description="Right end of the first graph.")
    ymin: float | None = Field(None, description="Bottom of the first graph.")
    ymax: float | None = Field(None, description="Top of the first graph.")
    save_file: str = Field("splot.log", description="Where the measurements are "
                                                    "appended. Empty: nowhere.")
    nerrsample: int = Field(0, ge=0, description="Monte-Carlo samples for the errors of k "
                                                 "and d (below 10: none).")
    sigma0: float | None = Field(None, description="Constant noise, for the errors.")
    invgain: float | None = Field(None, description="Inverse gain, for the errors.")
    function: Literal["spline3", "legendre", "chebyshev", "spline1"] = Field(
        "spline3", description="Continuum function of the t key.")
    order: int = Field(1, ge=1, description="Continuum order of the t key.")
    low_reject: float = Field(2.0, ge=0, description="t: low rejection, in sigmas.")
    high_reject: float = Field(4.0, ge=0, description="t: high rejection, in sigmas.")
    niterate: int = Field(10, ge=0, description="t: rejection iterations.")
    grow: float = Field(1.0, ge=0, description="t: rejection growing radius.")
    dispaxis: Literal[1, 2, 3] = Field(1, description="A 2-D image's dispersion axis when "
                                                      "it has no DISPAXIS (onedspec's).")
    nsum: str = Field("1", description="Lines (and bands) summed for a 2-D image's "
                                       "spectrum: '5' or '5 3' (onedspec's).")
    gcommands: str = Field("", description="Cursor commands, 'x y wcs key' a line (\\040 "
                                           "a space) or ':command', instead of the window.")
    answers: str = Field("", description="With gcommands: a file of what splot's "
                                         "questions read, a line each (boxsize, "
                                         "constant, next_image...).")


@register
class Splot(Task[SplotParams]):
    """IRAF's splot, from noao/onedspec/splot: the spectrum opened as onedspec opens
    it and each key's measurement as IRAF computes it -- what it prints and what it
    logs identical to IRAF 2.18's on 26 cursor files. The window is not ported yet:
    the keys come from --gcommands."""

    name = "splot"
    package = "onedspec"
    description = "Plot and measure spectra with the cursor's keys."
    Params = SplotParams

    def plan(self, params: SplotParams) -> list[Step]:
        return [Step("open", ", ".join(params.images), why=WHY["open"]),
                Step("keys", params.gcommands or "the window", why=WHY["keys"]),
                Step("log", params.save_file or "nowhere", why=WHY["log"])]

    def execute(self, params: SplotParams) -> TaskResult:
        from ..core import splot

        if not params.gcommands:
            advice.fail("the splot window is not ported yet",
                        "give the keys in a cursor file: --gcommands keys.txt, a line "
                        "'x y 1 key' each (\\040 for the space bar)")
        cursor = Path(params.gcommands)
        if not cursor.is_file():
            advice.fail(f"{cursor} does not exist", "give the cursor file's path",
                        FileNotFoundError)
        answers = []
        if params.answers:
            answers = Path(params.answers).read_text(encoding="utf-8").splitlines()
        nsum = [int(w) for w in params.nsum.split()] or [1]
        values = params.model_dump()
        values["nsum"] = (nsum[0], nsum[1] if len(nsum) > 1 else nsum[0])
        session = splot.Splot(values, iter(answers))
        session.run([str(i) for group in params.images for i in group.split(",")],
                    splot.cursor_reads(cursor.read_text(encoding="utf-8")))
        written = []
        if session.log and params.save_file:
            with open(params.save_file, "a", encoding="utf-8") as fd:
                fd.writelines(line + "\n" for line in session.log)
            written.append(Path(params.save_file))
        data: dict = {"text": session.out.splitlines()}
        warnings = [advice.warn(line, "see the splot manual") for line in
                    session.err.splitlines() if line.strip()]
        return TaskResult(outputs=written, steps=self.plan(params), data=data,
                          warnings=warnings, history=[])
