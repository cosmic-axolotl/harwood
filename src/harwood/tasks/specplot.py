"""specplot -- onedspec's stacked plot of several spectra.

Every aperture of every image named, each scaled and offset (by default to the
first one's mean) and stepped up by a separation that just clears one from the
next; keys and colon commands move, shift, scale and relabel them. The layout and
the commands are core/specplot.py, a port of IRAF's; what it prints and the
logfile's table are IRAF's.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import Field

from ..core import advice
from ..core.registry import register
from ..core.task import Step, Task, TaskParams, TaskResult

WHY = {
    "open": "Each aperture (and band) of each image as onedspec opens it: a 1-D image's "
            "world system, a line of a long slit summed over nsum, a multispec "
            "aperture and its dispersion.",
    "layout": "Each spectrum scaled to the first's mean (or offset to it, when a mean "
              "is not positive) and the step that just separates them, times fraction.",
    "keys": "The keys and colon commands, as IRAF's specplot reads them.",
    "log": "The table of every spectrum's coordinates, scale, offset and velocity, "
           "appended to the logfile at the end, as IRAF writes it.",
}


class SpecplotParams(TaskParams):
    spectra: list[str] = Field(..., description="Spectra to plot, all together.",
                               json_schema_extra={"positional": True})
    apertures: str = Field("", description="Apertures to plot, as ranges ('1,3-5'). "
                                           "Empty: all.")
    bands: str = Field("1", description="Bands of 3-D images to plot.")
    autolayout: bool = Field(True, description="Lay the spectra out automatically.")
    autoscale: bool = Field(True, description="Automatic layout: scale to the first's "
                                              "mean (no: offset to it).")
    fraction: float = Field(1.0, description="Fraction of the automatic separation step.")
    units: str = Field("", description="Coordinate units: angstroms, nm, km/s 6563 "
                                       "angstroms... Empty: the images' own.")
    transform: Literal["none", "log"] = Field("none", description="Flux transformation.")
    scale: str = Field("1.", description="Intensity scale: a value, @file (a line a "
                                         "spectrum) or a header keyword.")
    offset: str = Field("0.", description="Intensity offset: a value, @file or a "
                                          "header keyword.")
    step: float = Field(0.0, description="Separation step without the automatic "
                                         "layout.")
    ptype: str = Field("1", description="Plotting type: 1 lines, 0 histogram... "
                                        "negative: marks.")
    labels: Literal["none", "imname", "imtitle", "index", "user"] = Field(
        "user", description="Labels of the spectra.")
    ulabels: str = Field("", description="File of user labels, a line a spectrum.")
    xlpos: float = Field(1.02, description="Label x position, fraction of the range.")
    ylpos: float = Field(0.0, description="Label y position, fraction of the range "
                                          "above the mean.")
    sysid: bool = Field(True, description="Show the system line and the step.")
    yscale: bool = Field(False, description="Draw the y axis scale.")
    title: str = Field("", description="Plot title.")
    xlabel: str = Field("", description="X axis label. Empty: the units'.")
    ylabel: str = Field("", description="Y axis label.")
    xmin: float | None = Field(None, description="X axis left limit.")
    xmax: float | None = Field(None, description="X axis right limit.")
    ymin: float | None = Field(None, description="Y axis bottom limit.")
    ymax: float | None = Field(None, description="Y axis top limit.")
    logfile: str = Field("", description="Where the table of the spectra is appended "
                                         "at the end. Empty: nowhere.")
    dispaxis: Literal[1, 2, 3] = Field(1, description="A 2-D image's dispersion axis when "
                                                      "it has no DISPAXIS (onedspec's).")
    nsum: str = Field("1", description="Lines (and bands) summed for a 2-D image's "
                                       "spectrum: '5' or '5 3' (onedspec's).")
    gcommands: str = Field("", description="Cursor commands, 'x y wcs key' a line (\\040 "
                                           "a space) or ':command', instead of the window.")
    answers: str = Field("", description="With gcommands: a file of what the a, i and u "
                                         "keys read, a line each.")


@register
class Specplot(Task[SpecplotParams]):
    """IRAF's specplot, from noao/onedspec/t_specplot.x: the layout, the keys and the
    colon commands as IRAF 2.18 runs them -- what it prints, its logfile and each
    graph's window identical to IRAF's on 19 cursor files; the keys pressed in its
    window, or read from a cursor file (--gcommands)."""

    name = "specplot"
    package = "onedspec"
    description = "Plot several spectra stacked, and arrange them with the cursor."
    Params = SpecplotParams

    def plan(self, params: SpecplotParams) -> list[Step]:
        return [Step("open", ", ".join(params.spectra), why=WHY["open"]),
                Step("layout", "automatic" if params.autolayout else f"step {params.step}",
                     why=WHY["layout"]),
                Step("keys", params.gcommands or "the window", why=WHY["keys"]),
                Step("log", params.logfile or "nowhere", why=WHY["log"])]

    def execute(self, params: SpecplotParams) -> TaskResult:
        from ..core import specplot, splot

        nsum = [int(w) for w in params.nsum.split()] or [1]
        values = params.model_dump()
        values["nsum"] = (nsum[0], nsum[1] if len(nsum) > 1 else nsum[0])
        images = [str(i) for group in params.spectra for i in group.split(",") if i]
        if not params.gcommands:
            from ..viewer import specplot_panel

            panel = specplot_panel.run(values, images)
            session = panel.session
            if panel.error is not None:
                advice.fail(f"specplot stopped: {panel.error}", "see the message above")
        else:
            cursor = Path(params.gcommands)
            if not cursor.is_file():
                advice.fail(f"{cursor} does not exist", "give the cursor file's path",
                            FileNotFoundError)
            answers = []
            if params.answers:
                answers = Path(params.answers).read_text(encoding="utf-8").splitlines()
            session = specplot.Specplot(values, iter(answers))
            session.run(images, splot.cursor_reads(cursor.read_text(encoding="utf-8")))
        if session.fatal:
            advice.fail(f"specplot stopped: {session.fatal.rstrip(chr(7))}",
                        "units are angstroms, nm, microns, km/s <w> <units>, hertz...")
        written = []
        lines = getattr(session, "vshow_lines", None)
        if lines and params.logfile:
            with open(params.logfile, "a", encoding="utf-8") as fd:
                fd.writelines(line + "\n" for line in lines)
            written.append(Path(params.logfile))
        data: dict = {"text": session.out.replace("\a", "").splitlines(),
                      "spectra": len(session.sps), "step": float(session.step)}
        warnings = [advice.warn(line, "check the image's name and its apertures")
                    for line in session.err.splitlines() if line.strip()]
        return TaskResult(outputs=written, steps=self.plan(params), data=data,
                          warnings=warnings, history=[])
