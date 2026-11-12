"""plot -- IRAF's plot package: graph.

graph draws lists of y or of x,y (files, or what comes on standard input) and
image sections reduced to a vector, up to twenty curves on one pair of axes. The
reading, the window, the line types and colours, the marks and the histogram
steps are core/plotgraph.py, a port of IRAF's; the curves are drawn in the window,
or into an SVG file without a screen.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Literal

from pydantic import Field

from ..core import advice
from ..core.registry import register
from ..core.task import Step, Task, TaskParams, TaskResult

WHY = {
    "read": "Each operand as IRAF's graph takes it: a text file is a list (y alone, "
            "or x y), anything else an image whose section is averaged onto one axis.",
    "window": "The data's extent, or the window given; with the box, glabax's "
              "adjustments: no zero width, elog where a log axis reaches zero, and "
              "out to the next major tick with round (always on a log axis).",
    "draw": "Lines, histogram steps or marks, in the line types and colours of the "
            "lists, as GIO draws them.",
}


class GraphParams(TaskParams):
    input: list[str] = Field(
        default_factory=list,
        description="Lists (text files: y, or x y, a line each) and images or image "
                    "sections, overplotted. Empty, '-' or STDIN: read the list from "
                    "standard input (`... | harwood graph`).",
        json_schema_extra={"positional": True})
    wx1: float = Field(0.0, description="Left world x; wx1 = wx2 autoscales.")
    wx2: float = Field(0.0, description="Right world x.")
    wy1: float = Field(0.0, description="Lower world y; wy1 = wy2 autoscales.")
    wy2: float = Field(0.0, description="Upper world y.")
    wcs: Literal["logical", "physical", "world"] = Field(
        "logical", description="Coordinates of an image's axis.")
    axis: int = Field(1, ge=1, le=7, description="Image axis the projection is onto "
                                                 "(the mean over the others).")
    transpose: bool = Field(False, description="Swap x and y.")
    pointmode: bool = Field(False, description="Plot marks instead of lines.")
    marker: str = Field("box", description="Mark (point box plus cross circle hebar "
                                           "vebar hline vline diamond), or without "
                                           "pointmode the line: line, lhist, bhist.")
    szmarker: float = Field(5e-3, description="Mark size in NDC; 0 or less: read from "
                                              "a third column (negative scales it).")
    ltypes: str = Field("", description="Line types of the curves, 1-4 (solid, dashed, "
                                        "dotted, dot-dashed). Empty: one each.")
    colors: str = Field("", description="Colours of the curves, 1-9. Empty: one each.")
    logx: bool = Field(False, description="Log x axis.")
    logy: bool = Field(False, description="Log y axis.")
    box: bool = Field(True, description="Draw the axes, the labels and the title.")
    ticklabels: bool = Field(True, description="Label the ticks.")
    xlabel: str = Field("wcslabel", description="X label; wcslabel: the image's.")
    ylabel: str = Field("", description="Y label.")
    xformat: str = Field("wcsformat", description="X tick format; wcsformat: the image's.")
    yformat: str = Field("", description="Y tick format.")
    title: str = Field("imtitle", description="Title; imtitle: the first image's.")
    lintran: bool = Field(False, description="Map x from p1..p2 onto q1..q2.")
    p1: float = Field(0.0, description="lintran: start of the input range (p1 = p2 = 0: "
                                       "pixels 1 to n).")
    p2: float = Field(0.0, description="lintran: end of the input range.")
    q1: float = Field(0.0, description="lintran: start of the output range.")
    q2: float = Field(1.0, description="lintran: end of the output range.")
    vx1: float = Field(0.0, description="Viewport, NDC (IRAF's device; the window "
                                        "here fills its panel).")
    vx2: float = Field(0.0, description="Viewport right.")
    vy1: float = Field(0.0, description="Viewport bottom.")
    vy2: float = Field(0.0, description="Viewport top.")
    majrx: int = Field(5, description="Major divisions on x (fewer than 3: no ticks, "
                                      "no rounding).")
    minrx: int = Field(5, description="Minor divisions on x.")
    majry: int = Field(5, description="Major divisions on y.")
    minry: int = Field(5, description="Minor divisions on y.")
    overplot: bool = Field(False, description="IRAF's overplot on the device: refused, "
                                              "give all the curves to one graph.")
    append: bool = Field(False, description="IRAF's append to the device: refused.")
    device: str = Field("stdgraph", description="stdgraph: the window when there is a "
                                                "screen, else an SVG in this run's "
                                                "directory; a name ending in .svg "
                                                "writes there.")
    round: bool = Field(False, description="Round the window out to major ticks.")
    fill: bool = Field(True, description="Fill the viewport (no: unit aspect ratio, "
                                         "IRAF's device).")


def _fix(message: str) -> str:
    """What to do about each of graph's complaints."""
    if message.startswith("no args"):
        return ("the line has no number IRAF can read: a comment must begin in the "
                "first column with #")
    if message.startswith("only 1 arg"):
        return ("the list's first line has x and y, so every line needs both; a comma "
                "ends a number, separate them with spaces")
    if message.startswith("no szmark"):
        return "with szmarker 0 or less each line needs a third number, the mark's size"
    if message.startswith("Unrecognized marker"):
        return "markers are point box plus cross circle hebar vebar hline vline diamond"
    if "Cannot open" in message:
        return "a text file is a list and anything else an image: check the name"
    return "check the operand and the axis"


@register
class Graph(Task[GraphParams]):
    """IRAF's graph, pkg/plot/t_graph.x: lists and image projections, as IRAF 2.18
    reads them, with the window and every point drawn identical to IRAF's metacode
    on 77 cases."""

    name = "graph"
    package = "plot"
    description = "Graph lists (y, or x y) and image sections, overplotted."
    Params = GraphParams

    def plan(self, params: GraphParams) -> list[Step]:
        what = ", ".join(params.input) or "standard input"
        window = ("the data's" if params.wx1 == params.wx2 and params.wy1 == params.wy2
                  else f"{params.wx1:g}..{params.wx2:g}, {params.wy1:g}..{params.wy2:g}")
        return [Step("read", what, why=WHY["read"]),
                Step("window", window, why=WHY["window"]),
                Step("draw", f"marks ({params.marker})" if params.pointmode
                     else params.marker if params.marker in ("lhist", "bhist")
                     else "lines", why=WHY["draw"])]

    def execute(self, params: GraphParams) -> TaskResult:
        from ..core import graph as graphmod
        from ..core import plotgraph
        from .imutil import _plot

        values = params.model_dump()
        names = [n for n in params.input if n.strip()]
        stdin = None
        if not names or names in (["-"], ["STDIN"]):
            if sys.stdin is None or sys.stdin.isatty():
                advice.fail("graph has nothing to plot", "name lists or images, or "
                            "pipe a list in: `... | harwood graph`")
            stdin = sys.stdin.read()
        values["input"] = ",".join(names)
        try:
            plot = plotgraph.graph(values, stdin)
        except plotgraph.GraphError as error:
            advice.fail(f"graph stopped: {error}", "check the names and the axis")
        warnings = [advice.warn(line.removeprefix("Warning: "), _fix(line))
                    for line in plot.err.splitlines() if line.strip()]

        g = graphmod.Graph(plot.title_lines if plot.box else [],
                           plot.xlabel, plot.ylabel, logy=plot.ytran != "linear",
                           logx=plot.xtran != "linear")
        g.xtran, g.ytran = plot.xtran, plot.ytran
        x1, x2, y1, y2 = plot.window
        x1, x2, _ = plotgraph.fp_nondegenr(x1, x2)
        y1, y2, _ = plotgraph.fp_nondegenr(y1, y2)
        g.window = (float(x1), float(x2), float(y1), float(y2))
        for s in plot.strokes:
            if s.kind == "line":
                g.line(s.x, s.y, s.ltype, s.color)
            elif s.kind == "points":
                for x, y in zip(s.x, s.y, strict=True):
                    g.mark(x, y, "point", color=s.color)
            else:
                g.mark(s.x[0], s.y[0], s.kind, s.xsize, s.ysize, s.color)
        first = names[0] if names else "STDIN"
        where = _plot(g, params.device, f"graph · {first}",
                      Path(first).stem.split("[")[0] + ".graph.svg", task="graph")
        data = {"curves": [{"points": int(len(c.x)), "ltype": c.ltype, "color": c.color}
                           for c in plot.curves],
                "window": list(g.window), "xtran": plot.xtran, "ytran": plot.ytran}
        return TaskResult(data=data, steps=self.plan(params), warnings=warnings,
                          outputs=[where] if where else [])
