"""IRAF's utilities and lists packages: curfit, a curve through a list or an image, and
rimcursor, positions read off an image.

From iraf-community 2.18: pkg/utilities/{t_curfit.x,curfit.gx}, pkg/lists/rimcursor.x.
See LICENSES/IRAF-COPYRIGHT.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import Field

from ..core import advice
from ..core.registry import register
from ..core.task import Step, Task, TaskParams, TaskResult


class CurfitParams(TaskParams):
    input: list[str] = Field(
        ...,
        description="Lists of x y and perhaps a weight, or images. STDIN reads a list "
        "from the terminal.",
        json_schema_extra={"positional": True},
    )
    function: Literal["legendre", "chebyshev", "spline1", "spline3"] = Field(
        "legendre", description="The function to fit."
    )
    order: int = Field(4, ge=1, description="Its order, or for a spline how many pieces.")
    weighting: Literal["uniform", "user", "statistical", "instrumental"] = Field(
        "uniform",
        description="uniform, 1; user, the third column; statistical, 1/y; instrumental, "
        "the third column is a sigma and the weight is 1/sigma**2.",
    )
    axis: int = Field(1, ge=1, description="The axis an image is projected onto.")
    listdata: bool = Field(False, description="Only the x, fit, y and weight of each point.")
    verbose: bool = Field(False, description="Those too, after the usual output.")
    power: bool = Field(False, description="The polynomial in powers of x too.")
    calctype: Literal["real", "double"] = Field("double", description="The precision.")
    sample: str = Field("*", description="The ranges of x to fit, as 1:10,20:30.")
    naverage: int = Field(
        1, description="Bin the sample by this many points; means, or medians if negative."
    )
    low_reject: float = Field(3.0, ge=0, description="Reject below this many sigma.")
    high_reject: float = Field(3.0, ge=0, description="Reject above this many sigma.")
    niterate: int = Field(0, ge=0, description="How many times to reject and fit again.")
    grow: float = Field(0.0, ge=0, description="Reject too what is this close in x.")
    interactive: bool = Field(
        False, description="Fit it by hand first, in the icfit panel, with IRAF's keys."
    )
    cursor: str = Field(
        "", description="Those keys from a file, an 'x y 1 key' a line or just a key, for a "
                        "session with no window."
    )


def _divided(top, values, real, fallback):
    """arcz: top/value, and the fallback for a value too small to divide by."""
    import numpy as np

    tol = real(1.0e-20)
    out = np.zeros(values.size, real)
    for i, value in enumerate(values):
        out[i] = fallback if -tol < value < tol else real(top / value)
    return out


def _read_list(text: str, name: str, weighting: str, real, warnings: list) -> tuple:
    """cf_rlist, and the order xt_sort3 gives the points."""
    import numpy as np

    from ..core import artdata
    from .astutil import scan_doubles

    xs, ys, ws = [], [], []
    ncols = 0
    for lineno, line in enumerate(text.splitlines(), 1):
        if line.startswith("#") or not line.strip():
            continue
        values = scan_doubles(line)
        got = min(len(values), 3)
        if ncols == 0 and got > 0:
            ncols = got
        if got == 0:
            warnings.append(advice.warn(f"no args; {name}, line {lineno}: {line}",
                                        "this line is left out of the fit."))
            continue
        if got == 1 and ncols >= 2:
            warnings.append(advice.warn(f"only 1 arg; {name}, line {lineno}: {line}",
                                        "this line is left out of the fit."))
            continue
        if got == 2 and ncols == 3:
            warnings.append(advice.warn(f"only 2 args; {name}, line {lineno}: {line}",
                                        "this line is left out of the fit."))
            continue
        if got == 1:
            xs.append(float(len(xs) + 1))
            ys.append(values[0])
        else:
            xs.append(values[0])
            ys.append(values[1])
        ws.append(values[2] if got == 3 else 1.0)
    x, y, w = (np.array(v, real) for v in (xs, ys, ws))
    if weighting == "uniform":
        w = np.ones(x.size, real)
    elif weighting == "statistical":
        w = _divided(real(1), np.abs(y), real, real(1))
    elif weighting == "instrumental":
        w = _divided(real(1), w * w, real, real(0))
    order = artdata.qsort(list(range(x.size)), x)
    return x[order], y[order], w[order]


def _read_image(spec, axis: int, weighting: str, real) -> tuple:
    """cf_rimage and im_projection: the mean over the other axes."""
    import numpy as np

    from ..core import images

    data, _ = images.read(spec)
    data = np.asarray(data).astype(real)
    ndim = data.ndim
    if axis > ndim:
        advice.fail("Attempt to take projection over nonexistent axis",
                    f"this image has {ndim} axes; choose --axis up to that.")
    npix = data.shape[ndim - axis]
    y = np.zeros(npix, real)
    lines = data.reshape(-1, data.shape[-1])
    if axis == 1:
        for line in lines:
            y = y + line
    else:
        where = np.indices(data.shape[:-1]).reshape(ndim - 1, -1)[ndim - axis]
        for index, line in zip(where, lines, strict=True):
            total = real(0)
            for value in line:
                total = real(total + value)
            y[index] = real(y[index] + total)
    y = y / real(data.size // npix)
    x = np.arange(1, npix + 1, dtype=real)
    w = np.ones(npix, real)
    if weighting == "statistical":
        for i in range(npix):
            if y[i] > 0:
                w[i] = real(real(1) / y[i])
            elif y[i] < 0:
                w[i] = abs(real(real(1) / y[i]))
    return x, y, w


def _is_list(name: str, path: Path) -> bool:
    """cf_operand, with what a FITS file is known by instead of IRAF's test for text."""
    if name.endswith("]"):
        return False
    if len(name) == 1:
        return True
    if not path.is_file():
        return False
    with open(path, "rb") as file:
        return file.read(6) != b"SIMPLE"


@register
class Curfit(Task[CurfitParams]):
    """IRAF's curfit: a function fit to a list or to an image's projection, with its
    coefficients, their errors and the fit at each point. The digits are IRAF's."""

    name = "curfit"
    package = "utilities"
    description = "Fit a curve to a list or an image section."
    Params = CurfitParams

    def plan(self, params: CurfitParams) -> list[Step]:
        return [
            Step(
                "fit",
                f"{params.function} of order {params.order} to {', '.join(params.input)}",
                why="A curve and how far to trust each coefficient, with nothing to reduce.",
            )
        ]

    def execute(self, params: CurfitParams) -> TaskResult:
        import sys

        import numpy as np

        from ..core import icfitbits, images, iraffmt

        real = np.float32 if params.calctype == "real" else np.float64
        lines: list[str] = []
        warnings: list = []
        names = []
        for item in params.input:
            names += ["STDIN"] if item == "STDIN" else [str(s) for s in images.expand(item)]
        fits = []
        for name in names:
            if name == "STDIN":
                x, y, w = _read_list(sys.stdin.read(), name, params.weighting, real, warnings)
            else:
                spec = images.parse(name)
                if _is_list(name, spec.path):
                    x, y, w = _read_list(
                        spec.path.read_text(errors="replace"),
                        name,
                        params.weighting,
                        real,
                        warnings,
                    )
                else:
                    if not spec.path.is_file():
                        advice.fail(f"Cannot open image ({name})", "check the name.")
                    x, y, w = _read_image(spec, params.axis, params.weighting, real)
            ic = icfitbits.Ic(
                function=params.function,
                order=params.order,
                sample=params.sample,
                naverage=params.naverage,
                low_reject=params.low_reject,
                high_reject=params.high_reject,
                niterate=params.niterate,
                grow=params.grow,
            )
            fitted = None
            if params.interactive or params.cursor:
                ic, x, y, w, fitted = _session(ic, x, y, w, params, name, real)
            curve = icfitbits.fit(ic, x, y, w if fitted is None else fitted)
            lines += ic.messages
            warnings += [advice.warn(m, "check the sample ranges.") for m in ic.warnings]
            listed = [
                "\t"
                + " \t".join(
                    icfitbits.number(v, "e", 7, 14) for v in (x[i], curve(x[i]), y[i], w[i])
                )
                for i in range(x.size)
            ]
            if params.listdata:
                lines += listed
            else:
                when = iraffmt.cnvtime()
                lines += icfitbits.show(ic, curve, x, y, w, name, when)
                if params.verbose:
                    lines += ["", "# \t    X     \t     Yc   \t     Y    \t     W", *listed]
                if params.power:
                    coeff = curve.power()
                    if coeff is None:
                        warnings.append(
                            advice.warn(
                                "Cannot convert coefficients - wrong function type",
                                "only a legendre or a chebyshev is a polynomial in x.",
                            )
                        )
                    lines += ["# Power series coefficients would be:", "# \t\tcoefficient"]
                    for i in range(curve.ncoeff):
                        text = (
                            "INDEF".rjust(14)
                            if coeff is None
                            else icfitbits.number(coeff[i], "e", 7, 14)
                        )
                        lines.append(f"# \t{i + 1} \t{text}")
            fits.append({"name": name, "coefficients": [float(c) for c in curve.coeff]})
        return TaskResult(
            data={"fits": fits}, stdout=lines, warnings=warnings, steps=self.plan(params)
        )


def _session(ic, x, y, w, params, name: str, real):
    """icg_fit before the output: what it leaves is the parameters and the points, whose
    weights d and u have changed, and the fit is the exact one of those. q takes the last
    fit, so what was deleted since f is out of it and still shown with no weight."""
    import numpy as np

    from ..core import icfit, icfitsession
    from ..viewer import icfit_panel

    spec = icfit.FitSpec(function=ic.function, order=ic.order, sample=ic.sample,
                         naverage=ic.naverage, niterate=ic.niterate, low_reject=ic.low_reject,
                         high_reject=ic.high_reject, grow=ic.grow)
    if params.cursor:
        if not Path(params.cursor).exists():
            advice.fail(f"cursor file '{params.cursor}' does not exist",
                        "write one key a line, or use --interactive for the window",
                        FileNotFoundError)
        session = icfitsession.IcfitSession(x.astype(float), y.astype(float), spec,
                                            w.astype(float), name)
        lines = []
        for line in Path(params.cursor).read_text().splitlines():
            words = line.split()
            if words:
                lines.append(line if len(words) >= 4 and words[0][0] not in ":" else
                             f"0 0 1 {line.strip()}")
        session.run_commands("\n".join(lines))
    else:
        panel = icfit_panel.make(x.astype(float), y.astype(float), spec,
                                 title=f"curfit · {name}", weights=w.astype(float))
        session = panel.session
    done = session.spec
    ic.function, ic.order, ic.sample = done.function, done.order, done.sample
    ic.naverage, ic.niterate = done.naverage, done.niterate
    ic.low_reject, ic.high_reject, ic.grow = done.low_reject, done.high_reject, done.grow
    return (ic, np.asarray(session.x, real), np.asarray(session.y, real),
            np.asarray(session.wts, real), np.asarray(session.fitted_wts, real))


class RimcursorParams(TaskParams):
    image: str = Field(
        "",
        description="The image the coordinates refer to.",
        json_schema_extra={"positional": True},
    )
    wcs: str = Field("logical", description="The system to print: logical, physical or world.")
    wxformat: str = Field("", description="The format of x; or the WCS's, or else %0.15g.")
    wyformat: str = Field("", description="The format of y; or the WCS's, or else %0.15g.")
    cursor: str = Field(
        "",
        description="A file of cursor reads instead of the window: x y wcs key. STDIN reads "
        "them from the terminal.",
    )


def scan_cursor(line: str) -> tuple[list[float], int | None, str]:
    """clscan with gargd, gargd, gargi and gargstr: what is left after the code is kept
    as it is, blanks and all."""
    import re

    from ..core import iraffmt

    position = []
    at = 0
    for _ in range(2):
        found = re.compile(r"\s*(\S+)").match(line, at)
        value = iraffmt.ctod(found[1]) if found else None
        if value is None:
            return position, None, ""
        position.append(value)
        at = found.end()
    found = re.compile(r"\s*(\S+)").match(line, at)
    value = iraffmt.ctod(found[1]) if found else None
    if value is None:
        return position, None, ""
    return position, int(value), line[found.end() :]


def _cursor_formats(params: RimcursorParams, header) -> list:
    from ..core import iraffmt
    from ..core.implot import _wat_attributes

    formats = []
    for axis, given in ((1, params.wxformat), (2, params.wyformat)):
        own = header is not None and params.wcs == "world"
        text = given or (own and _wat_attributes(header, axis).get("format")) or "%0.15g"
        formats.append(iraffmt.Format(text))
    return formats


def cursor_line(line: str, header, formats, wcs: str) -> str:
    """A pass of t_rimcursor's loop: the same read, its coordinates transformed."""
    import numpy as np

    from ..core import mwcs

    position, code, rest = scan_cursor(line)
    if len(position) < 2:
        advice.fail("bad cursor read", "a cursor line is x y, and then a wcs and a key.")
    x, y = position
    if header is not None:
        ndim = max(int(header.get("NAXIS", 2)), 2)
        grid = np.ones((ndim, 1))
        grid[0, 0], grid[1, 0] = x, y
        out = mwcs.transform(header, wcs, grid)
        x, y = float(out[0, 0]), float(out[1, 0])
    text = f"{formats[0](x)} {formats[1](y)} "
    if code is not None:
        text += f"{code}{rest}"
    return text


@register
class Rimcursor(Task[RimcursorParams]):
    """IRAF's rimcursor: each cursor read as a line of a list, the position in the system
    asked for, ready for any task that takes a cursor file."""

    name = "rimcursor"
    package = "lists"
    description = "Read the image cursor, and print the positions."
    Params = RimcursorParams

    def plan(self, params: RimcursorParams) -> list[Step]:
        where = params.cursor or "the window"
        return [
            Step(
                "read",
                f"{where} in {params.wcs} coordinates",
                why="A list made by pointing, for what takes a list of positions.",
            )
        ]

    def execute(self, params: RimcursorParams) -> TaskResult:
        import sys

        from ..core import images

        header = None
        if params.image:
            try:
                _, header = images.read(params.image)
            except (OSError, ValueError, KeyError, IndexError):
                header = None
        if header is not None and params.wcs not in ("logical", "physical", "world"):
            advice.fail(
                f"MWCS: coordinate system not defined ({params.wcs})",
                "use logical, physical or world.",
            )
        formats = _cursor_formats(params, header)
        if not params.cursor:
            lines = _from_window(params, header, formats)
        else:
            if params.cursor == "STDIN":
                reads = sys.stdin.read().splitlines()
            else:
                file = Path(params.cursor)
                if not file.is_file():
                    advice.fail(f"Cannot open file ({params.cursor})", "check its name.")
                reads = file.read_text().splitlines()
            lines = [cursor_line(line, header, formats, params.wcs) for line in reads]
        return TaskResult(data={"reads": len(lines)}, stdout=lines, steps=self.plan(params))


def _from_window(params: RimcursorParams, header, formats) -> list[str]:
    """The window's cursor: any key is a read, a colon a command; closing it is the EOF."""
    from ..core import display, images

    screen = display.get(None)
    window = getattr(screen, "window", None)
    if window is None or not hasattr(window, "wait_for_examine"):
        advice.fail(
            "There is no window to read the cursor from",
            "run it where there is a screen, or give the reads with --cursor.",
        )
    if params.image:
        data, shown = images.read(params.image)
        screen.show(data, shown, name=params.image)
    lines: list[str] = []

    def read(rest: str, x: float, y: float) -> None:
        where = f"{float(x)!r} {float(y)!r}"
        line = cursor_line(f"{where} 101 {rest}", header, formats, params.wcs)
        lines.append(line)
        window.log("  " + line)

    def key(name: str, x: float, y: float) -> bool:
        read("\\040" if name == " " else name, x, y)
        return True

    def colon(text: str) -> str:
        position = getattr(window, "_cursor", None) or (0.0, 0.0)
        read(text if text.startswith(":") else ":" + text, *position)
        return ""

    window.key_hook, window.colon_hook = key, colon
    window.log("rimcursor: point and press a key; closing the window ends the list.")
    try:
        window.wait_for_examine()
    finally:
        window.key_hook = window.colon_hook = None
    return lines
