"""tv -- looking at an image and asking it questions.

`imexamine` is the task people mean when they say they used IRAF: point at a
star, press r, read the FWHM. It is the reason the viewer exists (section 8.5
puts its keys in the MVP), and the reason it is a task rather than a feature of
the window is the same as everywhere else in harwood -- it has parameters that
belong in the project, a manual, `--editpar`, and a record of what it measured.

It also has IRAF's other half, the one the tutorials never mention:
`imagecur`, a file of cursor positions read instead of the mouse. That is what
lets the same measurement run over ssh, in a script, or in a test, and get the
numbers the window would have shown.
"""

from __future__ import annotations

from typing import Literal

from pydantic import Field

from ..core import advice
from ..core.registry import register
from ..core.task import Step, Task, TaskParams, TaskResult

WHY = {
    "show": "The picture on the screen is scaled to the display limits and "
            "stretched; the pixels underneath are not touched. Every number the "
            "window reports -- the value under the cursor, a profile, a "
            "magnitude -- comes from the original floating-point data.",
    "limits": "zscale picks limits from a sample of the pixels near the "
              "median, so the sky and faint stars are both visible; minmax "
              "shows the full range, which on a frame with one bright star or "
              "a cosmic ray is mostly black.",
    "mark": "Each position from the list is drawn as a circle over the image, "
            "as tvmark does -- the quickest way to see what daofind found, and "
            "what it missed.",
    "read": "Each image goes into a frame of its own, in order, starting at "
            "--frame -- so n and p step through them as in IRAF and blink "
            "compares them.",
    "cursor": "Positions come from the file instead of the mouse, one key per "
              "line. Everything else is identical: the same code answers a key "
              "pressed over the image and a key read from here, so a script gets "
              "the numbers the window would have shown.",
    "examine": "The measurements use the original pixels in floating point, "
               "never the scaled picture on the screen. Every star is "
               "recentred before it is measured, so the cursor only has to land "
               "near it; the sky is the median of an annulus that starts "
               "`buffer` beyond the aperture, which keeps the star's own wings "
               "out of it. The FWHM fit weights each pixel by 1/r^2, so the "
               "core counts as much as the many pixels in the wings -- the "
               "weighting IRAF's rimexam uses, so the numbers compare.",
    "log": "Each measurement is appended as one line, as you make it -- the "
           "seeing of a night is something you want to read back the next "
           "morning, not remember.",
}


class DisplayParams(TaskParams):
    image: str = Field(..., description="Image to load. [SCI,1] and sections are accepted.",
                       json_schema_extra={"positional": True})
    frame: int = Field(1, ge=1, description="Frame to load the image into.")
    scale: Literal["zscale", "minmax", "99.5%"] = Field(
        "zscale", description="How the display limits are chosen. zscale is IRAF's "
                              "default and the one that shows sky and stars at once.",
    )
    coords: str = Field(
        "", description="Coordinate list to mark over the image, as tvmark does: a "
                        "daofind .coo, or any table with x and y columns.",
    )
    radius: float = Field(6.0, gt=0, description="Radius of the marks, in pixels.")
    backend: Literal["", "native", "headless", "ds9"] = Field(
        "", description="Where to draw. Empty picks the window when there is a "
                        "screen and a PNG when there is not.",
    )
    wait: bool = Field(
        True, description="Keep the window open until it is closed. --no-wait "
                          "returns at once; inside harwood gui it never waits.",
    )


@register
class Display(Task[DisplayParams]):
    """Load an image into a frame of the viewer. With --coords, each position
    in the list is circled, which is what tvmark is mostly used for: seeing
    what daofind found and what it missed.

    Without a screen -- over ssh, in a script -- the picture is written to a
    PNG instead, and the command says where.
    """

    name = "display"
    package = "tv"
    description = "Load an image into the viewer, with marks from a coordinate list."
    Params = DisplayParams

    def plan(self, params: DisplayParams) -> list[Step]:
        steps = [Step("show", f"{params.image} -> frame {params.frame}", why=WHY["show"]),
                 Step("limits", params.scale, why=WHY["limits"])]
        if params.coords:
            steps.append(Step("mark", f"{params.coords}, radius {params.radius:g}",
                              why=WHY["mark"]))
        return steps

    def execute(self, params: DisplayParams) -> TaskResult:
        from ..core import catalog, display, images

        data, header = images.read(params.image)
        screen = display.get(params.backend or None)
        screen.show(data, header, frame=params.frame, name=str(params.image),
                    scale=params.scale)
        steps = self.plan(params)

        if params.coords:
            table = catalog.read(params.coords)
            x, y = catalog.coordinates(table, params.coords)
            labels = (table["id"] if "id" in table.colnames else [""] * len(table))
            screen.mark([display.Marker(float(a), float(b), radius=params.radius,
                                        label=str(label))
                         for a, b, label in zip(x, y, labels, strict=True)],
                        frame=params.frame)
            steps[-1].measured = {"marked": len(table)}

        outputs = [screen.output] if hasattr(screen, "output") else []
        if params.wait and hasattr(screen, "run") and getattr(screen, "owns", False):
            screen.run()
        return TaskResult(outputs=outputs, steps=steps,
                          data={"shape": "x".join(str(n) for n in reversed(data.shape))})


class ImexamineParams(TaskParams):
    input: list[str] = Field(
        ..., description="Images to examine, each loaded into its own frame. "
                         "Globs, @lists and [SCI,1] extensions are accepted.",
        json_schema_extra={"positional": True},
    )
    frame: int = Field(1, ge=1, description="Frame the first image goes into.")
    imagecur: str = Field(
        "", description="Cursor file to read instead of the mouse: one 'x y "
                        "[frame] key' per line. Runs without a window, so it "
                        "works over ssh and in scripts.",
    )
    logfile: str = Field(
        "", description="Text file each measurement is appended to. Empty keeps "
                        "none, as IRAF's keeplog=no; the run itself is always in "
                        "harwood.log.jsonl.",
    )
    defkey: str = Field(
        "a", min_length=1, max_length=1,
        description="Key for a cursor-file line that gives only x and y, so a "
                    "plain list of positions from another program can be "
                    "examined as it is.",
    )
    # rimexam: the r and a keys
    radius: float = Field(
        5.0, ge=1, description="Aperture radius: the pixels summed for the flux, "
                               "used for centring, and fitted for the FWHM.",
    )
    buffer: float = Field(
        5.0, ge=0, description="Gap between the aperture and the sky annulus, so "
                               "the star's wings stay out of its own sky.",
    )
    width: float = Field(5.0, ge=1, description="Width of the sky annulus, in pixels.")
    rplot: float = Field(
        8.0, ge=1, description="Radius the radial profile is plotted out to. "
                               "Drawing only; the measurement uses `radius`.",
    )
    fittype: Literal["gaussian", "moffat"] = Field(
        "moffat", description="Profile fitted for the FWHM. Moffat has wings and "
                              "fits real seeing better than a gaussian.",
    )
    beta: float | None = Field(
        None, gt=0, description="Moffat beta held fixed. Empty fits it, as IRAF's "
                                "INDEF does.",
    )
    iterations: int = Field(
        3, ge=1, description="Passes of the r and a keys: after the first, the "
                             "aperture becomes three times the direct FWHM. 1 "
                             "keeps `radius`.",
    )
    magzero: float = Field(25.0, description="Zero point of the a key's magnitudes.")
    # imexamine: the m key
    ncstat: int = Field(5, ge=1, description="Columns in the m key's statistics box.")
    nlstat: int = Field(5, ge=1, description="Lines in the m key's statistics box.")
    # himexam: the h key
    ncolumns: int = Field(21, ge=2, description="Columns in the h key's histogram box.")
    nlines: int = Field(21, ge=2, description="Lines in the h key's histogram box.")
    nbins: int = Field(
        512, ge=1, description="Bins in the h key's histogram. Integer data never "
                               "get more bins than distinct values.",
    )
    # jimexam: the j and k keys
    naverage: int = Field(
        5, ge=1, description="Lines (j) or columns (k) averaged before the "
                             "gaussian fit.",
    )
    sigma: float = Field(
        1.0, gt=0, description="First guess of the j and k fit's sigma, in pixels. "
                               "It has to be near the truth for the fit to converge.",
    )


@register
class Imexamine(Task[ImexamineParams]):
    """Point at the image and press a key; the answer goes to the Examine
    panel and the session log, and to --logfile when one is named.

        r   radial profile and FWHM          j   gaussian fit along the line
        a   aperture photometry and FWHM     k   gaussian fit along the column
        m   statistics of a box              h   histogram of a box
        l   plot the line                    x   coordinates and value
        c   plot the column                  z   the 10x10 pixels around it
        n   next frame                       p   previous frame
        ?   list the keys                    q   end the task

    Every star is recentred before it is measured, so the cursor only has to
    land near it. Without a screen, put the positions in a file -- one
    `x y [frame] [key]` per line -- and pass it as --imagecur: the same code
    answers, so the numbers are the ones the window would have shown.

    The parameters are IRAF's, with IRAF's defaults and meanings: `radius` is
    the aperture, `rplot` how far the profile is drawn, and r and a widen the
    aperture to three times the direct FWHM, `iterations` times. The line
    reads as IRAF's: radius, magnitude, flux, sky, peak, ellipticity, position
    angle, Moffat beta, and IRAF's three FWHMs -- of a fit to the enclosed
    flux, of a fit to the pixels (MFWHM or GFWHM), and the direct one,
    measured without a model. The arithmetic is a port of IRAF's own
    (ierimexam.x and the starfocus code under it), and a test holds it to
    what IRAF 2.18 printed for the same pixels.
    """

    name = "imexamine"
    package = "tv"
    description = "Examine images with the cursor: profiles, photometry, statistics."
    Params = ImexamineParams

    def plan(self, params: ImexamineParams) -> list[Step]:
        last = params.frame + len(params.input) - 1
        frames = (f"frame {params.frame}" if last == params.frame
                  else f"frames {params.frame}-{last}")
        steps = [Step("read", f"{', '.join(params.input)} -> {frames}", why=WHY["read"])]
        if params.imagecur:
            steps.append(Step("cursor", params.imagecur, why=WHY["cursor"]))
        else:
            steps.append(Step("wait", "keys: " + " ".join(_keys()) + "  n p  ?  q",
                              why="Lower case belongs to the task and upper case "
                                  "to the window, as in IRAF's cursor mode."))
        steps.append(Step("examine", f"radius {params.radius:g}, buffer "
                                     f"{params.buffer:g}, width {params.width:g}, "
                                     f"rplot {params.rplot:g}, {params.fittype}, "
                                     f"iterations {params.iterations}",
                          why=WHY["examine"]))
        if params.logfile:
            steps.append(Step("log", params.logfile, why=WHY["log"]))
        return steps

    def execute(self, params: ImexamineParams) -> TaskResult:
        from ..core import examine, images

        specs = images.expand(params.input)
        if not specs:
            advice.fail(f"'{' '.join(params.input)}' matches no image",
                        "check the name, or run `harwood sections` on the template "
                        "to see what it matches", FileNotFoundError)
        settings = examine.Settings(**{
            name: getattr(params, name) for name in examine.Settings.__dataclass_fields__})

        if params.imagecur:
            results = self._from_cursor(params, specs, settings)
        else:
            results = self._interactive(params, specs, settings)

        rows = [{"image": name, "key": r.key, "x": round(r.x, 2), "y": round(r.y, 2),
                 "result": r.line.splitlines()[0].split("  ", 1)[-1]}
                for name, r in results]
        # What does not fit in a cell -- the z key's grid -- is printed whole.
        text = [line for _, r in results if "\n" in r.line
                for line in ["", *r.line.splitlines()]]
        steps = self.plan(params)
        measured, = [step for step in steps if step.action == "examine"]
        measured.measured = {"measurements": len(results),
                             "failed": sum(not r.ok for _, r in results)}
        data = {"rows": rows,
                "measurements": [{"image": name, "key": r.key, **_plain(r.values)}
                                 for name, r in results]}
        if text:
            data["text"] = text
        return TaskResult(data=data, steps=steps)

    def _from_cursor(self, params, specs, settings):
        from pathlib import Path

        from ..core import examine, images, logfile

        if not Path(params.imagecur).exists():
            advice.fail(f"cursor file '{params.imagecur}' does not exist",
                        "write one 'x y [frame] key' per line, or leave --imagecur "
                        "empty to use the mouse", FileNotFoundError)
        loaded: dict[int, tuple[str, object]] = {}
        results = []
        for x, y, frame, key in examine.read_cursor(params.imagecur, params.defkey):
            index = frame - params.frame
            if not 0 <= index < len(specs):
                advice.fail(f"{params.imagecur} asks for frame {frame}, and only "
                            f"frames {params.frame}-{params.frame + len(specs) - 1} "
                            "are loaded", "name more images, or correct the frame "
                            "column of the cursor file")
            if frame not in loaded:
                data, _ = images.read(specs[index])
                loaded[frame] = (str(specs[index]), data)
            name, data = loaded[frame]
            result = examine.examine(data, key, x, y, settings)
            results.append((name, result))
            logfile.write(params.logfile, "imexamine", name, result.line.splitlines())
        return results

    def _interactive(self, params, specs, settings):
        from ..core import display, images

        screen = display.get()
        window = getattr(screen, "window", None)
        if window is None:
            advice.fail("imexamine needs a window to point at, and there is no "
                        "screen here",
                        "put the positions in a file and pass --imagecur, which "
                        "gives the same numbers without one")
        window.configure_examine(settings, params.logfile)
        for offset, spec in enumerate(specs):
            data, header = images.read(spec)
            screen.show(data, header, frame=params.frame + offset, name=str(spec))
        if len(specs) > 1:
            window.select_frame(params.frame)
        window.log("imexamine: point at the image and press a key; ? lists them, "
                   "q ends the task.")
        start = len(window.measurements)
        window.wait_for_examine()
        return window.measurements[start:]


def _plain(values: dict) -> dict:
    """numpy scalars as Python numbers, so the record reads cleanly as JSON."""
    return {k: (v.item() if hasattr(v, "item") else v) for k, v in values.items()}


def _keys() -> list[str]:
    from ..core.examine import KEYS

    return list(KEYS)
