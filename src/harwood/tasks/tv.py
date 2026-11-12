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

from pathlib import Path
from typing import Literal

from pydantic import Field

from ..core import advice
from ..core.registry import register
from ..core.task import Step, Task, TaskParams, TaskResult, Template

WHY = {
    "render": "The same arithmetic as the window: limits, then the transfer "
              "function, then the colour map -- so the picture is what was on "
              "screen, at the image's own resolution.",
    "export": "A picture for a talk, a report or a message. It is not data: "
              "the FITS file keeps the numbers.",
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
    bpmask: str = Field(
        "BPM", description="Bad pixel mask whose pixels are left out of the zscale "
                           "sample. BPM is the file the image's BPM keyword names; "
                           "!KEY any other keyword; empty for none.",
    )
    zscale: bool = Field(
        True, description="Pick z1 and z2 from a sample of the pixels near the median, "
                          "so sky and faint stars show at once -- IRAF's default.",
    )
    contrast: float = Field(
        0.25, description="zscale's contrast: the fitted slope is divided by it, so "
                          "smaller values widen the range. 0 gives the sample's "
                          "extremes.",
    )
    zrange: bool = Field(
        True, description="With --no-zscale: z1 and z2 from the extremes of the "
                          "sample, instead of --z1 and --z2.",
    )
    zmask: str = Field(
        "", description="Pixels to sample instead of a uniform grid: a mask image or "
                        "a section such as [100:400,*]. Empty samples the whole image.",
    )
    nsample: int = Field(1000, ge=100, description="Most pixels sampled for zscale "
                                                   "and zrange.")
    z1: float | None = Field(None, description="Lowest value shown, with --no-zscale "
                                               "--no-zrange.")
    z2: float | None = Field(None, description="Highest value shown, with --no-zscale "
                                               "--no-zrange.")
    ztrans: Literal["linear", "log", "none", "user"] = Field(
        "linear", description="From z1..z2 to grey: linear; log (onto 1..1000, then "
                              "the logarithm); none (the values themselves, wrapping "
                              "every 256); user (the table in --lutfile).",
    )
    lutfile: str = Field(
        "", description="With --ztrans user: two columns, value and grey level 1-200.",
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


class ImPictureParams(TaskParams):
    input: list[str] = Field(..., description="Images to export. Globs, @lists, [SCI,1] "
                             "and sections are accepted.")
    output: Template = Field(..., description="The picture: a name ending in .png, .jpg, "
                             ".tif or .webp, a list of them, a directory (each image's "
                             "own name, as PNG), or a template like '%.fits%.png%'.")
    scale: Literal["zscale", "minmax", "99.5%"] = Field(
        "zscale", description="How the display limits are chosen, as in the window.")
    stretch: Literal["linear", "sqrt", "log", "asinh", "none", "user"] = Field(
        "linear", description="Transfer function from the limits to the colours. log, "
                              "none and user are display's ztrans of the same names.")
    lutfile: str = Field("", description="With --stretch user: the table, as display's "
                                         "--lutfile.")
    colormap: Literal["grey", "viridis", "cividis", "magma", "heat", "cool", "sepia"] = Field(
        "grey", description="Colour map. viridis, cividis and magma read the same with "
                            "colour-vision deficiencies; cividis was made for them.")
    z1: float | None = Field(None, description="Lower display limit; with z2, overrides "
                             "scale.")
    z2: float | None = Field(None, description="Upper display limit.")


@register
class ImPicture(Task[ImPictureParams]):
    """Write an image as a picture, as the window shows it: the same limits,
    stretch and colours give the same pixels as File > Export image. The
    whole frame at its own resolution, north as in the FITS file (row 1 at
    the bottom). PNG needs nothing installed; JPEG, TIFF and WebP need the
    window's extra, harwood[gui].

    Not IRAF's export, which writes raw and old raster formats with its own
    parameters: a different task, so a different name."""

    name = "impicture"
    package = "tv"
    description = "Write images as PNG, JPEG, TIFF or WebP pictures, as displayed."
    Params = ImPictureParams

    def plan(self, params: ImPictureParams) -> list[Step]:
        limits = (f"z1={params.z1:g} z2={params.z2:g}"
                  if params.z1 is not None and params.z2 is not None else params.scale)
        return [Step("render", f"{limits}, {params.stretch}, {params.colormap}",
                     why=WHY["render"]),
                Step("write", str(params.output), why=WHY["export"])]

    def execute(self, params: ImPictureParams) -> TaskResult:
        from ..core import images, render
        from ..viewer.png import save_rgb

        specs = images.expand(params.input)
        if not specs:
            advice.fail(f"'{' '.join(params.input)}' matches no image",
                        "check the name, or run `harwood sections` on the template",
                        FileNotFoundError)
        output = params.output
        if Path(output).is_dir():
            targets = [Path(output) / (spec.path.stem + ".png") for spec in specs]
        else:
            targets = images.outputs_for(specs, output)
        window = ((params.z1, params.z2)
                  if params.z1 is not None and params.z2 is not None else None)
        table = None
        if params.stretch == "user":
            if not params.lutfile:
                advice.fail("--stretch user needs a table", "give it with --lutfile")
            table = render.UserLut(params.lutfile)
        rows = []
        for spec, target in zip(specs, targets, strict=True):
            data, _ = images.read(spec)
            if data.ndim != 2:
                advice.fail(f"{spec} is {data.ndim}-dimensional; a picture is a plane",
                            "name one plane, e.g. 'cube.fits[*,*,1]', or one extension")
            low, high = window or render.limits(data, params.scale)
            pixels = render.rgb(data, params.scale, params.stretch, params.colormap,
                                window=(low, high), table=table)
            save_rgb(pixels, target)
            rows.append({"image": str(spec), "picture": str(target),
                         "z1": round(low, 4), "z2": round(high, 4)})
        return TaskResult(outputs=[Path(r["picture"]) for r in rows],
                          data={"rows": rows}, steps=self.plan(params))


@register
class Display(Task[DisplayParams]):
    """Load an image into a frame of the viewer. With --coords, each position
    in the list is circled, which is what tvmark is mostly used for: seeing
    what daofind found and what it missed.

    Without a screen -- over ssh, in a script -- the picture is written to a
    PNG instead, and the command says where.
    """

    name = "display"
    needs_window = True
    package = "tv"
    description = "Load an image into the viewer, with marks from a coordinate list."
    Params = DisplayParams

    def plan(self, params: DisplayParams) -> list[Step]:
        steps = [Step("show", f"{params.image} -> frame {params.frame}", why=WHY["show"]),
                 Step("limits", _how_limits(params), why=WHY["limits"])]
        if params.coords:
            steps.append(Step("mark", f"{params.coords}, radius {params.radius:g}",
                              why=WHY["mark"]))
        return steps

    def execute(self, params: DisplayParams) -> TaskResult:
        import numpy as np

        from ..core import catalog, display, images, mef

        data, header = images.read(params.image)
        _, _, _, quality = mef.load(params.image)
        z1, z2, table, notes = display_limits(params, data, header)
        steps = self.plan(params)
        steps[1].measured = {"z1": z1, "z2": z2}
        if quality.any():
            steps[1].measured["flagged in the DQ"] = int(np.count_nonzero(quality))
        screen = display.get(params.backend or None)
        stretch = {"linear": None, "log": "log", "none": "none", "user": "user"}[params.ztrans]
        screen.show(data, header, frame=params.frame, name=str(params.image),
                    window=(z1, z2), stretch=stretch, table=table,
                    quality=quality if quality.any() else None)

        if params.coords:
            table_xy = catalog.read(params.coords)
            x, y = catalog.coordinates(table_xy, params.coords)
            labels = (table_xy["id"] if "id" in table_xy.colnames else [""] * len(table_xy))
            screen.mark([display.Marker(float(a), float(b), radius=params.radius,
                                        label=str(label))
                         for a, b, label in zip(x, y, labels, strict=True)],
                        frame=params.frame)
            steps[-1].measured = {"marked": len(table_xy)}

        outputs = [screen.output] if hasattr(screen, "output") else []
        if params.wait and hasattr(screen, "run") and getattr(screen, "owns", False):
            screen.run()
        return TaskResult(outputs=outputs, steps=steps, warnings=notes,
                          # as IRAF prints them, %g of a single-precision value
                          data={"z1": float(f"{z1:.7g}"), "z2": float(f"{z2:.7g}"),
                                "shape": "x".join(str(n) for n in reversed(data.shape))})


def _how_limits(params: DisplayParams) -> str:
    if params.ztrans in ("none", "user"):
        return f"ztrans={params.ztrans}: z1=1 z2=200, the frame's own grey levels"
    if params.zscale:
        return f"zscale, contrast {params.contrast:g}, up to {params.nsample} pixels"
    if params.zrange:
        return f"zrange: the extremes of up to {params.nsample} pixels"
    return f"z1={params.z1} z2={params.z2}"


def display_limits(params: DisplayParams, data, header):
    """z1, z2 (and the user table) as t_display.x works them out.

    ztrans none and user switch zscale and zrange off and leave z1 and z2 at
    the frame's grey range, 1 and 200 -- in IRAF the user table's own range
    is read and then overwritten by these, so a table maps the values 1 to
    200. zrange would use the header's min and max when IRAF knows them to
    be current; a FITS file never says so, so it samples, as IRAF does."""
    from ..core import pixmask, render

    notes: list[str] = []
    table = None
    if params.ztrans == "user":
        if not params.lutfile:
            advice.fail("--ztrans user needs a table", "give it with --lutfile")
        table = render.UserLut(params.lutfile)
    if params.ztrans in ("none", "user"):
        return *render.DEVICE_GREY, table, notes
    if not params.zscale and not params.zrange:
        if params.z1 is None or params.z2 is None:
            advice.fail("with --no-zscale --no-zrange the limits are --z1 and --z2",
                        "give both, or leave --zscale on")
        return float(params.z1), float(params.z2), table, notes
    try:
        bpm = pixmask.read(params.bpmask, header, data.shape)
    except FileNotFoundError as error:
        # IRAF warns and goes on without the mask.
        notes.append(f"{error}; sampling without a bad pixel mask")
        bpm = None
    zmask = pixmask.read(params.zmask, header, data.shape) if params.zmask else None
    contrast = params.contrast if params.zscale else 0.0
    try:
        z1, z2 = render.zscale(data, contrast, params.nsample, zmask, bpm)
    except ValueError as error:
        advice.fail(str(error), "sample more of the image: a larger --nsample, or a "
                                "wider --zmask")
    return z1, z2, table, notes


class TvMarkParams(TaskParams):
    frame: int = Field(1, ge=1, description="Frame to mark.",
                       json_schema_extra={"positional": True})
    coords: str = Field("", description="Objects to mark: x and y in columns 1 and 2, "
                                        "a label in column 3 (a daofind .coo works).",
                        json_schema_extra={"positional": True})
    image: str = Field(
        "", description="Image to display in the frame first. Without it the frame must "
                        "already hold one (display it, in the same session).")
    logfile: str = Field("", description="File the cursor commands are logged to, in "
                                         "the form --commands reads back.")
    autolog: bool = Field(False, description="Log every command, not only those kept "
                                             "with k.")
    outimage: str = Field(
        "", description="Snapshot of the marked frame: FITS of its grey levels (1-200, "
                        "marks at their colour), or a .png picture. Written at the end "
                        "when marking the whole list; :write makes one any time.")
    deletions: str = Field(
        "", description="Extension of the file the deleted objects go to: coords.<ext>. "
                        "Empty keeps no such file.")
    commands: str = Field(
        "", description="Cursor commands to run instead of marking the whole list: "
                        "'x y frame key' or 'x y frame :command' per line -- a logfile "
                        "of an earlier run works.")
    mark: Literal["point", "circle", "rectangle", "line", "plus", "cross", "none"] = Field(
        "point", description="The mark drawn on each object.")
    radii: str = Field("0", description="Radii of the concentric circles, in image pixels: "
                                        "'5,10,20'.")
    lengths: str = Field("0", description="Lengths of the concentric rectangles, and their "
                                          "width/length after a blank: '10,20 0.5'.")
    font: str = Field("raster", description="The font; raster is the only one, as in IRAF.")
    color: int = Field(255, ge=0, le=255, description="Colour of the marks: IMTOOL's 202 "
                                                      "black, 203 white, 204 red, 205 "
                                                      "green, 206 blue, 207 yellow, 208 "
                                                      "cyan, 209 magenta ... 217 wheat.")
    label: bool = Field(False, description="Write each object's label (column 3) beside it.")
    number: bool = Field(False, description="Write each object's number in the list.")
    nxoffset: int = Field(0, description="Offset of the numbers in x, in pixels.")
    nyoffset: int = Field(0, description="Offset of the numbers in y, in pixels.")
    pointsize: int = Field(3, ge=1, description="Size of a point mark, rounded up to odd.")
    txsize: int = Field(1, ge=1, description="Size of text, numbers, + and x, in font cells "
                                             "of 6 x 7 pixels.")
    tolerance: float = Field(1.5, gt=0, description="How near the cursor an object must be "
                                                    "for d to delete it.")
    interactive: bool = Field(False, description="Mark with the cursor in the window: ? "
                                                 "lists the keys, q ends.")


@register
class TvMark(Task[TvMarkParams]):
    """IRAF's tvmark: objects from a list marked on a displayed frame, with
    the cursor or a command file, and the marked frame saved as an image.

    The marks are overlays on harwood's window; the snapshot draws them into
    the frame's grey levels -- the levels display computes, 1 to 200 -- at
    their colour number, so `display snap.fits --ztrans none` shows it as
    IRAF shows its snap. Every key and colon command goes through one
    engine, so a logfile replayed with --commands draws what the cursor did."""

    name = "tvmark"
    package = "tv"
    description = "Mark objects from a coordinate list on a displayed frame."
    Params = TvMarkParams

    def plan(self, params: TvMarkParams) -> list[Step]:
        how = ("with the cursor" if params.interactive else
               f"the commands in {params.commands}" if params.commands else
               "every object in the list")
        steps = [Step("mark", f"frame {params.frame}: {how}, {params.mark}",
                      why=WHY["mark"])]
        if params.outimage:
            steps.append(Step("snap", params.outimage, why=TVWHY["snap"]))
        return steps

    def execute(self, params: TvMarkParams) -> TaskResult:
        from ..core import display, tvmark

        settings = tvmark.settings_from(params)
        engine = tvmark.Engine(settings)
        if params.coords:
            if not Path(params.coords).exists():
                advice.fail(f"coordinate list '{params.coords}' does not exist",
                            "give the file daofind or a script wrote, or leave it empty "
                            "and mark with the cursor", FileNotFoundError)
            engine.items = tvmark.read_coords(params.coords)
        screen = display.get(None)
        data, header, window = _frame_contents(screen, params)

        if params.interactive:
            _interactive(screen, engine, params)
        elif params.commands:
            for x, y, _frame, key in tvmark.read_commands(params.commands):
                if key.startswith(":"):
                    engine.colon(key, x, y)
                else:
                    engine.key(key[:1], x, y)
                if engine.done:
                    break
        else:
            engine.key("l", 0.0, 0.0)
            if params.outimage:
                engine.snaps.append((params.outimage, list(engine.marks)))
        if not params.interactive and engine.marks:
            screen.mark(engine.marks, frame=params.frame)

        outputs = _write_snaps(engine, params, data, header, window)
        if params.logfile and engine.log:
            with open(params.logfile, "a", encoding="utf-8") as log:
                log.writelines(line + "\n" for line in engine.log)
            outputs.append(Path(params.logfile))
        if params.coords and (engine.deleted or engine.appended):
            tvmark.rewrite_coords(params.coords, engine.deleted, engine.appended)
            if params.deletions and engine.deleted:
                name = Path(f"{params.coords}.{params.deletions}")
                with open(name, "a", encoding="utf-8") as out:
                    out.writelines(f"{i.x:.3f} {i.y:.3f}" + (f" {i.label}" if i.label else "")
                                   + "\n" for i in engine.deleted)
                outputs.append(name)
        steps = self.plan(params)
        steps[0].measured = {"objects": len(engine.items), "marks": len(engine.marks),
                             "added": len(engine.appended), "deleted": len(engine.deleted)}
        data_out: dict = {"marked": len(engine.marks)}
        if engine.messages:
            data_out["text"] = engine.messages
        return TaskResult(outputs=outputs, steps=steps, data=data_out)


TVWHY = {
    "snap": "The frame as it looks marked, kept: a FITS of its grey levels with "
            "the marks at their colour number, as IRAF's snap image, or a PNG "
            "picture. The pixels of the data are not in it.",
}


def _frame_contents(screen, params: TvMarkParams):
    """The image in the frame and its display limits: --image displayed
    now, or what the window's frame already holds."""
    from ..core import images

    if params.image:
        data, header = images.read(params.image)
        defaults = DisplayParams(image=params.image)
        z1, z2, table, _ = display_limits(defaults, data, header)
        screen.show(data, header, frame=params.frame, name=params.image, window=(z1, z2))
        return data, header, (z1, z2)
    held = getattr(getattr(screen, "window", None), "frames", {}).get(params.frame)
    if held is None:
        advice.fail(f"frame {params.frame} holds no image to mark",
                    "display one first in the same session, or give it with --image")
    return held.data, held.header, held.limits(_fixed_or_zscale(held))


def _fixed_or_zscale(frame) -> str:
    return "z1 z2" if getattr(frame, "window", None) else "zscale"


def _interactive(screen, engine, params) -> None:
    window = getattr(screen, "window", None)
    if window is None or not hasattr(window, "wait_for_examine"):
        advice.fail("--interactive needs the window",
                    "run it where there is a screen, or replay a command file with "
                    "--commands")
    drawn = [0]

    def redraw() -> None:
        frame = window.frames.get(params.frame)
        if frame is None:
            return
        frame.markers = [m for m in frame.markers if m.tag != "tvmark"] + list(engine.marks)
        window._redraw()
        for message in engine.messages[drawn[0]:]:
            window.log("  " + message)
        drawn[0] = len(engine.messages)

    def key(name: str, x: float, y: float) -> bool:
        engine.key(name, x, y)
        redraw()
        if engine.done:
            window.examine_done.emit()
        return True

    def colon(line: str) -> str:
        position = getattr(window, "_cursor", None) or (0.0, 0.0)
        answer = engine.colon(line, *position)
        redraw()
        return answer

    window.key_hook, window.colon_hook = key, colon
    window.set_task_keys("tvmark", tvmark_keys())
    window.log("tvmark: point and press a key -- the Keys panel lists them; q ends.")
    try:
        window.wait_for_examine()
    finally:
        window.key_hook = window.colon_hook = None
        window.set_task_keys(None)


def tvmark_keys():
    from ..core import tvmark

    return tvmark.HELP


def _write_snaps(engine, params, data, header, window) -> list[Path]:
    """Each snapshot asked for: by :write name, or outimage; a :write with no
    name is outimage.snap.N, or the image's name .snap.N, N the first free."""
    import numpy as np
    from astropy.io import fits

    from ..core import render, tvmark

    written = []
    z1, z2 = window
    short = data.dtype == np.int16 or (header is not None and header.get("BITPIX") == 16
                                       and not header.get("BZERO"))
    levels = render.device_levels(data, z1, z2, short=short)
    for asked, marks in engine.snaps:
        target = Path(asked) if asked else _next_snap(params)
        snap = tvmark.snapshot(levels, marks, params.color)
        if target.suffix.lower() == ".png":
            from ..viewer.png import save_rgb

            grey = np.clip((snap.astype(float) - 1) / 199, 0, 1)
            rgb = render.lut("grey")[np.clip((grey * 255).astype(int), 0, 255)]
            for value in np.unique(snap[snap > 200]):
                rgb[snap == value] = RGB.get(tvmark.colour_name(int(value)), RGB["lime"])
            save_rgb(rgb[::-1].copy(), target)
        else:
            if target.suffix.lower() not in (".fits", ".fit"):
                target = target.with_name(target.name + ".fits")
            header_out = fits.Header({"OBJECT": f"tvmark snapshot of frame {params.frame}"})
            fits.PrimaryHDU(snap, header_out).writeto(target, overwrite=True)
        written.append(target)
    return written


def _next_snap(params) -> Path:
    stem = params.outimage or Path(params.image or "frame").name.removesuffix(".fits")
    number = 1
    while Path(f"{stem}.snap.{number}.fits").exists():
        number += 1
    return Path(f"{stem}.snap.{number}")


#: The overlay colours as RGB, for a PNG snapshot.
RGB = {"black": (0, 0, 0), "white": (255, 255, 255), "red": (255, 0, 0),
       "green": (0, 255, 0), "blue": (0, 0, 255), "yellow": (255, 255, 0),
       "cyan": (0, 255, 255), "magenta": (255, 0, 255), "coral": (255, 127, 80),
       "maroon": (128, 0, 0), "orange": (255, 165, 0), "khaki": (240, 230, 140),
       "orchid": (218, 112, 214), "turquoise": (64, 224, 208), "violet": (238, 130, 238),
       "wheat": (245, 222, 179), "lime": (0, 255, 0)}


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
    output: str = Field(
        "", description="Root of the images the t key writes, as root.001.fits. Empty is "
                        "the image's own name."
    )
    ncoutput: int = Field(101, ge=1, description="Columns of the t key's section.")
    nloutput: int = Field(101, ge=1, description="Lines of the t key's section.")
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
        l   plot the line                    e   contour plot of a box
        c   plot the column                  s   surface plot of a box
        x   coordinates and value            z   the 11x11 pixels around it
        b   box of two positions             y   set the origin for x
        u   vector centred on a position     v   vector between two positions
        ,   quick photometry                 .   quick profile and fit
        t   a section as a new image         o   overplot the next graph
        f   redraw the last graph            w   start or stop the logfile
        d   load an image (in the window)    g, i  change cursor: both are live here
        n   next frame                       p   previous frame
        ?   list the keys                    q   end the task
        :   a colon command -- `:radius 7` sets, `:radius` shows

    Colon commands are IRAF's: a name shared by several of IRAF's psets
    (width, rplot, naverage, ncolumns, floor...) changes the one of the last
    graph drawn, and `:unlearn` resets it. In a cursor file they go after the
    frame, `612 540 1 :radius 7`, and act on the lines that follow.

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
    measured without a model. Every key's arithmetic is a port of IRAF's own
    (ierimexam.x and the starfocus code under it, iejimexam.x with its
    Marquardt steps, iestatistics.x, iehimexam.x, ieprint.x, conrec), and
    tests hold r, a, m, j, k and z to what IRAF 2.18 printed for the same
    pixels.
    """

    name = "imexamine"
    needs_window = True
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
            name: getattr(params, name) for name in examine.Settings.__dataclass_fields__
            if hasattr(params, name)})
        # The task's naverage and sigma are for j and k alike; IRAF keeps a
        # kimexam beside jimexam, which colon commands can tell apart.
        settings.knaverage, settings.ksigma = params.naverage, params.sigma
        settings.logname = params.logfile

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
                data, header = images.read(specs[index])
                loaded[frame] = (str(specs[index]), examine.native(data, header), header)
            name, data, settings.header = loaded[frame]
            settings.image = name
            if key.startswith(":"):
                settings.redraw = False
                answer = examine.colon(settings, key)
                result = examine.Result(":", x, y, f"[{key}] {answer}",
                                        ok=not answer.startswith("Parameter does not"))
                results.append((name, result))
                logfile.write(params.logfile, "imexamine", name, [result.line])
                if settings.redraw and settings.graph and settings.last_position:
                    results.append((name, examine.examine(data, "f", x, y, settings)))
                continue
            result = examine.examine(data, key, x, y, settings)
            results.append((name, result))
            if settings.keeplog:
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


# ------------------------------------------------------------------- wcslab

WLWHY = {
    "grid": "The world system's ranges from twenty points of the edges, rounded to a "
            "step from IRAF's tables (seconds of time and of arc for RA and Dec); the "
            "lines of constant RA and Dec drawn in segments, a polar graph when the "
            "pole is in the image and a near-polar one when it is close.",
    "label": "Where each line crosses an edge: hms and dms, the one nearest the "
             "corner in full; the title two character heights beyond the labels.",
    "draw": "On the display over the image (imd), or a graph: alone, or added to "
            "the last one drawn (append, over contour).",
}

Side = Literal["default", "left", "right", "bottom", "top"]
LineType = Literal["solid", "dotted", "dashed", "dotdash"]


class WcslabParams(TaskParams):
    image: str = Field("", description="Image whose world coordinates label the grid; "
                                       "empty: wcspars' own (--usewcs), over log_x1.."
                                       "log_y2.", json_schema_extra={"positional": True})
    frame: int = Field(1, ge=1, description="Display frame the image is in (imd).",
                       json_schema_extra={"positional": True})
    usewcs: bool = Field(False, description="Use the WCS given here (ctype1 ... cd2_2) "
                                            "instead of the image's.")
    ctype1: str = Field("linear", description="wcspars: axis 1 type (linear, ra--tan "
                                              "...).")
    ctype2: str = Field("linear", description="wcspars: axis 2 type (linear, dec--tan "
                                              "...).")
    crpix1: float = Field(0.0, description="wcspars: reference pixel, x.")
    crpix2: float = Field(0.0, description="wcspars: reference pixel, y.")
    crval1: float = Field(0.0, description="wcspars: world at the reference, axis 1.")
    crval2: float = Field(0.0, description="wcspars: world at the reference, axis 2.")
    cd1_1: float = Field(1.0, description="wcspars: CD matrix.")
    cd1_2: float = Field(0.0, description="wcspars: CD matrix.")
    cd2_1: float = Field(0.0, description="wcspars: CD matrix.")
    cd2_2: float = Field(1.0, description="wcspars: CD matrix.")
    log_x1: float = Field(0.0, description="wcspars: left pixel, with no image.")
    log_x2: float = Field(1.0, description="wcspars: right pixel, with no image.")
    log_y1: float = Field(0.0, description="wcspars: bottom pixel, with no image.")
    log_y2: float = Field(1.0, description="wcspars: top pixel, with no image.")
    major_grid: bool = Field(True, description="Major grid lines (else major ticks).")
    minor_grid: bool = Field(False, description="Minor grid lines (else minor ticks).")
    dolabel: bool = Field(True, description="Label the major lines or ticks.")
    axis1_beg: str = Field("", description="First major axis 1 value (RA in hours, "
                                           "h:m:s too); all three of beg, end and int, or "
                                           "they are worked out.")
    axis1_end: str = Field("", description="Last major axis 1 value.")
    axis1_int: str = Field("", description="Axis 1 interval (RA in hours).")
    axis2_beg: str = Field("", description="First major axis 2 value (Dec in degrees).")
    axis2_end: str = Field("", description="Last major axis 2 value.")
    axis2_int: str = Field("", description="Axis 2 interval.")
    major_line: LineType = Field("solid", description="Major grid line type.")
    major_tick: float = Field(0.03, ge=0, le=1, description="Major tick, a fraction of "
                                                           "the viewport.")
    axis1_minor: int = Field(5, ge=1, description="Minor intervals per major, axis 1.")
    axis2_minor: int = Field(5, ge=1, description="Minor intervals per major, axis 2.")
    minor_line: LineType = Field("dotted", description="Minor grid line type.")
    minor_tick: float = Field(0.01, ge=0, le=1, description="Minor tick, a fraction of "
                                                           "the viewport.")
    tick_in: bool = Field(True, description="Ticks point into the graph.")
    axis1_side: str = Field("default", description="Sides to label axis 1 on: left, "
                                                   "right, bottom, top, or a list.")
    axis2_side: str = Field("default", description="Sides to label axis 2 on.")
    axis2_dir: str = Field("", description="Polar graph: the RA (hours) along which "
                                           "the declinations are labelled.")
    justify: Side = Field("default", description="Polar graph: the side the Dec labels "
                                                 "are written towards.")
    labout: bool = Field(True, description="Labels outside the axes.")
    rotate: bool = Field(True, description="Labels follow their lines' angle.")
    full_label: bool = Field(False, description="Every label in full (h m s, d m s).")
    label_size: float = Field(1.0, gt=0, description="Label size.")
    title: str = Field("imtitle", description="Title; imtitle: the image's name and "
                                              "title.")
    axis1_title: str = Field("", description="Axis 1 title.")
    axis2_title: str = Field("", description="Axis 2 title.")
    title_side: Literal["top", "bottom", "left", "right"] = Field(
        "top", description="Side of the title.")
    axis1_title_side: Side = Field("default", description="Side of axis 1's title.")
    axis2_title_side: Side = Field("default", description="Side of axis 2's title.")
    title_size: float = Field(1.0, gt=0, description="Title size.")
    axis_title_size: float = Field(1.0, gt=0, description="Axis titles' size.")
    graph_type: Literal["default", "normal", "polar", "near_polar"] = Field(
        "default", description="Graph type; default decides from the pole's place.")
    fill: bool = Field(True, description="Fill the viewport (else the image's shape).")
    vl: float | None = Field(None, ge=0, le=1, description="Viewport left (NDC).")
    vr: float | None = Field(None, ge=0, le=1, description="Viewport right.")
    vb: float | None = Field(None, ge=0, le=1, description="Viewport bottom.")
    vt: float | None = Field(None, ge=0, le=1, description="Viewport top.")
    overplot: bool = Field(False, description="Draw over the last graph, on the "
                                              "viewport given (stdgraph).")
    append: bool = Field(False, description="Draw over the last graph, on its own "
                                            "frame: wcslab after contour.")
    device: str = Field("imd", description="imd: over the image on the display; "
                                           "stdgraph: a graph (the window, or an SVG "
                                           "without a screen); or a file.svg.")
    wait: bool = Field(True, description="imd: keep the window open until it is "
                                         "closed. --no-wait returns at once; inside "
                                         "harwood gui it never waits.")


@register
class Wcslab(Task[WcslabParams]):
    """IRAF's wcslab (images.tv, pkg/images/tv/wcslab): every text it writes -- the
    hms and dms labels with their units, the titles -- in its place and every line
    it draws -- the grid, the ticks, the pole's plus -- identical to IRAF 2.18's
    (to two GKI units) on 47 cases: RA/Dec in every orientation, polar and near-
    polar, linear and other systems, and wlpars' options."""

    name = "wcslab"
    package = "tv"
    description = "Overlay a labelled world coordinate grid on an image or a graph."
    Params = WcslabParams

    def plan(self, params: WcslabParams) -> list[Step]:
        where = ("the last graph" if params.append or params.overplot
                 else f"frame {params.frame}" if params.device == "imd" else params.device)
        return [Step("grid", params.image or "wcspars' WCS", why=WLWHY["grid"]),
                Step("label", "labels" if params.dolabel else "no labels",
                     why=WLWHY["label"]),
                Step("draw", where, why=WLWHY["draw"])]

    def execute(self, params: WcslabParams) -> TaskResult:
        from ..core import display, iraffmt, wcslab
        from ..core import graph as graphmod
        from .imutil import _plot

        p = params.model_dump()
        if not params.image and not params.usewcs:
            p["usewcs"] = True
        notes = []
        to_graph = params.device != "imd" or params.append or params.overplot
        if not to_graph and display._autodetect() != "native":
            to_graph = True
            notes.append(advice.warn("there is no display window here: the grid is "
                                     "drawn as a graph instead",
                                     "run it where there is a screen to see it over "
                                     "the image"))
        viewport = None
        last = None
        if params.append or params.overplot:
            last = graphmod.last()
            if last is None:
                advice.fail("there is no graph to draw over",
                            "draw one first (contour wpix.fits --no-perimeter), or "
                            "leave out --append")
        try:
            if params.append and last is not None:
                shape = _shape_of(params, p)
                viewport = wcslab.map_viewport(*shape, None, None, None, None, False)
            elif params.device == "imd" and not params.overplot and params.image:
                ny, nx = _image_shape(params.image)
                (_, viewport) = wcslab.imd_viewport(nx, ny, vl=params.vl, vr=params.vr,
                                                    vb=params.vb, vt=params.vt)
                viewport = wcslab.map_viewport(1.0, float(nx), 1.0, float(ny), *viewport,
                                               params.fill)
            d, messages = wcslab.labelled(params.image, p, viewport)
        except wcslab.WcslabError as error:
            advice.fail(f"wcslab: {error}", "give a two-dimensional image")
        outputs = []
        if to_graph or not params.image:
            g = last if last is not None else graphmod.Graph([])
            if last is None:
                # the page, out to what the labels and titles need beyond it
                g.frame = False
                xs = [0.0, 1.0] + [t.x for t in d.texts if t.text.strip()]
                ys = [0.0, 1.0] + [t.y for t in d.texts if t.text.strip()]
                pad = 0.04
                x0, y0 = d.to_plot([min(xs) - pad, max(xs) + 2 * pad],
                                   [min(ys) - pad, max(ys) + pad])
                g.window = (float(x0[0]), float(x0[1]), float(y0[0]), float(y0[1]))
            for s in d.strokes:
                x, y = d.to_plot(s.x, s.y)
                g.line(x, y, s.ltype, 1)
            for t in d.texts:
                x, y = d.to_plot([t.x], [t.y])
                g.label(float(x[0]), float(y[0]), t.text, t.up - 90.0,
                        (_ANCHOR_H[t.h], _ANCHOR_V[t.v]), d.char_height(t.size))
            device = "stdgraph" if params.device == "imd" else params.device
            where = _plot(g, device, f"wcslab · {params.image}",
                          (Path(params.image.split("[")[0]).stem or "wcspars")
                          + ".wcslab.svg", task="wcslab")
            outputs = [where] if where else []
        else:
            outputs = _over_display(params, d)
        system = d.system
        axes = {}
        for i, name in enumerate(("axis1", "axis2")):
            beg, end, step = d.begin[i], d.end[i], d.interval[i]
            if system == "ra_dec":
                if i == 0:
                    beg, end, step = beg / 15, end / 15, step / 15
                axes[name] = {k: iraffmt.dtoc(v, 2, "h") for k, v in
                              (("beg", beg), ("end", end), ("int", step))}
            else:
                axes[name] = {k: float(f"{v:.7g}") for k, v in
                              (("beg", beg), ("end", end), ("int", step))}
        data = {"system": system, "graph_type": d.graph_type, **axes,
                "labels": sum(1 for t in d.texts if t.text.strip()) if params.dolabel else 0}
        if messages:
            data["text"] = messages
        return TaskResult(data=data, steps=self.plan(params), outputs=outputs,
                          warnings=notes + [advice.warn(m, "use solid, dotted, dashed "
                                                                   "or dotdash")
                                            for m in messages])


_ANCHOR_H = {"c": 0.5, "l": 0.0, "r": 1.0}
_ANCHOR_V = {"c": 0.5, "t": 0.0, "b": 1.0}


def _image_shape(image: str):
    from ..core import images

    data, _ = images.read(image)
    return data.shape


def _shape_of(params, p):
    if params.image:
        ny, nx = _image_shape(params.image)
        return 1.0, float(nx), 1.0, float(ny)
    return p["log_x1"], p["log_x2"], p["log_y1"], p["log_y2"]


def _over_display(params: WcslabParams, d) -> list[Path]:
    """The drawing over the image in its frame: the image displayed first when the
    frame does not hold it."""
    import numpy as np

    from ..core import display, images

    screen = display.get(None)
    held = getattr(getattr(screen, "window", None), "frames", {}).get(params.frame)
    data, header = images.read(params.image)
    if held is None or getattr(held, "data", None) is None or held.data.shape != data.shape:
        defaults = DisplayParams(image=params.image)
        z1, z2, table, _ = display_limits(defaults, data, header)
        screen.show(data, header, frame=params.frame, name=params.image, window=(z1, z2))
    marks = []
    for s in d.strokes:
        x, y = d.to_plot(s.x, s.y)
        marks.append(display.Marker(float(x[0]), float(y[0]), shape="path",
                                    points=tuple(zip(x.tolist(), y.tolist(), strict=True)),
                                    ltype=s.ltype, colour="lime", tag="wcslab"))
    for t in d.texts:
        if not t.text.strip():
            continue
        x, y = d.to_plot([t.x], [t.y])
        marks.append(display.Marker(float(x[0]), float(y[0]), shape="text", label=t.text,
                                    angle=t.up - 90.0,
                                    anchor=(_ANCHOR_H[t.h], _ANCHOR_V[t.v]),
                                    size=t.size, colour="lime", tag="wcslab"))
    screen.mark(marks, frame=params.frame)
    view = getattr(getattr(screen, "window", None), "view", None)
    if view is not None and d.texts:
        # the labels are beyond the image's edges, as in IRAF's frame: show them
        from ..core import pixels

        xs, ys = d.to_plot([t.x for t in d.texts] + [d.viewport[0], d.viewport[1]],
                           [t.y for t in d.texts] + [d.viewport[2], d.viewport[3]])
        sx, sy = pixels.to_scene(xs, ys)
        pad = 0.06 * max(float(np.ptp(sx)), float(np.ptp(sy)))
        view.setRange(xRange=(float(sx.min()) - pad, float(sx.max()) + pad),
                      yRange=(float(sy.min()) - pad, float(sy.max()) + pad), padding=0)
    outputs = [screen.output] if hasattr(screen, "output") else []
    if params.wait and hasattr(screen, "run") and getattr(screen, "owns", False):
        screen.run()
    return outputs
