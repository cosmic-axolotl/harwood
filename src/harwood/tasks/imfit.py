"""imfit -- IRAF's images.imfit fit1d, and imred.generic's background, which is a
fit1d of type difference.

`background` is what the long-slit course runs after transform: each line of the
rectified frame fitted across the slit on the sky's columns only (--sample), and
the fit subtracted -- the sky's lines gone, the galaxy left. The arithmetic is
core/fit1d.py, a port of IRAF's.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Literal

from pydantic import Field

from ..core import advice
from ..core.registry import register
from ..core.task import Step, Task, TaskParams, TaskResult

WHY = {
    "fit": "Each line (axis 1) or column (axis 2) fitted with ICFIT on the sample "
           "points -- for a sky, the columns off the object -- with rejection of what "
           "stands out.",
    "output": "The fit itself, the data minus it (a background subtracted), or the data "
              "over it (a normalization).",
    "write": "In real, a new image the size of the whole input -- zeros outside a "
             "section (ones for a ratio); an image that already exists is written in "
             "place, only where the section is, as IRAF writes it.",
}

FUNCTIONS = Literal["spline3", "legendre", "chebyshev", "spline1"]


class _FitFields(TaskParams):
    input: list[str] = Field(..., description="Images to fit. A section is fitted and "
                                              "written as a section.",
                             json_schema_extra={"positional": True})
    output: list[str] = Field(..., description="Output images, one each; the input "
                                               "itself to replace it.",
                              json_schema_extra={"positional": True})
    axis: Literal[1, 2] = Field(1, description="1: fit along the lines (across the "
                                               "columns); 2: along the columns.")
    interactive: bool = Field(False, description="Set the fit by hand (IRAF's "
                                                 "interactive=yes): the lines asked for "
                                                 "are fitted in the icfit window, and "
                                                 "what is set there fits the image.")
    lines: list[str] = Field(default_factory=list, description="With --interactive, the "
                             "answers to 'Fit line =': 'N' or 'N M' (the average of lines "
                             "N to M), one each. Left out, they are asked in the "
                             "terminal; an empty answer ends.")
    cursor: str = Field("", description="The fits' keys from a file, 'x y 1 key' or "
                                        "':command' a line, each fit up to its q -- "
                                        "IRAF's graphics cursor file.")
    sample: str = Field("*", description="Points to fit, as ranges: '10:50 60:110'.")
    naverage: int = Field(1, description="Points averaged (positive) or medianed "
                                         "(negative) into each one fitted.")
    order: int = Field(1, ge=1, description="Order: polynomial terms, or spline pieces.")
    low_reject: float = Field(0.0, ge=0.0, description="Low rejection, in sigmas. 0: none.")
    high_reject: float = Field(0.0, ge=0.0, description="High rejection, in sigmas. "
                                                        "0: none.")
    niterate: int = Field(1, ge=0, description="Rejection iterations.")
    grow: float = Field(0.0, ge=0.0, description="Rejection growing radius, in pixels.")


class Fit1dParams(_FitFields):
    type: Literal["fit", "difference", "ratio"] = Field(
        ..., description="What is written: the fit, data - fit, or data / fit.",
        json_schema_extra={"positional": True})
    bpm: str = Field("", description="Bad pixel masks (nonzero: left out of the fit), "
                                     "one or one each.")
    function: FUNCTIONS = Field("spline3", description="The fitting function.")


class BackgroundParams(_FitFields):
    function: FUNCTIONS = Field("chebyshev", description="The fitting function.")


def _pairs(inputs, outputs):
    from ..core import images

    ins, given = [], []
    for group in inputs:
        for piece in images.split_list(group):
            found = images.expand(piece)
            if not found:
                advice.fail(f"{piece} matches no image", "check the name",
                            FileNotFoundError)
            ins += found
            given += [piece] if len(found) == 1 else [str(s) for s in found]
    outs = [p for g in outputs for p in images.split_list(g) if p.strip()]
    if len(ins) != len(outs):
        advice.fail("Input and output image lists do not match",
                    f"{len(ins)} inputs and {len(outs)} outputs: give one each")
    return ins, given, outs


def _output(name: str):
    """imgimage and imgsection of an output name: its file and its section."""
    from ..core import images

    spec = images.parse(name)
    path = Path(spec.path)
    if path.suffix not in (".fits", ".fit", ".fz"):
        path = path.with_name(f"{path.name}.fits")
    return path, spec.section or ""


def run_fit1d(params, kind: str, function: str, bpms: list[str], task: str):
    """t_fit1d: each input fitted and its output written as f1d_immap maps it."""
    import numpy as np

    from ..core import fit1d, images, imsum, mef, pixels, pixmask

    ins, given, outs = _pairs(params.input, params.output)
    if len(bpms) > 1 and len(bpms) != len(ins):
        advice.fail("Input and mask lists do not match", "give one mask, or one each")
    spec = {k: getattr(params, k) for k in ("sample", "naverage", "order", "low_reject",
                                            "high_reject", "niterate", "grow")}
    spec["function"] = function
    answers = list(params.lines)
    written, warnings, rows, text = [], [], [], []
    bpm = ""
    for k, (inspec, name, output) in enumerate(zip(ins, given, outs, strict=True)):
        if k < len(bpms):
            bpm = bpms[k]
        hdul, hdu = images.open_hdu(images.ImageSpec(inspec.path, inspec.extension))
        try:
            stored = np.array(hdu.data)
            header = images.inherited(hdul, hdu)
        finally:
            hdul.close()
        if stored.ndim > 2:
            warnings.append(advice.warn(f"{name}: Image dimensions > 2 are not "
                                        "implemented", "fit a 1-D or 2-D image"))
            continue
        if params.axis > stored.ndim:
            warnings.append(advice.warn(f"{name}: Axis exceeds image dimension",
                                        "use --axis 1 for a 1-D image"))
            continue
        cut = (pixels.parse_section(inspec.section.strip("[]"), stored.shape)
               if inspec.section else tuple(slice(None) for _ in stored.shape))
        data = imsum.convert(imsum.ieee_in(stored), np.float32)[cut]
        opath, osect = _output(output)
        if opath.exists():
            hd, h = images.open_hdu(images.ImageSpec(opath, None))
            try:
                target = np.array(h.data)
                theader = images.inherited(hd, h)
            finally:
                hd.close()
            ocut = (pixels.parse_section(osect.strip("[]"), target.shape) if osect
                    else cut)
        else:
            fill = 1.0 if kind == "ratio" else 0.0
            target = np.full(stored.shape, fill, np.float32)
            theader = header.copy()
            ocut = (pixels.parse_section(osect.strip("[]"), target.shape) if osect
                    else cut)
        if target[ocut].shape != data.shape:
            warnings.append(advice.warn(f"{name}: Input and output images have "
                                        "different sizes",
                                        "give an output section the size of the input's"))
            continue
        mask = None
        if bpm:
            full = pixmask.read(bpm, header, stored.shape)
            mask = None if full is None else np.asarray(full)[cut]
        if params.interactive:
            spec = _by_hand(params, data, mask, name, header, spec, answers)
        data2 = data if data.ndim == 2 else data[None, :]
        mask2 = None if mask is None else (mask if mask.ndim == 2 else mask[None, :])
        axis = params.axis if data.ndim == 2 else 1
        values, messages = fit1d.fit1d(data2, axis, kind, spec, mask2)
        text += messages
        if data.ndim == 1:
            values = values[0]
        out = np.array(target)
        out[ocut] = imsum.convert(values, out.dtype)
        theader["IRAF-TLM"] = time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime())
        theader.add_history(f"{task}: {kind} of a {function} order {params.order} along "
                            f"axis {params.axis}")
        files, notes = mef.save(opath, out, theader, None, None, keep_type=True)
        warnings.extend(notes)
        written += files
        rows.append({"input": name, "output": output, "type": kind})
    return written, warnings, rows, text


def _answer(answers: list[str], prompt: str) -> str | None:
    """f1d_getline's getline: the next answer given, else the terminal's; None at the
    end (EOF, or no terminal to ask)."""
    import sys

    if answers:
        return answers.pop(0)
    if not sys.stdin.isatty():
        return None
    try:
        return input(prompt + " ")
    except EOFError:
        return None


def _by_hand(params, data, mask, name: str, header, spec: dict, answers: list[str]):
    """f1d_fit1d's interactive part: for each line asked for, the average of the lines
    (xt_21imavg) fitted in the icfit window or from the cursor file; the parameters it
    leaves are those the image is fitted with."""
    import numpy as np

    from ..core import longslit
    from .longslit import _by_hand as icfit_by_hand

    title_line = str(header.get("OBJECT", "")).strip()
    if data.ndim == 1:
        x = np.arange(1, data.size + 1, dtype=np.float32)
        title = f"{name}\n{title_line}"
        spec, _ = icfit_by_hand(params, x, data.astype(np.float32), spec, title, "")
        return spec
    axis = params.axis
    nlines, ncols = data.shape
    imlen = nlines if axis == 1 else ncols
    what = "line" if axis == 1 else "column"
    while True:
        text = _answer(answers, f"{name}: Fit {what} =")
        if text is None:
            break
        words = text.replace(",", " ").split()
        numbers = []
        for word in words[:2]:
            try:
                numbers.append(int(word))
            except ValueError:
                break
        if not numbers:
            break
        i = max(1, min(imlen, numbers[0]))
        j = i if len(numbers) == 1 else max(1, min(imlen, numbers[1]))
        title = f"{name}: Fit {what} = {i} - {j}\n{title_line}"
        if axis == 1:
            x, y = longslit.aimavg(data, 1, 1, ncols, i, j)
        else:
            x, y = longslit.aimavg(data, 2, i, j, 1, nlines)
        spec, _ = icfit_by_hand(params, x, y, spec, title,
                                "Column" if axis == 1 else "Line")
    return spec


@register
class Fit1d(Task[Fit1dParams]):
    """IRAF's fit1d, from images/imfit/src/fit1d.x: every line or column fitted with
    ICFIT, and the fit, the difference or the ratio written. Checked pixel for pixel
    against IRAF 2.18."""

    name = "fit1d"
    package = "imfit"
    description = "Fit a function to each line or column of images."
    Params = Fit1dParams

    def plan(self, params: Fit1dParams) -> list[Step]:
        return [Step("fit", f"{params.function} order {params.order} along axis "
                            f"{params.axis}, sample {params.sample}", why=WHY["fit"]),
                Step("output", params.type, why=WHY["output"]),
                Step("write", ", ".join(params.output), why=WHY["write"])]

    def execute(self, params: Fit1dParams) -> TaskResult:
        from ..core import images

        bpms = images.split_list(params.bpm) if params.bpm.strip() else []
        written, warnings, rows, text = run_fit1d(params, params.type, params.function,
                                                  bpms, "fit1d")
        data: dict = {"rows": rows}
        if text:
            data["text"] = text
        return TaskResult(outputs=written, steps=self.plan(params), data=data,
                          warnings=warnings, history=[f"fit1d: {len(rows)} image(s)"])


@register
class Background(Task[BackgroundParams]):
    """IRAF's background, imred/generic/background.cl: fit1d with the output of type
    difference -- the fit across the slit subtracted from each line. Checked pixel
    for pixel against IRAF 2.18."""

    name = "background"
    package = "generic"
    description = "Subtract a background fitted along the lines or columns."
    Params = BackgroundParams

    def plan(self, params: BackgroundParams) -> list[Step]:
        return [Step("fit", f"{params.function} order {params.order} along axis "
                            f"{params.axis}, sample {params.sample}", why=WHY["fit"]),
                Step("output", "difference", why=WHY["output"]),
                Step("write", ", ".join(params.output), why=WHY["write"])]

    def execute(self, params: BackgroundParams) -> TaskResult:
        written, warnings, rows, text = run_fit1d(params, "difference", params.function,
                                                  [], "background")
        data: dict = {"rows": rows}
        if text:
            data["text"] = text
        return TaskResult(outputs=written, steps=self.plan(params), data=data,
                          warnings=warnings, history=[f"background: {len(rows)} image(s)"])
