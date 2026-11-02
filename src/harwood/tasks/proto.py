"""proto -- the small utilities IRAF kept in its prototype package.

`fixpix` is the one a reduction needs: bad columns, hot pixels and the
occasional dead row replaced by interpolating across them, with a mask that
says which pixels they were. The arithmetic is `core.fixpix`, which ccdproc's
fixpix step uses too, so a frame fixed by either is fixed the same way -- the
way IRAF fixes it.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import Field

from ..core import advice
from ..core.registry import register
from ..core.task import Step, Task, TaskParams, TaskResult, format_field, uncertainty_field

WHY = {
    "regions": "A mask marks what no interpolation can recover. The value says the shape, "
               "so that a bad column is fixed from its neighbours across and a dead row "
               "from those above and below.",
    "extensions": "A multi-extension file is several images. This names them as a task "
                  "reads them, so the list can go to any task as an @file.",
    "through a mask": "The pixels a mask marks are left out of every number, and the "
                      "limits and clipping work on the rest.",
    "mask": "Non-zero marks a bad pixel, as in IRAF; the value can also say "
            "which way to interpolate (linterp, cinterp). BPM takes the mask "
            "named in each image's own BPM keyword, so a whole night can be "
            "fixed with the masks its headers already point to; DQ takes the "
            "image's own DQ plane -- whatever cosmicrays, ccdproc and the "
            "rest have flagged in it.",
    "fix": "Each bad region is interpolated across its narrowest side: a bad "
           "column along the lines, a bad row along the columns, a single "
           "pixel along its line. The replacement is linear between the good "
           "pixels either side, so it follows the sky's gradient -- and is "
           "still not a measurement, which is why the DQ keeps the pixel "
           "flagged. The variance is interpolated the same way: the noise a "
           "pixel there would have had.",
    "write": "The pixel type is kept: an integer image is rounded back to "
             "integers, as IRAF writes it. The input is not modified unless "
             "--output names it.",
}


class FixpixParams(TaskParams):
    images: list[str] = Field(
        ..., description="Images to fix. Globs and @lists expand.",
        json_schema_extra={"positional": True},
    )
    masks: list[str] = Field(
        ..., description="Bad pixel masks: one for all the images, or one each. "
                         "BPM uses the mask in each image's BPM keyword; DQ the "
                         "pixels its own DQ flags. A text file of 'x y' or "
                         "'x1 x2 y1 y2' lines is accepted too.",
    )
    linterp: int | None = Field(
        None, ge=1, description="Mask value that forces interpolation along "
                                "lines. Empty is IRAF's INDEF: the narrowest side.")
    cinterp: int | None = Field(
        None, ge=1, description="Mask value that forces interpolation along "
                                "columns. Empty is IRAF's INDEF.")
    verbose: bool = Field(False, description="Print 'FIXPIX: image ... with mask ...' "
                                             "for each image, as IRAF does.")
    pixels: bool = Field(
        False, description="With --verbose, list every pixel replaced: x, y, the value "
                           "it had, the value it got, and the good pixels it came from.")
    output: str = Field(
        "", description="Where to write: a directory or a prefix. Left out, this "
                        "run's own directory. IRAF fixes in place; name the "
                        "input's own directory to do the same.")
    uncertainty: Literal["VAR", "ERR", "UNCERT"] = uncertainty_field()
    format: Literal["harwood", "iraf"] = format_field()


@register
class Fixpix(Task[FixpixParams]):
    """Replace the pixels a mask marks by interpolating across them, as
    IRAF's proto.fixpix does: across the narrowest side of each bad region,
    linearly between the good pixels either side, with the edge taking its
    one good neighbour. The arithmetic is a port of IRAF's own (xtfixpix),
    checked pixel for pixel against IRAF 2.18 on a mask of every shape it
    decides differently on.
    """

    name = "fixpix"
    package = "proto"
    description = "Replace bad pixels by interpolating across them, as IRAF's fixpix."
    Params = FixpixParams

    def plan(self, params: FixpixParams) -> list[Step]:
        forced = []
        if params.linterp is not None:
            forced.append(f"{params.linterp} along lines")
        if params.cinterp is not None:
            forced.append(f"{params.cinterp} along columns")
        return [Step("mask", ", ".join(params.masks), why=WHY["mask"]),
                Step("fix", "across the narrowest side"
                            + (f"; mask value {' and '.join(forced)}" if forced else ""),
                     why=WHY["fix"]),
                Step("write", params.output or "this run's directory", why=WHY["write"])]

    def execute(self, params: FixpixParams) -> TaskResult:
        import numpy as np

        from ..core import dq, fixpix, images, mef, output
        from .crutil import _destination

        specs = images.expand(params.images)
        if not specs:
            advice.fail(f"'{' '.join(params.images)}' matches no image",
                        "check the name, or run `harwood sections` on the template",
                        FileNotFoundError)
        if len(params.masks) not in (1, len(specs)):
            advice.fail(f"{len(params.masks)} masks for {len(specs)} images",
                        "give one mask for all of them, or one for each, in order")

        rows, written, total, text, warnings = [], [], 0, [], []
        for number, spec in enumerate(specs):
            hdul, hdu = images.open_hdu(spec)
            try:
                data, header = np.array(hdu.data), images.inherited(hdul, hdu)
            finally:
                hdul.close()
            _, _, var, quality = mef.load(spec)
            mask_name = params.masks[0 if len(params.masks) == 1 else number]
            if mask_name.strip().upper() == "DQ":
                mask, where = (quality != 0).astype(np.int32), "DQ"
            else:
                mask, where = _mask(mask_name, header, spec, data.shape)
            listing: list | None = [] if params.verbose and params.pixels else None
            fixed, summary = fixpix.fix(data, mask, params.linterp, params.cinterp, listing)
            if var is not None:
                var, _ = fixpix.fix(var, mask, params.linterp, params.cinterp)
            # A pixel the mask named and the DQ had no reason for is a bad
            # pixel; one the DQ already explains keeps its reason.
            quality = quality | np.where((mask != 0) & (quality == 0), dq.BAD_PIXEL,
                                         0).astype(dq.DTYPE)
            if params.verbose:
                text.append(f"FIXPIX: image {spec} with mask {where}")
                text += fixpix.format_listing(listing or [],
                                              double=data.dtype == np.float64)

            header.add_history(f"fixpix: {summary['pixels']} pixels from {where} "
                               f"({summary['along lines']} along lines, "
                               f"{summary['along columns']} along columns)")
            _mark(header, where)
            name = Path(spec.path).name
            stem = name[:-5] if name.endswith(".fits") else name
            destination = _destination(params.output, stem, ".fits", spec, output)
            files, notes = mef.save(destination, fixed, header, var, quality,
                                    params.uncertainty, params.format)
            written.extend(files)
            if notes and not warnings:
                warnings.extend(notes)
            rows.append({"image": str(spec), "mask": where, **summary})
            total += summary["pixels"]

        steps = self.plan(params)
        steps[1].measured = {"images": len(specs), "pixels replaced": total}
        data = {"text": text, "rows": rows} if text else {"rows": rows}
        return TaskResult(outputs=written, steps=steps, data=data, warnings=warnings,
                          history=[f"fixpix: {total} pixels"])


def _mark(header, mask: str) -> None:
    """IRAF's record: FIXPIX = '<date> Bad pixel file is <mask>', or FIXPIX02
    and on when the frame was fixed before. The date is cnvdate's, local."""
    import time

    now = time.localtime()
    stamp = time.strftime("%b", now) + f" {now.tm_mday:2d} {now.tm_hour:2d}:{now.tm_min:02d}"
    value = f"{stamp} Bad pixel file is {mask}"
    if "FIXPIX" not in header:
        header["FIXPIX"] = value
        return
    for number in range(2, 100):
        if f"FIXPIX{number:02d}" not in header:
            header[f"FIXPIX{number:02d}"] = value
            return


def _mask(name: str, header, spec, shape):
    """The mask for one image, named as every IRAF mask is (core.pixmask):
    an image or .pl, BPM or !KEY for the file a keyword names, a section, a
    text file of 'x y' and 'x1 x2 y1 y2' lines; ^ inverts. A relative name
    not found here is looked for beside the image."""
    from ..core import pixmask

    resolved, _ = pixmask.resolve(name, header)
    if not resolved:
        keyword = name.lstrip("^").lstrip("!")
        advice.fail(f"{spec} has no {keyword} keyword",
                    "name the mask instead, or add the keyword with hedit")
    beside = Path(spec.path).parent
    workdir = "." if (Path(resolved).exists() or resolved.startswith("[")
                      or resolved == "EMPTY" or not (beside / resolved).exists()) else beside
    mask = pixmask.read(name, header, shape, workdir)
    return mask, resolved


# ------------------------------------------------------------- imextensions

class ImExtensionsParams(TaskParams):
    input: list[str] = Field(
        ..., description="Files to look in. Globs, @lists and an image section work.",
        json_schema_extra={"positional": True},
    )
    output: Literal["none", "list", "file"] = Field(
        "file", description="file: one extension per line, for an @file. list: all on a "
                            "line. none: only count them.")
    index: str = Field("1-", description="Which extensions, as a range: 1-, 2,4, 1-9x2. "
                                         "0 is the primary header.")
    extname: str = Field("", description="Patterns for EXTNAME, separated by commas.")
    extver: str = Field("", description="A range of EXTVER.")
    lindex: bool = Field(True, description="Name each with its index.")
    lname: bool = Field(False, description="Name each with its EXTNAME.")
    lver: bool = Field(False, description="Name each with its EXTVER.")
    ikparams: str = Field("", description="Text to add inside the brackets, for a task that "
                                          "reads it.")


def _decode_ranges(text: str, start: int = 0) -> list[tuple[int, int, int]]:
    """ix_decode_ranges and decode_ranges: first, last and step of each range, in
    commas or blanks; first defaults to `start`, last to no end."""
    limit = 2147483647
    ranges, ip, n = [], 0, len(text)

    def skip(at):
        while at < n and text[at] in " \t,":
            at += 1
        return at

    def integer(at):
        end = at
        while end < n and text[end].isdigit():
            end += 1
        return int(text[at:end]), end

    while True:
        ip = skip(ip)
        if ip >= n:
            return ranges or [(start, limit, 1)]
        first, last, step = start, limit, 1
        if text[ip].isdigit():
            first, ip = integer(ip)
        elif text[ip] not in "-x":
            raise ValueError(f"cannot read the range list '{text}'")
        ip = skip(ip)
        if ip < n and text[ip] == "-":
            ip = skip(ip + 1)
            if ip < n and text[ip].isdigit():
                last, ip = integer(ip)
            elif ip < n and text[ip] != "x":
                raise ValueError(f"cannot read the range list '{text}'")
        elif ip >= n or text[ip] != "x":
            last = first
        ip = skip(ip)
        if ip < n and text[ip] == "x":
            ip = skip(ip + 1)
            if ip < n and text[ip].isdigit():
                step, ip = integer(ip)
                if step == 0:
                    raise ValueError(f"cannot read the range list '{text}'")
            elif ip < n and text[ip] != "-":
                raise ValueError(f"cannot read the range list '{text}'")
        ranges.append((first, last, step))


def _in_ranges(ranges, number: int) -> bool:
    for first, last, step in ranges:
        low, high = min(first, last), max(first, last)
        if low <= number <= high and (number - first) % step == 0:
            return True
    return False


def _name_matches(name: str, patterns: str) -> bool:
    """xt_extmatch: any of the patterns, each of the whole name."""
    import re

    from ..core import evexpr

    if not patterns or "".join(patterns.split()) in ("", "*"):
        return True
    for pattern in re.split(r"(?<!\\),", patterns):
        if pattern == "*" or (pattern and evexpr.patmatch(name, f"^{pattern}$") > 0):
            return True
    return False


def _extensions(spec, params, indices, versions) -> list[str]:
    """xt_extn: the names of one file's image extensions."""
    from astropy.io import fits

    from ..core import images

    try:
        hdul = fits.open(str(spec.path), memmap=False)
    except OSError:
        return []
    names = []
    with hdul:
        if spec.extension is not None:
            try:
                picked = [images._resolve_extension(hdul, spec.extension)]
            except (KeyError, IndexError):
                picked = []
            picked = [k for k in picked if k < len(hdul)]
        else:
            picked = [k for k in range(len(hdul)) if _in_ranges(indices, k)]
            # decode_ranges sorts them: 5,2 is 2 and 5.
        for k in picked:
            header = hdul[k].header
            if str(header.get("XTENSION", "IMAGE")).strip() != "IMAGE":
                continue
            extname = str(header.get("EXTNAME", "")).strip()
            extver = header.get("EXTVER")
            if not _name_matches(extname, params.extname):
                continue
            if versions is not None and (extver is None or not _in_ranges(versions, extver)):
                continue
            name = extname if params.lname else ""
            ver = extver if params.lver else None
            text = str(spec.path)
            if params.lindex or (not name and ver is None):
                text += f"[{k}]"
            kernel = [part for part in (name, None if ver is None else str(ver)) if part]
            if name:
                extra = [params.ikparams] if params.ikparams else []
                text += "[" + ",".join(kernel + extra) + "]"
            elif ver is not None:
                extra = f",{params.ikparams}" if params.ikparams else ""
                text += f"[extver={ver}{extra}]"
            elif params.ikparams:
                text += f"[{params.ikparams}]"
            names.append(text + (f"[{spec.section.strip('[]')}]" if spec.section else ""))
    return names


@register
class ImExtensions(Task[ImExtensionsParams]):
    """IRAF's imextensions: the image extensions of FITS files, selected by index,
    name and version and named so that another task can read them."""

    name = "imextensions"
    package = "proto"
    description = "List the image extensions of FITS files."
    Params = ImExtensionsParams

    def plan(self, params: ImExtensionsParams) -> list[Step]:
        return [Step("list", ", ".join(params.input), why=WHY["extensions"])]

    def execute(self, params: ImExtensionsParams) -> TaskResult:
        from ..core import images

        indices = _decode_ranges(params.index)
        words = "".join(params.extver.split())
        versions = _decode_ranges(words, 1) if words else None
        names: list[str] = []
        for spec in images.expand(params.input):
            names += _extensions(spec, params, indices, versions)
        lines = {"file": names, "list": [",".join(names)] if names else [""],
                 "none": []}[params.output]
        return TaskResult(data={"nimages": len(names)}, stdout=lines, steps=self.plan(params))


# ------------------------------------------------------------ mimstatistics

MIS_FIELDS = ("image", "npix", "min", "max", "mean", "midpt", "mode", "stddev", "skew",
              "kurtosis", "mask")


class MimStatisticsParams(TaskParams):
    images: list[str] = Field(
        ..., description="Images to measure. Globs, @lists and sections work.",
        json_schema_extra={"positional": True},
    )
    imasks: str = Field("", description="Masks, none, one or one per image; 0 is good. A ^ "
                                        "before the list inverts them.")
    omasks: str = Field("", description="Masks to write, one per image: which pixels went "
                                        "into the statistics.")
    fields: str = Field("image,npix,mean,stddev,min,max",
                        description="What to print, of " + ", ".join(MIS_FIELDS) + ".")
    lower: float | None = Field(None, description="The lowest good value.")
    upper: float | None = Field(None, description="The highest good value.")
    nclip: int = Field(0, description="How many times to clip.")
    lsigma: float | None = Field(3.0, description="Clip this many sigma below the mean.")
    usigma: float | None = Field(3.0, description="Clip this many sigma above the mean.")
    binwidth: float = Field(0.1, description="The histogram's bins for midpt and mode, in "
                                             "sigma.")
    format: bool = Field(True, description="Column labels and fixed columns; otherwise free.")


def _fields(text: str) -> list[str]:
    """mst_fields: a comma list; strdic takes a unique abbreviation, and what it
    cannot place is left out."""
    found = []
    for word in text.split(","):
        word = word.strip().lower()
        if word in MIS_FIELDS:
            found.append(word)
            continue
        matches = [f for f in MIS_FIELDS if word and f.startswith(word)]
        if len(matches) == 1:
            found.append(matches[0])
    return found[:len(MIS_FIELDS)]


def _clip(values, params):
    """t_mimstatistics's loop: the limits the last pass was made with, and those
    it went on to set, which are what the output mask goes by."""
    import numpy as np

    from .imutil import MAX_REAL, imstat

    low, high = params.lower, params.upper
    nclip = params.nclip
    if nclip > 0 and params.lsigma is None and params.usigma is None:
        nclip = 0
    previous = None
    used = low, high
    for i in range(nclip + 1):
        used = low, high
        stats = imstat(values, ("npix", "mean", "stddev"), low, high)
        if i < nclip:
            mean, sigma = stats["mean"], stats["stddev"]
            defined = np.isfinite(mean) and np.isfinite(sigma)
            low = mean - params.lsigma * sigma if (params.lsigma or 0) > 0 and defined \
                else -MAX_REAL
            high = mean + params.usigma * sigma if (params.usigma or 0) > 0 and defined \
                else MAX_REAL
            if params.lower is not None:
                low = max(low, params.lower)
            if params.upper is not None:
                high = min(high, params.upper)
            if i > 0 and stats["npix"] == previous:
                break
            previous = stats["npix"]
    return used, (low, high)


@register
class MimStatistics(Task[MimStatisticsParams]):
    """IRAF's mimstatistics: imstatistics through a mask."""

    name = "mimstatistics"
    package = "proto"
    description = "Statistics of an image through a pixel mask."
    Params = MimStatisticsParams

    def plan(self, params: MimStatisticsParams) -> list[Step]:
        return [Step("measure", f"{', '.join(params.images)} through "
                                f"{params.imasks or 'no mask'}", why=WHY["through a mask"])]

    def execute(self, params: MimStatisticsParams) -> TaskResult:
        import numpy as np
        from astropy.io import fits

        from ..core import images, iraffmt, plio, provenance
        from .imutil import MAX_REAL, imstat

        specs = images.expand(params.images)
        text = params.imasks.strip()
        invert = text.startswith("^")
        masks = images.split_list(text[1:] if invert else text) if text else []
        outs = images.split_list(params.omasks) if params.omasks.strip() else []
        if len(masks) > 1 and len(masks) != len(specs):
            advice.fail("The input mask and image lists don't match",
                        "give no mask, one for all, or one for each image")
        if outs and len(outs) != len(specs):
            advice.fail("The output mask and image lists don't match",
                        "give one output mask for each image")
        fields = _fields(params.fields)
        if not fields:
            advice.fail(f"no field in '{params.fields}'", f"use {', '.join(MIS_FIELDS)}")
        lines, rows, written = [], [], []
        if params.format:
            lines.append("#" + "".join(f"{f.upper():>20}" if f in ("image", "mask")
                                       else f"{'MIDPT' if f == 'midpt' else f.upper():>10}"
                                       for f in fields))
        for k, spec in enumerate(specs):
            if not spec.path.exists():
                lines.append(f"Error reading image {spec} ...")
                continue
            data, header = images.read(spec)
            name = masks[k] if len(masks) > 1 else (masks[0] if masks else "")
            if name:
                good = _through(name, invert, spec, header, data.shape) != 0
            else:
                good = np.ones(data.shape, bool)
            values = data[good].astype(np.float32)
            used, (low, high) = _clip(values, params)
            stats = imstat(values, MIS_FIELDS[1:10], used[0], used[1],
                           binwidth=params.binwidth)
            row = {"image": str(spec), "mask": name or "EMPTY", **stats}
            rows.append({f: row[f] for f in fields})
            lines.append(_row(row, fields, params.format, iraffmt))
            if outs:
                lo = -MAX_REAL if low is None else np.float32(low)
                hi = MAX_REAL if high is None else np.float32(high)
                pixels = data.astype(np.float32)
                kept = good & (pixels >= lo) & (pixels <= hi)
                out = images.forget_limits(header.copy())
                # 1 is a pixel that did not count, as a mask has it.
                target = outs[k]
                if target.endswith((".fits", ".fit")):
                    provenance.stamp_header(out, [f"mimstatistics: mask of {spec}"])
                    fits.PrimaryHDU((~kept).astype(np.int16), out).writeto(target,
                                                                           overwrite=True)
                    written.append(Path(target))
                else:
                    written.append(plio.write(target, (~kept).astype(np.int64), header=out))
        return TaskResult(data={"rows": rows}, stdout=lines, outputs=written,
                          steps=self.plan(params))


def _through(name: str, invert: bool, spec, header, shape):
    """yt_mappm as this task calls it: ^ before the name is how it gets the pixels
    the mask marks 0, so the mask's sense is turned over twice for a ^ of the user's.
    A section on the mask's name is taken from the mask over the whole image."""
    from ..core import images, pixels, pixmask

    file, brackets = images.split_brackets(name)
    wanted = file if invert else f"^{file}"
    if not brackets:
        return pixmask.read(wanted, header, shape)
    whole = images.ImageSpec(spec.path, spec.extension, None)
    full = images.read_header(whole)
    size = tuple(int(full[f"NAXIS{n}"]) for n in range(int(full["NAXIS"]), 0, -1))
    mask = pixmask.read(wanted, full, size)[pixels.parse_section(brackets, size)]
    if mask.shape != tuple(shape):
        advice.fail(f"the mask {name} is not the shape of {spec}",
                    "give them sections of the same size")
    return mask


def _row(row: dict, fields: list[str], fixed: bool, iraffmt) -> str:
    """mst_print and mst_fprint."""
    parts = []
    for field in fields:
        value = row[field]
        if field in ("image", "mask"):
            parts.append(f"{value:>20}" if fixed else str(value))
        elif field == "npix":
            parts.append(f"{value:>10d}" if fixed else str(value))
        elif fixed:
            parts.append(iraffmt.printf(value, "g", 4, 10).rjust(10))
        else:
            parts.append(iraffmt.g(value))
    return " " + "".join(parts) if fixed else "  ".join(parts)


# ---------------------------------------------------------------- text2mask

class Text2MaskParams(TaskParams):
    text: str = Field(..., description="File of regions: column line, or c1 c2 l1 l2.",
                      json_schema_extra={"positional": True})
    mask: str = Field(..., description="The mask to make; .pl is added if not there.",
                      json_schema_extra={"positional": True})
    ncols: int = Field(..., ge=1, description="Columns of the mask.",
                       json_schema_extra={"positional": True})
    nlines: int = Field(..., ge=1, description="Lines of the mask.",
                        json_schema_extra={"positional": True})
    linterp: int = Field(1, ge=1, description="Value for regions narrower along lines: "
                                              "fixpix interpolates along the line.")
    cinterp: int = Field(2, ge=1, description="Value for regions narrower along columns.")
    square: int = Field(3, ge=1, description="Value for squares of more than a pixel.")
    pixel: int = Field(4, ge=1, description="Value for single pixels.")


def scan_ints(line: str) -> list[int]:
    """fscan and gargi: the integers at the start of a line, as far as they go. A
    real is cut to an integer; after anything else no more is read, so 8, 9 is an 8."""
    import re

    from ..core import iraffmt

    values = []
    for word in line.split():
        found = re.match(r"[-+]?(\d+\.?\d*|\.\d+)([eEdD][-+]?\d+)?", word)
        if found is None:
            break
        values.append(int(iraffmt.ctod(found[0])))
        if found.end() != len(word):
            break
    return values


def _region(values: list[int]) -> tuple[int, int, int, int] | None:
    """Two numbers are a pixel, four a rectangle, and the rest is not a region."""
    if len(values) >= 4:
        return values[0], values[1], values[2], values[3]
    if len(values) == 2:
        return values[0], values[0], values[1], values[1]
    return None


@register
class Text2Mask(Task[Text2MaskParams]):
    """IRAF's text2mask: a mask from a list of pixels and rectangles, coded by the
    shape of each so that fixpix interpolates across the narrow way."""

    name = "text2mask"
    package = "proto"
    description = "A pixel mask from a text file of regions."
    Params = Text2MaskParams

    def plan(self, params: Text2MaskParams) -> list[Step]:
        return [Step("write", f"{params.text} -> {params.mask}", why=WHY["regions"])]

    def execute(self, params: Text2MaskParams) -> TaskResult:
        import numpy as np

        from ..core import plio

        file = Path(params.text)
        if not file.is_file():
            advice.fail(f"Cannot open file ({params.text})", "check the name of the text file")
        nc, nl = params.ncols, params.nlines
        # IRAF keeps the codes in shorts, and 70000 is 4464.
        codes = np.array([params.linterp, params.cinterp, params.square, params.pixel],
                         dtype=np.int64).astype(np.int16).tolist()
        pixels = np.zeros((nl, nc), dtype=np.int64)
        mask = plio.Mask([nc, nl])
        regions = 0
        for line in file.read_text(errors="replace", encoding="utf-8").splitlines():
            region = _region(scan_ints(line))
            if region is None:
                continue
            c1, c2 = max(1, region[0]), min(nc, region[1])
            l1, l2 = max(1, region[2]), min(nl, region[3])
            width, height = c2 - c1 + 1, l2 - l1 + 1
            if width < 1 or height < 1:
                continue
            if width < height:
                value = codes[0]
            elif width > height:
                value = codes[1]
            else:
                value = codes[3] if width == 1 else codes[2]
            pixels[l1 - 1:l2, c1 - 1:c2] = value
            for y in range(l1 - 1, l2):
                mask.update(y, pixels[y])
            regions += 1
        written = plio.write(params.mask, mask, f"Pixel mask from file {params.text}")
        return TaskResult(data={"regions": regions, "bad": int((pixels != 0).sum())},
                          outputs=[written], steps=self.plan(params))


# ------------------------------------------------------------------ mskexpr

class MskExprParams(TaskParams):
    expr: str = Field(..., description="The expression, or @file. i is the reference image, m "
                                       "the reference mask, I J K the pixel.",
                      json_schema_extra={"positional": True})
    masks: str = Field(..., description="The masks to make; .pl is added if not there.",
                       json_schema_extra={"positional": True})
    refimages: str = Field("", description="A reference image for each mask: its size, its "
                                           "header, the operand i.")
    refmasks: str = Field("", description="A reference mask for each: the operand m.")
    dims: str = Field("512,512", description="The mask's axes with nothing to refer to.")
    depth: int = Field(0, description="The mask's depth in bits, up to and by default 27.")
    exprdb: str = Field("none", description="A file of macros for the expression.")
    verbose: bool = Field(True, description="Say what is being made.")


REGIONS = {"point": 2, "circle": 3, "ellipse": 5, "box": 4, "rectangle": 5, "vector": 5,
           "pie": 4, "cannulus": 4, "eannulus": 6, "rannulus": 6}


def _region_function(name: str, args: list, npix: int, y: int, axlen: list[int]):
    """me_fcn. None when it is not one of these, or has the wrong arguments."""
    import numpy as np

    from ..core import evexpr, evvexpr, mskregions

    def number(o):
        if o.type == evvexpr.CHAR:
            value = evexpr._ctod(o.value.strip()) if o.value.strip() else 0.0
            return np.float32(0.0 if value is None else value)
        return np.float32(o.value) if not o.length else np.float32(0.0)

    def done(stat):
        return evvexpr.Op(evvexpr.BOOL, np.asarray(stat, dtype=np.int32))

    nargs = len(args)
    ix0 = np.arange(1, npix + 1, dtype=np.int32)
    iy0 = np.full(npix, y, np.int32)
    width, height = axlen[0], axlen[1] if len(axlen) > 1 else 1
    if name in REGIONS:
        if nargs == REGIONS[name] + 2:
            ix, iy, values = args[0].value, args[1].value, args[2:]
        elif nargs == REGIONS[name]:
            ix, iy, values = ix0, iy0, args
        else:
            return None
        numbers = [number(a) for a in values]
        if name == "pie":
            return done(mskregions.pie(ix, iy, *numbers, width, height))
        return done(getattr(mskregions, name)(ix, iy, *numbers))
    if name == "polygon" or name == "pannulus":
        least = 6 if name == "polygon" else 7
        if nargs < least:
            return None
        vectors = args[0].length > 0 and args[1].length > 0
        ix, iy = (args[0].value, args[1].value) if vectors else (ix0, iy0)
        numbers = [number(a) for a in (args[2:] if vectors else args)]
        if name == "polygon":
            nver = len(numbers) // 2
            return done(mskregions.polygon(ix, iy, numbers[0:2 * nver:2],
                                           numbers[1:2 * nver:2]))
        nver = (len(numbers) - 1) // 2
        xin, yin = numbers[1:2 * nver + 1:2], numbers[2:2 * nver + 1:2]
        xout, yout = mskregions.expand(xin, yin, numbers[0])
        return done(mskregions.apolygon(ix, iy, xin, yin, xout, yout))
    if name in ("cols", "lines"):
        if nargs == 2:
            along, text = args[0].value, args[1]
        elif nargs == 1:
            along, text = (ix0 if name == "cols" else iy0), args[0]
        else:
            return None
        try:
            ranges = _decode_ranges(str(text.value), 1)
        except ValueError:
            ranges = _decode_ranges("-", 1)
        if name == "cols":
            stat = np.array([_in_ranges(ranges, k) for k in range(1, npix + 1)])
        else:
            stat = np.array([_in_ranges(ranges, int(k)) for k in along])
        return done(stat)
    return None


@register
class MskExpr(Task[MskExprParams]):
    """IRAF's mskexpr: a mask from an expression over a reference image and a reference
    mask, with regions: circle, box, polygon and the rest."""

    name = "mskexpr"
    package = "proto"
    description = "A pixel mask from an expression and geometric regions."
    Params = MskExprParams

    def plan(self, params: MskExprParams) -> list[Step]:
        return [Step("evaluate", params.expr, why=WHY["through a mask"]),
                Step("write", params.masks, why=WHY["regions"])]

    def execute(self, params: MskExprParams) -> TaskResult:
        import numpy as np

        from ..core import evvexpr, images, imexpr, plio

        text = params.expr
        if text.startswith("@"):
            file = Path(text[1:])
            if not file.is_file():
                advice.fail(f"no expression file '{file}'", "check the name after the @")
            text = file.read_text(encoding="utf-8").replace("\n", " ")
        if params.exprdb not in ("none", ""):
            text = imexpr.spaced(imexpr.expand(text, imexpr.read_macros(params.exprdb)))
        masks = images.split_list(params.masks)
        refs = images.split_list(params.refimages) if params.refimages.strip() else []
        rmasks = images.split_list(params.refmasks) if params.refmasks.strip() else []
        if refs and len(refs) != len(masks):
            advice.fail("The reference image and output mask lists are not the same size",
                        "give one reference image for each mask, or none")
        if rmasks and len(rmasks) != len(masks):
            advice.fail("The reference image and output mask lists are not the same size",
                        "give one reference mask for each output mask, or none")
        try:
            expression = evvexpr.Expression(text)
        except evvexpr.ExprError as e:
            advice.fail(f"in the expression: {e}", f"the expression was: {params.expr}")
        lines = [f"Expr: {text}"]
        written = []
        for k, name in enumerate(masks):
            target = name if name.endswith(".pl") else name + ".pl"
            if Path(target).exists():
                lines.append(f"Mask {target} already exists")
                continue
            ref = refs[k] if refs else None
            rmask = rmasks[k] if rmasks else None
            lines += _saying(target, ref, rmask)
            data, header = _evaluate(expression, ref, rmask, params, np, images, evvexpr,
                                     imexpr)
            mask = plio.Mask(list(reversed(data.shape)))
            for line, pixels in enumerate(data.reshape(-1, data.shape[-1])):
                mask.update(line, pixels)
            written.append(plio.write(target, mask, header=header))
        return TaskResult(data={"masks": len(written), "expression": text},
                          stdout=lines if params.verbose else [], outputs=written,
                          steps=self.plan(params))


def _saying(target: str, ref, rmask) -> list[str]:
    if ref and rmask:
        return [f"Creating mask {target}",
                f"    Using reference image {ref} and mask {rmask}"]
    if ref or rmask:
        return [f"Creating mask {target} using reference image {ref or rmask}"]
    return [f"Creating mask {target}"]


def _logical(name: str, shape, images, np):
    """yt_mappm with a logical match: the mask cut or padded to the image."""
    from ..core import pixmask

    file, invert = pixmask.resolve(name)
    data, _ = images.read(file)
    mask = (data <= 0).astype(np.int64) if invert else data.astype(np.int64)
    if mask.ndim != len(shape):
        advice.fail(f"Reference image and mask for {name} don't match",
                    "give them the same number of dimensions")
    out = np.zeros(shape, np.int64)
    box = tuple(slice(0, min(a, b)) for a, b in zip(shape, mask.shape, strict=True))
    out[box] = mask[box]
    return out


def _evaluate(expression, ref, rmask, params, np, images, evvexpr, imexpr):
    """me_mkmask: the expression once for each line, and an integer out of it."""
    image = mask = header = None
    kind = evvexpr.INT
    if ref:
        spec = images.parse(ref)
        if not spec.path.exists():
            advice.fail(f"Cannot open reference image {ref}", "check the name")
        data, header = images.read(spec)
        pixtype = imexpr.pixel_type(spec)
        kind = {imexpr.USHORT: evvexpr.INT, evvexpr.SHORT: evvexpr.INT}.get(pixtype, pixtype)
        image = data.astype(evvexpr.DTYPE[kind])
        shape = image.shape
    if rmask:
        if ref:
            mask = _logical(rmask, shape, images, np).astype(np.int32)
        else:
            spec = images.parse(rmask)
            if not spec.path.exists():
                advice.fail(f"Cannot open reference mask {rmask}", "check the name")
            data, header = images.read(spec)
            mask = data.astype(np.int32)
            shape = mask.shape
    if not ref and not rmask:
        axes = [int(part) for part in params.dims.replace(",", " ").split()]
        shape = tuple(reversed(axes))
    mask_header = images.read_header(images.parse(rmask)) if rmask else None
    npix = shape[-1]
    nlines = int(np.prod(shape[:-1])) if len(shape) > 1 else 1
    axlen = list(reversed(shape))
    image_lines = None if image is None else image.reshape(-1, npix)
    mask_lines = None if mask is None else mask.reshape(-1, npix)
    position = [1] * 7
    current = {"line": 0}

    def getop(name):
        line = current["line"]
        if name == "i" and image_lines is not None:
            return evvexpr.Op(kind, image_lines[line])
        if name == "m" and mask_lines is not None:
            return evvexpr.Op(evvexpr.INT, mask_lines[line])
        if len(name) > 2 and name[0] in "im" and name[1] == ".":
            source = header if name[0] == "i" and ref else mask_header
            return None if source is None else imexpr._keyword(source, name[2:], name)
        if len(name) == 1 and name.isupper():
            axis = ord(name) - ord("I") + 1
            if axis == 1:
                return evvexpr.Op(evvexpr.INT, np.arange(1, npix + 1, dtype=np.int32))
            at = position[axis - 1] if 1 <= axis <= 7 else 1
            return evvexpr.Op(evvexpr.INT, np.full(npix, at, np.int32))
        return None

    def ufcn(name, args):
        return _region_function(name, args, npix, position[1], axlen)

    out = np.zeros((nlines, npix), np.int64)
    limit = 2 ** (min(params.depth, 27) if params.depth > 0 else 27) - 1
    for line in range(nlines):
        current["line"] = line
        for axis in range(1, len(axlen)):
            span = int(np.prod(axlen[1:axis]))
            position[axis] = line // span % axlen[axis] + 1
        try:
            value = expression.evaluate(getop, False, ufcn)
        except evvexpr.ExprError as e:
            advice.fail(f"in the expression: {e}", f"the expression was: {params.expr}")
        if value.type in (evvexpr.REAL, evvexpr.DOUBLE):
            advice.fail("Type real expressions are not supported",
                        "a mask holds integers: use a comparison, or int() or nint()")
        if value.type == evvexpr.CHAR:
            advice.fail("Unknown expression value type", "a mask holds integers, not strings")
        # A scalar gives nothing: IRAF's raster op for it is PIX_CLR, and clears the line.
        if value.length:
            n = min(value.length, npix)
            out[line, :n] = np.asarray(value.value[:n], np.int64) & limit
    kept = None if header is None else images.forget_limits(header.copy())
    return out.reshape(shape), kept
