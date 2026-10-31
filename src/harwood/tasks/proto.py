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

        from ..core import images, iraffmt, provenance
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
                target = Path(outs[k] if Path(outs[k]).suffix else outs[k] + ".fits")
                if target.suffix == ".pl":
                    target = target.with_suffix(".fits")
                out = images.forget_limits(header.copy())
                provenance.stamp_header(out, [f"mimstatistics: mask of {spec}"])
                # 1 is a pixel that did not count, as a mask has it.
                fits.PrimaryHDU((~kept).astype(np.int16), out).writeto(target, overwrite=True)
                written.append(target)
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
