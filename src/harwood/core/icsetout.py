"""imcombine's output geometry and pixel masks: ic_setout and ic_mopen, ic_mget.

Offsets, x first and in pixels, say where each image starts in the output, which
is the box that holds them all, cut by outlimits. Masks are the file each image's
keyword names, and masktype and maskvalue turn their values into good and bad.

From iraf-community 2.18: pkg/images/immatch/src/imcombine/src/icsetout.x, icmask.x.
See LICENSES/IRAF-COPYRIGHT.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from . import advice, mwcs, plio

OFFTYPES = ("none", "wcs", "world", "physical", "grid")
MASKTYPES = ("none", "goodvalue", "badvalue", "goodbits", "badbits", "novalue")


def nint(value: float) -> int:
    """SPP's nint: a half goes away from zero."""
    return int(value + 0.5) if value >= 0 else -int(-value + 0.5)


@dataclass
class Setout:
    offsets: np.ndarray  # [image, axis], x first
    lengths: list[int]  # of the output, x first
    aligned: bool
    reloff: bool
    offtype: str


def _lens(shape) -> list[int]:
    return list(shape[::-1])


def _dictionary(word: str, names) -> str | None:
    """strdic: a whole name or the only one it begins."""
    found = [name for name in names if name.startswith(word)]
    return word if word in names else (found[0] if len(found) == 1 and word else None)


def _offset_file(path: Path, count: int, outdim: int) -> tuple[list[list[int]], bool]:
    if not path.exists():
        advice.fail(f"the offsets '{path}' are no type and no file",
                    f"none, wcs, world, physical, grid a b, or a file of {outdim} numbers "
                    "for each image")
    reloff = True
    found: list[list[int]] = []
    for line in path.read_text().splitlines():
        words = line.split()
        if not words:
            continue
        if words[0] == "#":
            kind = words[1].lower() if len(words) > 1 else ""
            reloff = {"absolute": False, "relative": True}.get(kind, reloff)
            continue
        try:
            found.append([nint(float(word)) for word in words[:outdim]])
        except ValueError:
            advice.fail("IMCOMBINE: error in offset list", f"line '{line}' of {path}")
        if len(found[-1]) < outdim:
            advice.fail("IMCOMBINE: error in offset list", f"line '{line}' of {path}")
        if len(found) == count:
            break
    if len(found) < count:
        advice.fail("IMCOMBINE: offset list too short",
                    f"{path} has {len(found)} lines for {count} images")
    return found, reloff


def setout(shapes, headers, offsets: str, outlimits: str, project: bool = False) -> Setout:
    """ic_setout. `shapes` of each image, numpy order, without the projected axis;
    `headers` of each, the one header repeated for a project."""
    nimages = len(shapes)
    outdim = len(shapes[0])
    if not project and any(len(shape) != outdim for shape in shapes):
        advice.fail("image dimensions are not the same", "combine images of one dimension")
    words = offsets.split()
    word = words[0] if words else "none"
    kind = word if word in OFFTYPES else "file"
    table = np.zeros((nimages, outdim), dtype=np.int64)
    reloff = True
    if kind in ("wcs", "world"):
        table = _world_offsets(headers, outdim, project)
    elif kind == "physical":
        table = _physical_offsets(headers, outdim, project)
    elif kind == "grid":
        amin = 1
        numbers = words[1:]
        for j in range(outdim):
            try:
                a, b = int(numbers[2 * j]), int(numbers[2 * j + 1])
            except (IndexError, ValueError):
                a, b = 1, 0
            for i in range(nimages):
                table[i, j] = ((i // amin) % a) * b
            amin *= a
    elif kind == "file":
        found, reloff = _offset_file(Path(word), nimages, outdim)
        table = np.array(found, dtype=np.int64)

    aligned = True
    lengths = [0] * outdim
    for j in range(outdim):
        amin = int(table[0, j])
        bmax = _lens(shapes[0])[j] + amin
        for i in range(1, nimages):
            a = int(table[i, j])
            b = _lens(shapes[i])[j] + a
            if a != amin or b != bmax or not reloff:
                aligned = False
            amin, bmax = min(a, amin), max(b, bmax)
        lengths[j] = bmax
        if reloff or amin < 0:
            table[:, j] -= amin
            lengths[j] -= amin

    limits = outlimits.split()
    for j in range(outdim):
        if len(limits) < 2 * (j + 1):
            break
        a, b = _limit(limits[2 * j]), _limit(limits[2 * j + 1])
        if a is not None:
            table[:, j] -= a - 1
            if (table[:, j] != 0).any():
                aligned = False
            lengths[j] -= a - 1
            if b is not None:
                lengths[j] = min(lengths[j], b - a + 1)
    if any(length < 1 for length in lengths):
        advice.fail(f"outlimits '{outlimits}' leave no output",
                    "the first limit past the images, or the second before the first")
    return Setout(table, lengths, aligned, reloff, kind)


def _limit(word: str) -> int | None:
    if word.upper().startswith("INDEF"):
        return None
    try:
        return int(word)
    except ValueError:
        advice.fail(f"the outlimits have '{word}' for a limit", "x1 x2 y1 y2, whole pixels")


def _world_offsets(headers, outdim: int, project: bool) -> np.ndarray:
    nimages = len(headers)
    table = np.zeros((nimages, outdim), dtype=np.int64)
    n = outdim + 1 if project else outdim
    wref = np.array([float(headers[0].get(f"CRVAL{i + 1}", 0.0)) for i in range(n)])
    lref = mwcs.logical_of(headers[0], wref)
    if project:
        lref[outdim] = 1
    for i in range(1, nimages):
        if project:
            wref[outdim] = i + 1
        coord = mwcs.logical_of(headers[i], wref)
        for j in range(outdim):
            table[i, j] = nint(lref[j] - coord[j])
    return table


def _physical_offsets(headers, outdim: int, project: bool) -> np.ndarray:
    nimages = len(headers)
    table = np.zeros((nimages, outdim), dtype=np.int64)
    n = outdim + 1 if project else outdim
    wref = np.array([float(headers[0].get(f"CRVAL{i + 1}", 0.0)) for i in range(n)])
    lref = mwcs.logical_of(headers[0], wref)
    if project:
        lref[outdim] = 1
    pref = mwcs.transform(headers[0], "physical", lref[:, None])[:, 0]
    for i in range(1, nimages):
        if project:
            pref[outdim] = i + 1
        coord = mwcs.logical_of_physical(headers[i], pref)
        for j in range(outdim):
            table[i, j] = nint(lref[j] - coord[j])
    return table


def flips(headers, outdim: int) -> list[str]:
    """offsets physical turns an image whose LTM has the other sign than the first's:
    the section to read each one with, '' for none."""
    out = [""]
    first = [float(headers[0].get(f"LTM{j + 1}_{j + 1}", 1.0)) for j in range(outdim)]
    for header in headers[1:]:
        axes = ["*" if first[j] * float(header.get(f"LTM{j + 1}_{j + 1}", 1.0)) >= 0 else "-*"
                for j in range(outdim)]
        out.append(f"[{','.join(axes)}]" if "-*" in axes else "")
    return out


def place(frame: np.ndarray, offset, lengths) -> tuple[tuple, tuple]:
    """Slices of the output and of `frame` where they meet; numpy order. None when
    they do not."""
    target, source = [], []
    for axis, (length, start) in enumerate(zip(_lens(frame.shape), offset, strict=True)):
        low = max(0, int(start))
        high = min(lengths[axis], length + int(start))
        if high <= low:
            return None
        target.append(slice(low, high))
        source.append(slice(low - int(start), high - int(start)))
    return tuple(target[::-1]), tuple(source[::-1])


def update_wcs(header, geometry: Setout, project: bool, outdim: int) -> None:
    """The end of ic_setout: the output's logical system follows the first image's
    offset, and with world offsets its physical system is reset to the logical. The
    projected axis goes, with what the header says of it."""
    moved = not geometry.aligned or not geometry.reloff
    if moved:
        for j in range(outdim):
            shift = int(geometry.offsets[0, j])
            if f"CRPIX{j + 1}" in header:
                header[f"CRPIX{j + 1}"] = float(header[f"CRPIX{j + 1}"]) + shift
            value = float(header.get(f"LTV{j + 1}", 0.0)) + shift
            if value:
                header[f"LTV{j + 1}"] = value
            else:
                header.remove(f"LTV{j + 1}", ignore_missing=True)
    if project:
        axis = outdim + 1
        for key in list(header.keys()):
            name = str(key)
            if name in (f"CRPIX{axis}", f"CRVAL{axis}", f"CDELT{axis}", f"CTYPE{axis}",
                        f"CUNIT{axis}", f"LTV{axis}", f"CROTA{axis}", f"WAT{axis}_001"):
                del header[name]
            elif name.startswith(("CD", "LTM")) and "_" in name:
                indices = name.split("_")
                row = indices[0][2:] if name.startswith("CD") else indices[0][3:]
                if row.isdigit() and axis in (int(row), int(indices[1])):
                    del header[name]
    if (moved or project) and geometry.offtype in ("wcs", "world"):
        for j in range(outdim):
            header.remove(f"LTV{j + 1}", ignore_missing=True)
            for i in range(outdim):
                header.remove(f"LTM{j + 1}_{i + 1}", ignore_missing=True)


@dataclass
class MaskType:
    kind: str  # none, goodvalue, badvalue, goodbits, badbits, novalue, boolean, ltval, gtval
    value: int
    key: str = "BPM"


def masktype(text: str, value: str) -> MaskType:
    """ic_mopen's reading of masktype and maskvalue."""
    words = text.split()
    kind = "none"
    key = "BPM"
    if words:
        if words[0].startswith("!"):
            key = words[0][1:]
            kind = "goodvalue" if len(words) == 1 else _dictionary(words[1], MASKTYPES)
        else:
            kind = _dictionary(words[0], MASKTYPES)
        if kind is None:
            advice.fail(f"invalid or ambiguous masktype ({text})",
                        "one of " + ", ".join(MASKTYPES) + ", or !KEYWORD, then one of them")
    number = str(value).strip()
    if number[:1] in ("<", ">"):
        kind = "ltval" if number[0] == "<" else "gtval"
        number = number[1:]
    try:
        mvalue = int(float(number))
    except ValueError:
        advice.fail("bad mask value", f"--maskvalue {value}: a whole number, or with < or >")
    if mvalue < 0:
        advice.fail("bad mask value", f"--maskvalue {value} is below zero")
    if mvalue == 0 and kind == "novalue":
        advice.fail("maskvalue cannot be 0 for masktype of 'novalue'", "give the value")
    if kind == "badbits" and mvalue == 0:
        kind = "none"
    elif mvalue == 0 and kind in ("goodvalue", "goodbits"):
        kind = "boolean"
    return MaskType(kind, mvalue, key)


def read_mask(mtype: MaskType, header, shape, image_path) -> np.ndarray | None:
    """The mask an image's keyword names, its raw values; None for no mask. A name with
    no directory is looked for where the image is too. It goes pixel by pixel from the
    first, whatever the image's LTV says, unless IRAF's pmmatch is set; past its end
    it marks nothing."""
    if mtype.key not in header:
        return None
    name = str(header[mtype.key]).strip()
    if not name:
        return None
    candidates = [Path(name)]
    if not Path(name).parent.parts and image_path is not None:
        candidates.append(Path(image_path).parent / name)
    for path in candidates:
        for found in (path, path.with_name(path.name + ".pl")):
            if found.exists():
                return _sized(_raw_values(found), shape)
    advice.fail(f"bad pixel mask not found ({name})",
                f"the {mtype.key} keyword of an image names it; make it, or --masktype none")


def _raw_values(path: Path) -> np.ndarray:
    if path.suffix == ".pl":
        return plio.read(path)[0].astype(np.int64)
    from astropy.io import fits

    return np.asarray(fits.getdata(path)).astype(np.int64)


def _sized(mask: np.ndarray, shape) -> np.ndarray:
    out = np.zeros(shape, dtype=np.int64)
    if mask.ndim != len(shape):
        advice.fail(f"a mask of {mask.ndim} dimensions for an image of {len(shape)}",
                    "give a mask of the image's shape")
    both = tuple(slice(0, min(a, b)) for a, b in zip(mask.shape, shape, strict=True))
    out[both] = mask[both]
    return out


def bad_pixels(mtype: MaskType, raw: np.ndarray | None, shape):
    """ic_mget: (bad, no value) from the raw values: bad pixels are excluded, and with
    novalue those not marked as having no value may still fill a pixel with no good
    one. An image with no mask has every pixel good, whatever the type."""
    if raw is None or mtype.kind == "none":
        return np.zeros(shape, dtype=bool), np.zeros(shape, dtype=bool)
    value = mtype.value
    none = np.zeros(shape, dtype=bool)
    kind = mtype.kind
    if kind == "boolean":
        return raw != 0, none
    if kind == "badbits":
        return (raw & value) != 0, none
    if kind == "badvalue":
        return raw == value, none
    if kind == "goodbits":
        return (raw & value) == 0, none
    if kind == "goodvalue":
        return raw != value, none
    if kind == "ltval":
        return raw >= value, none
    if kind == "gtval":
        return raw <= value, none
    return raw != 0, raw == value  # novalue
