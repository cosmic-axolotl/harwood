"""Pixel masks, named the way IRAF names them.

Every task that takes a mask -- display's bpmask and zmask, fixpix, the
combining tasks -- reads it through here, as every IRAF task reads it through
xtools/fixpix/ytpmmap.x, so a name that works for one works for all:

    mask.fits         an image; pixels above zero are flagged, negatives are 0
    mask.pl           IRAF's own compressed mask format
    BPM               the file named by the image's BPM keyword
    !KEYWORD          the file named by any keyword
    [100:200,*]       a section of the image itself, flagged with 1
    regions.txt       lines of "c1 c2 l1 l2" (or "c l"), flagged 2 for a
                      column-shaped box and 3 for a line-shaped one -- the
                      code fixpix reads to choose its direction
    EMPTY             nothing flagged
    ^mask.fits        any of the above, inverted

A mask that does not cover the image the same way -- a section of the
image, a trimmed frame against a mask of the whole detector -- is matched in
physical coordinates through LTV and LTM, as IRAF does with pmmatch=physical:
the mask follows the pixels, not the array index.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import numpy as np

from . import advice

INVERT = "^"


def resolve(name: str, header: Any = None) -> tuple[str, bool]:
    """The mask a name refers to, and whether it is inverted. A keyword the
    image does not have resolves to "" -- no mask, as in IRAF."""
    text = "".join(name.split())
    invert = text.count(INVERT) % 2 == 1
    text = text.replace(INVERT, "")
    if text.startswith("!"):
        text = str(header.get(text[1:], "")).strip() if header is not None else ""
    elif text == "BPM":
        text = str(header.get("BPM", "")).strip() if header is not None else ""
    return text, invert


def read(name: str, header: Any, shape: tuple[int, ...], workdir: Path | str = ".",
         ) -> np.ndarray | None:
    """The mask for an image of `shape` (numpy order) with `header`, as an
    int array of that shape; None when the name resolves to no mask."""
    text, invert = resolve(name, header)
    if not text:
        return None
    if text == "EMPTY":
        mask = np.zeros(shape, dtype=np.int32)
        return np.ones_like(mask) if invert else mask
    if text.startswith("["):
        return _section(text, shape, invert)
    path = Path(text)
    if not path.is_absolute():
        path = Path(workdir) / path
    if not path.exists() and path.suffix == "":
        for suffix in (".pl", ".fits", ".fit"):
            if path.with_suffix(suffix).exists():
                path = path.with_suffix(suffix)
                break
    if not path.exists():
        advice.fail(f"pixel mask '{text}' not found",
                    "give the mask's file name, or an empty value for no mask",
                    FileNotFoundError)
    if path.suffix.lower() == ".pl":
        from . import plio

        mask, mask_header = plio.read(path)
    elif _is_text(path):
        return _text(path, shape, invert)
    else:
        from astropy.io import fits

        with fits.open(path) as hdul:
            hdu = next((h for h in hdul if h.data is not None), None)
            if hdu is None:
                advice.fail(f"pixel mask '{path}' holds no image",
                            "name the extension with the mask, e.g. mask.fits[1]")
            mask = np.asarray(hdu.data)
            mask_header = hdu.header
        if not np.issubdtype(mask.dtype, np.integer):
            mask = np.nan_to_num(mask)
        mask = mask.astype(np.int64)
    mask = (mask <= 0).astype(np.int32) if invert else np.where(mask < 0, 0, mask)
    return match(mask.astype(np.int32), mask_header, header, shape)


def _is_text(path: Path) -> bool:
    with open(path, "rb") as handle:
        head = handle.read(80)
    return not head.startswith(b"SIMPLE") and not head.startswith(b"XTENSION") \
        and b"\0" not in head


def _text(path: Path, shape: tuple[int, ...], invert: bool) -> np.ndarray:
    """yt_pmtext: rectangles from a text file."""
    nl, nc = (1, shape[0]) if len(shape) == 1 else shape[-2:]
    mask = np.ones((nl, nc), dtype=np.int32) if invert else np.zeros((nl, nc), np.int32)
    for line in Path(path).read_text(encoding="latin-1").splitlines():
        words = line.split()
        try:
            numbers = [int(float(w)) for w in words[:4]]
        except ValueError:
            continue
        if len(numbers) == 4:
            c1, c2, l1, l2 = numbers
        elif len(numbers) == 2:
            c1, l1 = numbers
            c2, l2 = c1, l1
        else:
            continue
        c1, c2, l1, l2 = max(1, c1), min(nc, c2), max(1, l1), min(nl, l2)
        if c2 - c1 + 1 < 1 or l2 - l1 + 1 < 1:
            continue
        value = 0 if invert else (2 if c2 - c1 <= l2 - l1 else 3)
        mask[l1 - 1:l2, c1 - 1:c2] = value
    return mask.reshape(shape) if len(shape) == 1 else mask


def _section(text: str, shape: tuple[int, ...], invert: bool) -> np.ndarray:
    """yt_pmsection / zsc_pmsection: a mask set inside a section. Steps go
    in increasing order whichever way the section was written."""
    if len(shape) > 2:
        advice.fail("a section mask is only allowed for 1-D and 2-D images",
                    "use a mask image instead")
    sizes = list(reversed(shape))              # x first, as IRAF writes it
    body = text.strip()[1:-1] if text.strip().endswith("]") else ""
    parts = body.split(",")
    if not text.strip().endswith("]") or len(parts) != len(sizes):
        advice.fail(f"error in image section specification '{text}'",
                    "write one range per axis, e.g. [*:4,*:4] or [100:200,*]")
    ranges = []
    for part, size in zip(parts, sizes, strict=True):
        a, b, c = _axis(part.strip(), size, text)
        if c < 0:
            a, b, c = b, a, -c
        ranges.append(slice(a - 1, b, c))
    mask = np.zeros(shape, dtype=np.int32)
    mask[tuple(reversed(ranges))] = 1
    return (1 - mask) if invert else mask


_AXIS = re.compile(r"^(?:(?P<a>[+-]?\d+)(?::(?P<b>[+-]?\d+))?|(?P<flip>-\*?)|\*)?"
                   r"(?::(?P<c>[+-]?\d+))?$")


def _axis(part: str, size: int, text: str) -> tuple[int, int, int]:
    """One axis of a section, as the SPP parser reads it: a, a:b, *, -*,
    each optionally followed by :step."""
    found = _AXIS.match(part)
    if found is None or found["c"] == "0":
        advice.fail(f"error in image section specification '{text}'",
                    "ranges are first:last, first:last:step, *, -* or *:step")
    a, b = 1, size
    if found["a"] is not None:
        a = int(found["a"])
        b = int(found["b"]) if found["b"] is not None else a
    elif found["flip"]:
        a, b = b, a
    c = int(found["c"]) if found["c"] is not None else 1
    if a > b and c > 0:
        c = -c
    return a, b, c


def _lterm(header: Any, naxis: int) -> tuple[np.ndarray, np.ndarray]:
    """(LTM diagonal, LTV): logical = LTM * physical + LTV, per axis."""
    ltm = np.array([float(header.get(f"LTM{i}_{i}", 1.0)) if header is not None else 1.0
                    for i in range(1, naxis + 1)])
    ltv = np.array([float(header.get(f"LTV{i}", 0.0)) if header is not None else 0.0
                    for i in range(1, naxis + 1)])
    return ltm, ltv


def match(mask: np.ndarray, mask_header: Any, header: Any,
          shape: tuple[int, ...]) -> np.ndarray:
    """yt_match with pmmatch=physical: the mask as seen by the image's
    pixels. Where the scales agree it is a shift (padding with 0 off the
    mask); where they differ each image pixel takes the largest mask value
    falling inside it."""
    if mask.shape == tuple(shape) and not mask.any():
        return mask
    naxis = len(shape)
    if mask.ndim != naxis:
        advice.fail(f"the mask is {mask.ndim}-dimensional and the image {naxis}",
                    "use a mask made for this image")
    mltm, mltv = _lterm(mask_header, naxis)
    iltm, iltv = _lterm(header, naxis)
    # image logical -> physical -> mask logical, per axis: m = scale * l + shift
    scale = mltm / iltm
    shift = mltv - mltm * iltv / iltm
    scale = np.round(scale * 1e6) / 1e6
    shift = np.round(shift * 1e6) / 1e6
    if np.all(scale == 1) and np.all(shift == 0) and mask.shape == tuple(shape):
        return mask
    out = np.zeros(shape, dtype=np.int32)
    index = []
    for axis in range(naxis):              # x first
        n_img = shape[naxis - 1 - axis]
        n_mask = mask.shape[naxis - 1 - axis]
        lo = scale[axis] * (np.arange(1, n_img + 1) - 0.5) + shift[axis]
        hi = scale[axis] * (np.arange(1, n_img + 1) + 0.5) + shift[axis]
        first = np.maximum(1, np.rint(np.minimum(lo, hi) + 1e-5)).astype(int)
        last = np.minimum(n_mask, np.rint(np.maximum(lo, hi) - 1e-5)).astype(int)
        index.append((first, last))
    if naxis == 1:
        (first, last), = index
        for i in np.flatnonzero(last >= first):
            out[i] = mask[first[i] - 1:last[i]].max()
        return out
    (fx, lx), (fy, ly) = index
    rows = np.flatnonzero(ly >= fy)
    cols = np.flatnonzero(lx >= fx)
    if np.all(lx[cols] == fx[cols]) and np.all(ly[rows] == fy[rows]):
        out[np.ix_(rows, cols)] = mask[np.ix_(fy[rows] - 1, fx[cols] - 1)]
        return out
    for j in rows:
        block = mask[fy[j] - 1:ly[j]].max(axis=0)
        for i in cols:
            out[j, i] = block[fx[i] - 1:lx[i]].max()
    return out
