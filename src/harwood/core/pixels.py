"""The pixel convention boundary.

IRAF, FITS and DS9 count pixels from 1 and write them (x, y). NumPy counts
from 0 and indexes [y, x]. Section 2.2 of the design contract fixes the
harwood side: **everything a user sees is 1-based and (x, y)** -- the command
line, the viewer, reports, coordinate files. The conversion happens here, at
the boundary with the libraries, and nowhere else.

This is not a style preference. A user coming from IRAF who reads a centroid
at x=612.03 and gets 611.03 back has a silent, one-pixel systematic in every
measurement, and nothing in the output says so.

Three coordinate systems meet in this module:

    iraf    1-based, (x, y), pixel 1 spans [0.5, 1.5) and is centred at 1.0
    numpy   0-based, [row, col] = [y, x], pixel 0 is centred at 0.0
    scene   the viewer's image item, where pixel i spans [i, i+1)

    x_numpy = x_iraf - 1        x_scene = x_iraf - 0.5
"""

from __future__ import annotations

import re
from typing import NamedTuple

import numpy as np

__all__ = [
    "Position",
    "to_numpy",
    "from_numpy",
    "to_scene",
    "from_scene",
    "parse_section",
    "section_slices",
    "format_position",
]


class Position(NamedTuple):
    """A position in the harwood convention: 1-based, (x, y)."""

    x: float
    y: float


def to_numpy(x, y):
    """harwood (1-based, x, y) -> numpy (0-based, y, x), as (row, col)."""
    return np.asarray(y, dtype=float) - 1.0, np.asarray(x, dtype=float) - 1.0


def from_numpy(row, col) -> Position:
    """numpy (0-based, row, col) -> harwood (1-based, x, y)."""
    return Position(np.asarray(col, dtype=float) + 1.0, np.asarray(row, dtype=float) + 1.0)


def to_scene(x, y):
    """harwood (1-based centre) -> viewer scene coordinates (pixel edge).

    Section 8.4: pyqtgraph places pixel i over [i, i+1), so the centre of
    IRAF pixel 1 -- at 1.0 -- sits at scene 0.5. Getting this wrong shifts
    every measurement by half a pixel, which is why the synthetic-star test
    in tests/test_pixels.py is permanent.
    """
    return np.asarray(x, dtype=float) - 0.5, np.asarray(y, dtype=float) - 0.5


def from_scene(sx, sy) -> Position:
    """Viewer scene coordinates -> harwood (1-based, x, y)."""
    return Position(np.asarray(sx, dtype=float) + 0.5, np.asarray(sy, dtype=float) + 0.5)


_RANGE = re.compile(r"^\s*(\*|-?\d+)\s*(?::\s*(-?\d+)\s*)?(?::\s*(-?\d+)\s*)?$")


def parse_section(section: str, shape: tuple[int, ...]) -> tuple[slice, ...]:
    """Parse an IRAF image section into numpy slices.

    Accepts what IRAF accepts, in IRAF's own order and origin::

        [100:400,*]     columns 100 to 400 inclusive, every row
        [*,50:60]       every column, rows 50 to 60
        [100:400:2,*]   every other column
        [-*,*]          reversed along x

    The section is written (x, y) and 1-based and *inclusive on both ends*;
    what comes back is (row, col) and 0-based and exclusive on the stop, so
    that ``data[parse_section(...)]`` just works.
    """
    text = section.strip()
    if text.startswith("[") and text.endswith("]"):
        text = text[1:-1]
    parts = [p.strip() for p in text.split(",")]
    if len(parts) > len(shape):
        raise ValueError(
            f"section '{section}' has {len(parts)} axes but the image has {len(shape)}"
        )

    slices: list[slice] = []
    # A section is written x first; numpy indexes y first.
    for axis, part in enumerate(parts):
        length = shape[len(shape) - 1 - axis]
        slices.append(_one_axis(part, length, section))
    while len(slices) < len(shape):
        slices.append(slice(None))
    return tuple(reversed(slices))


def _one_axis(part: str, length: int, whole: str) -> slice:
    reverse = part.startswith("-")
    if reverse:
        part = part[1:]
    if part == "*":
        return slice(None, None, -1) if reverse else slice(None)

    match = _RANGE.match(part)
    if match is None:
        raise ValueError(f"cannot read '{part}' in section '{whole}'")
    first, last, step = match.groups()
    if first == "*":
        raise ValueError(f"cannot read '{part}' in section '{whole}'")

    start = int(first)
    stop = int(last) if last is not None else start
    stride = int(step) if step is not None else 1
    if start < 1 or stop > length:
        raise ValueError(
            f"section '{whole}' asks for {start}:{stop} but the axis has {length} pixels"
        )
    return slice(start - 1, stop, stride)


def section_slices(section: str | None, shape: tuple[int, ...]) -> tuple[slice, ...]:
    """Like :func:`parse_section`, but ``None`` means the whole image."""
    if section is None:
        return tuple(slice(None) for _ in shape)
    return parse_section(section, shape)


def format_position(x, y, precision: int = 2) -> str:
    """Render a position the way every harwood surface should render it."""
    return f"x={float(x):.{precision}f} y={float(y):.{precision}f}"
