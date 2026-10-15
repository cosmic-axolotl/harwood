"""IRAF's fixpix: bad pixels replaced by interpolating across them.

A port of `xtools/fixpix` (xtfixpix.x and xtfp.gx, iraf-community 2.18),
which both IRAF's `fixpix` task and ccdproc's fixpix step use. What the
manual compresses into "across the narrowest dimension" is, in the code:

1. every bad pixel is labelled with the length of its run of bad pixels
   along the line;
2. column by column, each run of bad pixels along the column is compared
   with that: a pixel is interpolated along the column only when its line
   run is *longer* than its column run. Equal lengths go to the line, so a
   single hot pixel is fixed along its line, a bad column along its lines,
   and a bad row along its columns;
3. pixels marked for the line take the line's linear interpolation between
   the good pixels either side of the whole run; pixels marked for the
   column take the column's, between the good pixels above and below;
4. a run that reaches the edge takes the one good neighbour it has, unchanged
   (IRAF extrapolates with zero slope).

`linterp` and `cinterp` name mask values that force the direction, as in
IRAF. Integer images are rounded back to integers, as IRAF does.

Translated from IRAF, Copyright (c) 1986 Association of Universities for
Research in Astronomy Inc.; the notice it is distributed under is in
LICENSES/IRAF-COPYRIGHT.
"""

from __future__ import annotations

import numpy as np

LDEF, CDEF = 1, 2           # FP_LDEF, FP_CDEF


def _runs(flags: np.ndarray):
    """(start, stop) of each run of True, stop exclusive."""
    padded = np.concatenate(([False], flags, [False]))
    edges = np.flatnonzero(padded[1:] != padded[:-1])
    return list(zip(edges[::2], edges[1::2], strict=True))


def directions(mask: np.ndarray, linterp: int | None = None,
               cinterp: int | None = None) -> np.ndarray:
    """For each bad pixel, LDEF (along the line) or CDEF (along the column).

    0 where the pixel is good. This is xt_fpsinterp and the first half of
    xt_fpinit.
    """
    from . import advice

    mask = np.asarray(mask).astype(np.int64)
    nl, nc = mask.shape
    lvalin = linterp if linterp is not None and linterp >= 1 else None
    cvalin = cinterp if cinterp is not None and cinterp >= 1 else None
    if lvalin is not None and lvalin == cvalin:
        advice.fail("linterp and cinterp cannot be the same mask value",
                    "give them different values, or leave one empty")

    label = np.zeros_like(mask)
    forced_line = np.zeros(mask.shape, bool)
    forced_col = np.zeros(mask.shape, bool)
    for row in range(nl):
        values = mask[row]
        for start, stop in _runs(values != 0):
            segment = values[start:stop]
            label[row, start:stop] = nc + (stop - start)
            if lvalin is not None:
                forced_line[row, start:stop] |= segment == lvalin
            if cvalin is not None:
                forced_col[row, start:stop] |= segment == cvalin

    out = np.zeros_like(mask)
    for col in range(nc):
        bad = mask[:, col] != 0
        for start, stop in _runs(bad):
            column_run = stop - start
            for row in range(start, stop):
                if forced_line[row, col]:
                    out[row, col] = LDEF
                elif forced_col[row, col] or label[row, col] > nc + column_run:
                    # Along the column only when the line run is longer:
                    # equal lengths stay on the line.
                    out[row, col] = CDEF
                else:
                    out[row, col] = LDEF
    return out


def fix(data: np.ndarray, mask: np.ndarray, linterp: int | None = None,
        cinterp: int | None = None) -> tuple[np.ndarray, dict]:
    """Return the fixed image and a summary of what was done.

    The input is not modified. The output keeps the input's dtype; integer
    images are rounded, as IRAF writes them.
    """
    data = np.asarray(data)
    if np.asarray(mask).shape != data.shape:
        from . import advice

        advice.fail(f"the mask is {np.asarray(mask).shape[::-1]} and the image "
                    f"{data.shape[::-1]} (x by y)",
                    "use the mask made for this detector and this trim, or trim "
                    "the mask the same way")
    how = directions(mask, linterp, cinterp)
    nl, nc = data.shape
    source = data.astype(np.float64)
    fixed = source.copy()

    # Column interpolation reads the original columns: the endpoints are good
    # pixels, which nothing changes.
    for col in range(nc):
        bad = how[:, col] != 0
        for start, stop in _runs(bad):
            rows = np.arange(start, stop)
            todo = rows[how[rows, col] == CDEF]
            if todo.size == 0:
                continue
            l1, l2 = start - 1, stop                     # 0-based endpoints
            if l1 < 0 and l2 >= nl:
                continue
            c = source[l1, col] if l1 >= 0 else source[l2, col]
            slope = (source[l2, col] - c) / (l2 - l1) if l2 < nl else 0.0
            fixed[todo, col] = c + slope * (todo - l1)

    for row in range(nl):
        bad = how[row] != 0
        for start, stop in _runs(bad):
            cols = np.arange(start, stop)
            todo = cols[how[row, cols] == LDEF]
            if todo.size == 0:
                continue
            c1, c2 = start - 1, stop
            if c1 < 0 and c2 >= nc:
                continue
            a = source[row, c1] if c1 >= 0 else source[row, c2]
            slope = (source[row, c2] - a) / (c2 - c1) if c2 < nc else 0.0
            fixed[row, todo] = a + slope * (todo - c1)

    if np.issubdtype(data.dtype, np.integer):
        # nint: half away from zero, negatives included.
        result = np.where(fixed >= 0, np.floor(fixed + 0.5),
                          np.ceil(fixed - 0.5)).astype(data.dtype)
    else:
        result = fixed.astype(data.dtype)
    summary = {"pixels": int((how != 0).sum()),
               "along lines": int((how == LDEF).sum()),
               "along columns": int((how == CDEF).sum())}
    return result, summary
