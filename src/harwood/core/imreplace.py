"""imreplace without the task: images/imutil/src/imrep.gx, as IRAF 2.18 runs it.

Pixels between lower and upper -- either INDEF, no limit on that side -- become
value, and with a radius of 1 or more every pixel within that distance of one
too. What IRAF does is decided by the image's pixel type, and the ported code
keeps every consequence:

* lower, upper and value are CL reals: single precision, whatever the image.
* An integer image (short, int, and ushort read as long) takes value truncated
  toward zero, and wrapped into the type (99999 in a short is -31073), and
  limits made integers: upper is int(upper), lower int(lower) when it is
  whole and int(lower + 1) when not. int() truncates, so for a negative limit
  that is not whole the window moves by one: lower -100.5 starts at -99,
  upper -20.5 stops at -20.
* A real image compares in single precision; a double one in double, against
  the limits as reals.
* With a radius, the pixels in range are marked INDEF in a ring buffer of
  1 + 2 int(radius) lines (no more than the image has) and a line is written
  once the one int(radius) below it has been read. The buffer cannot tell a
  mark from a pixel that equals INDEF, so a short image's -32767 is replaced
  wherever it is. With no lower limit the window starts at -MAX_PIXEL, so a
  short's -32768 is never in it. A line is written only after the line
  int(radius) further on is read: a 1-D image (IMIO counts its lines in the
  first axis) and an image with no more lines than int(radius) are left as
  they were. In a cube the line counter goes back to 0 at the last line of
  each band, so a band's last lines are written only when the next band's
  reading writes them, and the cube's last ones not at all.

Every line imreplace reads comes through IRAF's IEEE input (osb/ieee.gx, NaN
mapping on, as the FITS kernel sets it): in a real or double image a NaN or an
infinity is read as 0, and so is anything with a zero exponent -- -0 and the
numbers below the type's normal range -- which then compares as 0 and is written
back as 0. Lines a radius never writes keep what the file had.

`replace` takes the pixels in their working type (`working_type` says which)
and returns them in it; the task converts back to what the file stores.
"""

from __future__ import annotations

import numpy as np

from .jit import jit
from .scombine import fp_equalr

f32 = np.float32

#: IRAF's INDEF and MAX_ of each type imrep.gx is generic over.
INDEF = {"short": -32767, "int": -2147483647, "long": -9223372036854775807,
         "real": 1.6e38, "double": 1.6e308}
MAX_PIXEL = {"short": 32767, "int": 2147483647, "long": 9223372036854775807,
             "real": float(np.finfo(f32).max), "double": float(np.finfo(np.float64).max)}
DTYPES = {"short": np.int16, "int": np.int32, "long": np.int64, "real": f32,
          "double": np.float64}


def working_type(dtype) -> str:
    """The IRAF type a FITS image's pixels are read as: short, int, real and
    double as they are, ushort and 64-bit as long (imrepl). An 8-bit image is
    read as short; IRAF cannot write one back in place and stops."""
    kind = np.dtype(dtype)
    found = {("i", 1): "short", ("u", 1): "short", ("i", 2): "short",
             ("i", 4): "int", ("u", 2): "long", ("u", 4): "long", ("i", 8): "long",
             ("f", 4): "real", ("f", 8): "double"}.get((kind.kind, kind.itemsize))
    if found is None:
        raise ValueError(f"Unsupported image pixel datatype: {kind}")
    return found


def _integer(x: float, kind: str):
    """A double assigned to an integer PIXEL: truncated (C's cast), then the low
    bits of a 32- or 64-bit integer, as gcc's conversion leaves them."""
    value = int(np.trunc(x))
    bits = 64 if kind == "long" else 32
    value &= (1 << bits) - 1
    if value >= 1 << (bits - 1):
        value -= 1 << bits
    return DTYPES[kind](np.array(value, dtype=np.int64).astype(DTYPES[kind]))


def newval(value: float, kind: str):
    """newval = double(value), value a CL real, into the image's type."""
    v = float(f32(value))
    if kind in ("real", "double"):
        return DTYPES[kind](v)
    return _integer(v, kind)


def window(lower: float | None, upper: float | None, kind: str, radius: bool):
    """floor and ceil, as imrep (radius False) or imrrep sets them: the limits in
    the image's type; a side left INDEF is None for imrep and +-MAX_PIXEL for
    imrrep."""
    def integral_floor(low):
        low = f32(low)
        whole = f32(int(low))
        return int(low) if fp_equalr(low, whole) else int(f32(low + f32(1.0)))

    if kind in ("real", "double"):
        floor = None if lower is None else DTYPES[kind](float(f32(lower)))
        ceil = None if upper is None else DTYPES[kind](float(f32(upper)))
    else:
        floor = None if lower is None else integral_floor(lower)
        ceil = None if upper is None else int(f32(upper))
    if radius:
        if floor is None:
            floor = -MAX_PIXEL[kind]
        if ceil is None:
            ceil = MAX_PIXEL[kind]
    return floor, ceil


def replace(data: np.ndarray, value: float, lower: float | None, upper: float | None,
            radius: float = 0.0, kind: str = "real") -> np.ndarray:
    """IRAF's imreplace on pixels of `kind`, any number of dimensions; a copy."""
    data = np.array(data, dtype=DTYPES[kind])
    original, data = data, _ieee_in(data, kind)
    new = newval(value, kind)
    radius = max(0.0, float(f32(radius)))
    if lower is None and upper is None:
        data[...] = new
        return data
    if radius < 1.0:
        floor, ceil = window(lower, upper, kind, radius=False)
        if floor is None:
            hit = data <= ceil
        elif ceil is None:
            hit = data >= floor
        else:
            hit = (data >= floor) & (data <= ceil)
        data[hit] = new
        return data
    floor, ceil = window(lower, upper, kind, radius=True)
    shape = data.shape
    nc = shape[-1]
    if data.ndim == 1:
        nl, counters = 1, np.zeros(1, dtype=np.int64)
    else:
        nl = shape[-2]
        counters = _counters(data.ndim, nl, data.size // nc)
    work = np.array(floor, dtype=DTYPES[kind]), np.array(ceil, dtype=DTYPES[kind])
    out = _rrep(data.reshape(-1, nc), original.reshape(-1, nc).copy(), counters, nl,
                work[0][()], work[1][()], f32(radius), DTYPES[kind](INDEF[kind]), new)
    return out.reshape(shape)


def _ieee_in(data: np.ndarray, kind: str) -> np.ndarray:
    """ieevupk with NaN mapping: NaN and infinities to the native NaN, which the
    FITS kernel leaves 0, and a zero exponent (either zero, a subnormal) to 0."""
    if kind not in ("real", "double"):
        return data
    with np.errstate(invalid="ignore"):
        odd = ~np.isfinite(data) | (np.abs(data) < np.finfo(data.dtype).tiny)
    return np.where(odd, data.dtype.type(0), data)


def _counters(ndim: int, nl: int, total: int) -> np.ndarray:
    """v[2] - 1 after imgnl (and impnl) has read each line: imgnln advances v[2]
    of a 2-D image past its last line, and imloop sends it back to 1 at the end
    of each band of a cube (the next band in v[3])."""
    line = np.arange(total) % nl + 1
    if ndim == 2:
        return line.astype(np.int64)
    return np.where(line < nl, line, 0).astype(np.int64)


@jit
def _rrep(lines, out, counters, nl, floor, ceil, radius, indef, newval):
    """imrrep's loop: read a line, mark INDEF around what is in range, write the
    line nradius back; lines it never writes stay as `out` has them."""
    total, nc = lines.shape
    radius2 = radius * radius
    nradius = int(radius)
    nbufs = min(1 + 2 * nradius, nl)
    buf = np.zeros((nbufs, nc), dtype=lines.dtype)
    written = 0
    for r in range(total):
        j = counters[r]
        slot2 = j % nbufs
        for i in range(1, nc + 1):
            val1 = lines[r, i - 1]
            val2 = buf[slot2, i - 1]
            if val1 >= floor and val1 <= ceil:
                for k in range(max(1, j - nradius), min(nl, j + nradius) + 1):
                    slot = k % nbufs
                    y2 = np.float32((k - j) ** 2)
                    for col in range(max(1, i - nradius), min(nc, i + nradius) + 1):
                        if np.float32((col - i) ** 2) + y2 > radius2:
                            continue
                        buf[slot, col - 1] = indef
            else:
                if val2 != indef:
                    buf[slot2, i - 1] = val1
        if j > nradius:
            while written < total:
                k = counters[written]
                slot = k % nbufs
                for i in range(nc):
                    val1 = buf[slot, i]
                    if val1 == indef:
                        out[written, i] = newval
                    else:
                        out[written, i] = val1
                    buf[slot, i] = 0
                written += 1
                if j != nl:
                    break
    return out
