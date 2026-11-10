"""imsum without the task: images/imutil/src/t_imsum.x and imsum.gx, as IRAF 2.18
runs them.

The sum, average or median of images, pixel by pixel, with the nlow lowest and
nhigh highest of each pixel rejected. What IRAF does follows from three types:

* the type each image is read as (short, unsigned short, int, real, double; an
  8-bit image as short, a scaled one as real);
* the calculation type: the highest of the inputs by imsum_set's precedence
  (short and unsigned short together make int) unless calctype names one. An
  unsigned short calculation has no routine of its own and runs in real;
* the output pixel type: the calculation type unless pixtype names one.

Each pixel read is converted to the calculation type as IMIO converts it -- a
real into an integer type truncated, wrapped into the type as gcc's conversion
leaves it -- after IRAF's IEEE input has read a NaN, an infinity, -0 or a
subnormal as 0. Integer sums wrap, an integer average truncates, and the
result reaches the output type the same way.

With nothing rejected the images are summed fifteen at a time (IMS_MAX), the
partial sum kept in the output image: in the output's pixel type, so a short
output of a real average of more than fifteen frames is truncated, and can
wrap, between passes. With rejection, imrej chooses between sorting and pairs
by its timing constants, and a median of 3 or 5 is amed3 or amed5. The median
option rejects n/2 low and n - n/2 - 1 high: of an even number, the upper of
the two middle values.
"""

from __future__ import annotations

import numpy as np

from .jit import jit

f32 = np.float32

IMS_MAX = 15

#: dtstring's names, which verbose prints.
NAMES = {"short": "short", "ushort": "unsigned short", "int": "integer", "long": "long",
         "real": "real", "double": "double"}
#: The numpy type a calculation or an output runs in. IRAF writes long as BITPIX 32.
CALC = {"short": np.int16, "int": np.int32, "long": np.int64, "real": f32,
        "double": np.float64}
STORE = {"short": np.int16, "ushort": np.uint16, "int": np.int32, "long": np.int32,
         "real": f32, "double": np.float64}
_LETTERS = {"u": "ushort", "s": "short", "i": "int", "l": "long", "r": "real",
            "d": "double"}

TMINSW, TMXMNSW, TMED3, TMED5 = f32(1.00), f32(1.46), f32(0.18), f32(0.55)


class ImsumError(ValueError):
    """What IRAF's imsum stops with, in its words."""


def pixel_type(dtype, scaled: bool = False) -> str:
    """IM_PIXTYPE of a FITS image as astropy hands its data over."""
    if scaled:
        return "real"
    kind = np.dtype(dtype)
    found = {("i", 1): "short", ("u", 1): "short", ("i", 2): "short",
             ("u", 2): "ushort", ("i", 4): "int", ("u", 4): "long", ("i", 8): "long",
             ("f", 4): "real", ("f", 8): "double"}.get((kind.kind, kind.itemsize))
    if found is None:
        raise ImsumError(f"Unsupported image pixel datatype: {kind}")
    return found


def max_type(kinds: list[str]) -> str:
    """imsum_set's precedence, from the first image's type on."""
    top = kinds[0]
    for kind in kinds[1:]:
        if kind == "short":
            if top == "ushort":
                top = "int"
        elif kind == "ushort":
            if top == "short":
                top = "int"
        elif kind == "int":
            if top in ("ushort", "short"):
                top = kind
        elif kind == "long":
            if top in ("ushort", "short", "int"):
                top = kind
        elif kind == "real":
            if top != "double":
                top = kind
        elif kind == "double":
            top = kind
    return top


def types(kinds: list[str], calctype: str, pixtype: str) -> tuple[str, str]:
    """The calculation and output types: calctype by its first letter (s i l r d),
    pixtype by its (u s i l r d), each empty for the default."""
    calctype, pixtype = calctype.strip(), pixtype.strip()
    if not calctype:
        calc = max_type(kinds)
    elif calctype[0] in "silrd":
        calc = _LETTERS[calctype[0]]
    else:
        raise ImsumError("Unrecognized datatype")
    if not pixtype:
        pix = calc
    elif pixtype[0] in _LETTERS:
        pix = _LETTERS[pixtype[0]]
    else:
        raise ImsumError("Unrecognized datatype")
    return calc, pix


def rejections(option: str, nimages: int, low_reject: float,
               high_reject: float) -> tuple[int, int]:
    """nlow and nhigh: for the median n/2 and n - n/2 - 1, else a fraction of the
    images below 1 and a number from 1 on, truncated."""
    if option == "median":
        nlow = nimages // 2
        return nlow, nimages - nlow - 1
    low, high = f32(low_reject), f32(high_reject)
    nlow = int(f32(low * f32(nimages))) if low < 1 else int(low)
    nhigh = int(f32(high * f32(nimages))) if high < 1 else int(high)
    if nlow + nhigh >= nimages:
        raise ImsumError("Number of pixels rejected >= number of images")
    return nlow, nhigh


# ------------------------------------------------------------- conversions

def ieee_in(data: np.ndarray) -> np.ndarray:
    """ieevupk with NaN mapping: NaN and infinities read as 0, and so is a zero
    exponent (-0, a subnormal)."""
    if data.dtype.kind != "f":
        return data
    with np.errstate(invalid="ignore"):
        odd = ~np.isfinite(data) | (np.abs(data) < np.finfo(data.dtype).tiny)
    return np.where(odd, data.dtype.type(0), data)


def convert(data: np.ndarray, to) -> np.ndarray:
    """acht: b = a in Fortran, as gcc compiles it. A float to an integer type is
    truncated through a 32-bit integer (64 for a 64-bit one) -- NaN and what
    does not fit become its minimum -- and keeps the low bits; an integer to a
    narrower one keeps the low bits too."""
    to = np.dtype(to)
    data = np.asarray(data)
    if to.kind == "f":
        return data.astype(to)
    if data.dtype.kind == "f":
        wide = 64 if to.itemsize == 8 else 32
        low, high = -(2.0 ** (wide - 1)), 2.0 ** (wide - 1)
        with np.errstate(invalid="ignore"):
            t = np.trunc(data.astype(np.float64))
            fits = np.isfinite(t) & (t >= low) & (t < high)
        whole = np.where(fits, t, 0).astype(np.int64)
        whole = np.where(fits, whole, np.int64(-(2 ** (wide - 1))))
        return whole.astype(to)
    return data.astype(np.int64).astype(to)


def read_as(stored: np.ndarray, kind: str, calc: str) -> np.ndarray:
    """imgnl$t of an image of pixel type `kind` in the calculation's type."""
    data = ieee_in(np.asarray(stored, dtype=STORE[kind] if kind != "long" else np.int64))
    return convert(data, CALC["real" if calc == "ushort" else calc])


def _idiv(values: np.ndarray, by: int) -> np.ndarray:
    """An integer array divided by an integer, truncated toward zero, in its type."""
    wide = values.astype(np.int64)
    q = np.abs(wide) // abs(by) * (np.sign(wide) * np.sign(by))
    return q.astype(values.dtype)


def _add(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """aadd in the calculation type: integers wrap."""
    if a.dtype.kind == "f":
        return (a + b).astype(a.dtype)
    return (a.astype(np.int64) + b.astype(np.int64)).astype(a.dtype)


def _divide(values: np.ndarray, count: int) -> np.ndarray:
    """adivk by `const`, a PIXEL holding the count."""
    if values.dtype.kind == "f":
        return (values / values.dtype.type(count)).astype(values.dtype)
    return _idiv(values, count)


# ---------------------------------------------------------------- the sum

def combine(frames, calc: str, pix: str, option: str, nlow: int, nhigh: int) -> np.ndarray:
    """imsum$t: `frames` the images' pixels already in the calculation type, all of
    one shape; the result in the output's storage type."""
    work = CALC["real" if calc == "ushort" else calc]
    nimages = len(frames)
    naccept = nimages - nlow - nhigh
    shape = frames[0].shape
    store = STORE[pix]
    if nlow == 0 and nhigh == 0:
        stored = None
        done = 0
        for start in range(0, nimages, IMS_MAX):
            block = frames[start:start + IMS_MAX]
            done += len(block)
            if stored is None:
                buf = np.zeros(shape, dtype=work)
            else:
                kind = "long" if pix == "long" else pix
                buf = read_as(stored, kind, calc)
            for frame in block:
                buf = _add(buf, frame)
            if done == nimages and option == "average":
                buf = _divide(buf, naccept)
            stored = convert(buf, store)
        return stored
    stack = np.stack([np.asarray(f, dtype=work).ravel() for f in frames])
    integer = stack.dtype.kind != "f"
    if integer:
        total = _reject(stack.astype(np.int64), nlow, nhigh).astype(work)
    else:
        total = _reject(stack, nlow, nhigh)
    if naccept > 1 and option == "average":
        total = _divide(total, naccept)
    return convert(total.reshape(shape), store)


def _plan(nvecs: int, nlow: int, nhigh: int):
    """imrej's choice: (sorting?, npairs, nlow1, nhigh1, median of 0, 3 or 5)."""
    naccept = nvecs - nlow - nhigh
    minrej = min(nlow, nhigh)
    npairs = minrej
    nlow1, nhigh1 = nlow - npairs, nhigh - npairs
    if naccept == 1 and npairs > 0:
        if npairs == 1:
            tmedian, npairs, median = TMED3, npairs - 1, 3
        else:
            tmedian, npairs, median = TMED5, npairs - 2, 5
    else:
        tmedian, median = f32(0), 0
    time1 = f32(TMINSW * f32(minrej + naccept))
    time2 = f32(f32(tmedian + f32(TMXMNSW * f32(npairs))) + f32(TMINSW * f32(nlow1 + nhigh1)))
    return bool(time1 < time2), npairs, nlow1, nhigh1, median


def _reject(stack: np.ndarray, nlow: int, nhigh: int) -> np.ndarray:
    """imrej$t over every pixel: the sum of what is kept, in the stack's type
    (integers summed in 64 bits; wrapping is the caller's)."""
    nvecs = stack.shape[0]
    sorting, npairs, nlow1, nhigh1, median = _plan(nvecs, nlow, nhigh)
    return _imrej(stack.copy(), nlow, nhigh, sorting, npairs, nlow1, nhigh1, median)


@jit
def _imrej(a, nlow, nhigh, sorting, npairs, nlow1, nhigh1, median):
    nvecs, npts = a.shape
    naccept = nvecs - nlow - nhigh
    b = np.zeros(npts, dtype=a.dtype)
    for p in range(npts):
        i = nvecs
        if naccept == nvecs:
            s = a[0, p]
            for j in range(1, naccept):
                s = s + a[j, p]
            b[p] = s
            continue
        if sorting:
            if nlow < nhigh:
                for _ in range(nlow + naccept):
                    _minsw(a, i, p)
                    i -= 1
                s = a[nhigh, p]
                for j in range(nhigh + 1, nhigh + naccept):
                    s = s + a[j, p]
            else:
                for _ in range(nhigh + naccept):
                    _maxsw(a, i, p)
                    i -= 1
                s = a[nlow, p]
                for j in range(nlow + 1, nlow + naccept):
                    s = s + a[j, p]
            b[p] = s
            continue
        for _ in range(npairs):
            _mxmnsw(a, i, p)
            i -= 2
        for _ in range(nlow1):
            _minsw(a, i, p)
            i -= 1
        for _ in range(nhigh1):
            _maxsw(a, i, p)
            i -= 1
        if median == 0:
            s = a[0, p]
            for j in range(1, naccept):
                s = s + a[j, p]
            b[p] = s
        elif median == 3:
            b[p] = _amed3(a[0, p], a[1, p], a[2, p])
        else:
            b[p] = _amed5(a[0, p], a[1, p], a[2, p], a[3, p], a[4, p])
    return b


@jit
def _minsw(a, nvecs, p):
    """The minimum of the first nvecs swapped into the last of them."""
    kmin = 0
    for j in range(1, nvecs):
        if a[j, p] < a[kmin, p]:
            kmin = j
    k = nvecs - 1
    if k != kmin:
        temp = a[k, p]
        a[k, p] = a[kmin, p]
        a[kmin, p] = temp


@jit
def _maxsw(a, nvecs, p):
    kmax = 0
    for j in range(1, nvecs):
        if a[j, p] > a[kmax, p]:
            kmax = j
    k = nvecs - 1
    if k != kmax:
        temp = a[k, p]
        a[k, p] = a[kmax, p]
        a[kmax, p] = temp


@jit
def _mxmnsw(a, nvecs, p):
    """The maximum to the last and the minimum to the next to last, as mxmnsw
    does it: with `else if`, and the minimum followed when the swap moved it."""
    kmax = 0
    kmin = 0
    for j in range(1, nvecs):
        if a[j, p] > a[kmax, p]:
            kmax = j
        elif a[j, p] < a[kmin, p]:
            kmin = j
    k = nvecs - 1
    temp = a[k, p]
    a[k, p] = a[kmax, p]
    a[kmax, p] = temp
    j = nvecs - 2
    if kmin == k:
        temp = a[j, p]
        a[j, p] = a[kmax, p]
        a[kmax, p] = temp
    else:
        temp = a[j, p]
        a[j, p] = a[kmin, p]
        a[kmin, p] = temp


@jit
def _amed3(a, b, c):
    if a < b:
        if b < c:
            return b
        if a < c:
            return c
        return a
    if b > c:
        return b
    if a > c:
        return c
    return a


@jit
def _amed5(a, b, c, d, e):
    if b < a:
        a, b = b, a
    if c < a:
        a, c = c, a
    if d < a:
        a, d = d, a
    if e < a:
        a, e = e, a
    if b > e:
        b, e = e, b
    if c > e:
        c, e = e, c
    if d > e:
        d, e = e, d
    if b < c:
        if c < d:
            return c
        if b < d:
            return d
        return b
    if c > d:
        return c
    if b > d:
        return d
    return b


# ---------------------------------------------------------------- headers

def hparam_value(card) -> float:
    """imgetd: the card's value as IRAF's ctod reads its text, which is not always
    the nearest double (62.99999999999999 is read a little above)."""
    from .iraffmt import ctod

    text = card.image[10:].split("/")[0].strip().strip("'").strip()
    value = ctod(text)
    if value is None:
        raise ImsumError(f"{card.keyword} is not a number: {text!r}")
    return value


def hparam_format(value: float) -> str:
    """imputd: %0.*g from NDIGITS_DP - 1 (15) down to NDIGITS_RP while the text
    is 20 characters or more. The last digits are dtoc3's, not the nearest
    decimal's: 62.99999999999999 as read is written 63.0000000000001."""
    from .iraffmt import NDIGITS_DP, NDIGITS_RP, printf

    for digits in range(NDIGITS_DP - 1, NDIGITS_RP - 1, -1):
        text = printf(float(value), "g", digits, double=True)
        if len(text) < 20:
            break
    return text
