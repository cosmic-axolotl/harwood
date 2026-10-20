"""Numbers written the way IRAF's printf writes them.

IRAF's %g is not C's. A whole real keeps its point (5.), an exponent has no
padding or plus sign (1.500000E-7) and keeps the mantissa's zeros, and a
single-precision value gets 7 significant digits, a double 16. When a task
prints what IRAF printed -- hedit's record of an edit, a header value -- it
has to come out of sys/fmtio/dtoc3.x, so that is what is ported here.

    g(5.0)          '5.'
    g(1.5e-7)       '1.500000E-7'
    g(1.334567)     '1.334567'
    g(x, double=True)   16 digits
"""

from __future__ import annotations

import math

NDIGITS_RP = 7          # mach.h: significant digits of a real
NDIGITS_DP = 16         # ... and of a double
MAX_DIGITS = 25
MAX_EXPONENTD = 308
EPSILOND = 2.220446049250313e-16
SZ_OBUF = 1024
#: IRAF's INDEF; a single-precision INDEFR (1.6e38 rounded) counts too.
INDEF = 1.5999e38


def g(value: float, double: bool = False, decpl: int | None = None) -> str:
    """printf("%g") of a real (or, with double, a double) as IRAF does it."""
    return printf(value, "g", decpl, double=double)


def printf(value: float, fmt: str = "g", decpl: int | None = None, width: int | None = None,
           double: bool = False) -> str:
    """fmtio's handling of %e, %f and %g: the default precision, then
    shorter until the number fits the width."""
    precision = NDIGITS_DP if double else NDIGITS_RP
    if decpl is None or decpl == 0:
        decpl = precision
    w = SZ_OBUF if width is None else width
    while True:
        text = dtoc3(value, decpl, fmt, w + 1)
        decpl -= 1
        if len(text) <= w or decpl <= 0:
            return text


def _equal(x: float, y: float) -> bool:
    """dt_equald: equal to within 32 epsilon after normalising."""
    if x == y:
        return True
    if x == 0 or y == 0:
        return False
    nx, ex = _norm(x)
    ny, ey = _norm(y)
    return ex == ey and 1.0 + abs(nx - ny) <= 1.0 + EPSILOND * 32.0


def _norm(x: float) -> tuple[float, int]:
    tol = EPSILOND * 10.0
    a, e = abs(x), 0
    if a > 0:
        while a < 1.0 - tol:
            a *= 10.0
            e -= 1
            if a == 0:
                return 0.0, 0
        while a >= 10.0 + tol:
            a /= 10.0
            e += 1
    return (-a if x < 0 else a), e


def _scale(v: float) -> tuple[float, int]:
    """dtcscl: v scaled to 0.1 <= v < 1, and the exponent that took."""
    if _equal(v, 0.0):
        return v, 0
    e = -1
    while v < 0.1:
        v *= 10.0
        e -= 1
        if e <= -MAX_EXPONENTD or _equal(v, 0.0):
            e = 0
            break
    while v >= 1.0:
        v /= 10.0
        e += 1
        if e >= MAX_EXPONENTD:
            break
    return v, e


def dtoc3(val: float, decpl: int, a_fmt: str = "g", width: int = SZ_OBUF + 1) -> str:
    """sys/fmtio/dtoc3.x: f (decimal places), e and g (significant digits)."""
    if val is None or (isinstance(val, float) and math.isnan(val)) or abs(val) >= INDEF:
        return "INDEF"
    v, w = abs(val), abs(width)
    fmt = a_fmt.lower()
    squeeze = fmt == "g"
    neg = val < 0.0
    d = abs(decpl) if squeeze else max(0, decpl)
    v, e = _scale(v)
    max_size = 2 if neg else 1
    no_digits = min(MAX_DIGITS, d)
    length = abs(e)
    e_size = 1
    i = 10
    while i <= 10000:
        if length < i:
            break
        e_size += 1
        i *= 10
    e_size += max_size + 1
    if e < 0:
        e_size += 1
    f_size = 0
    if squeeze:
        e_size += d
        f_size = max(d, e + 1) if e > 0 else d - e
        f_size += max_size
    elif fmt == "f":
        f_size = max(e, 1) + max_size + d
    if squeeze:
        if f_size <= e_size:
            exp_format = False
            if e > 0:
                no_digits = f_size - max_size
            max_size = f_size
        else:
            exp_format = True
            max_size = e_size
        d = w
    elif fmt == "f":
        exp_format = f_size > w
        if exp_format:
            no_digits = max(1, w - e_size)
            max_size = no_digits + e_size
        else:
            no_digits = e + d + 1
            max_size = f_size
    else:
        exp_format = True
        max_size = e_size + d
        d = w
    if no_digits >= 0:
        v = v + 0.5 * 10.0 ** (-no_digits)
    no_digits = max(1, min(MAX_DIGITS, no_digits))
    if v >= 1.0:
        v /= 10.0
        e += 1
        if not exp_format:
            max_size += 1
            no_digits = min(MAX_DIGITS, no_digits + 1)
    if max_size > w:
        return "*" * w
    digits = []
    for _ in range(no_digits):
        v *= 10.0
        j = int(v + EPSILOND)
        v -= j
        if int(v * 10.0 + EPSILOND) >= 10:
            j += 1
            v = max(v - 1, 0.0)
        digits.append(str(j) if j < 10 else "9")
    out = ["-"] if neg else []
    if exp_format:
        out += [digits[0], "."] + digits[1:no_digits] + ["E"]
        out += ["-", str(-e)] if e < 0 else [str(e)]
    elif e >= no_digits:
        out += digits[:no_digits] + ["0"] * (e - no_digits + 1)
        if decpl > 0:
            out.append(".")
            if not squeeze:
                out += ["0"] * d
    else:
        if e < 0:
            out.append("1" if d == 0 and e == -1 and digits[0] >= "5" else "0")
            out.append(".")
            k = 1
            while k < -e and d > 0:
                out.append("0")
                d -= 1
                k += 1
            i = 0
        else:
            e += 2
            out += digits[:e - 1]
            i = e - 1
            if decpl > 0:
                out.append(".")
        j = 1
        while i < no_digits and j <= d:
            out.append(digits[i])
            i += 1
            j += 1
        if squeeze:
            while len(out) > 1 and out[-1] == "0":
                out.pop()
        else:
            k = 1
            while k < d + e - no_digits and k <= d:
                out.append("0")
                k += 1
    return "".join(out)
