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
import re
import time

NDIGITS_RP = 7          # mach.h: significant digits of a real
NDIGITS_DP = 16         # ... and of a double
MAX_DIGITS = 25
MAX_EXPONENTD = 308
EPSILOND = 2.220446049250313e-16
SZ_OBUF = 1024
#: IRAF's INDEF; a single-precision INDEFR (1.6e38 rounded) counts too.
INDEF = 1.5999e38


_DAYS = ("Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun")
_MONTHS = ("Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct",
           "Nov", "Dec")


def cnvtime(when: float | None = None) -> str:
    """IRAF's cnvtime, local time: Wed 15:49:56 21-Oct-2026. The names are
    IRAF's, in English whatever the locale -- strftime's %a and %b follow it,
    and a Portuguese desktop would write qua and out."""
    t = time.localtime(when)
    return (f"{_DAYS[t.tm_wday]} {t.tm_hour:02d}:{t.tm_min:02d}:{t.tm_sec:02d} "
            f"{t.tm_mday:02d}-{_MONTHS[t.tm_mon - 1]}-{t.tm_year}")


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


def dtoc(value: float, decpl: int, fmt: str = "g", width: int = SZ_OBUF + 1) -> str:
    """sys/fmtio/dtoc.x: dtoc3, and %h and %m, sexagesimal; %H and %M divide
    by 15 first, degrees to hours."""
    kind = fmt.lower()
    if kind in "feg" or value is None or (isinstance(value, float) and math.isnan(value)):
        return dtoc3(value, decpl, kind, width)
    val = value / 15.0 if fmt.isupper() else value
    neg = val < 0.0
    val = abs(val)
    h = 0
    if kind == "h":
        h = int(val)
        val = (val - h) * 60.0
    m = int(val)
    val = (val - m) * 60.0
    s = int(val)
    val -= s
    scale = int(10.0 ** decpl)
    f = int(val * scale + 0.5)
    while f >= scale:
        f -= scale
        s += 1
    while s >= 60:
        s -= 60
        m += 1
    while m >= 60:
        m -= 60
        h += 1
    text = ("-" if neg else "") + (f"{h}:{m:02d}:" if kind == "h" else f"{h * 60 + m}:")
    text += f"{s:02d}" + (f".{f:0{decpl}d}" if decpl > 0 else "")
    return dtoc3(value, decpl, "g", width) if len(text) > abs(width) else text


class Format:
    """One of fmtio's printf formats and a double to go in it: %d, %f, %e, %g,
    %s, %h, %m, %H, %M, with a width, a precision, - and 0. Text around it
    goes through; without a % there is no number to print."""

    _SPEC = re.compile(r"%(-?)(0?)(\d*)(?:\.(\d+))?([dfegshmHM])")

    def __init__(self, text: str):
        self.text = text
        self.match = self._SPEC.search(text)
        if self.match is None and "%" in text:
            raise ValueError(f"cannot read the format '{text}': use one of %d %f %e %g "
                             "%h %m %H %M, as in %8.2f")

    def __call__(self, value: float) -> str:
        if self.match is None:
            return self.text
        left, zero, width, decpl, kind = self.match.groups()
        size = int(width) if width else None
        places = None if decpl is None else int(decpl)
        if kind == "d":
            number = str(int(value))
        elif kind in "hmHM":
            number = dtoc(value, 1 if places is None else places, kind,
                          SZ_OBUF + 1 if size is None else size)
        elif kind == "f" and places == 0:
            number = dtoc3(value, 0, "f")
        else:
            number = printf(value, "g" if kind == "s" else kind, places, size, double=True)
        if size is not None:
            if left:
                number = number.ljust(size)
            elif zero:
                number = number.zfill(size)
            else:
                number = number.rjust(size)
        return self.text[:self.match.start()] + number + self.text[self.match.end():]


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


def pow_di(x: float, n: int) -> float:
    """f2c's pow_di, which Fortran's double ** integer compiles to: for a
    negative power the reciprocal first, then repeated squaring -- so
    10.0 ** -2 is 0.1 * 0.1 = 0.010000000000000002, not 0.01."""
    result = 1.0
    if n != 0:
        if n < 0:
            n, x = -n, 1.0 / x
        while True:
            if n & 1:
                result *= x
            n >>= 1
            if n:
                x *= x
            else:
                break
    return result


def ctod(text: str) -> float | None:
    """sys/fmtio/ctod.x: a number as IRAF reads it -- the digits gathered
    into an integer and scaled by a power of ten (pow_di), which is not
    always the nearest double to the decimal: 2.25 comes out as
    2.2500000000000004.
    Sexagesimal h:m:s, d or e exponents and INDEF are accepted; None when
    the text is not a number."""
    s = str(text).strip()
    if s.startswith("INDEF"):
        return 1.6e308
    i, n = 0, len(s)
    neg = s[:1] == "-"
    if s[:1] in "+-" and s:
        i += 1
    while i < n and s[i] == "0":
        i += 1
    dval, scalar = 0.0, 60.0
    digits: list[str] = []
    e = 0
    while True:
        while i < n and s[i].isdigit() and len(digits) < MAX_DIGITS:
            digits.append(s[i])
            i += 1
        e = 0
        while i < n and s[i].isdigit():
            e += 1
            i += 1
        scalar /= 60.0
        if i < n and s[i] in "'\":dDhHmMsS" and i > 0:
            i += 1
            value = 0.0
            for d in digits:
                value = value * 10.0 + int(d)
            dval += value * scalar * pow_di(10.0, e)
            digits = []
            while i < n and s[i] in " '\":dDhHmMsS":
                i += 1
        else:
            break
    if i < n and s[i] == ".":
        i += 1
        if not digits:
            while i < n and s[i] == "0":
                i += 1
                e -= 1
        while i < n and s[i].isdigit() and len(digits) < MAX_DIGITS:
            digits.append(s[i])
            e -= 1
            i += 1
        while i < n and s[i].isdigit():
            i += 1
    if not any(ch.isdigit() for ch in s[:i]):
        return None
    value = 0.0
    for d in digits:
        value = value * 10.0 + int(d)
    if e != 0:
        value = value * pow_di(10.0, e)
    expon = 0
    found = re.match(r"[eEdD]([+-]?\d+)", s[i:])
    if found:
        expon = int(found.group(1))
        i += found.end()
    dval += value * scalar
    if expon:
        dval *= pow_di(10.0, expon)
    rest = s[i:].strip()
    if rest and not rest.startswith("/"):
        return None
    return -dval if neg else dval
