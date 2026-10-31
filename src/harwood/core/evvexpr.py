"""IRAF's vector expression evaluator, evvexpr, without IRAF.

imexpr goes through sys/fmtio/evvexpr.gy, which is not hedit's evexpr
(core/evexpr.py): operands are vectors as well as scalars, there are five
numeric types with C's arithmetic on each, the trigonometry is in radians, and
a mean rejects points unless told how not to. The rules that decide the numbers:

    types    bool, str, short, int, long, real, double. A binary operation runs
             in the wider of the two, so short * short wraps at 16 bits and
             int / int truncates.
    lengths  a scalar goes with any vector; two vectors give the shorter.
    real     single precision. A function of a real is the double function of
             it, rounded: f2c's sin(x), not sinf.

The grammar is the .gy's, whose precedence is not C's: && and || are one level
and looser than & | ^, -a ** 2 is -(a ** 2), ** is left associative, and the
last arm of ?: takes all the expression there is.

From iraf-community 2.18: sys/fmtio/evvexpr.gy, xvvgettok.x, sys/vops. See
LICENSES/IRAF-COPYRIGHT.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np

from . import evexpr, iraffmt

BOOL, CHAR, SHORT, INT, LONG, REAL, DOUBLE = "b", "c", "s", "i", "l", "r", "d"
DTYPE = {BOOL: np.int32, SHORT: np.int16, INT: np.int32, LONG: np.int64, REAL: np.float32,
         DOUBLE: np.float64}
_ORDER = {SHORT: 0, INT: 1, LONG: 2, REAL: 3, DOUBLE: 4}
_BY_ORDER = (SHORT, INT, LONG, REAL, DOUBLE)
f32 = np.float32
#: math.h's, a real: RADTODEG and DEGTORAD are off by 1.2e-8 for a double too.
RADIAN = f32(57.295779513082320877)
LOG10E = 0.43429448190325182765
INDEFR = f32(1.6e38)
INDEFD = 1.6e308
EPSILONR = f32(1.192092896e-7)
EPSILOND = 2.220446049250313e-16
MAX_REAL = f32(3.4028235e38)
#: The tolerance of advzr and advzd: a divisor this small divides by nothing.
_TOL = {REAL: f32(1.0e-20), DOUBLE: 1.0e-20}


class ExprError(ValueError):
    """An expression IRAF would also refuse, with IRAF's reason."""


@dataclass
class Op:
    """An operand: a type, and a numpy scalar, an array or (str) a string."""

    type: str
    value: Any

    @property
    def length(self) -> int:
        """O_LEN: 0 for a scalar."""
        return self.value.size if isinstance(self.value, np.ndarray) else 0


def scalar(kind: str, value) -> Op:
    return Op(kind, DTYPE[kind](value))


# ---------------------------------------------------------------- the lexer

_SINGLE = {"+": "PLUS", "-": "MINUS", ":": "COLON", "@": "AT", "^": "BXOR", "~": "BNOT",
           "(": "(", ")": ")", ",": ","}
_DOUBLED = {"*": ("*", "EXPON", "STAR"), "/": ("/", "CONCAT", "SLASH"),
            "?": ("=", "SE", "QUEST"), "<": ("=", "LE", "LT"), ">": ("=", "GE", "GT"),
            "!": ("=", "NE", "LNOT"),
            "=": ("=", "EQ", "EQ"), "&": ("&", "LAND", "BAND"), "|": ("|", "LOR", "BOR")}
_STARTS_NAME = set("ABCDEFGHJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz")


def tokens(expr: str) -> list[tuple[str, Any]]:
    """xvv_gettok over the whole expression."""
    out: list[tuple[str, Any]] = []
    ip, n = 0, len(expr)
    while True:
        while ip < n and expr[ip] in " \t":
            ip += 1
        if ip >= n:
            out.append(("EOS", None))
            return out
        ch = expr[ip]
        if ch in _STARTS_NAME:
            ip = _identifier(expr, ip, out)
        elif ch in "I.0123456789":
            ip = _number(expr, ip, out)
        elif ch in "'\"":
            end = expr.find(ch, ip + 1)
            if end < 0:
                raise ExprError("missing closing quote in string constant")
            out.append(("CONSTANT", Op(CHAR, expr[ip + 1:end])))
            ip = end + 1
        elif ch in _DOUBLED:
            second, two, one = _DOUBLED[ch]
            if expr[ip + 1:ip + 2] == second:
                out.append((two, None))
                ip += 2
            else:
                out.append((one, None))
                ip += 1
        elif ch in _SINGLE:
            out.append((_SINGLE[ch], None))
            ip += 1
        elif ch == "\n":
            ip += 1
        else:
            ip = _identifier(expr, ip, out)


def _identifier(expr: str, ip: int, out: list) -> int:
    start = ip
    while ip < len(expr) and (expr[ip].isalnum() or expr[ip] in evexpr.IDENT_CHARS):
        ip += 1
    if ip == start:
        raise ExprError(f"syntax error at '{expr[ip]}'")
    out.append(("IDENTIFIER", expr[start:ip]))
    return ip


def _number(expr: str, ip: int, out: list) -> int:
    kind, nchars = evexpr.lexnum(expr, ip)
    if kind == evexpr.NON or nchars == 0:
        return _identifier(expr, ip, out)
    text = expr[ip:ip + nchars]
    end = ip + nchars
    if kind == evexpr.REAL:
        value = evexpr.INDEFR if text == "INDEF" else iraffmt.ctod(text)
        # numbuf is copied from after the number, not from it: 1.5d0 is a real,
        # and in 2.5*d the d makes the 2.5 a double.
        after = expr[end:end + nchars]
        double = "d" in after or "D" in after or nchars > iraffmt.NDIGITS_RP + 3
        out.append(("CONSTANT", scalar(DOUBLE if double else REAL, value)))
    else:
        value = evexpr.number(text, kind).value
        out.append(("CONSTANT", Op(INT, np.array(value, np.int64).astype(np.int32)[()])))
    return end


# --------------------------------------------------------------- the grammar

_LEVEL = {"LAND": 2, "LOR": 2, "BAND": 3, "BOR": 3, "BXOR": 3, "EQ": 4, "NE": 4, "SE": 4,
          "LT": 5, "GT": 5, "LE": 5, "GE": 5, "CONCAT": 6, "PLUS": 7, "MINUS": 7, "STAR": 8,
          "SLASH": 8, "EXPON": 10}
_NONASSOC = {4, 5}
_QUEST, _UNARY = 1, 9
_BITWISE_OR_ARITHMETIC = {"PLUS", "MINUS", "STAR", "SLASH", "EXPON", "CONCAT", "BAND", "BOR",
                          "BXOR"}


class Expression:
    """A parsed expression; evaluate() runs it, once for each image line."""

    def __init__(self, text: str):
        self.tokens = tokens(text)
        self.at = 0
        self.tree = self._exprlist()
        if self._peek() != "EOS":
            raise ExprError("syntax error")

    def _peek(self) -> str:
        return self.tokens[self.at][0]

    def _take(self, kind: str | None = None):
        token = self.tokens[self.at]
        if kind is not None and token[0] != kind:
            raise ExprError("syntax error")
        self.at += 1
        return token[1]

    def _exprlist(self):
        last = self._expr(0)
        while self._peek() == ",":
            self._take()
            last = self._expr(0)
        return last

    def _expr(self, level: int):
        left = self._unary()
        while True:
            kind = self._peek()
            if kind == "QUEST":
                if level > _QUEST:
                    return left
                self._take()
                middle = self._expr(0)
                self._take("COLON")
                left = ("quest", left, middle, self._expr(0))
                continue
            at = _LEVEL.get(kind)
            if at is None or at < level:
                return left
            self._take()
            right = self._expr(at + 1)
            op = "binop" if kind in _BITWISE_OR_ARITHMETIC else "boolop"
            left = (op, kind, left, right)
            if at in _NONASSOC and _LEVEL.get(self._peek()) == at:
                raise ExprError("syntax error")

    def _unary(self):
        kind = self._peek()
        if kind in ("MINUS", "LNOT", "BNOT"):
            self._take()
            return ("unop", kind, self._expr(_UNARY))
        return self._primary()

    def _primary(self):
        kind = self._peek()
        if kind == "(":
            self._take()
            inside = self._expr(0)
            self._take(")")
            return inside
        if kind == "AT":
            self._take()
            name = self._take("CONSTANT")
            if name.type != CHAR:
                raise ExprError("syntax error")
            return ("operand", name.value)
        if kind == "IDENTIFIER" or kind == "CONSTANT":
            value = self._take()
            if self._peek() == "(":
                if kind == "CONSTANT":
                    if value.type != CHAR:
                        raise ExprError("illegal function name")
                    value = value.value
                self._take()
                args = []
                if self._peek() != ")":
                    args.append(self._expr(0))
                    while self._peek() == ",":
                        self._take()
                        args.append(self._expr(0))
                self._take(")")
                return ("call", value, args)
            if kind == "CONSTANT":
                return ("constant", value)
            if value in ("yes", "no"):
                return ("constant", scalar(BOOL, value == "yes"))
            return ("operand", value)
        raise ExprError("syntax error")

    def evaluate(self, getop: Callable[[str], Op | None], rangecheck: bool = True) -> Op:
        with np.errstate(all="ignore"):
            return _run(self.tree, getop, rangecheck)


def _run(node, getop, check: bool) -> Op:
    kind = node[0]
    if kind == "constant":
        return node[1]
    if kind == "operand":
        o = getop(node[1])
        if o is None:
            raise ExprError(f"unknown operand `{node[1]}'")
        return o
    if kind == "unop":
        return unop(node[1], _run(node[2], getop, check))
    if kind == "binop":
        return binop(node[1], _run(node[2], getop, check), _run(node[3], getop, check), check)
    if kind == "boolop":
        return boolop(node[1], _run(node[2], getop, check), _run(node[3], getop, check))
    if kind == "quest":
        return quest(_run(node[1], getop, check), _run(node[2], getop, check),
                     _run(node[3], getop, check))
    return call(node[1], [_run(arg, getop, check) for arg in node[2]], check)


# -------------------------------------------------------------- the operators

def newtype(a: str, b: str) -> str:
    """xvv_newtype: bool and str go only with themselves."""
    if a == b:
        return a
    if a in _ORDER and b in _ORDER:
        wide = max(_ORDER[a], _ORDER[b])
        return _BY_ORDER[wide]
    raise ExprError("operands have incompatible types")


def chtype(o: Op, kind: str) -> Op:
    """xvv_chtype."""
    if o.type == kind:
        return o
    if kind == CHAR:
        if o.length:
            raise ExprError("chtype: invalid operand type")
        return Op(CHAR, _text(o))
    if o.type == CHAR:
        if kind == BOOL:
            return scalar(BOOL, o.value[:1] in ("y", "Y"))
        return scalar(kind, _gctod(o.value))
    if kind == BOOL:
        out = o.value != 0
        return Op(BOOL, out.astype(np.int32) if o.length else np.int32(out))
    return Op(kind, _cast(o.value, kind))


def _cast(value, kind: str):
    if isinstance(value, np.ndarray):
        return value.astype(DTYPE[kind])
    return np.array(value).astype(DTYPE[kind])[()]


def _gctod(text: str) -> float:
    value = evexpr._ctod(text.strip()) if text.strip() else None
    return 0.0 if value is None else value


def _text(o: Op) -> str:
    if o.type == BOOL:
        return "YES" if o.value else "NO"
    if o.type in (REAL, DOUBLE):
        return iraffmt.g(float(o.value), o.type == DOUBLE)
    return str(int(o.value))


def unop(kind: str, a: Op) -> Op:
    if kind == "MINUS":
        if a.type in (BOOL, CHAR):
            raise ExprError("negation of a nonarithmetic operand")
        return Op(a.type, _keep(np.negative(a.value), a.type))
    if kind == "LNOT":
        if a.type in (CHAR, REAL, DOUBLE):
            raise ExprError("not of a nonlogical")
        value = a.value == 0
        if a.length:
            return Op(BOOL, value.astype(np.int32))
        # A scalar short or int stays what it was, as O_VAL$T(out) writes it,
        # in an operand that says bool.
        return scalar(BOOL, value)
    if a.type in (BOOL, CHAR, REAL, DOUBLE):
        raise ExprError("boolean not of a noninteger operand")
    return Op(a.type, _keep(np.invert(a.value), a.type))


def _keep(value, kind: str):
    """The result in the operand's own type: numpy promotes less than it did,
    but never mind that for a scalar."""
    if isinstance(value, np.ndarray):
        return value.astype(DTYPE[kind]) if value.dtype != DTYPE[kind] else value
    return DTYPE[kind](value)


def _length(a: Op, b: Op) -> int:
    if a.length and b.length:
        return min(a.length, b.length)
    return a.length or b.length


def _cut(o: Op, nelem: int):
    return o.value[:nelem] if o.length else o.value


def binop(kind: str, a: Op, b: Op, check: bool) -> Op:
    dtype = newtype(a.type, b.type)
    if dtype == BOOL:
        raise ExprError("binop: operation illegal for boolean operands")
    if dtype == CHAR:
        if kind != "CONCAT":
            raise ExprError("binop: operation illegal for string operands")
        return Op(CHAR, a.value + b.value)
    a, b = chtype(a, dtype), chtype(b, dtype)
    if kind == "CONCAT":
        return Op(dtype, np.concatenate([np.atleast_1d(a.value), np.atleast_1d(b.value)]))
    nelem = _length(a, b)
    x, y = _cut(a, nelem), _cut(b, nelem)
    if kind in ("BAND", "BOR", "BXOR"):
        if dtype in (REAL, DOUBLE):
            raise ExprError("binop: bitwise boolean operands must be an integer type")
        f = {"BAND": np.bitwise_and, "BOR": np.bitwise_or, "BXOR": np.bitwise_xor}[kind]
        return Op(dtype, _keep(f(x, y), dtype))
    if kind == "PLUS":
        return Op(dtype, _keep(np.add(x, y), dtype))
    if kind == "MINUS":
        return Op(dtype, _keep(np.subtract(x, y), dtype))
    if kind == "STAR":
        return Op(dtype, _keep(np.multiply(x, y), dtype))
    if kind == "SLASH":
        return Op(dtype, _divide(x, y, dtype, check, a.length, b.length))
    return Op(dtype, _power(x, y, dtype))


def _divide(x, y, dtype: str, check: bool, len1: int, len2: int):
    zero = DTYPE[dtype](0)
    if not check:
        if dtype in (REAL, DOUBLE):
            return _keep(np.divide(x, y), dtype)
        if np.any(np.asarray(y) == 0):
            raise ExprError("integer divide by zero")
        return _truncate(x, y, dtype)
    if len2 == 0:
        # Against a scalar the test is for 0 itself; advz has a tolerance.
        if y == 0:
            return np.full(len1, zero, DTYPE[dtype]) if len1 else zero
        return _keep(np.divide(x, y), dtype) if dtype in (REAL, DOUBLE) \
            else _truncate(x, y, dtype)
    if len1 == 0:
        x = np.full(np.shape(y), x, DTYPE[dtype])
    if dtype in (REAL, DOUBLE):
        tol = _TOL[dtype]
        bad = (y < tol) & (y > -tol)
        return np.where(bad, zero, _keep(np.divide(x, np.where(bad, 1, y)), dtype)) \
            .astype(DTYPE[dtype])
    bad = y == 0
    return np.where(bad, zero, _truncate(x, np.where(bad, 1, y), dtype)).astype(DTYPE[dtype])


def _truncate(x, y, dtype: str):
    """C's integer division, toward zero, in the type's width."""
    wide = np.asarray(x, np.int64), np.asarray(y, np.int64)
    out = np.abs(wide[0]) // np.abs(wide[1]) * (np.sign(wide[0]) * np.sign(wide[1]))
    return _keep(out[()] if out.ndim == 0 else out, dtype)


def _power(x, y, dtype: str):
    if dtype in (REAL, DOUBLE):
        out = np.power(np.asarray(x, np.float64), np.asarray(y, np.float64))
        return _keep(out[()] if out.ndim == 0 else out, dtype)
    bits = {SHORT: np.int16, INT: np.int32, LONG: np.int64}[dtype]
    shape = np.broadcast(x, y).shape
    xs = np.broadcast_to(x, shape).ravel().tolist()
    ys = np.broadcast_to(y, shape).ravel().tolist()
    wrap = 1 << (8 * np.dtype(bits).itemsize)
    out = np.array([_pow_ii(p, q, wrap) for p, q in zip(xs, ys, strict=True)], np.int64)
    out = out.astype(bits).reshape(shape)
    return out[()] if out.ndim == 0 else out


def _pow_ii(x: int, n: int, wrap: int) -> int:
    """libf2c's pow_ii."""
    if n <= 0:
        if n == 0 or x == 1:
            return 1
        if x != -1:
            return 0            # 1/x for a 0 too, as IRAF's build does it.
        n = -n
    return pow(x, n, wrap) if x >= 0 else (1 if n % 2 == 0 else -1) * pow(-x, n, wrap)


def boolop(kind: str, a: Op, b: Op) -> Op:
    a = Op(INT, a.value) if a.type == BOOL else a
    b = Op(INT, b.value) if b.type == BOOL else b
    dtype = newtype(a.type, b.type)
    if dtype == CHAR:
        return scalar(BOOL, _compare_text(kind, a.value, b.value))
    a, b = chtype(a, dtype), chtype(b, dtype)
    nelem = _length(a, b)
    x, y = _cut(a, nelem), _cut(b, nelem)
    if kind in ("LAND", "LOR"):
        if dtype in (REAL, DOUBLE):
            raise ExprError("boolop: illegal switch")
        f = np.logical_and if kind == "LAND" else np.logical_or
        result = f(x != 0, y != 0)
    elif kind == "SE":
        raise ExprError("boolop: illegal operation")
    else:
        result = {"LT": np.less, "LE": np.less_equal, "GT": np.greater, "GE": np.greater_equal,
                  "EQ": np.equal, "NE": np.not_equal}[kind](x, y)
    if nelem:
        return Op(BOOL, result.astype(np.int32))
    return scalar(BOOL, result)


def _compare_text(kind: str, a: str, b: str) -> bool:
    if kind == "SE":
        # xvv_patmatch: * is any characters, as in a file name.
        return evexpr.patmatch(a, b.replace("*", "?*")) > 0
    if kind not in ("LT", "LE", "GT", "GE", "EQ", "NE"):
        raise ExprError("boolop: illegal operation")
    return {"LT": a < b, "LE": a <= b, "GT": a > b, "GE": a >= b, "EQ": a == b,
            "NE": a != b}[kind]


def quest(cond: Op, a: Op, b: Op) -> Op:
    if cond.type in (SHORT, LONG):
        cond = chtype(cond, BOOL)
    elif cond.type not in (BOOL, INT):
        raise ExprError("evvexpr: nonboolean condition operand")
    if (not cond.length and (not a.length or a.type == CHAR)
            and (not b.length or b.type == CHAR)):
        return a if cond.value != 0 else b
    if a.type == CHAR or b.type == CHAR:
        raise ExprError("evvexpr: character and vector in cond expr")
    dtype = newtype(a.type, b.type)
    a, b = chtype(a, dtype), chtype(b, dtype)
    nelem = _length(a, b)
    if cond.length:
        nelem = min(cond.length, nelem) if nelem else cond.length
        return Op(dtype, np.where(cond.value[:nelem] != 0, _cut(a, nelem), _cut(b, nelem))
                  .astype(DTYPE[BOOL] if dtype == BOOL else DTYPE[dtype]))
    chosen = a if cond.value != 0 else b
    return Op(dtype, chosen.value[:nelem] if chosen.length
              else np.full(nelem, chosen.value, DTYPE[dtype]))


# -------------------------------------------------------------- the functions

_FLOATING = {"acos": np.arccos, "asin": np.arcsin, "cos": np.cos, "cosh": np.cosh,
             "exp": np.exp, "sin": np.sin, "sinh": np.sinh, "tan": np.tan, "tanh": np.tanh,
             "atan": np.arctan}
_NUMPY_NAME = {"acos": "arccos", "asin": "arcsin", "atan": "arctan"}
_ONE_ARG = set(_FLOATING) - {"atan"} | {
    "abs", "bool", "deg", "double", "hiv", "int", "len", "log", "log10", "long", "lov", "nint",
    "rad", "real", "short", "sort", "sqrt", "str", "sum"}
_TWO_ARGS = {"mod", "repl", "shift"}
_AT_LEAST_ONE = {"max", "min", "atan", "atan2", "mean", "stddev", "median"}
FUNCTIONS = _ONE_ARG | _TWO_ARGS | _AT_LEAST_ONE


def call(name: str, args: list[Op], check: bool = True) -> Op:
    """xvv_callfcn."""
    if name not in FUNCTIONS:
        raise ExprError(f"unrecognized macro or function `{name}'")
    if name in _AT_LEAST_ONE:
        if not args:
            raise ExprError(f"function `{name}' requires at least 1 arguments")
    else:
        wanted = 2 if name in _TWO_ARGS else 1
        if len(args) != wanted:
            raise ExprError(f"function `{name}' requires {wanted} arguments")
    if name in _FLOATING or name in ("atan2", "deg", "log", "log10", "rad", "sqrt"):
        return _floating(name, args, check)
    if name in ("mod", "min", "max") or (name == "median" and len(args) > 1):
        return _same_type(name, args)
    return _RESTS[name](args)


def _double_of(value, kind: str, f):
    """A function of a real is the double function of it, rounded to a real."""
    out = f(np.asarray(value, np.float64))
    return _keep(out[()] if out.ndim == 0 else out, kind)


def _libm(name: str):
    """A double is the bits libm gives, and numpy's own can be a bit away."""
    f = getattr(math, name)

    def call_it(x: float) -> float:
        try:
            return f(x)
        except (ValueError, OverflowError):
            return getattr(np, _NUMPY_NAME.get(name, name))(x)

    return np.frompyfunc(call_it, 1, 1)


def _floating(name: str, args: list[Op], check: bool) -> Op:
    kind = DOUBLE if any(a.type in (DOUBLE, LONG) for a in args) else REAL
    args = [chtype(a, kind) for a in args]
    v = args[0].value
    if name in _FLOATING and not (name == "atan" and len(args) > 1):
        if kind == DOUBLE:
            out = _libm(name)(np.asarray(v, np.float64))
            return Op(kind, np.asarray(out, np.float64) if args[0].length else np.float64(out))
        return Op(kind, _double_of(v, kind, _FLOATING[name]))
    if name in ("atan", "atan2"):
        if len(args) == 1:
            return Op(kind, _double_of(v, kind, np.arctan))
        nelem = _length(args[0], args[1])
        out = np.arctan2(np.asarray(_cut(args[0], nelem), np.float64),
                         np.asarray(_cut(args[1], nelem), np.float64))
        return Op(kind, _keep(out[()] if out.ndim == 0 else out, kind))
    if name == "deg":
        return Op(kind, _keep(np.multiply(v, RADIAN), kind))
    if name == "rad":
        return Op(kind, _keep(np.divide(v, RADIAN), kind))
    if name == "sqrt":
        # A scalar 0 needs no root, and a vector's does; all the same number.
        bad = (v < 0) if check else np.zeros_like(v, bool)
        return Op(kind, _guarded(v, bad, np.sqrt, kind))
    bad = (v <= 0) if check else np.zeros_like(v, bool)
    f = np.log if name == "log" else _lg10
    return Op(kind, _guarded(v, bad, f, kind))


def _lg10(x):
    """libf2c's r_lg10 and d_lg10: log10e * log(x), not log10(x)."""
    return LOG10E * np.log(x)


def _guarded(v, bad, f, kind: str):
    safe = np.where(bad, 1, v)
    out = np.where(bad, 0, _double_of(safe, kind, f))
    return _keep(out[()] if np.ndim(out) == 0 else out, kind)


def _same_type(name: str, args: list[Op]) -> Op:
    kind = args[0].type
    for a in args[1:]:
        kind = newtype(kind, a.type)
    if kind in (BOOL, CHAR):
        raise ExprError(f"{name}: illegal operand type")
    args = [chtype(a, kind) for a in args]
    lengths = [a.length for a in args if a.length]
    nelem = min(lengths) if lengths else 0
    values = [_cut(a, nelem) for a in args]
    if name == "mod":
        if len(args) != 2:
            raise ExprError("function `mod' requires 2 arguments")
        return Op(kind, _mod(values[0], values[1], kind))
    if name in ("min", "max"):
        out = values[0]
        f = np.maximum if name == "max" else np.minimum
        for v in values[1:]:
            out = f(out, v)
        return Op(kind, _keep(out, kind) if lengths else DTYPE[kind](out))
    if len(args) not in (3, 4, 5):
        raise ExprError("median: wrong number of arguments")
    stack = np.sort(np.stack(np.broadcast_arrays(*values)), axis=0)
    # amed4 takes the lower of the two in the middle.
    middle = (len(args) - 1) // 2
    out = stack[middle]
    return Op(kind, out if nelem else DTYPE[kind](out))


def _mod(x, y, kind: str):
    if kind in (REAL, DOUBLE):
        # libf2c's r_mod and d_mod, in double.
        p, q = np.asarray(x, np.float64), np.asarray(y, np.float64)
        out = p - q * np.trunc(p / q)
        return _keep(out[()] if out.ndim == 0 else out, kind)
    if np.any(np.asarray(y) == 0):
        raise ExprError("integer divide by zero")
    p, q = np.asarray(x, np.int64), np.asarray(y, np.int64)
    out = np.abs(p) % np.abs(q) * np.sign(p)
    return _keep(out[()] if out.ndim == 0 else out, kind)


def _abs(args):
    a = args[0]
    if a.type in (BOOL, CHAR):
        raise ExprError("abs: illegal operand type")
    return Op(a.type, _keep(np.abs(a.value), a.type))


def _integer_of(a: Op, name: str) -> Op:
    if a.type == BOOL:
        return Op(INT, a.value)
    if a.type == CHAR:
        raise ExprError(f"{name}: illegal operand type")
    return a


def _hiv(args, f=np.max):
    a = _integer_of(args[0], "hiv")
    return Op(a.type, DTYPE[a.type](f(a.value)))


def _lov(args):
    return _hiv(args, np.min)


def _sum(args):
    a = _integer_of(args[0], "sum")
    if a.type == DOUBLE:
        v = np.add.accumulate(a.value)[-1] if a.length else a.value
        return Op(DOUBLE, np.float64(v))
    if a.type == LONG:
        v = np.add.accumulate(np.asarray(a.value, np.float64).ravel())[-1]
        return Op(REAL, f32(v))
    v = np.add.accumulate(np.asarray(a.value, f32).ravel())[-1]
    return Op(REAL, f32(v))


def _awvg(values, lcut, hcut):
    """awvg: lcut and hcut as the type gives them; the sums always in double."""
    if hcut == lcut:
        kept = values
    else:
        kept = values[(values >= float(lcut)) & (values <= float(hcut))]
    n = kept.size
    if n == 0:
        return n, None, None
    total = float(np.add.accumulate(kept)[-1])
    if n == 1:
        return n, total, None
    square = float(np.add.accumulate(kept * kept)[-1])
    temp = (square - (total / n) * total) / (n - 1)
    return n, total / n, 0.0 if temp < 0 else math.sqrt(temp)


def _arav(a: Op, ksig):
    """arav: reject and measure again until nothing goes. ksig 0 is Chauvenet's
    relation, which is what a mean gets when not told otherwise."""
    real = a.type in (SHORT, INT, REAL)
    values = np.asarray(a.value, np.float64).ravel()
    to = f32 if real else float
    lcut, hcut = -MAX_REAL, MAX_REAL
    if not real:
        lcut, hcut = float(lcut), float(hcut)
    ngpix = 2147483647
    mean = sigma = None
    while True:
        old = ngpix
        ngpix, m, s = _awvg(values, lcut, hcut)
        mean = None if m is None else to(m)
        sigma = None if s is None else to(s)
        if ngpix <= 1 or sigma is None or sigma <= (EPSILONR if real else EPSILOND):
            break
        if ksig == 0.0:
            lgpx = to(LOG10E * math.log(float(f32(ngpix))))
            k1, k2, k3 = to(float(f32(-0.1042))), to(float(f32(1.1695))), to(float(f32(.8895)))
            deviation = to(to(to(lgpx * to(to(k1 * lgpx) + k2)) + k3) * sigma)
        else:
            deviation = to(sigma * abs(ksig))
        lcut, hcut = to(mean - deviation), to(mean + deviation)
        if ngpix >= old:
            break
    return mean, sigma


def _moment(args, which: int):
    a = _integer_of(args[0], "mean")
    if not a.length:
        # arav of no pixels.
        return Op(DOUBLE, np.float64(INDEFD)) if a.type == DOUBLE else Op(REAL, INDEFR)
    ksig = 0.0
    if len(args) == 2:
        if args[1].length:
            raise ExprError("mean: ksigma arg must be a scalar")
        k = args[1] if args[1].type in (REAL, DOUBLE) else chtype(args[1], REAL)
        ksig = float(k.value) if a.type in (LONG, DOUBLE) else f32(k.value)
    result = _arav(a, ksig)[which]
    if a.type == DOUBLE:
        return Op(DOUBLE, np.float64(INDEFD if result is None else result))
    return Op(REAL, INDEFR if result is None else f32(result))


def _mean(args):
    return _moment(args, 0)


def _stddev(args):
    return _moment(args, 1)


def _median(args):
    a = _integer_of(args[0], "median")
    if not a.length:
        return a
    n = a.length
    if n <= 2:
        return Op(a.type, a.value[0])
    # asok: the (n + 1) / 2 th smallest, the lower of the two when n is even.
    return Op(a.type, np.partition(a.value, (n + 1) // 2 - 1)[(n + 1) // 2 - 1])


def _len(args):
    return scalar(INT, args[0].length)


def _repl(args):
    a = _integer_of(args[0], "repl")
    if args[1].length:
        raise ExprError("repl: replication factor must be a scalar")
    n = max(1, int(chtype(args[1], INT).value))
    if a.length:
        return Op(a.type, np.tile(a.value, n))
    return Op(a.type, np.full(n, a.value, DTYPE[a.type]))


def _shift(args):
    a = args[0]
    if a.type in (BOOL, CHAR):
        raise ExprError("shift: illegal operand type")
    if args[1].length:
        raise ExprError("shift: shift arg must be a scalar")
    by = int(chtype(args[1], INT).value)
    n = a.length
    return Op(a.type, np.roll(a.value, max(-n, min(n, by))) if n else a.value)


def _sort(args):
    a = args[0]
    if a.type in (BOOL, CHAR):
        raise ExprError("sort: illegal operand type")
    return Op(a.type, np.sort(a.value) if a.length else a.value)


def _to(kind: str):
    def convert(args):
        a = args[0]
        if a.type == CHAR:
            if kind == BOOL:
                return scalar(BOOL, a.value[:1] in ("y", "Y"))
            return scalar(kind, _gctod(a.value))
        return chtype(Op(INT, a.value) if a.type == BOOL else a, kind)
    return convert


def _nint(args):
    a = args[0]
    if a.type == CHAR:
        return scalar(INT, _nearest(_gctod(a.value)))
    if a.type in (REAL, DOUBLE):
        return Op(INT, _keep(_nearest(a.value), INT))
    return chtype(Op(INT, a.value) if a.type == BOOL else a, INT)


def _nearest(x):
    """libf2c's i_nint: floor(x + .5), or for a negative -floor(.5 - x), in double."""
    v = np.asarray(x, np.float64)
    out = np.where(v >= 0, np.floor(v + .5), -np.floor(.5 - v))
    return np.asarray(out).astype(np.int32)[()]


def _str(args):
    a = args[0]
    return a if a.type == CHAR else Op(CHAR, _text(a))


_RESTS = {"abs": _abs, "hiv": _hiv, "lov": _lov, "sum": _sum, "mean": _mean, "stddev": _stddev,
          "median": _median, "len": _len, "repl": _repl, "shift": _shift, "sort": _sort,
          "bool": _to(BOOL), "short": _to(SHORT), "int": _to(INT), "long": _to(LONG),
          "real": _to(REAL), "double": _to(DOUBLE), "nint": _nint, "str": _str}
