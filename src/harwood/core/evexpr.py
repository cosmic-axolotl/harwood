"""IRAF's expression evaluator, evexpr, without IRAF.

hedit's parenthesised values and hselect's selection go through
sys/fmtio/evexpr.y in IRAF, and the answers depend on its rules more than
one would guess: an integer divided by an integer is an integer (7/2 is 3),
a real is single precision, the trigonometric functions work in degrees,
`//` joins strings, `?=` is a pattern match, `a ? b : c` chooses, and an
operand is whatever the caller says a name means -- a header keyword, for
these tasks. This is that grammar and those types, ported:

    bool    yes, no, and comparisons
    int     1, 17b (octal), 1fx (hex)
    real    1.5, 1e3, 12:30:45 (sexagesimal), INDEF -- float32
    str     'text' or "text"

Operators, loosest first: ?:, ||, &&, == != ?=, < <= > >=, //, + -, * /,
**, unary - and !. Functions: abs acos asin atan atan2 bool cos exp int log
log10 max min mod nint real sin sqrt str tan. A name is letters, digits and
_.$@#%&;[]\\^{}~ ; anything else in a keyword, as in DATE-OBS, is written
@"DATE-OBS".

    evaluate("(EXPTIME * 2)", lookup)    -> Operand("int", 30)
    encode(operand)                      -> "30", as hedit records it
"""

from __future__ import annotations

import math
import re
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np

from . import iraffmt

f32 = np.float32
INDEFR = f32(1.6e38)
IDENT_CHARS = "_.$@#%&;[]\\^{}~"


class ExprError(ValueError):
    """An expression IRAF would also refuse, with IRAF's reason."""


@dataclass
class Operand:
    type: str          # "bool", "str", "int", "real"
    value: Any


def evaluate(expr: str, getop: Callable[[str], Operand] | None = None) -> Operand:
    """The value of an expression; `getop(name)` gives each operand."""
    parser = _Parser(_tokens(expr), getop)
    result = parser.expr()
    if parser.peek()[0] != "EOS":
        raise ExprError("syntax error")
    return result


def encode(o: Operand) -> str:
    """he_encodeop: %b, %s, %d or %g, IRAF's."""
    if o.type == "bool":
        return "yes" if o.value else "no"
    if o.type == "str":
        return o.value
    if o.type == "int":
        return str(o.value)
    return iraffmt.g(float(o.value))


# ------------------------------------------------------------ numbers


_CLASSES = {"+": 0, "-": 0, ":": 5, ".": 6}
for _c in "01234567":
    _CLASSES[_c] = 1
for _c in "89":
    _CLASSES[_c] = 2
for _c in "ACF":
    _CLASSES[_c] = 3
for _c in "DE":
    _CLASSES[_c] = 4
_CLASSES["X"], _CLASSES["B"] = 7, 8
_OTHER = 9

ACC, RVT = "acc", "rvt"
OCT, DEC, HEX, REAL, NON = "octal", "decimal", "hex", "real", "nonnum"
START, UNM, ODH, DHR, QRF, HX, QHX, QRN, OHN, RFR, RRX, QRX, HRX, RNM, REX = range(1, 16)
# lexdata.inc: one row per state, columns +- 0-7 8-9 ACF ED : . X B other
_ACTION = {
    START: [UNM, ODH, DHR, NON, NON, QRN, QRF, NON, NON, NON],
    UNM: [RVT, ODH, DHR, RVT, RVT, RVT, QRF, RVT, RVT, RVT],
    ODH: [DEC, ACC, DHR, HX, QHX, QRN, RFR, HEX, OHN, DEC],
    DHR: [DEC, ACC, ACC, HX, QHX, QRN, RFR, HEX, HX, DEC],
    QRF: [RVT, RFR, RFR, RVT, RVT, RVT, RVT, RVT, RVT, RVT],
    HX: [RVT, ACC, ACC, ACC, ACC, RVT, RVT, HEX, ACC, RVT],
    QHX: [QRX, HRX, HRX, HX, HX, RVT, RVT, HEX, HX, RVT],
    QRN: [RVT, RNM, RNM, RVT, RVT, ACC, RVT, RVT, RVT, RVT],
    OHN: [OCT, HX, HX, HX, HX, OCT, OCT, HEX, HX, OCT],
    RFR: [REAL, ACC, ACC, REAL, RRX, REAL, REAL, REAL, REAL, REAL],
    RRX: [QRX, REX, REX, RVT, RVT, RVT, RVT, RVT, RVT, RVT],
    QRX: [RVT, REX, REX, RVT, RVT, RVT, RVT, RVT, RVT, RVT],
    HRX: [REAL, ACC, ACC, HX, HX, REAL, REAL, HEX, REAL, REAL],
    RNM: [REAL, ACC, ACC, REAL, QRX, ACC, QRF, REAL, REAL, REAL],
    REX: [REAL, ACC, ACC, REAL, REAL, REAL, REAL, REAL, REAL, REAL],
}


def lexnum(text: str, start: int = 0) -> tuple[str, int]:
    """sys/fmtio/lexnum.x: the kind of number at `start` and its length
    (NON, 0 when there is none). Leading blanks are skipped."""
    while start < len(text) and text[start] in " \t":
        start += 1
    if text.startswith("INDEF", start):
        return REAL, 5
    ip, state, stack = start, START, []
    while True:
        ch = text[ip].upper() if ip < len(text) else ""
        cc = _CLASSES.get(ch, _OTHER)
        act = _ACTION[state][cc]
        if act == ACC:
            ip += 1
            continue
        if act == RVT:
            while True:
                ip, state = stack.pop()
                kind = _ACTION[state][_OTHER]
                if kind != RVT or not stack:
                    break
            break
        if act in (OCT, DEC, HEX, REAL, NON):
            kind = act
            if (kind == OCT and cc == 8) or (kind == HEX and cc == 7):
                ip += 1
            break
        if len(stack) >= 15:
            kind = NON
            break
        stack.append((ip, state))
        ip += 1
        state = act
    if kind in (NON, RVT):
        return NON, 0
    return kind, ip - start


def number(text: str, kind: str) -> Operand:
    """gctol / gctod of a token lexnum has classified."""
    if kind == OCT:
        return Operand("int", int(text.rstrip("bB"), 8))
    if kind == HEX:
        return Operand("int", int(text.rstrip("xX"), 16))
    if kind == DEC:
        return Operand("int", int(text))
    if text == "INDEF":
        return Operand("real", INDEFR)
    return Operand("real", f32(_ctod(text)))


def _ctod(text: str) -> float:
    """gctod: decimal, with d or e exponent, or sexagesimal h:m:s."""
    sign = -1.0 if text.startswith("-") else 1.0
    body = text.lstrip("+-")
    if ":" in body:
        value, scale = 0.0, 1.0
        for part in body.split(":"):
            value += float(part or 0) / scale
            scale *= 60.0
        return sign * value
    return sign * float(body.replace("d", "e").replace("D", "e"))


# ------------------------------------------------------------ tokens


def _tokens(expr: str) -> list[tuple[str, Any]]:
    out, ip, n = [], 0, len(expr)
    while True:
        while ip < n and expr[ip] in " \t":
            ip += 1
        if ip >= n:
            out.append(("EOS", None))
            return out
        ch = expr[ip]
        if ch.isalpha() and ch != "I":
            ip = _ident(expr, ip, out)
            continue
        if ch == "I" or ch == "." or ch.isdigit():
            kind, length = lexnum(expr, ip)
            if kind != NON:
                out.append(("CONST", number(expr[ip:ip + length], kind)))
                ip += length
            else:
                ip = _ident(expr, ip, out)
            continue
        if ch in "'\"":
            end = expr.find(ch, ip + 1)
            if end < 0:
                raise ExprError("missing closing quote in string constant")
            out.append(("CONST", Operand("str", expr[ip + 1:end])))
            ip = end + 1
            continue
        two = expr[ip:ip + 2]
        if two in ("**", "//", "?=", "<=", ">=", "!=", "==", "&&", "||"):
            out.append(({"==": "=", "&&": "&", "||": "|"}.get(two, two), None))
            ip += 2
            continue
        if ch in "+-*/?:@<>!=&|(),":
            out.append((ch, None))
            ip += 1
            continue
        if ch == "\n":
            ip += 1
            continue
        ip = _ident(expr, ip, out)


def _ident(expr: str, ip: int, out: list) -> int:
    start = ip
    ip += 1
    while ip < len(expr) and (expr[ip].isalnum() or expr[ip] in IDENT_CHARS):
        ip += 1
    out.append(("IDENT", expr[start:ip]))
    return ip


# ------------------------------------------------------------ parser


class _Parser:
    def __init__(self, tokens: list, getop) -> None:
        self.tokens, self.at, self.getop = tokens, 0, getop

    def peek(self):
        return self.tokens[self.at]

    def take(self, kind: str | None = None):
        token = self.tokens[self.at]
        if kind is not None and token[0] != kind:
            raise ExprError("syntax error")
        self.at += 1
        return token

    def expr(self) -> Operand:
        cond = self.binary(0)
        if self.peek()[0] == "?":
            self.take()
            yes = self.expr()
            self.take(":")
            no = self.binary(0)
            if cond.type != "bool":
                raise ExprError("nonboolean condition operand")
            return yes if cond.value else no
        return cond

    # loosest first; each level is left-associative except the comparisons
    LEVELS = [("|",), ("&",), ("=", "!=", "?="), ("<", ">", "<=", ">="), ("//",),
              ("+", "-"), ("*", "/"), ("**",)]

    def binary(self, level: int) -> Operand:
        if level == len(self.LEVELS):
            return self.unary()
        left = self.binary(level + 1)
        ops = self.LEVELS[level]
        while self.peek()[0] in ops:
            op = self.take()[0]
            right = self.binary(level + 1)
            left = _apply(op, left, right)
            if level in (2, 3) and self.peek()[0] in ops:
                raise ExprError("syntax error")       # nonassoc
        return left

    def unary(self) -> Operand:
        kind = self.peek()[0]
        if kind == "-":
            self.take()
            o = self.unary()
            if o.type in ("bool", "str"):
                raise ExprError("negation of a nonarithmetic operand")
            return Operand(o.type, -o.value if o.type == "int" else f32(-o.value))
        if kind == "!":
            self.take()
            o = self.unary()
            if o.type != "bool":
                raise ExprError("not of a nonlogical")
            return Operand("bool", not o.value)
        return self.primary()

    def primary(self) -> Operand:
        kind, value = self.take()
        if kind == "(":
            inner = self.expr()
            self.take(")")
            return inner
        if kind == "@":
            name = self.take("CONST")[1]
            return self.operand(str(name.value))
        if kind in ("IDENT", "CONST") and self.peek()[0] == "(":
            name = value if kind == "IDENT" else value.value
            if kind == "CONST" and value.type != "str":
                raise ExprError("illegal function name")
            self.take("(")
            args = []
            if self.peek()[0] != ")":
                args.append(self.expr())
                while self.peek()[0] == ",":
                    self.take()
                    args.append(self.expr())
            self.take(")")
            return call(name, args)
        if kind == "CONST":
            return value
        if kind == "IDENT":
            if value == "yes":
                return Operand("bool", True)
            if value == "no":
                return Operand("bool", False)
            return self.operand(value)
        raise ExprError("syntax error")

    def operand(self, name: str) -> Operand:
        if self.getop is None:
            raise ExprError(f"illegal operand `{name}'")
        return self.getop(name)


# ------------------------------------------------------------ operations


_TYPES = {("bool", "bool"): "bool", ("str", "str"): "str", ("int", "int"): "int",
          ("int", "real"): "real", ("real", "int"): "real", ("real", "real"): "real"}


def _newtype(a: Operand, b: Operand) -> str:
    kind = _TYPES.get((a.type, b.type))
    if kind is None:
        raise ExprError("operands have incompatible types")
    return kind


def _r(o: Operand):
    return f32(o.value)


def _idiv(a: int, b: int) -> int:
    if b == 0:
        raise ExprError("integer divide by zero")
    q = abs(a) // abs(b)
    return q if (a >= 0) == (b >= 0) else -q


def _ipow(a: int, b: int) -> int:
    if b >= 0:
        return a ** b
    return 1 if a == 1 else (1 if b % 2 == 0 else -1) if a == -1 else 0


def _apply(op: str, a: Operand, b: Operand) -> Operand:
    if op in ("|", "&", "=", "!=", "?=", "<", ">", "<=", ">="):
        return _boolop(op, a, b)
    kind = _newtype(a, b)
    if kind == "bool":
        raise ExprError("operation illegal for boolean operands")
    if kind == "str":
        if op != "//":
            raise ExprError("operation illegal for string operands")
        return Operand("str", a.value + b.value)
    if op == "//":
        raise ExprError("concatenation of a nonstring operand")
    if kind == "int":
        x, y = a.value, b.value
        value = {"+": lambda: x + y, "-": lambda: x - y, "*": lambda: x * y,
                 "/": lambda: _idiv(x, y), "**": lambda: _ipow(x, y)}[op]()
        return Operand("int", value)
    x, y = _r(a), _r(b)
    with np.errstate(all="ignore"):
        if op == "**":
            value = f32(x ** f32(b.value)) if b.type == "int" else f32(x ** y)
        else:
            value = {"+": x + y, "-": x - y, "*": x * y, "/": x / y}[op]
    return Operand("real", f32(value))


def _boolop(op: str, a: Operand, b: Operand) -> Operand:
    kind = _newtype(a, b)
    if op in ("|", "&") and kind != "bool":
        raise ExprError("AND or OR of nonlogical")
    if op in ("<", ">", "<=", ">=") and kind == "bool":
        raise ExprError("order comparison of a boolean operand")
    if kind == "real":
        x, y = _r(a), _r(b)
    else:
        x, y = a.value, b.value
    if op == "|":
        result = bool(x) or bool(y)
    elif op == "&":
        result = bool(x) and bool(y)
    elif op in ("<", ">=",):
        result = (x < y) != (op == ">=")
    elif op in (">", "<="):
        result = (x > y) != (op == "<=")
    elif op == "?=" and kind == "str":
        result = patmatch(x, y.replace("*", "?*")) > 0
    else:
        result = (x == y) != (op == "!=")
    return Operand("bool", bool(result))


# ------------------------------------------------------------ functions


FUNCTIONS = ("abs", "acos", "asin", "atan", "atan2", "bool", "cos", "exp", "int", "log",
             "log10", "max", "min", "mod", "nint", "real", "sin", "sqrt", "str", "tan")


def _nint(x: float) -> int:
    """Fortran NINT: halves away from zero."""
    return int(math.floor(abs(x) + 0.5)) * (1 if x >= 0 else -1)


def call(name: str, args: list[Operand]) -> Operand:
    """xev_callfcn, for the intrinsic functions."""
    if name not in FUNCTIONS:
        raise ExprError(f"unknown function `{name}' called")
    n = len(args)
    need = 2 if name == "mod" else -1 if name in ("max", "min", "atan", "atan2") else 1
    if need > 0 and n != need:
        raise ExprError(f"function `{name}' requires {need} arguments")
    if need < 0 and n < 1:
        raise ExprError(f"function `{name}' requires at least 1 arguments")
    if name not in ("str", "bool"):
        for o in args[:2]:
            if o.type not in ("int", "real"):
                raise ExprError(f"bad argument to function `{name}'")
    first = args[0]
    rval = [f32(o.value) for o in args[:2]] if name not in ("str", "bool") else []
    dtor, rtod = (lambda x: f32(x / f32(57.2957795))), (lambda x: f32(x * f32(57.2957795)))
    real = lambda x: Operand("real", f32(x))  # noqa: E731
    with np.errstate(all="ignore"):
        if name == "abs":
            return Operand("int", abs(first.value)) if first.type == "int" \
                else real(abs(rval[0]))
        if name == "acos":
            return real(rtod(np.arccos(rval[0])))
        if name == "asin":
            return real(rtod(np.arcsin(rval[0])))
        if name == "cos":
            return real(np.cos(dtor(rval[0])))
        if name == "sin":
            return real(np.sin(dtor(rval[0])))
        if name == "tan":
            return real(np.tan(dtor(rval[0])))
        if name == "exp":
            return real(np.exp(rval[0]))
        if name == "log":
            return real(np.log(rval[0]))
        if name == "log10":
            return real(np.log10(rval[0]))
        if name == "sqrt":
            return real(np.sqrt(rval[0]))
        if name in ("atan", "atan2"):
            return real(rtod(np.arctan(rval[0]) if n == 1 else np.arctan2(rval[0], rval[1])))
    if name == "mod":
        if "real" in (args[0].type, args[1].type):
            return real(math.fmod(float(rval[0]), float(rval[1])))
        x, y = args[0].value, args[1].value
        return Operand("int", x - _idiv(x, y) * y)
    if name == "nint":
        return Operand("int", _nint(float(rval[0])))
    if name in ("max", "min"):
        if any(o.type not in ("int", "real") for o in args):
            raise ExprError(f"bad argument to function `{name}'")
        pick = max if name == "max" else min
        if all(o.type == "int" for o in args):
            return Operand("int", pick(o.value for o in args))
        return real(pick(f32(o.value) for o in args))
    if name == "bool":
        if first.type == "bool":
            return first
        if first.type == "str":
            return Operand("bool", len(first.value) != 0)
        if first.type == "int":
            return Operand("bool", first.value != 0)
        return Operand("bool", abs(f32(first.value)) > 0.001)
    if name == "int":
        return Operand("int", first.value if first.type == "int" else int(rval[0]))
    if name == "real":
        return real(rval[0])
    # str
    if first.type == "bool":
        return Operand("str", "yes" if first.value else "no")
    if first.type == "str":
        return first
    if first.type == "int":
        return Operand("str", str(first.value))
    return Operand("str", iraffmt.g(float(first.value)))


# ------------------------------------------------------------ patterns


def patmatch(text: str, pattern: str) -> int:
    """sys/fmtio/patmatch.x: where a match of the pattern ends, plus one
    (1-based), or 0 for none. ^ and $ anchor, ? is any character, # optional
    blanks, [..] a class, * the closure of what precedes, {..} ignores case,
    \\ escapes."""
    found = re.search(_regex(pattern), text)
    return found.end() + 1 if found else 0


def _regex(pattern: str) -> str:
    out: list[str] = []
    fold = False
    i, n = 0, len(pattern)
    while i < n:
        ch = pattern[i]
        if ch == "^" and i == 0:
            out.append("^")
        elif ch == "$" and i == n - 1:
            out.append("$")
        elif ch == "?":
            out.append(".")
        elif ch == "#":
            out.append("[ \\t]*")
        elif ch == "{":
            fold = True
        elif ch == "}":
            fold = False
        elif ch == "%":
            pass
        elif ch == "[":
            end = pattern.find("]", i + 1)
            if end < 0:
                raise ExprError("bad pattern: no closing ]")
            body = pattern[i + 1:end]
            negate = body.startswith("^")
            body = body[1:] if negate else body
            out.append("[" + ("^" if negate else "") + body.replace("\\", "\\\\")
                       .replace("]", "\\]") + "]")
            i = end
        elif ch == "*":
            if not out or out[-1] in ("^", "$") or out[-1].endswith("*"):
                out.append(".")
            out.append("*")
        elif ch == "\\" and i + 1 < n:
            i += 1
            out.append(re.escape(pattern[i]))
        elif fold and ch.isalpha():
            out.append(f"[{ch.lower()}{ch.upper()}]")
        else:
            out.append(re.escape(ch))
        i += 1
    return "".join(out)
