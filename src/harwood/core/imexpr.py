"""imexpr: an image from an expression, one line at a time, as IRAF does it.

The expression is evaluated once for each line of the output, and that is what
decides the shape of things: an image with fewer lines starts over when it runs
out (so a line or a plane is applied to a whole frame), the first line evaluated
says how long the lines are and what type the pixels are, and a function of a
vector, as hiv or median, is a function of one line.

From iraf-community 2.18: pkg/images/imutil/src/imexpr.gx. The expressions are
core/evvexpr.py. See LICENSES/IRAF-COPYRIGHT.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from . import advice, evexpr, evvexpr, images, iraffmt, pixels
from .evvexpr import DOUBLE, INT, LONG, REAL, SHORT, Op

USHORT = "u"
LINELEN = 32768         # DEF_LINELEN, for I before the length of a line is known
MAXDIM = 7
#: The pixels of an output of each type: IRAF's FITS kernel writes a long in 32 bits.
STORED = {USHORT: np.uint16, SHORT: np.int16, INT: np.int32, LONG: np.int32, REAL: np.float32,
          DOUBLE: np.float64}
_ORDER = {0: 0, SHORT: 3, INT: 4, LONG: 5, REAL: 6, DOUBLE: 7}
_NAME = re.compile(r"[A-Za-z_][A-Za-z0-9_$.]*")


@dataclass
class Operand:
    """A letter of the expression: an image, or the Op of a number or keyword."""

    name: str
    op: Op | None = None
    data: Any = None              # lines[n, x]; the whole of a 1D image in `op`
    ndim: int = 0
    pixtype: str = ""
    header: Any = None
    spec: Any = None
    above: tuple = ()


@dataclass
class Result:
    data: Any
    type: str
    header: Any = None
    section: Any = None
    expression: str = ""
    operands: dict = field(default_factory=dict)


def operand_names(expr: evvexpr.Expression) -> list[str]:
    """ie_getops: the single lowercase letters of the expression that are not
    the name of a function."""
    found = set()
    tokens = expr.tokens
    for k, (kind, value) in enumerate(tokens):
        if (kind == "IDENTIFIER" and len(value) == 1 and value.islower()
                and tokens[k + 1][0] != "("):
            found.add(value)
    return sorted(found)


def pixel_type(spec) -> str:
    """IM_PIXTYPE: BITPIX and the scaling on disk, which astropy hides."""
    from astropy.io import fits

    if spec.path.suffix == ".pl":
        return INT
    with fits.open(str(spec.path), memmap=False, do_not_scale_image_data=True) as hdul:
        index = images._resolve_extension(hdul, spec.extension)
        h = hdul[index].header
        bitpix = int(h.get("BITPIX", -32))
        bscale, bzero = float(h.get("BSCALE", 1.0)), float(h.get("BZERO", 0.0))
    if bitpix == -64:
        return DOUBLE
    if bitpix == -32:
        return REAL
    if bitpix == 16 and bscale == 1.0 and bzero == 32768.0:
        return USHORT
    if bscale != 1.0 or bzero != 0.0:
        return REAL
    return {8: SHORT, 16: SHORT, 32: INT, 64: LONG}.get(bitpix, REAL)


def _minimum(pixtype: str, intype: str) -> str:
    """O_TYPE: a ushort is read as an int, and intype can only raise it."""
    kind = INT if pixtype == USHORT else pixtype
    return intype if _ORDER[intype or 0] > _ORDER[kind] else kind


def _number(text: str) -> Op | None:
    """A numeric constant given as an operand: all of the text is the number."""
    kind, nchars = evexpr.lexnum(text)
    if kind == evexpr.NON or nchars != len(text):
        return None
    if kind == evexpr.REAL:
        value = evvexpr.INDEFR if text == "INDEF" else iraffmt.ctod(text)
        double = "d" in text.lower() or nchars > iraffmt.NDIGITS_RP + 3
        return evvexpr.scalar(DOUBLE if double else REAL, value)
    return evvexpr.scalar(INT, evexpr.number(text, kind).value)


def _open_image(name: str, value: str, intype: str) -> Operand:
    spec = images.parse(value)
    if not spec.path.exists():
        advice.fail(f"Cannot open image ({value})", f"check the name given to operand {name}")
    pixtype = pixel_type(spec)
    data, header = images.read(spec)
    kept = [axis for axis in range(1, data.ndim + 1)
            if not spec.section or data.shape[data.ndim - axis] > 1] or [1]
    data = data.reshape([data.shape[data.ndim - axis] for axis in reversed(kept)])
    kind = _minimum(pixtype, intype)
    typed = data.astype(evvexpr.DTYPE[kind])
    out = Operand(name, ndim=data.ndim, pixtype=pixtype, header=header, spec=spec)
    # NAXIS2, NAXIS3 ...: the axes above a line.
    out.above = tuple(reversed(data.shape[:-1]))
    if data.ndim == 1:
        out.op = Op(kind, typed)
    else:
        out.op = Op(kind, None)
        out.data = typed.reshape(-1, data.shape[-1])
    return out


def _keyword(header, name: str, reference: str) -> Op:
    """imgftype and the imget's: a real keyword is a double."""
    key = name.upper()
    if key not in header:
        raise evvexpr.ExprError(f"param {reference} not found")
    value = header[key]
    if isinstance(value, (bool, np.bool_)):
        return evvexpr.scalar(evvexpr.BOOL, value)
    if isinstance(value, (int, np.integer)):
        return evvexpr.scalar(INT, value)
    if isinstance(value, (float, np.floating)):
        return evvexpr.scalar(DOUBLE, value)
    return Op(evvexpr.CHAR, str(value).strip())


def read_macros(path: str | Path) -> dict[str, tuple[int, str]]:
    """ie_getexprdb: symbol -> (number of arguments, replacement text)."""
    file = Path(path)
    if not file.is_file():
        advice.fail(f"no expression database '{path}'", "check the name given to --exprdb")
    lines = file.read_text(encoding="utf-8").replace("'", '"').splitlines()
    macros: dict[str, tuple[int, str]] = {}
    k = 0
    while k < len(lines):
        line = lines[k]
        k += 1
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        found = _NAME.match(stripped)
        if found is None:
            continue
        name, rest = found[0], stripped[found.end():]
        rest = rest[1:] if rest.startswith(" ") else rest
        args: list[str] = []
        if rest.startswith("("):
            close = rest.index(")")
            args = [a.strip() for a in rest[1:close].split(",") if a.strip()]
            rest = rest[close + 1:]
        rest = rest.lstrip()
        if rest[:1] in (":", "="):
            rest = rest[1:]
        text = [_comment(rest)]
        while k < len(lines) and lines[k][:1] in (" ", "\t"):
            text.append(_comment(lines[k]))
            k += 1
        body = " ".join(part.strip() for part in text if part.strip())
        for n, arg in enumerate(args, 1):
            word = rf"(?<![A-Za-z0-9_$.]){re.escape(arg)}(?![A-Za-z0-9_$.])"
            body = re.sub(word, f"${n}", body)
        macros[name] = (len(args), body)
    return macros


def _comment(text: str) -> str:
    return text.split("#", 1)[0]


def expand(text: str, macros: dict[str, tuple[int, str]], depth: int = 0) -> str:
    """ie_expandtext: the macros in an expression, replaced until none is left."""
    if depth > 50:
        advice.fail("the macros of the expression database call each other forever",
                    "look for a macro that uses itself")
    out, ip = [], 0
    while ip < len(text):
        ch = text[ip]
        if ch in "'\"":
            end = text.find(ch, ip + 1)
            end = len(text) if end < 0 else end + 1
            out.append(text[ip:end])
            ip = end
            continue
        found = _NAME.match(text, ip)
        if found is None:
            out.append(ch)
            ip += 1
            continue
        name, ip = found[0], found.end()
        if name not in macros:
            out.append(name)
            continue
        nargs, body = macros[name]
        args: list[str] = []
        if nargs:
            at = ip
            while at < len(text) and text[at] in " \t":
                at += 1
            if text[at:at + 1] != "(":
                advice.fail(f"macro {name} wants {nargs} arguments",
                            f"write {name}(...) in the expression")
            args, ip = _arguments(text, at)
        for n, arg in enumerate(args, 1):
            body = body.replace(f"${n}", arg)
        out.append(expand(body, macros, depth + 1))
    return "".join(out)


_TOKEN = re.compile(r"""\"[^\"]*\"|\'[^\']*\'|[A-Za-z_][A-Za-z0-9_$.]*"""
                    r"""|\d+\.?\d*(?:[eEdD][-+]?\d+)?|\.\d+(?:[eEdD][-+]?\d+)?"""
                    r"""|&&|\|\||==|!=|<=|>=|\*\*|//|\?=|\S""")


def spaced(text: str) -> str:
    """gt_expand's text: a blank after every token."""
    return " ".join(_TOKEN.findall(text))


def _arguments(text: str, at: int) -> tuple[list[str], int]:
    level, start, args, ip = 0, at + 1, [], at
    while ip < len(text):
        ch = text[ip]
        if ch == "(":
            level += 1
        elif ch == ")":
            level -= 1
            if level == 0:
                args.append(text[start:ip].strip())
                return args, ip + 1
        elif ch == "," and level == 1:
            args.append(text[start:ip].strip())
            start = ip + 1
        ip += 1
    advice.fail("a macro call has no closing parenthesis", "check the expression")


def run(text: str, values: dict[str, str | None], output: str, dims: str = "auto",
        intype: str = "auto", outtype: str = "auto", refim: str = "auto",
        rangecheck: bool = True, exprdb: str = "none") -> Result:
    """t_imexpr, up to the pixels of the output."""
    if text.startswith("@"):
        file = Path(text[1:])
        if not file.is_file():
            advice.fail(f"no expression file '{file}'", "check the name after the @")
        text = file.read_text(encoding="utf-8").replace("\n", " ")
    if exprdb != "none" and exprdb:
        text = expand(text, read_macros(exprdb))
    expression = evvexpr.Expression(text)

    minimum = {"i": INT, "l": INT, "r": REAL, "d": DOUBLE}.get(intype[:1], 0) \
        if not intype.startswith("auto") else 0
    target = images.parse(output)
    operands: dict[str, Operand] = {}
    parameters: list[tuple[Operand, str]] = []
    for name in operand_names(expression):
        value = values.get(name)
        if value is None:
            advice.fail(f"operand {name} of the expression has no value",
                        f"give it: --{name} image.fits, or a number, or a.EXPTIME")
        value = str(value).strip()
        if value.startswith(".") and (len(value) == 1 or value[1] == "["):
            advice.fail("'.' is IRAF's last output, which harwood does not keep",
                        "name the image")
        # IRAF takes an image called x.fits for keyword fits of operand x.
        if (len(value) > 1 and value[0].islower() and value[1] == "."
                and not images.parse(value).path.exists()):
            operand = Operand(name)
            parameters.append((operand, value))
        else:
            number = _number(value)
            if number is not None:
                operand = Operand(name, op=number)
            else:
                operand = _open_image(name, value, minimum)
                if operand.spec.path.resolve() == target.path.resolve():
                    advice.fail("input and output images cannot be the same",
                                "write the result to another image")
        operands[name] = operand

    for operand, value in parameters:
        source = operands.get(value[0])
        if source is not None and source.header is not None:
            header = source.header
        else:
            other = values.get(value[0])
            if other is None or not images.parse(other).path.exists():
                raise evvexpr.ExprError(f"bad image parameter reference {value}")
            header = images.read_header(images.parse(other))
        operand.op = _keyword(header, value[2:], value)

    reference = _reference(operands, refim)
    axlen, section, existing = _shape(operands, dims, target)
    kind = _output_type(outtype, reference)
    if existing is not None:
        kind = existing
    nlines = int(np.prod(axlen[1:])) if len(axlen) > 1 else 1
    position = [1] * MAXDIM

    def getop(name: str) -> Op | None:
        if len(name) == 1 and name.islower():
            found = operands.get(name)
            if found is None:
                return None
            if found.data is not None:
                return Op(found.op.type, found.data[line % found.data.shape[0]])
            return found.op
        if len(name) > 2 and name[0].islower() and name[1] == ".":
            found = operands.get(name[0])
            if found is None or found.header is None:
                return None
            return _keyword(found.header, name[2:], name)
        if len(name) == 1 and name.isupper():
            n = axlen[0] if axlen[0] > 0 else LINELEN
            axis = ord(name) - ord("I") + 1
            if axis == 1:
                return Op(INT, np.arange(1, n + 1, dtype=np.int32))
            at = position[axis - 1] if 1 <= axis <= MAXDIM else 1
            return Op(INT, np.full(n, at, np.int32))
        return None

    stored = None
    for line in range(nlines):
        for axis in range(1, len(axlen)):
            span = int(np.prod(axlen[1:axis]))
            position[axis] = line // span % axlen[axis] + 1
        out = expression.evaluate(getop, rangecheck)
        if out.type == evvexpr.CHAR:
            advice.fail("expression type incompatible with image",
                        "an image holds numbers; this expression gives a string")
        if stored is None:
            line_type = INT if out.type == evvexpr.BOOL else out.type
            kind = kind or line_type
            if axlen[0] == 0:
                axlen[0] = max(out.length, 1)
            stored = np.zeros([nlines, axlen[0]], STORED[kind])
        if out.length:
            npix = min(out.length, axlen[0])
            stored[line, :npix] = _write(out.value[:npix], STORED[kind])
        else:
            stored[line, :] = _write(np.asarray(out.value), STORED[kind])
    shape = [axlen[axis] for axis in range(len(axlen) - 1, -1, -1)]
    return Result(stored.reshape(shape), kind,
                  None if reference is None else reference.header, section, text)


def _write(values, dtype):
    """imio's conversion on the way out: C's cast, whatever the range."""
    return np.asarray(values).astype(dtype)


def _reference(operands: dict[str, Operand], refim: str) -> Operand | None:
    if refim == "auto" or not refim:
        best = None
        for operand in operands.values():
            if operand.ndim and (best is None or operand.ndim > best.ndim):
                best = operand
        return best
    found = operands.get(refim)
    if found is None or not found.ndim:
        raise evvexpr.ExprError(f"bad wcsimage reference image {refim}")
    return found


def _shape(operands: dict[str, Operand], dims: str, target):
    """The axes of the output; axis 1 stays 0 until the first line says."""
    if target.section:
        return _shape_of_section(target)
    if dims.strip() != "auto":
        axes = [int(part) for part in re.split(r"[,\s]+", dims.strip()) if part]
        if not axes:
            advice.fail(f"cannot read --dims '{dims}'", "write it as 512,512")
        return axes, None, None
    axes = [0]
    ndim = 1
    for operand in operands.values():
        if not operand.ndim:
            continue
        ndim = max(ndim, operand.ndim)
        while len(axes) < ndim:
            axes.append(1)
        for axis, npix in enumerate(operand.above, 1):
            if npix > 1:
                axes[axis] = npix if axes[axis] <= 1 else min(axes[axis], npix)
    return axes, None, None


def _shape_of_section(target):
    if not target.path.exists():
        advice.fail(f"Cannot open image ({target.path})",
                    "writing into a section needs the image to exist already")
    kind = pixel_type(target)
    whole = images.read(images.ImageSpec(target.path, target.extension, None))[0]
    slices = pixels.parse_section(target.section, whole.shape)
    picked = whole[slices].shape
    keep = [n for n in reversed(picked) if n > 1] or [1]
    return keep, slices, kind


def _output_type(outtype: str, reference: Operand | None):
    if outtype.startswith("auto") or not outtype:
        return None
    if outtype.startswith("ref"):
        return reference.pixtype if reference is not None else None
    found = {"u": USHORT, "s": SHORT, "i": INT, "l": LONG, "r": REAL,
             "d": DOUBLE}.get(outtype[:1])
    if found is None:
        advice.fail(f"bad outtype '{outtype}'", "use ushort, short, int, long, real or double")
    return found
