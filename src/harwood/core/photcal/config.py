"""photcal's configuration file: the lexer, the grammar and the evaluator.

From photcal/parser (parser.y, prlexer.x, prtable.x, prcode.x, preval.gx,
prexit.x). A configuration declares the catalog and observation columns,
then the transformation section: fitting parameters and constants, set
equations, and equations `NAME : reference = fit`, each with optional
derivative, weight, error and plot equations.

What IRAF does that a fresh parser would not, and is kept:

  * keywords may be abbreviated to three letters, so an identifier like
    `err` or `fit` is a keyword, while functions must be spelled out;
  * identifiers are letters and digits only;
  * `**` associates to the left and unary minus binds tightest, so
    -x**2 is (-x)**2;
  * an expression is kept as the concatenation of its tokens, which is how
    it is written into the fitted parameters file: V+v1+v2*(B-V)+v3*XV;
  * a parameter's index in an equation is its order of first appearance in
    that equation's fit expression, given when the expression is read, so
    a derivative or weight equation that names it uses the index of the
    last equation that did;
  * an equation takes the initial values its parameters have where it is
    written; a `fit` after it does not change them;
  * evaluation stops at the first undefined value -- a variable, a
    division by zero, the log of a non-positive number -- and the whole
    expression is undefined; 0 ** x is 0.

INDEF is NaN here.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from ..evexpr import DEC, REAL, lexnum
from ..iraffmt import ctod

f32 = np.float32
INDEF = f32(np.nan)
DEF_PFITDELTA = f32(0.1)
ABBREVIATE = 3
MAX_LEVELS = 6                   # pr_evs .. pr_evx: six levels of reference

KEYWORDS = ["catalog", "observations", "extinction", "transformations", "\n",
            "fit", "constant", "delta", "\n", "error", "weight", "minimum",
            "maximum", "\n", "set", "derivative", "plot"]
FUNCTIONS = ["abs", "acos", "asin", "atan", "cos", "exp", "log", "log10", "sin",
             "\n", "sqrt", "tan"]
_KEYWORD_TOKENS = {"catalog": "CATSECT", "observations": "OBSSECT",
                   "extinction": "EXTSECT", "transformations": "TRNSECT",
                   "fit": "FITID", "constant": "CONSTID", "delta": "DELTAID",
                   "error": "ERRORID", "weight": "WEIGHTID", "minimum": "MINID",
                   "maximum": "MAXID", "set": "SETID", "derivative": "DERIVID",
                   "plot": "PLOTID"}
_PUNCTUATION = {"(": "LPAR", ")": "RPAR", "+": "PLUS", "-": "MINUS", "/": "SLASH",
                "=": "EQUAL", ",": "COMMA", ":": "COLON", ";": "SEMICOLON"}


class ConfigError(ValueError):
    """The configuration could not be parsed; `messages` are IRAF's."""

    def __init__(self, messages: list[str]):
        super().__init__("\n".join(messages))
        self.messages = messages


def strdic(word: str, entries: list[str]) -> tuple[int, str]:
    """sys/fmtio/strdic.x: the 1-based entry `word` matches exactly or
    abbreviates uniquely, and the entry; 0 when none or ambiguous."""
    match, found = 0, ""
    for k, entry in enumerate(entries, start=1):
        if word and entry.startswith(word):
            if len(entry) == len(word):
                return k, entry
            if match:
                return 0, entry
            match, found = k, entry
    return match, found


def ctor(token: str) -> float | None:
    """ctor: the number a token starts with (sexagesimal and INDEF too), in
    single precision; None when it does not start with one."""
    value = ctod(token)
    if value is None:
        return None
    return float(INDEF) if value > 1e300 else float(f32(value))


# ---------------------------------------------------------------- symbols


@dataclass
class Variable:
    """An observation or catalog column. `number` is its 1-based index among
    the variables of its kind; the spare variables @E_x and @W_x hold the
    error and weight columns."""
    name: str
    kind: str                     # "obs" or "cat"
    column: int
    number: int
    spare: bool = False
    errcol: int | None = None
    wtscol: int | None = None


@dataclass
class Param:
    name: str
    kind: str                     # "fit" or "const"
    value: np.float32
    number: int
    delta: np.float32 | None = None


@dataclass
class Expr:
    text: str
    code: list

    def __bool__(self) -> bool:
        return bool(self.code)


@dataclass
class SetEquation:
    name: str
    number: int
    eq: Expr
    error: Expr | None = None
    errmin: Expr | None = None
    errmax: Expr | None = None
    weight: Expr | None = None
    wtsmin: Expr | None = None
    wtsmax: Expr | None = None


@dataclass
class Equation:
    """A transformation equation, `name : ref = fit`."""
    name: str
    number: int
    ref: Expr
    fit: Expr
    params: list[Param]
    values: np.ndarray            # PTEQSPARVAL: the initial, then fitted, values
    plist: list[int]              # 1-based indices of the fitted parameters
    refvars: list[tuple[Variable, int]]   # catalog first, then observations
    fitvars: list[tuple[Variable, int]]
    derivatives: list[Expr | None] = field(default_factory=list)
    error: Expr | None = None
    errmin: Expr | None = None
    errmax: Expr | None = None
    weight: Expr | None = None
    wtsmin: Expr | None = None
    wtsmax: Expr | None = None
    xplot: Expr | None = None
    yplot: Expr | None = None

    @property
    def nfit(self) -> int:
        return len(self.plist)


@dataclass
class Config:
    path: str
    symbols: dict = field(default_factory=dict)
    obsvars: list[Variable] = field(default_factory=list)
    catvars: list[Variable] = field(default_factory=list)
    params: list[Param] = field(default_factory=list)
    sets: list[SetEquation] = field(default_factory=list)
    equations: list[Equation] = field(default_factory=list)
    mincol: int = 1               # 2 when there is a catalog: column 1 is the id
    warnings: list[str] = field(default_factory=list)

    @property
    def nobs(self) -> int:
        return len(self.obsvars)

    @property
    def ncat(self) -> int:
        return len(self.catvars)

    @property
    def nvars(self) -> int:
        return self.nobs + self.ncat

    def column_map(self, kind: str) -> dict[int, list[int]]:
        """pr_obsmap / pr_catmap: input column -> 0-based indices of the
        variables read from it, into the evaluator's row (catalog variables
        after the observations)."""
        out: dict[int, list[int]] = {}
        for var in (self.obsvars if kind == "obs" else self.catvars):
            out.setdefault(var.column, []).append(self.index(var))
        return out

    def index(self, var: Variable) -> int:
        return var.number - 1 + (self.nobs if var.kind == "cat" else 0)

    def error_index(self, var: Variable) -> int | None:
        """pr_findmap1 of the variable's error column: the row index of the
        first variable read from that column."""
        if var.errcol is None:
            return None
        found = self.column_map(var.kind).get(var.errcol)
        return found[0] if found else None


# ------------------------------------------------------------- evaluator


def evaluate(code: list, v, p, level: int = 1) -> np.float32:
    """pr_eval: the expression's value for the variables `v` (a row,
    observations then catalog) and parameters `p`; NaN when undefined."""
    stack: list = []
    with np.errstate(all="ignore"):
        for op, arg in code:
            if op == "num":
                x = arg
            elif op == "var":
                x = f32(v[arg])
            elif op == "par":
                x = f32(p[arg])
            elif op == "set":
                x = (evaluate(arg.eq.code, v, p, level + 1) if level < MAX_LEVELS
                     else INDEF)
            elif op == "trn":
                x = (evaluate(arg.fit.code, v, arg.values, level + 1)
                     if level < MAX_LEVELS else INDEF)
            elif op == "ext":
                x = INDEF
            else:
                x = None
            if x is not None:
                stack.append(f32(x))
                if np.isnan(stack[-1]):
                    return INDEF
                continue
            if op == "u-":
                stack[-1] = -stack[-1]
            elif op == "u+":
                pass
            elif op in "+-*/" or op == "**":
                b = stack.pop()
                a = stack[-1]
                if op == "+":
                    stack[-1] = f32(a + b)
                elif op == "-":
                    stack[-1] = f32(a - b)
                elif op == "*":
                    stack[-1] = f32(a * b)
                elif op == "/":
                    if b == 0:
                        return INDEF
                    stack[-1] = f32(a / b)
                else:
                    stack[-1] = f32(a ** b) if a != 0 else f32(0.0)
            else:
                x = stack[-1]
                if ((op in ("acos", "asin") and abs(x) > 1)
                        or (op in ("log", "log10") and x <= 0)
                        or (op == "sqrt" and x < 0)):
                    return INDEF
                fn = {"abs": np.abs, "acos": np.arccos, "asin": np.arcsin,
                      "atan": np.arctan, "cos": np.cos, "exp": np.exp,
                      "log": np.log, "log10": np.log10, "sin": np.sin,
                      "sqrt": np.sqrt, "tan": np.tan}[op]
                stack[-1] = f32(fn(x))
    return stack[-1] if stack else INDEF


# ----------------------------------------------------------------- lexer


@dataclass
class Token:
    kind: str
    text: str
    line: int
    source: str                   # the line it is on, newline and all
    value: float | None = None


def _tokens(text: str) -> list[Token]:
    """pr_lexer over the whole file. A last line without its newline ends
    in the end of string, which the lexer takes for an illegal character:
    IRAF cannot parse such a file."""
    out = []
    lines = text.split("\n")
    for number, body in enumerate(lines, start=1):
        last = number == len(lines)
        if last and not body:
            break
        line = body if last else body + "\n"
        pos, n = 0, len(body)
        while True:
            if pos >= n:
                if last:
                    out.append(Token("ERR", "", number, line))
                    return out
                break
            ch = line[pos]
            if ch in " \t\r\f\v":
                pos += 1
                continue
            if ch.isalpha():
                end = pos
                while end < n and line[end].isalnum():
                    end += 1
                word = line[pos:end]
                pos = end
                k, key = strdic(word, KEYWORDS)
                if k > 0 and len(word) >= ABBREVIATE and key != "\n":
                    out.append(Token(_KEYWORD_TOKENS[key], key, number, line))
                    continue
                k, key = strdic(word, FUNCTIONS)
                if k == 0 or word != key:
                    out.append(Token("IDENTIFIER", word, number, line))
                else:
                    out.append(Token("F_" + word.upper(), word, number, line))
                continue
            if ch.isdigit() or ch == ".":
                kind, length = lexnum(line, pos)
                token = line[pos:pos + length]
                pos += length
                if kind in (DEC, REAL) and length:
                    out.append(Token("INUMBER" if kind == DEC else "RNUMBER", token,
                                     number, line, ctor(token)))
                else:
                    out.append(Token("ERR", token or ch, number, line))
                    pos += 0 if length else 1
                continue
            if ch == "*":
                if line[pos + 1:pos + 2] == "*":
                    out.append(Token("EXPON", "**", number, line))
                    pos += 2
                else:
                    out.append(Token("STAR", "*", number, line))
                    pos += 1
                continue
            if ch == "#":
                # pos = strlen (line): the newline, or the comment's last
                # character when there is none.
                if not last:
                    break
                pos = n if pos == n - 1 else n - 1
                continue
            if ch in _PUNCTUATION:
                out.append(Token(_PUNCTUATION[ch], ch, number, line))
            else:
                out.append(Token("ERR", ch, number, line))
            pos += 1
    out.append(Token("EOFILE", out[-1].text if out else "", len(lines), ""))
    return out


# ---------------------------------------------------------------- parser


class _SyntaxError(Exception):
    pass


_FUNCTION_TOKENS = {"F_" + name.upper() for name in FUNCTIONS if name != "\n"}


class _Parser:
    """parser.y as recursive descent, its actions run in the order yacc
    reduces: identifiers are checked, and parameters numbered, in the order
    they are read."""

    def __init__(self, text: str, path: str):
        self.tokens = _tokens(text)
        self.k = 0
        self.cfg = Config(path)
        self.errors: list[str] = []
        self.messages: list[str] = []      # errors and warnings as printed
        self.syntax_error = False
        self.section = None
        self._tables()

    # -- plumbing
    def _tables(self) -> None:
        """The temporary tables an equation's variables and parameters are
        counted in while it is read (TROBSTABLE ... TPARTABLE)."""
        self.trcat: dict = {}
        self.trobs: dict = {}
        self.tfcat: dict = {}
        self.tfobs: dict = {}
        self.tpar: list[Param] = []

    @property
    def tok(self) -> Token:
        return self.tokens[self.k]

    def _next(self) -> Token:
        token = self.tokens[self.k]
        self.k += 1
        return token

    def _expect(self, kind: str) -> Token:
        if self.tok.kind != kind:
            raise _SyntaxError
        return self._next()

    def _at(self, *kinds: str) -> bool:
        return self.tok.kind in kinds

    def _where(self, at: str) -> str:
        """Where pr_error says it is: the line of the last token the lexer
        read. That is the token just taken, unless the grammar had to look
        at the next one to know the rule had ended ("next") -- after any
        expression, an empty or missing section -- or the parse is over
        ("end")."""
        k = {"last": self.k - 1, "next": self.k, "end": len(self.tokens) - 1}[at]
        token = self.tokens[max(k, 0)]
        return f"near line {token.line}: {token.source}"

    def error(self, message: str, at: str = "last") -> None:
        text = f"** Error {self._where(at)}{message}"
        self.errors.append(text)
        self.messages.append(text)

    def warning(self, message: str, at: str = "last") -> None:
        text = f"** Warning {self._where(at)}{message}"
        self.cfg.warnings.append(text)
        self.messages.append(text)

    # -- the grammar
    def parse(self) -> Config:
        try:
            self._catalog()
            self._observation()
            if self._at("EXTSECT"):
                self._next()
                if self._at("SEMICOLON"):
                    self._next()
            self._transform()
            self._expect("EOFILE")
        except _SyntaxError:
            # yyerror's "syntax error" goes to the standard error; the
            # lexer has read the token it failed on.
            self.syntax_error = True
            self.error(f"Cannot continue parsing at '{self.tok.text}'", at="next")
        if not self.errors:
            self._exit()
        if self.errors:
            raise ConfigError(self.messages)
        return self.cfg

    def _columns(self, kind: str) -> int:
        count = 0
        while True:
            if self._at("IDENTIFIER"):
                name = self._next().text
                column = self._expect("INUMBER").text
                self._incol(kind, name, column)
            elif self._at("ERRORID", "WEIGHTID"):
                which = self._next().kind
                self._expect("LPAR")
                name = self._expect("IDENTIFIER").text
                self._expect("RPAR")
                column = self._expect("INUMBER").text
                self._spare_column(which == "ERRORID", name, column)
            else:
                return count
            count += 1

    def _catalog(self) -> None:
        if not self._at("CATSECT"):
            return
        self._next()
        self.cfg.mincol = 2
        empty = not self._columns("cat")
        if self._at("SEMICOLON"):
            self._next()
            if empty:
                self.warning("The catalog section is empty")
        elif empty:
            self.warning("The catalog section is empty", at="next")

    def _observation(self) -> None:
        if not self._at("OBSSECT"):
            self.warning("The observation section is undefined", at="next")
            return
        self._next()
        empty = not self._columns("obs")
        if self._at("SEMICOLON"):
            self._next()
            if empty:
                self.warning("The observation section is empty")
        elif empty:
            self.warning("The observation section is empty", at="next")

    def _transform(self) -> None:
        self._expect("TRNSECT")
        if self._at("SEMICOLON"):
            self._next()
            self.warning("The transformation section is empty")
            return
        if self._at("EOFILE"):
            self.warning("The transformation section is empty", at="next")
            return
        while not self._at("EOFILE"):
            self._statement()

    def _statement(self) -> None:
        kind = self.tok.kind
        if kind == "IDENTIFIER":
            name = self._next().text
            self._expect("COLON")
            self.section = "ref"
            ref = self._stmt()
            self._expect("EQUAL")
            self.section = "fit"
            fit = self._stmt()
            self._treq(name, ref, fit)
        elif kind == "DERIVID":
            self._next()
            self._expect("LPAR")
            name = self._expect("IDENTIFIER").text
            self._expect("COMMA")
            param = self._expect("IDENTIFIER").text
            self._expect("RPAR")
            self._expect("EQUAL")
            self.section = "der"
            self._derivative(name, param, self._stmt())
        elif kind == "PLOTID":
            self._next()
            self._expect("LPAR")
            name = self._expect("IDENTIFIER").text
            self._expect("RPAR")
            self._expect("EQUAL")
            self.section = "plot"
            x = self._stmt()
            self._expect("COMMA")
            y = self._stmt()
            self._plot(name, x, y)
        elif kind in ("FITID", "CONSTID", "DELTAID"):
            self._next()
            while True:
                name = self._expect("IDENTIFIER").text
                self._expect("EQUAL")
                sign = ""
                if self._at("PLUS", "MINUS"):
                    sign = self._next().text
                if not self._at("INUMBER", "RNUMBER"):
                    raise _SyntaxError
                value = sign + self._next().text
                if kind == "DELTAID":
                    self._delta(name, value)
                else:
                    self._param("fit" if kind == "FITID" else "const", name, value)
                if not self._at("COMMA"):
                    break
                self._next()
        elif kind == "SETID":
            self._next()
            name = self._expect("IDENTIFIER").text
            self._expect("EQUAL")
            self.section = "set"
            self._seteq(name, self._stmt())
        elif kind in ("ERRORID", "WEIGHTID"):
            self._next()
            self._expect("LPAR")
            name = self._expect("IDENTIFIER").text
            self._expect("RPAR")
            self._expect("EQUAL")
            self.section = "err" if kind == "ERRORID" else "wts"
            eq = self._stmt()
            self.section = "lmt"
            low = high = None
            if self._at("MINID"):
                self._next()
                self._expect("EQUAL")
                low = self._stmt()
            if self._at("MAXID"):
                self._next()
                self._expect("EQUAL")
                high = self._stmt()
            self._limits(kind == "ERRORID", name, eq, low, high)
        else:
            raise _SyntaxError

    # -- expressions
    def _stmt(self) -> Expr:
        code: list = []
        text = self._expr(code, 0)
        return Expr(text, code)

    _BINARY = {"PLUS": (1, "+"), "MINUS": (1, "-"), "STAR": (2, "*"),
               "SLASH": (2, "/"), "EXPON": (3, "**")}

    def _expr(self, code: list, minimum: int) -> str:
        text = self._unary(code)
        while self.tok.kind in self._BINARY:
            level, op = self._BINARY[self.tok.kind]
            if level <= minimum:
                break
            symbol = self._next().text
            text = text + symbol + self._expr(code, level)
            code.append((op, None))
        return text

    def _unary(self, code: list) -> str:
        if self._at("PLUS", "MINUS"):
            symbol = self._next().text
            text = symbol + self._unary(code)
            code.append(("u+" if symbol == "+" else "u-", None))
            return text
        if self.tok.kind in _FUNCTION_TOKENS:
            name = self._next().text
            self._expect("LPAR")
            inner = self._expr(code, 0)
            self._expect("RPAR")
            code.append((name, None))
            return f"{name}({inner})"
        if self._at("LPAR"):
            self._next()
            inner = self._expr(code, 0)
            self._expect("RPAR")
            return f"({inner})"
        if self._at("INUMBER", "RNUMBER"):
            token = self._next()
            code.append(("num", f32(token.value)))
            return token.text
        if self._at("IDENTIFIER"):
            name = self._next().text
            self._chkid(name)
            self._cgen(name, code)
            return name
        raise _SyntaxError

    def _cgen(self, name: str, code: list) -> None:
        """pr_cgen of an identifier: nothing when it is undefined."""
        sym = self.cfg.symbols.get(name)
        if isinstance(sym, Variable):
            # The catalog section comes first but equations come last, so
            # the number of observation variables is known here.
            code.append(("var", self.cfg.index(sym)))
        elif isinstance(sym, Param):
            code.append(("par", sym.number - 1))
        elif isinstance(sym, SetEquation):
            code.append(("set", sym))
        elif isinstance(sym, Equation):
            code.append(("trn", sym))

    def _chkid(self, name: str) -> None:
        """pr_chkid: is the identifier allowed here, and count it."""
        sym = self.cfg.symbols.get(name)
        if sym is None:
            self.error(f"Undefined identifier in expression [{name}]")
            return
        var = isinstance(sym, Variable)
        par = isinstance(sym, Param)
        seq = isinstance(sym, SetEquation)
        teq = isinstance(sym, Equation)
        allowed = {"set": var or seq,
                   "ref": var or seq or teq,
                   "fit": var or par or seq,
                   "der": True, "plot": True,
                   "err": var or par or seq, "wts": var or par or seq,
                   "lmt": var or par or seq}[self.section]
        if not allowed:
            what = {"set": "set", "ref": "reference", "fit": "fit",
                    "err": "error", "wts": "weight", "lmt": "min or max"}[self.section]
            self.error(f"Illegal identifier in {what} equation [{name}]")
        if self.section == "ref" and var:
            table = self.trcat if sym.kind == "cat" else self.trobs
            table[name] = (sym, table.get(name, (sym, 0))[1] + 1)
        elif self.section == "fit" and var:
            table = self.tfcat if sym.kind == "cat" else self.tfobs
            table[name] = (sym, table.get(name, (sym, 0))[1] + 1)
        elif self.section == "fit" and par and sym not in self.tpar:
            self.tpar.append(sym)
            sym.number = len(self.tpar)

    # -- actions
    def _incol(self, kind: str, name: str, column: str, spare: bool = False) -> None:
        cfg = self.cfg
        if name in cfg.symbols:
            self.error(f"Input variable [{name}] declared more than once")
            return
        col = int(column)
        if col < cfg.mincol:
            self.error(f"Column out of range or reserved for matching name [{name}]")
        table = cfg.obsvars if kind == "obs" else cfg.catvars
        var = Variable(name, kind, col, len(table) + 1, spare)
        table.append(var)
        cfg.symbols[name] = var

    def _spare_column(self, error: bool, name: str, column: str) -> None:
        sym = self.cfg.symbols.get(name)
        which = "error" if error else "weight"
        if not isinstance(sym, Variable):
            self.error(f"Attempt to define {which} column for undefined variable [{name}]")
            return
        col = int(column)
        if col < self.cfg.mincol:
            self.error(f"{which.capitalize()} column out of range or reserved for "
                       f"matching name [{name}]")
        if error:
            sym.errcol = col
        else:
            sym.wtscol = col
        self._incol(sym.kind, f"@{'E' if error else 'W'}_{name}", column, spare=True)

    def _param(self, kind: str, name: str, text: str) -> None:
        value = ctor(text)
        if value is None or np.isnan(value):
            self.error(f"Constant or fitting parameter value undefined for [{name}]")
            value = 0.0
        sym = self.cfg.symbols.get(name)
        if sym is None:
            param = Param(name, kind, f32(value), len(self.cfg.params) + 1)
            self.cfg.params.append(param)
            self.cfg.symbols[name] = param
        elif isinstance(sym, Param):
            sym.kind = kind
            sym.value = f32(value)
            self.warning(f"Constant or fitting parameter [{name}] redefined")
        else:
            self.error(f"Constant or fitting parameter [{name}] declared more than once")

    def _delta(self, name: str, text: str) -> None:
        sym = self.cfg.symbols.get(name)
        if sym is None:
            self.error(f"Attempt to define delta for undefined parameter [{name}]")
            return
        value = ctor(text)
        if value is None or np.isnan(value):
            self.error(f"Delta value undefined for parameter [{name}]")
        elif value <= 0:
            self.error(f"Delta value for parameter [{name}] must be positive")
        if isinstance(sym, Param):
            sym.delta = f32(value) if value is not None else None
        else:
            self.error(f"Attempt to define a delta for a non-parameter [{name}]")

    def _seteq(self, name: str, eq: Expr) -> None:
        if name in self.cfg.symbols:
            self.error(f"Set equation declared more than once [{name}]", at="next")
            return
        seq = SetEquation(name, len(self.cfg.sets) + 1, eq)
        self.cfg.sets.append(seq)
        self.cfg.symbols[name] = seq

    def _treq(self, name: str, ref: Expr, fit: Expr) -> None:
        if name in self.cfg.symbols:
            self.error(f"Transformation equation [{name}] declared more than once", at="next")
            return
        params = list(self.tpar)
        plist = [k for k, p in enumerate(params, start=1) if p.kind == "fit"]
        eq = Equation(
            name, len(self.cfg.equations) + 1, ref, fit, params,
            np.array([p.value for p in params], dtype=f32), plist,
            list(self.trcat.values()) + list(self.trobs.values()),
            list(self.tfcat.values()) + list(self.tfobs.values()),
            [None] * len(params))
        self.cfg.equations.append(eq)
        self.cfg.symbols[name] = eq
        self._tables()

    def _derivative(self, name: str, param: str, eq: Expr) -> None:
        symp = self.cfg.symbols.get(param)
        if symp is None:
            self.error(f"Derivative with respect of undefined parameter [{param}]", at="next")
            return
        if not isinstance(symp, Param):
            self.error(f"Derivative with respect of non-parameter [{param}]", at="next")
            return
        syme = self.cfg.symbols.get(name)
        if syme is None:
            self.error(f"Derivative of undefined equation [{name}]", at="next")
        elif not isinstance(syme, Equation):
            self.error(f"Derivative of non-transformation equation [{name}]", at="next")
        elif symp in syme.params:
            syme.derivatives[syme.params.index(symp)] = eq
        else:
            self.warning(f"Derivative with respect to unappropiate parameter [{param}]",
                         at="next")

    def _plot(self, name: str, x: Expr, y: Expr) -> None:
        sym = self.cfg.symbols.get(name)
        if isinstance(sym, Equation):
            sym.xplot, sym.yplot = x, y
        elif sym is None:
            self.error(f"Plot of undefined equation [{name}]", at="next")
        else:
            self.error(f"Plot of non-transformation equation [{name}]", at="next")

    def _limits(self, error: bool, name: str, eq: Expr, low, high) -> None:
        sym = self.cfg.symbols.get(name)
        which = "Error" if error else "Weight"
        if sym is None:
            self.error(f"{which} of undefined equation [{name}]", at="next")
        elif not isinstance(sym, Equation | SetEquation):
            self.error(f"{which} of non transformation or set equation [{name}]", at="next")
        elif error:
            sym.error, sym.errmin, sym.errmax = eq, low, high
        else:
            sym.weight, sym.wtsmin, sym.wtsmax = eq, low, high

    def _exit(self) -> None:
        """pr_exit: column checks and the default deltas. Its tables hold
        the symbols last declared first, which orders the warnings and
        decides which of two equations sharing a parameter sees the
        default delta already set."""
        for table in (self.cfg.catvars, self.cfg.obsvars):
            self._excol(table[::-1])
        for eq in self.cfg.equations[::-1]:
            for k, param in enumerate(eq.params):
                if param.delta is None:
                    param.delta = DEF_PFITDELTA
                elif eq.derivatives[k] is not None:
                    self.warning(f"Parameter delta and derivative defined for "
                                 f"[{param.name}] in equation [{eq.name}]", at="end")

    def _excol(self, table: list[Variable]) -> None:
        """pr_excol, which never checks the last row against itself."""
        for i, one in enumerate(table[:-1]):
            if one.spare:
                continue
            cols = [c for c in (one.errcol, one.wtscol) if c is not None]
            if one.column in cols or (len(cols) == 2 and cols[0] == cols[1]):
                self.warning(f"Duplicate column for input variable [{one.name}]", at="end")
            for two in table[i + 1:]:
                if two.spare:
                    continue
                shared_error = one.errcol is not None and one.errcol in (two.errcol,
                                                                         two.wtscol)
                shared_weight = one.wtscol is not None and one.wtscol in (two.wtscol,
                                                                          two.errcol)
                if one.column == two.column or shared_error or shared_weight:
                    self.warning(f"Duplicate column for input variables "
                                 f"[{one.name}] and [{two.name}]", at="end")


def parse(path: str | Path) -> Config:
    """pr_parse: the configuration file, or ConfigError with IRAF's
    messages. Warnings are in Config.warnings."""
    text = Path(path).read_text(errors="replace", encoding="utf-8")
    return _Parser(text, str(path)).parse()


def parse_text(text: str, path: str = "<config>") -> Config:
    return _Parser(text, path).parse()
