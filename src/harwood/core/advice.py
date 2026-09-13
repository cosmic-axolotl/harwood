"""Saying that something is wrong, and what to do about it.

A warning that only says what happened leaves the reader exactly where they
were. "Only 3 of 12 features land on a catalogued line" is a fact; it is not
help until it is followed by "try a smaller --tolerance, or --prominence 2".
So every notice harwood raises carries two halves: what happened, and what to
do next. The second half is not optional -- a test enforces it.

**Colour is never the only channel.** About one man in twelve cannot reliably
separate red from green, which is precisely the pair that "red means bad,
green means good" depends on. Three things make these readable without it:

* every notice is labelled in words -- `note`, `warning`, `problem` -- so the
  severity survives being piped into a file, read by a screen reader, or
  looked at by someone who sees no colour at all;
* each level has its own symbol, so the shape differs as well as the hue;
* the colours themselves come from the Okabe-Ito palette, chosen so that each
  pair stays distinguishable under the common forms of colour blindness.
  Vermilion and orange read as different from each other under deuteranopia
  and protanopia, where pure red and pure green collapse together.

The three levels mean different things and the distinction is worth keeping:

``note``      something worth knowing. The result is fine.
``warning``   the result is usable but something about it should be checked.
``problem``   the result is probably wrong. It is still returned, because
              seeing a wrong answer and being told why beats seeing nothing.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

Level = Literal["note", "warning", "problem"]

#: Okabe-Ito, the standard colour-blind-safe palette. Never the only signal --
#: the word and the symbol carry the same meaning on their own.
COLOURS: dict[str, str] = {
    "note": "#56B4E9",       # sky blue
    "warning": "#E69F00",    # orange
    "problem": "#D55E00",    # vermilion
}

#: A second channel, so severity survives greyscale and colour blindness both.
SYMBOLS: dict[str, str] = {
    "note": "●",        # a filled circle
    "warning": "▲",     # a triangle
    "problem": "✖",     # a cross
}

ORDER: dict[str, int] = {"note": 0, "warning": 1, "problem": 2}


@dataclass
class Advice:
    """One notice: what happened, and what to do about it."""

    message: str
    fix: str = ""
    level: Level = "warning"

    @property
    def colour(self) -> str:
        return COLOURS.get(self.level, COLOURS["warning"])

    @property
    def symbol(self) -> str:
        return SYMBOLS.get(self.level, SYMBOLS["warning"])

    def plain(self) -> str:
        """The whole notice as one line of text, for a log or a file."""
        return (f"{self.level}: {self.message}"
                + (f" -- try: {self.fix}" if self.fix else ""))

    def __str__(self) -> str:
        return self.message

    def __contains__(self, text: str) -> bool:
        """`"--override" in advice` -- searching the whole notice.

        Both halves count. A suggestion moved out of the prose and into `fix`
        is still part of what the notice says, and a caller asking whether the
        advice mentions something means the advice, not one field of it.
        """
        return str(text) in self.message or str(text) in self.fix


def note(message: str, fix: str = "") -> Advice:
    """Worth knowing. The result is fine."""
    return Advice(message, fix, "note")


def warn(message: str, fix: str = "") -> Advice:
    """Usable, but something about it should be checked."""
    return Advice(message, fix, "warning")


def problem(message: str, fix: str = "") -> Advice:
    """Probably wrong, and returned anyway so it can be seen and understood."""
    return Advice(message, fix, "problem")


def as_advice(item) -> Advice:
    """Accept a bare string as well as an Advice.

    Warnings were plain strings before this existed, and a task written
    against the older shape should keep working rather than crash -- but it
    gets no suggestion, which is what the test is for.
    """
    return item if isinstance(item, Advice) else Advice(str(item))


def worst(items) -> Level | None:
    """The most severe level in a set of notices."""
    levels = [as_advice(item).level for item in items]
    return max(levels, key=lambda level: ORDER.get(level, 1)) if levels else None


def fail(message: str, fix: str = "", kind: type[Exception] = ValueError):
    """Raise an error that carries a suggestion, keeping its usual type.

    The type matters: callers catch `ValueError` and `FileNotFoundError` for
    the reasons they always did, and a new exception class would quietly break
    that. So the suggestion is attached as an attribute instead, and the
    renderer reads it off whatever was raised.
    """
    error = kind(message)
    error.fix = fix
    raise error


class HarwoodError(Exception):
    """A failure that says what to do next.

    The same two halves as `Advice`, for the case where the task cannot
    produce a result at all. A plain ValueError still works and still prints;
    it simply has no separate suggestion line.
    """

    def __init__(self, message: str, fix: str = "") -> None:
        super().__init__(message)
        self.message = message
        self.fix = fix

    def advice(self) -> Advice:
        return Advice(self.message, self.fix, "problem")
