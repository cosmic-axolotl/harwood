"""IRAF's range lists, as xtools/rngranges.x reads them.

Aperture lists, line lists and the like are written the same way in every
IRAF task, and harwood reads them the same way everywhere:

    ""  or  *       everything
    3               one
    1,3,5           several (commas or blanks)
    2-6  or  2:6    from 2 to 6
    1-9x2  1:9:2    from 1 to 9 in steps of 2
    10x5            from 10 up, in steps of 5
    @file           the ranges written in a file, one or more per line
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

from . import advice

RNG_MAXINT = 2 ** 31 // 2          # MAX_INT/2, rngranges.x's top
_NUMBER = re.compile(r"[+-]?(\d+\.?\d*|\.\d+)([eEdD][+-]?\d+)?")


@dataclass
class Ranges:
    """(start, end, step) intervals; `in` tests an integer element."""

    intervals: list[tuple[float, float, float]] = field(default_factory=list)

    def nearest(self, x: float) -> float:
        """rng_nearest: x minus the nearest element."""
        best, distance = self.intervals[0][0], abs(x - self.intervals[0][0])
        for x1, x2, dx in self.intervals:
            count = int((x2 - x1) / dx + 1)
            j = max(0, min(count - 1, _nint((x - x1) / dx)))
            gap = abs((x - x1) - j * dx)
            if gap < distance:
                best, distance = x1 + j * dx, gap
        return x - best

    def __contains__(self, x: int) -> bool:          # rng_elementi
        return abs(self.nearest(float(x))) < 0.49

    @property
    def everything(self) -> bool:
        return self.intervals == [(0.0, float(RNG_MAXINT), 1.0)]


def _nint(x: float) -> int:
    return int(x + 0.5) if x >= 0 else -int(-x + 0.5)


def parse(text: str, r1: float = 0.0, r2: float = float(RNG_MAXINT),
          dr: float = 1.0) -> Ranges:
    """rng_open: the range list in `text`, within r1..r2, default step dr."""
    ranges = Ranges()
    for word in re.split(r"[\s,]+", str(text or "").strip()):
        if not word:
            continue
        if word.startswith("@"):
            path = Path(word[1:])
            if not path.exists():
                advice.fail(f"range file '{path}' does not exist",
                            "write the ranges in it, or give them directly: 1-5,8")
            for line in path.read_text().splitlines():
                for inner in re.split(r"[\s,]+", line.strip()):
                    if inner:
                        if inner.startswith("@"):
                            _error(f"Cannot nest @files ({inner})")
                        ranges.intervals.append(_one(inner, r1, r2, dr))
            continue
        ranges.intervals.append(_one(word, r1, r2, dr))
    if not ranges.intervals:
        ranges.intervals.append(_one("*", r1, r2, dr))
    return ranges


def _error(what: str):
    advice.fail(f"Range syntax error: {what}",
                "write ranges as 1,3,5 or 2-6 (2:6), with a step as 1-9x2 (1:9:2)")


def _one(word: str, r1: float, r2: float, dr: float) -> tuple[float, float, float]:
    if word.count(":") > 2:
        _error(f"Too many colons ({word})")
    text = word.replace(":", "-", 1).replace(":", "x", 1) if ":" in word else word
    if text == "*":
        return float(r1), float(r2), float(dr)
    first = _NUMBER.match(text)
    if first is None:
        _error(f"({word})")
    x1, rest = float(first.group(0).replace("d", "e").replace("D", "e")), \
        text[first.end():]
    x2, dx = x1, dr
    if rest.startswith("-"):
        second = _NUMBER.match(rest[1:])
        if second is None:
            _error(f"({word})")
        x2, rest = float(second.group(0)), rest[1 + second.end():]
        if rest.startswith("x"):
            third = _NUMBER.match(rest[1:])
            if third is None:
                _error(f"({word})")
            dx = float(third.group(0))
    elif rest.startswith("x"):
        third = _NUMBER.match(rest[1:])
        if third is None:
            _error(f"({word})")
        dx = float(third.group(0))
        x2 = min(r1, r2) if dx < 0 else max(r1, r2)
    lo, hi = min(r1, r2), max(r1, r2)
    if not (lo <= x1 <= hi and lo <= x2 <= hi):
        _error(f"Range out of bounds ({word})")
    return x1, x2, dx
