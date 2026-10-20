"""tvmark: marking objects on a displayed frame, without the frame buffer.

IRAF's tvmark draws straight into the display's frame buffer and can save
that buffer as an image (the "snap"). harwood keeps the two halves apart:
the marks are overlays on the window, and a snapshot is made by drawing the
same marks into the frame's grey levels -- the levels `display` computes,
1 to 200 -- so `display snap.fits --ztrans none` shows it as IRAF's does.

Everything the task does goes through `Engine`, one cursor command at a
time, so the window's keys, a command file and the plain "mark them all"
run are the same code:

    keys     + x .  c r  s v b f e  - m p n l o a d  k q
    colon    :show :move N :next N M :text str :save :restore :write [name]
             and :mark :radii :lengths :color :number :label :txsize
             :pointsize :font :autolog :logfile :coords :outimage :frame

Coordinates are 1-based image pixels, as everywhere a user can see them.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from . import advice
from .display import Marker

MARKS = ("point", "circle", "rectangle", "line", "plus", "cross", "none")

#: IMTOOL's graphics colours, which tvmark's `color` numbers mean.
COLOURS = {202: "black", 203: "white", 204: "red", 205: "green", 206: "blue",
           207: "yellow", 208: "cyan", 209: "magenta", 210: "coral", 211: "maroon",
           212: "orange", 213: "khaki", 214: "orchid", 215: "turquoise", 216: "violet",
           217: "wheat"}

HELP = {
    "+": "mark the cursor position with +", "x": "mark the cursor position with x",
    ".": "mark the cursor position with a dot",
    "c": "concentric circles (radii) at the cursor",
    "r": "concentric rectangles (lengths) at the cursor",
    "s": "line segment: press at both ends", "v": "circle: centre, then a point on it",
    "b": "rectangle: two corners", "f": "filled rectangle: two corners",
    "e": "erase a region back to the saved frame: two corners",
    "-": "move to the previous object", "m": "move to the next object",
    "p": "mark the previous object", "n": "mark the next object",
    "l": "mark every object in the list", "o": "rewind the list",
    "a": "add the cursor position to the list, and mark it",
    "d": "delete the object nearest the cursor (within tolerance)",
    "k": "keep the last command in the log", "q": "quit",
}


@dataclass
class Settings:
    mark: str = "point"
    radii: list[float] = field(default_factory=lambda: [0.0])
    lengths: list[float] = field(default_factory=lambda: [0.0])
    ratio: float = 1.0
    color: int = 255
    label: bool = False
    number: bool = False
    nxoffset: int = 0
    nyoffset: int = 0
    pointsize: int = 3
    txsize: int = 1
    tolerance: float = 1.5
    font: str = "raster"
    autolog: bool = False
    logfile: str = ""
    coords: str = ""
    outimage: str = ""
    deletions: str = ""
    frame: int = 1


@dataclass
class Item:
    """One object of the coordinate list."""

    x: float
    y: float
    label: str = ""
    line: int | None = None            #: where it is in the coordinate file, 0-based


def parse_radii(text: str) -> list[float]:
    values = [float(v) for v in str(text).replace(",", " ").split()]
    return values or [0.0]


def parse_lengths(text: str) -> tuple[list[float], float]:
    """'l1,l2,...,lN ratio': the lengths, and width/length after a blank."""
    head, _, tail = str(text).strip().partition(" ")
    lengths = [float(v) for v in head.split(",") if v.strip()] or [0.0]
    ratio = float(tail.split()[0]) if tail.strip() else 1.0
    return lengths, ratio


def read_coords(path: Path | str) -> list[Item]:
    """x and y in columns 1 and 2, an optional label in column 3; lines
    starting with # and lines that are not numbers are skipped."""
    items = []
    lines = Path(path).read_text(encoding="latin-1").splitlines()
    for number, line in enumerate(lines):
        words = line.split()
        if not words or words[0].startswith("#"):
            continue
        try:
            x, y = float(words[0]), float(words[1])
        except (ValueError, IndexError):
            continue
        items.append(Item(x, y, words[2] if len(words) > 2 else "", number))
    return items


def rewrite_coords(path: Path | str, deleted: list[Item], appended: list[Item]) -> None:
    """The coordinate file without the deleted objects' lines and with the
    added ones at the end; every other line -- a header, the other
    columns of a daofind list -- as it was."""
    path = Path(path)
    lines = path.read_text(encoding="latin-1").splitlines() if path.exists() else []
    gone = {item.line for item in deleted if item.line is not None}
    kept = [line for number, line in enumerate(lines) if number not in gone]
    kept += [f"{item.x:.3f} {item.y:.3f}" for item in appended if item not in deleted]
    path.write_text("".join(line + "\n" for line in kept), encoding="latin-1")


def colour_name(color: int) -> str:
    """The overlay colour for a tvmark colour number. 255 (and anything
    that is not one of IMTOOL's) draws in the window's own mark colour."""
    return COLOURS.get(int(color), "lime")


@dataclass
class Engine:
    settings: Settings
    items: list[Item] = field(default_factory=list)
    marks: list[Marker] = field(default_factory=list)
    log: list[str] = field(default_factory=list)
    deleted: list[Item] = field(default_factory=list)
    appended: list[Item] = field(default_factory=list)
    snaps: list[tuple[str, list[Marker]]] = field(default_factory=list)  #: :write
    messages: list[str] = field(default_factory=list)
    index: int = 0                     #: the next object, 0-based
    pending: tuple[str, float, float] | None = None
    saved: list[Marker] | None = None
    last: str = ""
    done: bool = False
    erased: list[tuple[float, float, float, float]] = field(default_factory=list)

    # ------------------------------------------------------------ marking

    def _style(self) -> dict:
        return {"colour": colour_name(self.settings.color), "tag": "tvmark"}

    def mark_at(self, x: float, y: float, kind: str | None = None, text: str = "") -> None:
        """The default mark (or `kind`) at one position, with its number or
        label when those are on."""
        s, kind = self.settings, kind or self.settings.mark
        style = self._style()
        size = 3.0 * s.txsize                         # half of a 6x7 font cell
        if kind == "point":
            self.marks.append(Marker(x, y, "point", (s.pointsize | 1) / 2, **style))
        elif kind == "plus":
            self.marks.append(Marker(x, y, "plus", size, **style))
        elif kind == "cross":
            self.marks.append(Marker(x, y, "cross", size, **style))
        elif kind == "circle":
            for radius in s.radii:
                self.marks.append(Marker(x, y, "circle", max(radius, 0.5), **style))
        elif kind == "rectangle":
            for length in s.lengths:
                self.marks.append(Marker(x, y, "rectangle", max(length, 1.0) / 2,
                                         width=s.ratio, **style))
        if text:
            self.marks.append(Marker(x + s.nxoffset, y + s.nyoffset, "text",
                                     float(s.txsize), label=text, **style))

    def mark_item(self, number: int) -> None:
        """Object `number` (0-based) of the list, numbered or labelled."""
        item = self.items[number]
        s = self.settings
        text = item.label if s.label else str(number + 1) if s.number else ""
        self.mark_at(item.x, item.y, text=text)

    # ------------------------------------------------------------ commands

    def key(self, key: str, x: float, y: float) -> None:
        """One keystroke at (x, y)."""
        if key in ("s", "v", "b", "f", "e"):
            if self.pending is None or self.pending[0] != key:
                self.pending = (key, x, y)
                self.messages.append(f"{key}: now the second point")
                self._remember(f"{x:g} {y:g} {self.settings.frame} {key}")
                return
            _, x0, y0 = self.pending
            self.pending = None
            self._two(key, x0, y0, x, y)
        elif key == "+":
            self.mark_at(x, y, "plus")
        elif key == "x":
            self.mark_at(x, y, "cross")
        elif key == ".":
            self.mark_at(x, y, "point")
        elif key == "c":
            self.mark_at(x, y, "circle")
        elif key == "r":
            self.mark_at(x, y, "rectangle")
        elif key in ("-", "m"):
            if not self.items:
                self.messages.append("the coordinate list is empty")
                return
            self.index = (self.index - 2 if key == "-" else self.index) % len(self.items)
            item = self.items[self.index]
            self.messages.append(f"object {self.index + 1}: {item.x:g} {item.y:g}")
            self.index += 1
        elif key in ("n", "p"):
            if not self.items:
                self.messages.append("the coordinate list is empty")
                return
            if key == "p":
                self.index = max(0, self.index - 2)
            if self.index >= len(self.items):
                self.messages.append("end of the coordinate list")
                return
            self.mark_item(self.index)
            self.index += 1
        elif key == "l":
            for number in range(len(self.items)):
                self.mark_item(number)
            self.index = len(self.items)
        elif key == "o":
            self.index = 0
            self.messages.append("coordinate list rewound")
        elif key == "a":
            item = Item(x, y, "")
            self.items.append(item)
            self.appended.append(item)
            self.mark_item(len(self.items) - 1)
        elif key == "d":
            self._delete(x, y)
        elif key == "k":
            if self.last:
                self.log.append(self.last)
            return
        elif key == "q":
            self.done = True
            return
        elif key == "?":
            self.messages += [f"{k}  {what}" for k, what in HELP.items()]
            return
        else:
            self.messages.append(f"'{key}' is not a tvmark key; ? lists them")
            return
        self._remember(f"{x:g} {y:g} {self.settings.frame} {key}")

    def _two(self, key: str, x0: float, y0: float, x1: float, y1: float) -> None:
        style = self._style()
        if key == "s":
            self.marks.append(Marker(x0, y0, "line", 0.0, x2=x1, y2=y1, **style))
        elif key == "v":
            self.marks.append(Marker(x0, y0, "circle", math.hypot(x1 - x0, y1 - y0),
                                     **style))
        elif key in ("b", "f"):
            half_x, half_y = abs(x1 - x0) / 2, abs(y1 - y0) / 2
            self.marks.append(Marker((x0 + x1) / 2, (y0 + y1) / 2,
                                     "box" if key == "b" else "fill", max(half_x, 0.5),
                                     width=(half_y / half_x) if half_x else 1.0, **style))
        else:                                   # e: back to the saved frame there
            lo_x, hi_x, lo_y, hi_y = min(x0, x1), max(x0, x1), min(y0, y1), max(y0, y1)
            base = self.saved if self.saved is not None else []
            inside = [m for m in self.marks if lo_x <= m.x <= hi_x and lo_y <= m.y <= hi_y]
            keep = [m for m in self.marks if m not in inside]
            keep += [m for m in base if lo_x <= m.x <= hi_x and lo_y <= m.y <= hi_y]
            self.marks = keep
            self.erased.append((lo_x, hi_x, lo_y, hi_y))
        self._remember(f"{x1:g} {y1:g} {self.settings.frame} {key}")

    def _delete(self, x: float, y: float) -> None:
        if not self.items:
            self.messages.append("the coordinate list is empty")
            return
        distance = [math.hypot(i.x - x, i.y - y) for i in self.items]
        nearest = int(np.argmin(distance))
        if distance[nearest] > self.settings.tolerance:
            self.messages.append(f"no object within {self.settings.tolerance:g} pixels")
            return
        item = self.items.pop(nearest)
        self.deleted.append(item)
        self.mark_at(item.x, item.y, "cross")
        self.messages.append(f"deleted object at {item.x:g} {item.y:g}")
        if nearest < self.index:
            self.index -= 1

    def _remember(self, command: str) -> None:
        self.last = command
        if self.settings.autolog:
            self.log.append(command)

    def colon(self, text: str, x: float = 0.0, y: float = 0.0) -> str:
        """A colon command; returns what it has to say."""
        s = self.settings
        words = text.lstrip(":").split(None, 1)
        if not words:
            return ""
        name, rest = words[0], (words[1] if len(words) > 1 else "").strip()
        answer = ""
        if name == "show":
            answer = "  ".join(f"{k}={_shown(getattr(s, k))}" for k in (
                "frame", "coords", "logfile", "autolog", "outimage", "mark", "radii",
                "lengths", "font", "color", "number", "label", "txsize", "pointsize"))
        elif name == "move":
            target = int(rest) - 1
            if not 0 <= target < len(self.items):
                return f"no object {rest} in a list of {len(self.items)}"
            self.index = target + 1
            answer = f"object {target + 1}: {self.items[target].x:g} {self.items[target].y:g}"
        elif name == "next":
            parts = rest.split()
            first = int(parts[0]) - 1 if parts else self.index
            last = int(parts[1]) - 1 if len(parts) > 1 else len(self.items) - 1
            for number in range(max(0, first), min(last, len(self.items) - 1) + 1):
                self.mark_item(number)
                self.index = number + 1
        elif name == "text":
            self.marks.append(Marker(x, y, "text", float(s.txsize), label=rest.strip("'\""),
                                     **self._style()))
        elif name == "save":
            self.saved = list(self.marks)
            answer = "frame saved"
        elif name == "restore":
            if self.saved is None:
                return "nothing has been saved; :save first"
            self.marks = list(self.saved)
            answer = "frame restored"
        elif name == "write":
            self.snaps.append((rest or "", list(self.marks)))
            answer = f"snapshot {'to ' + rest if rest else 'requested'}"
        elif name in ("mark", "font", "logfile", "coords", "outimage"):
            if rest:
                if name == "mark" and rest not in MARKS:
                    return f"mark is one of {', '.join(MARKS)}"
                setattr(s, name, rest)
            answer = f"{name} = {getattr(s, name)}"
        elif name == "radii":
            if rest:
                s.radii = parse_radii(rest)
            answer = f"radii = {_shown(s.radii)}"
        elif name == "lengths":
            if rest:
                s.lengths, s.ratio = parse_lengths(rest)
            answer = f"lengths = {_shown(s.lengths)} {s.ratio:g}"
        elif name in ("color", "txsize", "pointsize", "frame"):
            if rest:
                setattr(s, name, int(float(rest)))
            answer = f"{name} = {getattr(s, name)}"
        elif name in ("number", "label", "autolog"):
            if rest:
                setattr(s, name, rest.lower() in ("yes", "y", "true", "1"))
            answer = f"{name} = {'yes' if getattr(s, name) else 'no'}"
        else:
            return f"':{name}' is not a tvmark command; :show lists the parameters"
        self._remember(f"{x:g} {y:g} {s.frame} :{name}" + (f" {rest}" if rest else ""))
        return answer


def _shown(value) -> str:
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, list):
        return ",".join(f"{v:g}" for v in value)
    return str(value)


def read_commands(path: Path | str) -> list[tuple[float, float, int, str]]:
    """A command file, or the logfile of an earlier run, in IRAF's image
    cursor format: 'x y frame key' or 'x y frame :command args' per line
    ('x y key' without the frame is read too)."""
    commands = []
    for number, line in enumerate(Path(path).read_text(encoding="latin-1").splitlines(), 1):
        words = line.split()
        if not words or words[0].startswith("#"):
            continue
        if words[0].startswith(":"):
            commands.append((0.0, 0.0, 0, line.strip()))
            continue
        try:
            x, y = float(words[0]), float(words[1])
            if len(words) >= 4:
                frame, key = int(words[2]), line.split(None, 3)[3].strip()
            else:
                frame, key = 0, words[2]
        except (ValueError, IndexError):
            advice.fail(f"{path}:{number}: '{line.strip()}' is not 'x y frame key'",
                        "write the position, the frame and the key, as the logfile does")
        commands.append((x, y, frame, key))
    return commands


# ------------------------------------------------------------ the snapshot


def snapshot(levels: np.ndarray, marks: list[Marker], color: int) -> np.ndarray:
    """The frame's grey levels with the marks drawn in at their colour
    number, as tvmark writes into the frame buffer."""
    out = np.array(levels, dtype=np.int16, copy=True)
    for mark in marks:
        value = _colour_number(mark.colour, color)
        for x, y in _pixels(mark):
            if 0 <= y < out.shape[0] and 0 <= x < out.shape[1]:
                out[y, x] = value
    return out


def _colour_number(name: str, default: int) -> int:
    for number, colour in COLOURS.items():
        if colour == name:
            return number
    return default


def _pixels(mark: Marker):
    """The 0-based pixels a mark covers."""
    cx, cy = mark.x - 1.0, mark.y - 1.0
    r = mark.radius
    if mark.shape == "point":
        half = int(r)
        for dy in range(-half, half + 1):
            for dx in range(-half, half + 1):
                yield int(round(cx)) + dx, int(round(cy)) + dy
    elif mark.shape in ("plus", "cross"):
        n = int(round(r))
        for d in range(-n, n + 1):
            if mark.shape == "plus":
                yield int(round(cx)) + d, int(round(cy))
                yield int(round(cx)), int(round(cy)) + d
            else:
                yield int(round(cx)) + d, int(round(cy)) + d
                yield int(round(cx)) + d, int(round(cy)) - d
    elif mark.shape == "circle":
        steps = max(16, int(2 * math.pi * r * 2))
        for k in range(steps):
            t = 2 * math.pi * k / steps
            yield int(round(cx + r * math.cos(t))), int(round(cy + r * math.sin(t)))
    elif mark.shape in ("rectangle", "box", "fill"):
        hx, hy = r, r * (mark.width or 1.0)
        x0, x1 = int(round(cx - hx)), int(round(cx + hx))
        y0, y1 = int(round(cy - hy)), int(round(cy + hy))
        if mark.shape == "fill":
            for y in range(y0, y1 + 1):
                for x in range(x0, x1 + 1):
                    yield x, y
        else:
            for x in range(x0, x1 + 1):
                yield x, y0
                yield x, y1
            for y in range(y0, y1 + 1):
                yield x0, y
                yield x1, y
    elif mark.shape == "line":
        x1, y1 = (mark.x2 or mark.x) - 1.0, (mark.y2 or mark.y) - 1.0
        n = int(max(abs(x1 - cx), abs(y1 - cy))) + 1
        for k in range(n + 1):
            t = k / n
            yield int(round(cx + t * (x1 - cx))), int(round(cy + t * (y1 - cy)))
    elif mark.shape == "text":
        yield from _text_pixels(mark.label, int(round(cx)), int(round(cy)),
                                max(1, int(r)))


#: A 5x7 raster font, as IRAF's tvmark uses: rows top to bottom, five bits each.
FONT = {
    "0": "0E 11 13 15 19 11 0E", "1": "04 0C 04 04 04 04 0E", "2": "0E 11 01 02 04 08 1F",
    "3": "1F 02 04 02 01 11 0E", "4": "02 06 0A 12 1F 02 02", "5": "1F 10 1E 01 01 11 0E",
    "6": "06 08 10 1E 11 11 0E", "7": "1F 01 02 04 08 08 08", "8": "0E 11 11 0E 11 11 0E",
    "9": "0E 11 11 0F 01 02 0C", "A": "0E 11 11 11 1F 11 11", "B": "1E 11 11 1E 11 11 1E",
    "C": "0E 11 10 10 10 11 0E", "D": "1C 12 11 11 11 12 1C", "E": "1F 10 10 1E 10 10 1F",
    "F": "1F 10 10 1E 10 10 10", "G": "0E 11 10 17 11 11 0F", "H": "11 11 11 1F 11 11 11",
    "I": "0E 04 04 04 04 04 0E", "J": "07 02 02 02 02 12 0C", "K": "11 12 14 18 14 12 11",
    "L": "10 10 10 10 10 10 1F", "M": "11 1B 15 15 11 11 11", "N": "11 11 19 15 13 11 11",
    "O": "0E 11 11 11 11 11 0E", "P": "1E 11 11 1E 10 10 10", "Q": "0E 11 11 11 15 12 0D",
    "R": "1E 11 11 1E 14 12 11", "S": "0F 10 10 0E 01 01 1E", "T": "1F 04 04 04 04 04 04",
    "U": "11 11 11 11 11 11 0E", "V": "11 11 11 11 11 0A 04", "W": "11 11 11 15 15 15 0A",
    "X": "11 11 0A 04 0A 11 11", "Y": "11 11 11 0A 04 04 04", "Z": "1F 01 02 04 08 10 1F",
    ".": "00 00 00 00 00 0C 0C", ",": "00 00 00 00 0C 04 08", "-": "00 00 00 1F 00 00 00",
    "+": "00 04 04 1F 04 04 00", ":": "00 0C 0C 00 0C 0C 00", "/": "00 01 02 04 08 10 00",
    "_": "00 00 00 00 00 00 1F", "(": "02 04 08 08 08 04 02", ")": "08 04 02 02 02 04 08",
    "#": "0A 0A 1F 0A 1F 0A 0A", "*": "00 04 15 0E 15 04 00", "=": "00 00 1F 00 1F 00 00",
    "'": "0C 04 08 00 00 00 00", " ": "00 00 00 00 00 00 00",
}


def _text_pixels(text: str, x0: int, y0: int, size: int):
    """Text with its lower left corner at (x0, y0): 6 x 7 pixel cells times
    txsize, rows counted up as the image's are."""
    for n, char in enumerate(text):
        rows = FONT.get(char.upper(), FONT["#"] if char.strip() else FONT[" "]).split()
        for r, row in enumerate(rows):
            bits = int(row, 16)
            for c in range(5):
                if bits & (1 << (4 - c)):
                    for sy in range(size):
                        for sx in range(size):
                            yield (x0 + (6 * n + c) * size + sx,
                                   y0 + (6 - r) * size + sy)


def settings_from(params) -> Settings:
    lengths, ratio = parse_lengths(params.lengths)
    return Settings(mark=params.mark, radii=parse_radii(params.radii), lengths=lengths,
                    ratio=ratio, color=params.color, label=params.label,
                    number=params.number, nxoffset=params.nxoffset,
                    nyoffset=params.nyoffset, pointsize=params.pointsize,
                    txsize=params.txsize, tolerance=params.tolerance, font=params.font,
                    autolog=params.autolog, logfile=params.logfile, coords=params.coords,
                    outimage=params.outimage, deletions=params.deletions,
                    frame=params.frame)

