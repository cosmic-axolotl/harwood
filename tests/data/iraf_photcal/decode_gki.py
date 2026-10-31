"""IRAF's graphics metacode (GKI, what stdvdm writes to uparm$vdm) as what the graph
tests compare: python decode_gki.py NAME.vdm [--over-lines | --everything] writes NAME.json.

A frame is a clear of the screen. A mark is drawn as five points: a plus or a cross
through its centre, a box or a diamond closed on itself. What d and u put over a
finished frame is left out, the frame after the next redraw showing where things
then stand, which is what the panels redraw at once. In mkapfile those come after
the frame's lines; in fitparams, whose title gives the number of points, after the
last whole set of them."""
import json
import re
import sys
from pathlib import Path

import numpy as np

CLEAR, LINE, TEXT, PLSET, WCS = 6, 9, 11, 15, 26
MARKS = ("plus", "cross", "diamond", "box")


def instructions(path):
    words = np.fromfile(path, dtype="<i2")
    i = 0
    while i < len(words):
        opcode, length = int(words[i + 1]), int(words[i + 2])
        yield opcode, words[i + 3:i + length]
        i += length


def mark(points):
    """The kind of mark five points are and its centre, or None."""
    if len(points) != 5:
        return None
    a, b, c, d, e = points
    if abs((a + b) / 2 - c).max() <= 1 and abs((d + e) / 2 - c).max() <= 1:
        return ("plus" if a[0] == b[0] else "cross"), c
    if abs(a[0] - b[0]) <= 1 and abs(d[1] - e[1]) <= 1 and abs(c[0] - a[0]) <= 1:
        return "plus", c            # cut by the edge of the plot, as implot's are
    if (a == e).all():
        return ("box" if a[0] == b[0] or a[1] == b[1] else "diamond"), points[:4].mean(axis=0)
    return None


def frames(path, over_lines=False):
    """`over_lines`: marks after a frame's lines are the frame's too, as implot's."""
    out, frame, style = [], None, 1
    for opcode, data in instructions(path):
        if opcode == CLEAR:
            frame = {"window": None, "text": [], "lines": [], "drawn": [],
                     **{k: [] for k in MARKS}}
            out.append(frame)
            style = 1
        elif frame is None:
            continue
        elif opcode == WCS:
            # Eight reals, each in a slot of eight bytes: window, then viewport.
            values = np.frombuffer(data[1:].tobytes()[:64], dtype="<f4")[::2]
            frame["window"] = [round(float(v), 7) for v in values]
        elif opcode == PLSET:
            style = int(data[0])
        elif opcode == TEXT:
            text = "".join(chr(c) for c in data[3:3 + int(data[2])])
            if not text.startswith("Community"):
                frame["text"].append(text)
            frame["lines"] = []     # those were the axes: their labels come after
        elif opcode == LINE:
            points = data[1:1 + 2 * int(data[0])].astype(int).reshape(-1, 2) & 0xffff
            found = mark(points)
            if found:
                centre = np.round(found[1]).astype(int).tolist()
                if found[0] in ("diamond", "box"):
                    frame[found[0]].append(centre)
                elif over_lines or not frame["lines"]:
                    frame["drawn"].append((found[0], centre))
            else:
                frame["lines"].append([style, points.tolist()])
    return [_without_what_came_after(frame) for frame in out]


def everything(path):
    """Frames of several graphs, in the order drawn: the window and viewport of each
    WCS at the end of the frame, every mark (kind, x, y, width, height) and every
    line, the axes' too."""
    out, frame, style = [], None, 1
    for opcode, data in instructions(path):
        if opcode == CLEAR:
            frame = {"windows": [], "text": [], "marks": [], "lines": []}
            out.append(frame)
            style = 1
        elif frame is None:
            continue
        elif opcode == WCS:
            values = np.frombuffer(data[1:].tobytes(), dtype="<f4")[::2]
            tables = [values[i:i + 8] for i in range(0, len(values), 12)]
            frame["windows"] = [[round(float(v), 7) for v in table] for table in tables
                                if table[0] != table[1] and table[2] != table[3]]
        elif opcode == PLSET:
            style = int(data[0])
        elif opcode == TEXT:
            text = "".join(chr(c) for c in data[3:3 + int(data[2])])
            if not text.startswith("Community"):
                frame["text"].append(text)
        elif opcode == LINE:
            points = data[1:1 + 2 * int(data[0])].astype(int).reshape(-1, 2) & 0xffff
            found = mark(points)
            if found:
                size = (points.max(axis=0) - points.min(axis=0)).tolist()
                frame["marks"].append([found[0], *np.round(found[1]).astype(int).tolist(),
                                       *size])
            else:
                frame["lines"].append([style, points.tolist()])
    return out


def _without_what_came_after(frame):
    found = re.search(r"total=(\d+)", " ".join(frame["text"]))
    drawn = frame.pop("drawn")
    if found:
        drawn = drawn[:len(drawn) // int(found[1]) * int(found[1])]
    for kind, centre in drawn:
        frame[kind].append(centre)
    return frame


if __name__ == "__main__":
    path = Path(sys.argv[1])
    read = everything(path) if "--everything" in sys.argv else frames(
        path, "--over-lines" in sys.argv)
    text = json.dumps(read, separators=(",", ":"))
    path.with_suffix(".json").write_text(text + "\n")
