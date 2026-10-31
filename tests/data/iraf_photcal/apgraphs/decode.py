"""IRAF's metacode (GKI) as what the graph tests compare: python decode.py NAME.vdm
writes NAME.json. A frame is a clear of the screen; a mark is drawn as five points
whose third is its centre. The marks d and u put over a finished frame, after its
lines, are left out: the frame after the next g shows where things then stand, and
that is what the panel redraws at once."""
import json
import sys
from pathlib import Path

import numpy as np

CLEAR, LINE, TEXT, PLSET, WCS = 6, 9, 11, 15, 26


def instructions(path):
    words = np.fromfile(path, dtype="<i2")
    i = 0
    while i < len(words):
        opcode, length = int(words[i + 1]), int(words[i + 2])
        yield opcode, words[i + 3:i + length]
        i += length


def is_mark(points):
    return (len(points) == 5 and abs((points[0] + points[1]) / 2 - points[2]).max() <= 1
            and abs((points[3] + points[4]) / 2 - points[2]).max() <= 1)


def frames(path):
    out, frame, style = [], None, 1
    for opcode, data in instructions(path):
        if opcode == CLEAR:
            frame = {"window": None, "text": [], "plus": [], "cross": [], "lines": []}
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
            frame["text"].append("".join(chr(c) for c in data[3:3 + int(data[2])]))
        elif opcode == LINE:
            points = data[1:1 + 2 * int(data[0])].astype(int).reshape(-1, 2) & 0xffff
            if is_mark(points):
                if not frame["lines"]:
                    kind = "plus" if points[0][0] == points[1][0] else "cross"
                    frame[kind].append(points[2].tolist())
            elif frame["plus"] or frame["cross"]:
                frame["lines"].append([style, points.tolist()])
    return out


if __name__ == "__main__":
    path = Path(sys.argv[1])
    keep = [{**f, "text": [t for t in f["text"] if not t.startswith("Community")]}
            for f in frames(path)]
    path.with_suffix(".json").write_text(json.dumps(keep, separators=(",", ":")) + "\n")
