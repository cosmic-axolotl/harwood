"""What IRAF's graph did in each case, as graph.json: both streams and, from the
metacode, the frame's window, viewport and transforms (0 linear, 1 log, 2 elog),
its texts, and every polyline and polymarker in the order drawn, with the line type
and colour it was drawn in (GKI_PLSET's), how many texts came before it and its
width (x100): the curves are what comes after the last text (the title, which
glabax writes last) in width 1, the axes' last polyline being flushed late."""

import json
import sys
from pathlib import Path

import numpy as np
from gr_cases import CASES

OPEN, CLEAR, LINE, MARKER, TEXT, PLSET, WCS = 1, 6, 9, 10, 11, 15, 26


def instructions(path):
    words = np.fromfile(path, dtype="<i2")
    i = 0
    while i < len(words):
        opcode, length = int(words[i + 1]), int(words[i + 2])
        yield opcode, words[i + 3:i + length]
        i += length


def frame(path):
    out = {"window": None, "transforms": None, "text": [], "drawn": []}
    style = [1, 1]
    for opcode, data in instructions(path):
        if opcode == WCS:
            values = np.frombuffer(data[1:].tobytes(), dtype="<f4")[::2]
            ints = np.frombuffer(data[1:].tobytes(), dtype="<i4")[::2]
            out["window"] = [float(v) for v in values[:8]]
            out["transforms"] = [int(t) for t in ints[8:10]]
        elif opcode == PLSET:
            style = [int(data[0]), int(data[2]), int(data[1])]
        elif opcode == TEXT:
            text = "".join(chr(c) for c in data[3:3 + int(data[2])])
            if not text.startswith("Community"):
                out["text"].append(text)
        elif opcode in (LINE, MARKER):
            points = data[1:1 + 2 * int(data[0])].astype(int).reshape(-1, 2) & 0xffff
            out["drawn"].append(["line" if opcode == LINE else "marker", *style[:2],
                                 points.tolist(), len(out["text"]), style[2]])
    return out


results = Path(sys.argv[1])
gold = {}
for case in CASES:
    work = results / case
    entry = {"out": (work / "out.txt").read_text(encoding="utf-8"),
             "err": (work / "err.txt").read_text(encoding="utf-8")}
    if (work / "vdm").exists():
        entry["frame"] = frame(work / "vdm")
    gold[case] = entry
Path(__file__).with_name("graph.json").write_text(json.dumps(gold, separators=(",", ":")),
                                                   encoding="utf-8")
