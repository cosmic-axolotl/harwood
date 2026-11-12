"""What IRAF's surface did in each case, as surface.json: both streams and, from
the metacode, the texts (with where they are, GKI) and the polylines -- the mesh,
and with label yes the axes and their stroked labels."""

import json
import sys
from pathlib import Path

import numpy as np
from srf_cases import CASES

LINE, TEXT, WCS = 9, 11, 26


def instructions(path):
    words = np.fromfile(path, dtype="<i2")
    i = 0
    while i < len(words):
        opcode, length = int(words[i + 1]), int(words[i + 2])
        yield opcode, words[i + 3:i + length]
        i += length


results = Path(sys.argv[1])
gold = {}
for case in CASES:
    work = results / case
    entry = {"out": (work / "out.txt").read_text(encoding="utf-8", errors="replace"),
             "err": (work / "err.txt").read_text(encoding="utf-8", errors="replace"),
             "texts": [], "lines": []}
    if (work / "vdm").exists():
        for opcode, data in instructions(work / "vdm"):
            if opcode == TEXT:
                entry["texts"].append([int(data[0]), int(data[1]),
                                       "".join(chr(c) for c in data[3:3 + int(data[2])])])
            elif opcode == LINE:
                pts = (data[1:1 + 2 * int(data[0])].astype(int) & 0xffff).reshape(-1, 2)
                entry["lines"].append(pts.tolist())
    gold[case] = entry
Path(__file__).with_name("surface.json").write_text(json.dumps(gold, separators=(",", ":")),
                                                     encoding="utf-8")
