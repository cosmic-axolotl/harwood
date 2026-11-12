"""What IRAF's wcslab did in each case, as wcslab.json: both streams and, from the
metacode, the polylines (GKI, with their line type) and the texts (GKI, with the
attributes of the TXSET in force: up, size, horizontal and vertical justification).
"""

import json
import sys
from pathlib import Path

import numpy as np
from wcl_cases import CASES

LINE, TEXT, PLSET, TXSET = 9, 11, 15, 17
JUSTIFY = {0: "n", 1: "c", 2: "l", 3: "r", 6: "t", 7: "b"}


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
    ltype, tx = 1, [90, 100, "c", "c"]
    if (work / "vdm").exists():
        for opcode, data in instructions(work / "vdm"):
            if opcode == PLSET:
                ltype = int(data[0])
            elif opcode == TXSET:
                tx = [int(data[0]), int(data[1]), JUSTIFY.get(int(data[4]), "?"),
                      JUSTIFY.get(int(data[5]), "?")]
            elif opcode == TEXT:
                entry["texts"].append([int(data[0]), int(data[1]),
                                       "".join(chr(c) for c in data[3:3 + int(data[2])]),
                                       *tx])
            elif opcode == LINE:
                pts = (data[1:1 + 2 * int(data[0])].astype(int) & 0xffff).reshape(-1, 2)
                entry["lines"].append([ltype, pts.tolist()])
    gold[case] = entry
Path(__file__).with_name("wcslab.json").write_text(json.dumps(gold, separators=(",", ":")),
                                                    encoding="utf-8")
