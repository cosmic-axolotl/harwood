"""What RVSAO's emsao did in each case, as emsao.json: its standard output and
error, and the emsao.log it wrote (without its first line, the time stamp)."""

import json
import sys
from pathlib import Path

from em_cases import CASES

results = Path(sys.argv[1])
gold = {}
for case in CASES:
    work = results / case
    log = work / "emsao.log"
    gold[case] = {"out": (work / "out.txt").read_text(encoding="utf-8"),
                  "err": (work / "err.txt").read_text(encoding="utf-8"),
                  "log": log.read_text(encoding="utf-8") if log.exists() else None}
Path(__file__).with_name("emsao.json").write_text(json.dumps(gold, indent=0),
                                                   encoding="utf-8")
