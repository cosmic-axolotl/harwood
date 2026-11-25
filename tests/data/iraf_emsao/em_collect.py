"""What RVSAO's emsao did in each case, as emsao.json: its standard output and
error, and the emsao.log it wrote (without its first line, the time stamp). The
cases found in <results> replace theirs in emsao.json; the others are kept."""

import json
import sys
from pathlib import Path

from em_cases import CASES

results = Path(sys.argv[1])
target = Path(__file__).with_name("emsao.json")
gold = json.loads(target.read_text(encoding="utf-8")) if target.exists() else {}
for case in CASES:
    work = results / case
    if not work.is_dir():
        continue
    log = work / "emsao.log"
    gold[case] = {"out": (work / "out.txt").read_text(encoding="utf-8"),
                  "err": (work / "err.txt").read_text(encoding="utf-8"),
                  "log": log.read_text(encoding="utf-8") if log.exists() else None}
gold = {case: gold[case] for case in CASES if case in gold}
target.write_text(json.dumps(gold, indent=0), encoding="utf-8")
