"""What RVSAO's xcsao did in each case, as xcsao.json: its standard output and
error, and the xcsao.log it wrote (without its first line, the time stamp)."""

import json
import sys
from pathlib import Path

from xc_cases import CASES

results = Path(sys.argv[1])
target = Path(__file__).with_name("xcsao.json")
# cases not run this time keep what they had
gold = json.loads(target.read_text(encoding="utf-8")) if target.exists() else {}
for case in CASES:
    work = results / case
    if not work.exists():
        continue
    log = work / "xcsao.log"
    gold[case] = {"out": (work / "out.txt").read_text(encoding="utf-8", errors="replace"),
                  "err": (work / "err.txt").read_text(encoding="utf-8", errors="replace"),
                  "log": log.read_text(encoding="utf-8") if log.exists() else None}
Path(__file__).with_name("xcsao.json").write_text(json.dumps(gold, indent=0),
                                                   encoding="utf-8")
