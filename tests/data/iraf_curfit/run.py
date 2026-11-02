"""IRAF 2.18's curfit, for tests/test_curfit_iraf.py.

curfit_images.py <folder> in harwood's python, and then in the one with PyRAF:

    python3 run.py <folder>
"""

import io
import json
import re
import sys
from pathlib import Path

from curfit_cases import CASES, LISTS, SESSIONS
from pyraf import iraf

iraf.utilities(_doprint=0)
folder = Path(sys.argv[1]).resolve()
iraf.chdir(str(folder))
for name, text in LISTS.items():
    (folder / name).write_text(text, encoding="utf-8")
if not (folder / "im2.imh").exists():
    iraf.images(_doprint=0)
    iraf.imutil(_doprint=0)
    iraf.imcopy("fits.fits[0]", "im2", verbose="no")
    iraf.imcopy("cube.fits[0]", "im3", verbose="no")


def run(kw, **more):
    out, err = io.StringIO(), io.StringIO()
    args = {k: v for k, v in kw.items() if k not in ("input", "cmds")}
    args = {k: ("yes" if v is True else "no" if v is False else v) for k, v in args.items()}
    message = ""
    try:
        iraf.flprcache()
        iraf.curfit(kw["input"], Stdout=out, Stderr=err, mode="h", **args, **more)
    except Exception as e:  # noqa: BLE001
        message = re.sub(r"\s+", " ", str(e))[-200:]
    return {
        "stdout": out.getvalue().split("\n")[:-1],
        "stderr": err.getvalue().split("\n")[:-1],
        "error": message,
    }


gold = {"cases": {}, "sessions": {}}
for name, kw in CASES.items():
    gold["cases"][name] = {"params": kw, **run(kw, interactive="no")}
SWAP = {"low_reject": "low", "high_reject": "high", "niterate": "niterate", "sample": "sample",
        "naverage": "naverage", "grow": "grow"}
for name, kw in SESSIONS.items():
    cursor = folder / "cursor"
    cursor.write_text("".join(f":{SWAP[k]} {v}\n" for k, v in kw["cmds"].items()) + "f\nq\n",
                      encoding="utf-8")
    gold["sessions"][name] = {
        "params": kw,
        **run(kw, interactive="yes", device="stdvdm", cursor=str(cursor)),
    }
(Path(__file__).parent / "curfit.json").write_text(json.dumps(gold, indent=0),
                                                   encoding="utf-8")
print(len(gold["cases"]), len(gold["sessions"]))
