"""IRAF 2.18's ccdinstrument, for tests/test_ccdinstrument_iraf.py.

ccdinst_inputs.py <folder> in harwood's python, and then in the one with PyRAF:

    python3 run.py <folder>
"""

import io
import json
import re
import shutil
import sys
from pathlib import Path

from ccdinst_cases import CASES
from pyraf import iraf

iraf.noao(_doprint=0)
iraf.imred(_doprint=0)
iraf.ccdred(_doprint=0)
fresh = Path(sys.argv[1]).resolve()
work = fresh.parent / "ccdinst_work"
gold = {}
for name, kw in CASES.items():
    kw = dict(kw)
    shutil.rmtree(work, ignore_errors=True)
    shutil.copytree(fresh, work)
    (work / "o2.dat").write_text("exptime OTHER\nsubset  'OF ME' 'a b'\n", encoding="utf-8")
    iraf.chdir(str(work))
    stdin = kw.pop("stdin", "")
    images = kw.pop("images")
    iraf.ccdred.instrument = kw.pop("instrument", "inst.dat")
    iraf.ccdred.ssfile = "subs"
    kw.setdefault("edit", True)
    kw.setdefault("parameters", "basic")
    kw["edit"] = "yes" if kw["edit"] else "no"
    text = io.StringIO()
    message = ""
    before = {p.name: p.read_bytes() for p in work.iterdir()}
    try:
        iraf.flprcache()
        iraf.ccdinstrument(images, Stdin=io.StringIO(stdin), Stdout=text, Stderr=text,
                           mode="h", **kw)
    except Exception as e:  # noqa: BLE001
        message = re.sub(r"\s+", " ", str(e))[-160:]
    files = {
        p.name: p.read_text(errors="replace", encoding="utf-8")
        for p in work.iterdir()
        if p.is_file() and before.get(p.name) != p.read_bytes() and p.suffix != ".par"
    }
    gold[name] = {"text": text.getvalue(), "files": files, "error": message}
(Path(__file__).parent / "ccdinstrument.json").write_text(json.dumps(gold, indent=0),
                                                          encoding="utf-8")
print(len(gold))
