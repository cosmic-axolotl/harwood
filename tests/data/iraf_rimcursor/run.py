"""IRAF 2.18's rimcursor, for tests/test_rimcursor_iraf.py.

rimcursor_inputs.py <folder> in harwood's python, and then in the one with PyRAF:

    python3 run.py <folder>
"""

import io
import json
import re
import sys
from pathlib import Path

from pyraf import iraf
from rimcursor_cases import BAD, CASES, CURSOR

iraf.lists(_doprint=0)
folder = Path(sys.argv[1]).resolve()
iraf.chdir(str(folder))


def run(image, cursor, **kw):
    (folder / "cur").write_text(cursor, encoding="utf-8")
    text = io.StringIO()
    message = ""
    try:
        iraf.flprcache()
        iraf.rimcursor(image, cursor="cur", Stdout=text, Stderr=text, mode="h", **kw)
    except Exception as e:  # noqa: BLE001
        message = re.sub(r"\s+", " ", str(e))[-160:]
    return {"text": text.getvalue(), "error": message}


gold = {"cases": {}, "bad": {}}
for name, kw in CASES.items():
    kw = dict(kw)
    image = kw.pop("image")
    gold["cases"][name] = {"params": {"image": image, **kw}, **run(image, CURSOR, **kw)}
for name, cursor in BAD.items():
    gold["bad"][name] = {"cursor": cursor, **run("", cursor)}
(Path(__file__).parent / "rimcursor.json").write_text(json.dumps(gold, indent=0),
                                                      encoding="utf-8")
print(len(gold["cases"]))
for k, v in {**gold["cases"], **gold["bad"]}.items():
    print(k, repr(v["text"][:220]), v["error"][-70:])
