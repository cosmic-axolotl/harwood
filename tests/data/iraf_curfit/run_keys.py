"""IRAF 2.18's curfit in sessions with keys, for tests/test_curfit_iraf.py.

curfit_images.py <folder> in harwood's python, and then in the one with PyRAF:

    HOME=<a scratch folder with .iraf/uparm> python3 run_keys.py <folder>
"""

import io
import json
import sys
from pathlib import Path

from curfit_cases import LISTS
from curfit_keys_cases import KEYS
from pyraf import iraf

iraf.utilities(_doprint=0)
folder = Path(sys.argv[1]).resolve()
iraf.chdir(str(folder))
for name, text in LISTS.items():
    (folder / name).write_text(text)
gold = {}
for name, (cursor, kw) in KEYS.items():
    (folder / "cursor").write_text(cursor)
    out, err = io.StringIO(), io.StringIO()
    args = {k: ("yes" if v is True else "no" if v is False else v) for k, v in kw.items()}
    iraf.flprcache()
    iraf.curfit("wide.txt", Stdout=out, Stderr=err, mode="h", interactive="yes",
                device="stdvdm", cursor="cursor", **args)
    gold[name] = {"stdout": out.getvalue().split("\n")[:-1],
                  "stderr": err.getvalue().split("\n")[:-1]}
(Path(__file__).parent / "curfit_keys.json").write_text(json.dumps(gold, indent=0))
print(len(gold))
