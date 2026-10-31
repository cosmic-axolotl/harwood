"""IRAF 2.18's listpixels and minmax, for tests/test_imutil_lists_iraf.py.

list_images.write(folder) in harwood's python, and then in the one with PyRAF:

    python3 run.py <folder>
"""

import io
import json
import re
import shutil
import sys
import time
from pathlib import Path

from cases import LISTPIXELS, MINMAX
from pyraf import iraf

iraf.images(_doprint=0)
iraf.imutil(_doprint=0)
fresh = Path(sys.argv[1]).resolve()
here = Path(__file__).parent.resolve()
work = fresh.parent / "work"


def header(path):
    """What minmax leaves in the header it may write."""
    raw, out = path.read_bytes(), {}
    for i in range(0, len(raw), 80):
        card = raw[i:i + 80].decode("latin1")
        if card.startswith("END"):
            break
        if card[:8].strip() in ("DATAMIN", "DATAMAX"):
            text = card[10:].split("/")[0].strip()
            out[card[:8].strip()] = None if text == "INDEF" else float(text)
    return out


def renew():
    shutil.rmtree(work, ignore_errors=True)
    shutil.copytree(fresh, work)
    iraf.chdir(str(work))


def run(task, params):
    out = io.StringIO()
    try:
        task(Stdout=out, Stderr=out, **params)
    except Exception as e:  # noqa: BLE001
        out.write("ERROR " + re.sub(r"\s+", " ", str(e))[:200] + "\n")
    return out.getvalue().split("\n")[:-1]


lists = {}
for name, params in LISTPIXELS.items():
    renew()
    lists[name] = {"params": params, "stdout": run(iraf.listpixels, dict(params))}
(here / "listpixels.json").write_text(json.dumps(lists, indent=0))

mm = {}
for name, steps in MINMAX.items():
    renew()
    done = []
    for params in steps:
        time.sleep(1.1)     # the limits are as new as the file, to the second
        lines = run(iraf.minmax, dict(params))
        first = re.split(r"[,\[]", params["images"])[0]
        file = work / first
        done.append({"params": params, "stdout": lines,
                     "header": header(file) if file.exists() else {},
                     "minval": str(iraf.minmax.minval), "maxval": str(iraf.minmax.maxval),
                     "minpix": str(iraf.minmax.minpix), "maxpix": str(iraf.minmax.maxpix)})
    mm[name] = done
(here / "minmax.json").write_text(json.dumps(mm, indent=0))
