"""IRAF 2.18's cosmicrays, for tests/test_cosmicrays_iraf.py.

    python crf_inputs.py <folder>                 (harwood's)
    yes yes | HOME=<scratch with .iraf/uparm> python3 run.py <folder> <results>  (with PyRAF:
                                                  answer is a query, whatever the task's mode)
    python crf_collect.py <results>               (harwood's: results to crf.json)
"""

import io
import shutil
import sys
from pathlib import Path

from crf_cases import CASES, SESSIONS
from pyraf import iraf

iraf.noao(_doprint=0)
iraf.imred(_doprint=0)
iraf.crutil(_doprint=0)
fresh = Path(sys.argv[1]).resolve()
results = Path(sys.argv[2]).resolve()
BASE = dict(crmasks="m.pl", threshold=25.0, fluxratio=2.0, npasses=5, window="5",
            interactive="no", train="no", objects="", savefile="", plotfile="",
            graphics="stdvdm", cursor="", answer="yes")


def run(name, params):
    work = results / name
    shutil.rmtree(work, ignore_errors=True)
    shutil.copytree(fresh, work)
    iraf.chdir(str(work))
    merged = {k: v for k, v in {**BASE, **params}.items() if k not in ("input", "keys")}
    if "keys" in params:
        (work / "keys.cur").write_text(params["keys"])
    text = io.StringIO()
    error = ""
    try:
        iraf.flprcache()
        iraf.cosmicrays(params.get("input", "f.fits"), "o.fits", mode="h", Stdout=text,
                        Stderr=text, **merged)
        if (work / "m.pl").exists():
            iraf.imcopy("m.pl", "m.fits", verbose="no", Stdout=io.StringIO())
    except Exception as e:  # noqa: BLE001
        error = str(e)[-200:]
    (work / "stdout.txt").write_text(text.getvalue())
    (work / "error.txt").write_text(error)
    print(name, error[-100:], flush=True)


for name, params in CASES.items():
    run(name, params)
for name, keys in SESSIONS.items():
    run("s-" + name, dict(interactive="yes", cursor="keys.cur", keys=keys))
