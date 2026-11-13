"""IRAF 2.18's fitcoords by hand (the igsfit graph), for tests/test_igsfit_iraf.py:

    python ../iraf_transform/tr_inputs.py <folder>                      (harwood's)
    HOME=<scratch with .iraf/uparm> python3 run.py <folder> <results>    (PyRAF)
    python ig_collect.py <results>                                    (harwood's)

Each case on a fresh copy of the inputs, the task unlearned first, graphics on
stdvdm: what it printed goes to said.txt, the metacode to case.vdm (taken from
uparm$vdm, then removed), the plot log to plotfile."""

import io
import os
import shutil
import sys
from pathlib import Path

from ig_cases import CASES
from pyraf import iraf

if __name__ == "__main__":
    iraf.noao(_doprint=0)
    iraf.twodspec(_doprint=0)
    iraf.longslit(_doprint=0)
    fresh = Path(sys.argv[1]).resolve()
    results = Path(sys.argv[2]).resolve()
    vdm = Path(os.environ["HOME"]) / ".iraf" / "uparm" / "vdm"
    only = sys.argv[3:]
    for case, (images, params, cursor, answers) in CASES.items():
        if only and case not in only:
            continue
        work = results / case
        shutil.rmtree(work, ignore_errors=True)
        shutil.copytree(fresh, work)
        iraf.chdir(str(work))
        (work / "cursor").write_text(cursor, encoding="utf-8")
        vdm.unlink(missing_ok=True)
        out, err = io.StringIO(), io.StringIO()
        try:
            iraf.unlearn("fitcoords")
            iraf.fitcoords(images, interactive="yes", graphics="stdvdm",
                           cursor=str(work / "cursor"), plotfile="plotfile", Stdout=out,
                           Stderr=err, Stdin=io.StringIO(answers), **params)
        except Exception as error:  # noqa: BLE001 -- IRAF's refusal is a result
            err.write(f"ERROR: {error}\n")
        iraf.flpr()
        (work / "said.txt").write_text(out.getvalue(), encoding="utf-8")
        (work / "err.txt").write_text(err.getvalue(), encoding="utf-8")
        if vdm.exists():
            shutil.move(vdm, work / "case.vdm")
        print(case, flush=True)
