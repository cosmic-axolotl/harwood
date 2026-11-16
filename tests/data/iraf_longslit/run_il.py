"""IRAF 2.18's illumination by hand, for tests/test_illumination_bins_iraf.py:

    python ls_inputs.py <folder>                                        (harwood's)
    HOME=<scratch with .iraf/uparm> python3 run_il.py <folder> <results>   (PyRAF)
    python il_collect.py <results>                                      (harwood's)

Each case in a process of its own, on a fresh copy of the images, the task unlearned,
graphics on stdvdm (uparm$vdm, kept as g.vdm), the cursor a file; xt_answer reads the
standard input, which is the case's answers."""

import os
import shutil
import subprocess
import sys
from pathlib import Path

from il_cases import CASES


def one(case: str) -> None:
    from pyraf import iraf

    images, params, _, _ = CASES[case]
    iraf.noao(_doprint=0)
    iraf.twodspec(_doprint=0)
    iraf.longslit(_doprint=0)
    iraf.unlearn("illumination")
    iraf.longslit.dispaxis = 1
    params = dict(params)
    vdm = Path(iraf.osfn("uparm$")) / "vdm"
    vdm.unlink(missing_ok=True)
    try:
        iraf.illumination(images, params.pop("output", "o"), interactive="yes",
                          graphics="stdvdm", cursor="cursor", Stdout="out.txt",
                          Stderr="err.txt", **params)
    except Exception as error:  # noqa: BLE001 -- IRAF's refusal is a result
        with open("err.txt", "a", encoding="utf-8") as fd:
            fd.write("ERROR " + str(error).splitlines()[-1] + "\n")
    iraf.flpr()
    if vdm.exists():
        os.rename(vdm, "g.vdm")


if __name__ == "__main__":
    if sys.argv[1] == "--one":
        one(sys.argv[2])
        sys.exit()
    here = Path(__file__).resolve().parent
    fresh = Path(sys.argv[1]).resolve()
    results = Path(sys.argv[2]).resolve()
    only = sys.argv[3:]
    for case, (_, _, cursor, answers) in CASES.items():
        if only and case not in only:
            continue
        work = results / case
        shutil.rmtree(work, ignore_errors=True)
        shutil.copytree(fresh, work)
        (work / "cursor").write_text(cursor, encoding="utf-8")
        (work / "answers").write_text(answers, encoding="utf-8")
        with open(work / "answers", encoding="utf-8") as stdin:
            done = subprocess.run([sys.executable, str(here / "run_il.py"), "--one", case],
                                  cwd=work, stdin=stdin, capture_output=True, text=True,
                                  timeout=300, env={**os.environ, "PYTHONPATH": str(here)})
        if done.returncode:
            (work / "err.txt").write_text(done.stderr[-2000:], encoding="utf-8")
        print(case, done.returncode, flush=True)
