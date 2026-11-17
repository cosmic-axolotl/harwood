"""IRAF 2.18's ccdproc under the ccdred package's parameters, for
tests/test_ccdproc_package_iraf.py:

    python pkg_inputs.py <folder>                   (harwood's)
    python3 run_pkg.py <folder> <results>           (PyRAF)
    python pkg_collect.py <results>                 (harwood's: results to pkg.json)

Each case in a process of its own (the cursor file and the answers, on the standard
input), in place as IRAF does it with no output list, the logfile on."""

import os
import shutil
import subprocess
import sys
from pathlib import Path

from pkg_cases import CASES
from run import BASE


def one(case: str) -> None:
    from pyraf import iraf

    images, package, params, _, _ = CASES[case]
    iraf.noao(_doprint=0)
    iraf.imred(_doprint=0)
    iraf.ccdred(_doprint=0)
    iraf.ccdred.instrument = ""
    iraf.ccdred.pixeltype = "real real"
    iraf.ccdred.verbose = "no"
    iraf.ccdred.logfile = "logfile"
    iraf.ccdred.plotfile = ""
    iraf.ccdred.backup = ""
    iraf.ccdred.ssfile = "subsets"
    iraf.ccdred.graphics = "stdvdm"
    iraf.ccdred.cursor = "cursor" if Path("cursor").stat().st_size else ""
    for key, value in package.items():
        setattr(iraf.ccdred, key, value)
    try:
        iraf.flprcache()
        iraf.ccdproc(images, mode="h", Stdout="out.txt", Stderr="err.txt",
                     **{**BASE, **params})
    except Exception as e:  # noqa: BLE001
        Path("error.txt").write_text(str(e)[-300:], encoding="utf-8")


if __name__ == "__main__":
    if sys.argv[1] == "--one":
        one(sys.argv[2])
        sys.exit()
    here = Path(__file__).resolve().parent
    fresh = Path(sys.argv[1]).resolve()
    results = Path(sys.argv[2]).resolve()
    only = sys.argv[3:]
    for case, (_, _, _, cursor, answers) in CASES.items():
        if only and case not in only:
            continue
        work = results / case
        shutil.rmtree(work, ignore_errors=True)
        shutil.copytree(fresh, work)
        (work / "cursor").write_text(cursor, encoding="utf-8")
        (work / "answers").write_text(answers, encoding="utf-8")
        with open(work / "answers", encoding="utf-8") as stdin:
            done = subprocess.run([sys.executable, str(here / "run_pkg.py"), "--one", case],
                                  cwd=work, stdin=stdin, capture_output=True, text=True,
                                  timeout=300, env={**os.environ, "PYTHONPATH": str(here)})
        if done.returncode:
            (work / "error.txt").write_text(done.stderr[-2000:], encoding="utf-8")
        print(case, done.returncode, flush=True)
