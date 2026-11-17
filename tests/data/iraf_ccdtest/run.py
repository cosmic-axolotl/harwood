"""IRAF 2.18's ccdtest, for tests/test_ccdtest_iraf.py:

    HOME=<scratch with .iraf/uparm> python3 run.py <results>       (PyRAF)
    python ccdt_collect.py <results>                                (harwood's)

Each case in a new process: mkimage's seed starts at 1."""

import os
import shutil
import subprocess
import sys
from pathlib import Path

from ccdt_cases import CASES


def one(case: str) -> None:
    from pyraf import iraf

    instrument, calls = CASES[case]
    iraf.noao(_doprint=0)
    iraf.imred(_doprint=0)
    iraf.ccdred(_doprint=0)
    iraf.ccdtest(_doprint=0)
    iraf.ccdred.instrument = instrument
    for task in ("mkimage", "artobs", "subsection"):
        iraf.unlearn(task)
    for task, args, params in calls:
        getattr(iraf, task)(*args, mode="h", **params)


if __name__ == "__main__":
    if sys.argv[1] == "--one":
        one(sys.argv[2])
        sys.exit()
    here = Path(__file__).resolve().parent
    results = Path(sys.argv[1]).resolve()
    for case in CASES:
        work = results / case
        shutil.rmtree(work, ignore_errors=True)
        work.mkdir(parents=True)
        done = subprocess.run([sys.executable, str(here / "run.py"), "--one", case],
                              cwd=work, capture_output=True, text=True, timeout=300,
                              stdin=subprocess.DEVNULL,
                              env={**os.environ, "PYTHONPATH": str(here)})
        (work / "said.txt").write_text(done.stdout + done.stderr, encoding="utf-8")
        print(case, done.returncode, flush=True)
