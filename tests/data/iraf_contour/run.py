"""IRAF 2.18's contour, for tests/test_contour_iraf.py:

    python ct_inputs.py <folder>                                        (harwood's)
    HOME=<scratch with .iraf/uparm> python3 run.py <folder> <results>    (PyRAF)
    python ct_collect.py <results>                                     (harwood's)

Each case in a process of its own, contour unlearned, its plot to stdvdm (kept as
vdm), both streams kept."""

import os
import shutil
import subprocess
import sys
from pathlib import Path

from ct_cases import CASES


def one(case: str) -> None:
    from pyraf import iraf

    image, params = CASES[case]
    iraf.plot(_doprint=0)
    iraf.unlearn("contour")
    vdm = Path(iraf.osfn("uparm$")) / "vdm"
    vdm.unlink(missing_ok=True)
    try:
        iraf.contour(image, device="stdvdm", Stdout="out.txt", Stderr="err.txt", **params)
    except Exception as error:  # noqa: BLE001 -- IRAF's refusal is a result
        with open("err.txt", "a", encoding="utf-8") as fd:
            fd.write(f"ERROR: {error}\n")
    if vdm.exists():
        shutil.copy(vdm, "vdm")
    iraf.flpr()


if __name__ == "__main__":
    if sys.argv[1] == "--one":
        one(sys.argv[2])
        sys.exit()
    here = Path(__file__).resolve().parent
    fresh, results = Path(sys.argv[1]).resolve(), Path(sys.argv[2]).resolve()
    for case in CASES:
        work = results / case
        shutil.rmtree(work, ignore_errors=True)
        shutil.copytree(fresh, work)
        done = subprocess.run([sys.executable, str(here / "run.py"), "--one", case],
                              cwd=work, stdin=subprocess.DEVNULL, capture_output=True,
                              text=True, timeout=300,
                              env={**os.environ, "PYTHONPATH": str(here)})
        if done.returncode:
            (work / "err.txt").write_text(done.stderr[-2000:], encoding="utf-8")
        shutil.rmtree(work / "pyraf", ignore_errors=True)
        print(case, done.returncode, flush=True)
