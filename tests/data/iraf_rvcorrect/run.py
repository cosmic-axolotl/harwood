"""IRAF 2.18's rvcorrect, for tests/test_rvcorrect_iraf.py:

    python rv_inputs.py <folder>                                        (harwood's)
    HOME=<scratch with .iraf/uparm> python3 run.py <folder> <results>    (PyRAF)
    python rv_collect.py <results>                                     (harwood's)

Each case in a process of its own, on a fresh copy of the inputs, rvcorrect
unlearned; what it prints, the parameters it records and, after imupdate, the
images' cards."""

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

from rv_cases import CASES


def one(case: str) -> None:
    from pyraf import iraf

    iraf.noao(_doprint=0)
    iraf.astutil(_doprint=0)
    iraf.unlearn("rvcorrect")
    params = {"observatory": "kpno", **CASES[case]}
    try:
        iraf.rvcorrect(Stdout="out.txt", Stderr="err.txt", **params)
    except Exception as error:  # noqa: BLE001 -- IRAF's refusal is a result
        with open("err.txt", "a", encoding="utf-8") as fd:
            fd.write(f"ERROR: {error}\n")
    recorded = {}
    if "files" not in params and "images" not in params:   # only rvc_cl records them
        for k in ("hjd", "vhelio", "vlsr"):
            recorded[k] = repr(float(iraf.rvcorrect.getParam(k, native=1, prompt=0)))
    Path("params.json").write_text(json.dumps(recorded), encoding="utf-8")
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
