"""IRAF 2.18's graph, for tests/test_plotgraph_iraf.py:

    python gr_inputs.py <folder>                                        (harwood's)
    HOME=<scratch with .iraf/uparm> python3 run.py <folder> <results>    (PyRAF)
    python gr_collect.py <results>                                     (harwood's)

Each case in a process of its own, on a fresh copy of the inputs, graph unlearned,
its plot to stdvdm (kept as vdm). A case with standard input gets it from a file
(Stdin=): graph then reads STDIN whatever its input says."""

import os
import shutil
import subprocess
import sys
from pathlib import Path

from gr_cases import CASES, LISTS


def one(case: str) -> None:
    from pyraf import iraf

    operands, params, stdin = CASES[case]
    iraf.plot(_doprint=0)
    iraf.unlearn("graph")
    vdm = Path(iraf.osfn("uparm$")) / "vdm"
    vdm.unlink(missing_ok=True)
    extra = {"Stdin": "stdin.txt"} if stdin is not None else {}
    try:
        iraf.graph(operands, device="stdvdm", Stdout="out.txt", Stderr="err.txt",
                   **extra, **params)
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
    fresh = Path(sys.argv[1]).resolve()
    results = Path(sys.argv[2]).resolve()
    only = sys.argv[3:]
    for case, (_, _, stdin) in CASES.items():
        if only and case not in only:
            continue
        work = results / case
        shutil.rmtree(work, ignore_errors=True)
        shutil.copytree(fresh, work)
        (work / "lists").write_text(LISTS, encoding="utf-8")
        if stdin is not None:
            (work / "stdin.txt").write_text(stdin, encoding="utf-8")
        done = subprocess.run([sys.executable, str(here / "run.py"), "--one", case],
                              cwd=work, stdin=subprocess.DEVNULL, capture_output=True,
                              text=True, timeout=300,
                              env={**os.environ, "PYTHONPATH": str(here)})
        if done.returncode:
            (work / "err.txt").write_text(done.stderr[-2000:], encoding="utf-8")
        shutil.rmtree(work / "pyraf", ignore_errors=True)
        print(case, done.returncode, flush=True)
