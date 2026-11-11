"""IRAF 2.18's specplot, for tests/test_specplot_iraf.py:

    python ../iraf_splot/sp_inputs.py <folder>                         (harwood's)
    HOME=<scratch with .iraf/uparm> python3 run.py <folder> <results>    (PyRAF)
    python spp_collect.py <results>                                    (harwood's)

Each case in a process of its own, on a fresh copy of the spectra, specplot
unlearned, its graphics to stdvdm (kept as vdm), the cursor from a file and its
prompts (scan) reading the case's answers on its standard input."""

import os
import shutil
import subprocess
import sys
from pathlib import Path

from spp_cases import CASES


def one(case: str) -> None:
    from pyraf import iraf

    spectra, params, _, _ = CASES[case]
    iraf.noao(_doprint=0)
    iraf.onedspec(_doprint=0)
    iraf.unlearn("specplot")
    vdm = Path(iraf.osfn("uparm$")) / "vdm"
    vdm.unlink(missing_ok=True)
    names = ",".join(f"{n}.fits" for n in spectra.split(","))
    try:
        iraf.specplot(names, graphics="stdvdm", cursor="cursor", Stdout="out.txt",
                      Stderr="err.txt", Stdin="answers", **params)
    except Exception as error:  # noqa: BLE001 -- IRAF's refusal is a result
        with open("err.txt", "a", encoding="utf-8") as fd:
            fd.write(f"ERROR: {error}\n")
    if vdm.exists():
        shutil.copy(vdm, "vdm")


if __name__ == "__main__":
    if sys.argv[1] == "--one":
        one(sys.argv[2])
        sys.exit()
    here = Path(__file__).resolve().parent
    fresh = Path(sys.argv[1]).resolve()
    results = Path(sys.argv[2]).resolve()
    for case, (_, _, cursor, answers) in CASES.items():
        work = results / case
        shutil.rmtree(work, ignore_errors=True)
        shutil.copytree(fresh, work)
        (work / "labels.txt").write_text("first one\nsecond\n", encoding="utf-8")
        (work / "cursor").write_text(cursor, encoding="utf-8")
        (work / "answers").write_text(answers, encoding="utf-8")
        with open(work / "answers", encoding="utf-8") as stdin:
            done = subprocess.run([sys.executable, str(here / "run.py"), "--one", case],
                                  cwd=work, stdin=stdin, capture_output=True, text=True,
                                  timeout=300, env={**os.environ, "PYTHONPATH": str(here)})
        if done.returncode:
            (work / "err.txt").write_text(done.stderr[-2000:], encoding="utf-8")
        print(case, done.returncode, flush=True)
