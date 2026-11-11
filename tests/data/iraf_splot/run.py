"""IRAF 2.18's splot, for tests/test_splot_iraf.py:

    python sp_inputs.py <folder>                                      (harwood's)
    HOME=<scratch with .iraf/uparm> python3 run.py <folder> <results>    (PyRAF)
    python sp_collect.py <results>                                    (harwood's)

Each case in a process of its own, on a fresh copy of the inputs (and lines.dat),
splot unlearned, its graphics to stdvdm (uparm$vdm, kept as vdm), its cursor from a
file; PyRAF asks splot's queries on its standard input, which is the case's
answers."""

import shutil
import subprocess
import sys
from pathlib import Path

from sp_cases import CASES


def one(case: str) -> None:
    from pyraf import iraf

    image, params, _, _ = CASES[case]
    iraf.noao(_doprint=0)
    iraf.onedspec(_doprint=0)
    iraf.unlearn("splot")
    params = dict(params)
    iraf.onedspec.nsum = str(params.pop("nsum", 1))
    iraf.onedspec.dispaxis = 1
    vdm = Path(iraf.osfn("uparm$")) / "vdm"
    vdm.unlink(missing_ok=True)
    try:
        iraf.splot(f"{image}.fits", graphics="stdvdm", cursor="cursor",
                   Stdout="out.txt", Stderr="err.txt", **params)
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
        shutil.copy(here / "lines.dat", work / "lines.dat")
        (work / "cursor").write_text(cursor, encoding="utf-8")
        (work / "answers").write_text(answers, encoding="utf-8")
        with open(work / "answers", encoding="utf-8") as stdin:
            done = subprocess.run([sys.executable, str(here / "run.py"), "--one", case],
                                  cwd=work, stdin=stdin, capture_output=True, text=True,
                                  timeout=300, env={**__import__("os").environ,
                                                    "PYTHONPATH": str(here)})
        if done.returncode:
            (work / "err.txt").write_text(done.stderr[-2000:], encoding="utf-8")
        print(case, done.returncode, flush=True)
