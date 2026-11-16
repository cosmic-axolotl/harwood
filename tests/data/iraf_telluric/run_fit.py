"""IRAF 2.18's telluric and skytweak by hand, for tests/test_telluric_fit_iraf.py:

    python twk_inputs.py <folder>                                          (harwood's)
    HOME=<scratch with .iraf/uparm> python3 run_fit.py <folder> <results>     (PyRAF)
    python twkfit_collect.py <results>                                     (harwood's)

Each case in a process of its own, on a fresh copy of the spectra, the task unlearned;
the graphs go to stdgraph, whose metacode PyRAF writes to g.vdm (StdoutG); the cursor
is a file; PyRAF asks 'Search interactively?' on its standard input, which is the
case's answers."""

import shutil
import subprocess
import sys
from pathlib import Path

from twkfit_cases import CASES


def one(case: str) -> None:
    from pyraf import iraf

    task, params, _, _ = CASES[case]
    iraf.noao(_doprint=0)
    iraf.onedspec(_doprint=0)
    iraf.unlearn(task)
    params = {k: ("yes" if v else "no") if isinstance(v, bool) else v
              for k, v in params.items()}
    try:
        getattr(iraf, task)(params.pop("input"), params.pop("output"), params.pop("cal"),
                            interactive="yes", cursor="cursor", Stdout="out.txt",
                            Stderr="err.txt", StdoutG="g.vdm", **params)
    except Exception as error:  # noqa: BLE001 -- IRAF's refusal is a result
        with open("err.txt", "a", encoding="utf-8") as fd:
            fd.write("ERROR " + str(error).splitlines()[-1] + "\n")
    iraf.flpr()


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
            done = subprocess.run([sys.executable, str(here / "run_fit.py"), "--one", case],
                                  cwd=work, stdin=stdin, capture_output=True, text=True,
                                  timeout=300, env={**__import__("os").environ,
                                                    "PYTHONPATH": str(here)})
        if done.returncode:
            (work / "err.txt").write_text(done.stderr[-2000:], encoding="utf-8")
        print(case, done.returncode, flush=True)
