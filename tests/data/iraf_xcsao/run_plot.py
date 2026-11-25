"""RVSAO 2.8.5's xcsao with its graph and cursor (xcplot), for tests/test_xcplot_rvsao.py:

    python xc_inputs.py <folder>                                        (harwood's)
    sh ../iraf_emsao/build_fixed.sh <rvsao folder> <fixed>              (once)
    HOME=<scratch with .iraf/uparm> python3 run_plot.py <rvsao folder> \
        <fixed>/rvsao <folder> <results> [case ...]                      (PyRAF)
    python xp_collect.py <results>                                     (harwood's)

Each case in a process of its own, on a fresh copy of the spectra (the templates
among them, tempdir ""), xcsao and contpars unlearned, the log to STDOUT only; the
graphs go to stdgraph, whose metacode PyRAF writes to g.vdm (StdoutG); the cursor is
a file, and what xcplot asks with getline is read from the case's answers, its
standard input. The cases in FIXED run on build_fixed.sh's RVSAO, whose d (in
plotspec, the spectrum's editor) does not end in a segmentation violation."""

import os
import shutil
import subprocess
import sys
from pathlib import Path

from xp_cases import CASES, FIXED


def one(rvsao: str, case: str) -> None:
    from pyraf import iraf

    spectra, params, _, _ = CASES[case]
    iraf.set(rvsao=rvsao.rstrip("/") + "/")
    iraf.task(**{"rvsao.pkg": "rvsao$rvsao.cl"})
    iraf.rvsao(Stdout="/dev/null")
    iraf.unlearn("xcsao")
    iraf.unlearn("contpars")
    params = {"displot": "yes", "curmode": "yes", **params}
    try:
        iraf.xcsao(spectra, cursor="cursor", logfiles="STDOUT", Stdout="out.txt",
                   Stderr="err.txt", StdoutG="g.vdm", **params)
    except Exception as error:  # noqa: BLE001 -- RVSAO's refusal is a result
        with open("err.txt", "a", encoding="utf-8") as fd:
            fd.write("ERROR " + str(error).splitlines()[-1] + "\n")
    iraf.flpr()


if __name__ == "__main__":
    if sys.argv[1] == "--one":
        one(sys.argv[2], sys.argv[3])
        sys.exit()
    here = Path(__file__).resolve().parent
    rvsao, fixed, fresh, results = (Path(a).resolve() for a in sys.argv[1:5])
    only = sys.argv[5:]
    for case, (_, _, cursor, answers) in CASES.items():
        if only and case not in only:
            continue
        work = results / case
        shutil.rmtree(work, ignore_errors=True)
        shutil.copytree(fresh, work)
        (work / "cursor").write_text(cursor, encoding="utf-8")
        (work / "answers").write_text(answers, encoding="utf-8")
        with open(work / "answers", encoding="utf-8") as stdin:
            done = subprocess.run([sys.executable, str(here / "run_plot.py"), "--one",
                                   str(fixed if case in FIXED else rvsao),
                                   case],
                                  cwd=work, stdin=stdin, capture_output=True, text=True,
                                  timeout=300,
                                  env={**os.environ, "PYTHONPATH": str(here)})
        if done.returncode:
            (work / "err.txt").write_text(done.stderr[-2000:], encoding="utf-8")
        shutil.rmtree(work / "pyraf", ignore_errors=True)
        print(case, done.returncode, flush=True)
