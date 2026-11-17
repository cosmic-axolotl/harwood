"""RVSAO 2.8.5's xcsao, for tests/test_xcsao_rvsao.py:

    sh ../iraf_emsao/build.sh <rvsao folder>                            (once)
    python xc_inputs.py <folder>                                        (harwood's)
    HOME=<scratch with .iraf/uparm> python3 run.py <rvsao folder>/rvsao-2.8.5 \\
        <folder> <results>                                              (PyRAF)
    python xc_collect.py <results>                                     (harwood's)

Each case in a process of its own (x_rvsao.e keeps state between tasks), on a
fresh copy of the spectra, xcsao and contpars unlearned, the templates in the
folder (tempdir ""), displot and curmode off, the log to STDOUT and xcsao.log."""

import os
import shutil
import subprocess
import sys
from pathlib import Path

from xc_cases import CASES


def one(rvsao: str, case: str) -> None:
    from pyraf import iraf

    spectra, params, cont = CASES[case]
    iraf.set(rvsao=rvsao.rstrip("/") + "/")
    iraf.task(**{"rvsao.pkg": "rvsao$rvsao.cl"})
    iraf.rvsao(Stdout="/dev/null")
    iraf.unlearn("xcsao")
    iraf.unlearn("contpars")
    for key, value in cont.items():
        setattr(iraf.contpars, key, value)
    try:
        iraf.xcsao(spectra, displot="no", curmode="no", logfiles="STDOUT,xcsao.log",
                   Stdout="out.txt", Stderr="err.txt", **params)
    except Exception as error:  # noqa: BLE001 -- RVSAO's refusal is a result
        with open("err.txt", "a", encoding="utf-8") as fd:
            fd.write(f"ERROR: {error}\n")
    iraf.flpr()


if __name__ == "__main__":
    if sys.argv[1] == "--one":
        one(sys.argv[2], sys.argv[3])
        sys.exit()
    here = Path(__file__).resolve().parent
    rvsao, fresh, results = (Path(a).resolve() for a in sys.argv[1:4])
    only = sys.argv[4:]
    for case in CASES:
        if only and case not in only:
            continue
        work = results / case
        shutil.rmtree(work, ignore_errors=True)
        shutil.copytree(fresh, work)
        done = subprocess.run([sys.executable, str(here / "run.py"), "--one", str(rvsao),
                               case], cwd=work, stdin=subprocess.DEVNULL,
                              capture_output=True, text=True, timeout=300,
                              env={**os.environ, "PYTHONPATH": str(here)})
        if done.returncode:
            (work / "err.txt").write_text(done.stderr[-2000:], encoding="utf-8")
        print(case, done.returncode, flush=True)
