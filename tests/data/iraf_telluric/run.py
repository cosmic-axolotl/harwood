"""IRAF 2.18's telluric and skytweak, for tests/test_telluric_iraf.py: what each says
and warns of, and every image in the case's folder after it.

    python twk_inputs.py <folder>                                    (harwood's)
    HOME=<scratch with .iraf/uparm> python3 run.py <folder> <results>   (with PyRAF)
    python twk_collect.py <results>                                  (harwood's)
"""

import io
import shutil
import sys
from pathlib import Path

from pyraf import iraf
from twk_cases import CASES


def run(task, params):
    """(what it printed, what it warned of): kept apart, and flpr ends the process,
    which flushes them before the next case."""
    params = {k: ("yes" if v else "no") if isinstance(v, bool) else v
              for k, v in params.items()}
    out, err = io.StringIO(), io.StringIO()
    getattr(iraf, task).unlearn()
    try:
        getattr(iraf, task)(params.pop("input"), params.pop("output"), params.pop("cal"),
                            interactive="no", Stdout=out, Stderr=err, **params)
    except Exception as error:  # noqa: BLE001
        err.write("ERROR " + str(error).splitlines()[-1] + "\n")
    iraf.flpr()
    return out.getvalue(), err.getvalue()


if __name__ == "__main__":
    iraf.noao(_doprint=0)
    iraf.onedspec(_doprint=0)
    fresh = Path(sys.argv[1]).resolve()
    results = Path(sys.argv[2]).resolve()
    for name, (task, params) in CASES.items():
        work = results / name
        shutil.rmtree(work, ignore_errors=True)
        shutil.copytree(fresh, work)
        iraf.chdir(str(work))
        said, warned = run(task, dict(params))
        (work / "said.txt").write_text(said, encoding="utf-8")
        (work / "warned.txt").write_text(warned, encoding="utf-8")
        print(name, flush=True)
