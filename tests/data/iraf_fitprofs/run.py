"""IRAF 2.18's fitprofs, for tests/test_fitprofs_iraf.py: what it says, and the pixels
and header of every image it writes.

    python fpr_inputs.py <folder>                                    (harwood's)
    HOME=<scratch with .iraf/uparm> python3 run.py <folder> <results>   (with PyRAF)
    python fpr_collect.py <results>                                  (harwood's)
"""

import io
import shutil
import sys
from pathlib import Path

from fpr_cases import BEFORE, CASES
from pyraf import iraf


def run(params):
    """(what it printed, what it warned of): IRAF's STDOUT and STDERR interleave by
    their buffers, so each is kept apart; flpr ends the process, which flushes them
    before the next case."""
    params = {k: ("yes" if v else "no") if isinstance(v, bool) else v
              for k, v in params.items()}
    params.setdefault("logfile", "")
    params.setdefault("plotfile", "")
    out, err = io.StringIO(), io.StringIO()
    iraf.fitprofs.unlearn()
    try:
        iraf.fitprofs(params.pop("input"), Stdout=out, Stderr=err, **params)
    except Exception as error:  # noqa: BLE001
        err.write("ERROR " + str(error).splitlines()[-1] + "\n")
    iraf.flpr()
    return out.getvalue(), err.getvalue()


if __name__ == "__main__":
    iraf.noao(_doprint=0)
    iraf.onedspec(_doprint=0)
    fresh = Path(sys.argv[1]).resolve()
    results = Path(sys.argv[2]).resolve()
    for name, params in CASES.items():
        work = results / name
        shutil.rmtree(work, ignore_errors=True)
        shutil.copytree(fresh, work)
        iraf.chdir(str(work))
        if name in BEFORE:
            run(dict(CASES[BEFORE[name]]))
        said, warned = run(dict(params))
        (work / "said.txt").write_text(said, encoding="utf-8")
        (work / "warned.txt").write_text(warned, encoding="utf-8")
        print(name, flush=True)
