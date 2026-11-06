"""IRAF 2.18's sfit and continuum, for tests/test_sfit_iraf.py: what each says, and
the pixels of every image in the case's folder after it, and their SFIT and SFITB.

    python sft_inputs.py <folder>                                    (harwood's)
    HOME=<scratch with .iraf/uparm> python3 run.py <folder> <results>   (with PyRAF)
    python sft_collect.py <results>                                  (harwood's)
"""

import io
import shutil
import sys
from pathlib import Path

from pyraf import iraf
from sft_cases import CASES


def run(task, params):
    params = {k: ("yes" if v else "no") if isinstance(v, bool) else v
              for k, v in params.items()}
    params.setdefault("logfiles", "STDOUT")
    text = io.StringIO()
    getattr(iraf, task).unlearn()
    try:
        getattr(iraf, task)(params.pop("input"), params.pop("output", "o"),
                            interactive="no", Stdout=text, Stderr=text, **params)
    except Exception as error:  # noqa: BLE001
        text.write("ERROR " + str(error).splitlines()[-1] + "\n")
    return text.getvalue()


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
        (work / "said.txt").write_text(run(task, dict(params)), encoding="utf-8")
        print(name, flush=True)
