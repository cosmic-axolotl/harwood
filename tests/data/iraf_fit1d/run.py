"""IRAF 2.18's fit1d and background, for tests/test_fit1d_iraf.py:

    python f1_inputs.py <folder>                                      (harwood's)
    HOME=<scratch with .iraf/uparm> python3 run.py <folder> <results>    (PyRAF)
    python f1_collect.py <results>                                    (harwood's)

Each case on a fresh copy of the inputs, interactive=no, the task unlearned first;
what it printed goes to said.txt."""

import io
import shutil
import sys
from pathlib import Path

from f1_cases import CASES, INTERACTIVE
from pyraf import iraf

if __name__ == "__main__":
    iraf.noao(_doprint=0)
    iraf.imred(_doprint=0)
    iraf.generic(_doprint=0)
    iraf.images(_doprint=0)
    iraf.imfit(_doprint=0)
    fresh = Path(sys.argv[1]).resolve()
    results = Path(sys.argv[2]).resolve()
    for case, (task, image, output, params) in {**CASES, **INTERACTIVE}.items():
        work = results / case
        shutil.rmtree(work, ignore_errors=True)
        shutil.copytree(fresh, work)
        iraf.chdir(str(work))
        text = io.StringIO()
        params = dict(params)
        answers = params.pop("_answers", None)
        cursor = params.pop("_cursor", None)
        extra = {"interactive": "no"}
        if cursor is not None:
            (work / "cursor").write_text(cursor, encoding="utf-8")
            extra = {"interactive": "yes", "graphics": "stdvdm",
                     "cursor": str(work / "cursor"), "Stdin": io.StringIO(answers)}
        try:
            iraf.unlearn(task)
            if task == "background":
                iraf.background(image, output, Stdout=text, Stderr=text, **extra,
                                **params)
            else:
                iraf.fit1d(image, output, params.pop("type"), Stdout=text, Stderr=text,
                           **extra, **params)
        except Exception as error:  # noqa: BLE001 -- IRAF's refusal is a result
            text.write(f"ERROR: {error}\n")
        (work / "said.txt").write_text(text.getvalue(), encoding="utf-8")
        print(case, flush=True)
