"""IRAF 2.18's fitcoords and transform, for tests/test_transform_iraf.py:

    python tr_inputs.py <folder>                                      (harwood's)
    HOME=<scratch with .iraf/uparm> python3 run.py <folder> <results>    (PyRAF)
    python tr_collect.py <results>                                    (harwood's)

Each case on a fresh copy of the inputs, fitcoords with interactive=no; each
task unlearned before it runs. What a case printed goes to said.txt; its
database/, logfile and outputs stay in its folder."""

import io
import shutil
import sys
from pathlib import Path

from pyraf import iraf
from tr_cases import FITCOORDS, TRANSFORM, WCS


def fitcoords(images, params, text):
    params = dict(params)
    again = params.pop("_again", None)
    iraf.unlearn("fitcoords")
    iraf.fitcoords(images, interactive="no", plotfile="", Stdout=text, Stderr=text,
                   **params)
    if again is not None:
        iraf.unlearn("fitcoords")
        iraf.fitcoords(images, interactive="no", plotfile="", Stdout=text,
                       Stderr=text, **again)


if __name__ == "__main__":
    iraf.noao(_doprint=0)
    iraf.twodspec(_doprint=0)
    iraf.longslit(_doprint=0)
    fresh = Path(sys.argv[1]).resolve()
    results = Path(sys.argv[2]).resolve()
    for group, cases in (("fitcoords", FITCOORDS), ("transform", TRANSFORM)):
        for case, spec in cases.items():
            work = results / group / case
            shutil.rmtree(work, ignore_errors=True)
            shutil.copytree(fresh, work)
            iraf.chdir(str(work))
            text = io.StringIO()
            try:
                if group == "fitcoords":
                    fitcoords(spec[0], spec[1], text)
                else:
                    quiet = io.StringIO()
                    for images, params in spec[0]:
                        fitcoords(images, dict(params, logfiles=""), quiet)
                    params = dict(spec[1])
                    iraf.unlearn("transform")
                    iraf.transform(params.pop("input"), params.pop("output"),
                                   params.pop("fitnames"), Stdout=text, Stderr=text,
                                   **params)
            except Exception as error:  # noqa: BLE001 -- IRAF's refusal is a result
                text.write(f"ERROR: {error}\n")
            (work / "said.txt").write_text(text.getvalue(), encoding="utf-8")
            print(group, case, flush=True)
    for case, (section, params) in WCS.items():
        work = results / "transform" / case
        shutil.rmtree(work, ignore_errors=True)
        shutil.copytree(fresh, work)
        iraf.chdir(str(work))
        text = io.StringIO()
        quiet = io.StringIO()
        try:
            fitcoords("arc", {"logfiles": ""}, quiet)
            iraf.unlearn("transform")
            iraf.transform("gal", "w", "arc", logfiles="", Stdout=quiet, Stderr=quiet)
            source = "w"
            if section:
                iraf.imcopy(f"w{section}", "ws", verbose="no")
                source = "ws"
            iraf.unlearn("transform")
            iraf.transform(source, "o", "", Stdout=text, Stderr=text, **params)
        except Exception as error:  # noqa: BLE001 -- IRAF's refusal is a result
            text.write(f"ERROR: {error}\n")
        (work / "said.txt").write_text(text.getvalue(), encoding="utf-8")
        print("wcs", case, flush=True)
