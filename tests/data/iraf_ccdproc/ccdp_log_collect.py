"""IRAF's log, from run_log.py's folder, as ccdp_log.json: the logfile, what verbose wrote
to the standard error, and every processing flag's text in the header of the image, the
date (cnvdate's) cut off each line."""

import json
import re
import sys
from pathlib import Path

from astropy.io import fits
from ccdp_cases import CASES

FLAGS = ("FIXPIX", "OVERSCAN", "TRIM", "ZEROCOR", "DARKCOR", "FLATCOR", "ILLUMCOR",
         "FRINGCOR", "READCOR", "SCANCOR", "CCDPROC")
#: cnvdate's: "Nov 17  9:56"
DATE = re.compile(r"[A-Z][a-z]{2} [ \d]\d [ \d]\d:\d\d ")


def undated(text: str) -> str:
    return DATE.sub("", text)


if __name__ == "__main__":
    results = Path(sys.argv[1])
    gold = {}
    for name, (image, _) in CASES.items():
        work = results / name
        header = fits.getheader(work / image)
        log = work / "logfile"
        gold[name] = {
            "logfile": undated(log.read_text(encoding="utf-8")).splitlines()
            if log.exists() else [],
            "verbose": undated((work / "err.txt").read_text(encoding="utf-8")).splitlines(),
            "flags": {key: undated(str(header[key])) for key in FLAGS if key in header}}
    (Path(__file__).parent / "ccdp_log.json").write_text(json.dumps(gold, indent=0),
                                                         encoding="utf-8")
    print(len(gold))
