"""IRAF 2.18's imexamine, for tests/test_imexamine_keys_iraf.py.

    python iek_make.py <folder>         (harwood's: the image)
    HOME=<a scratch folder> python3 run.py <folder> <results>
                                        (the one with PyRAF; each case's text and metacode)

With a HOME of its own: the colon commands are learned, into uparm.
"""

import io
import os
import shutil
import sys
from pathlib import Path

from iek_cases import CASES
from pyraf import iraf

iraf.images(_doprint=0)
iraf.tv(_doprint=0)
fresh = Path(sys.argv[1]).resolve()
results = Path(sys.argv[2]).resolve()
vdm = Path(os.path.expanduser("~/.iraf/uparm/vdm"))
for name, cursor in CASES.items():
    work = results / name
    shutil.rmtree(work, ignore_errors=True)
    shutil.copytree(fresh, work)
    iraf.chdir(str(work))
    (work / "keys.cur").write_text(cursor, encoding="utf-8")
    # The psets stay as the last case left them
    for key, value in (("fittype", "moffat"), ("radius", 5.0), ("rplot", 8.0),
                       ("beta", "INDEF"), ("background", "yes"), ("center", "yes"),
                       ("magzero", 25.0)):
        setattr(iraf.rimexam, key, value)
    iraf.vimexam.naverage = 1
    vdm.unlink(missing_ok=True)
    text = io.StringIO()
    error = ""
    try:
        iraf.flprcache()
        iraf.imexamine("f.fits", frame=1, imagecur="keys.cur", use_display="no",
                       logfile="out.log", keeplog="yes", graphics="stdvdm", output="",
                       ncoutput=21, nloutput=21, mode="h", Stdout=text, Stderr=text)
    except Exception as e:  # noqa: BLE001
        error = str(e)[-200:]
    (work / "stdout.txt").write_text(text.getvalue(), encoding="utf-8")
    (work / "error.txt").write_text(error, encoding="utf-8")
    if vdm.exists():
        shutil.copy(vdm, work / "metacode")
    print(name, error[-100:])
vdm.unlink(missing_ok=True)
