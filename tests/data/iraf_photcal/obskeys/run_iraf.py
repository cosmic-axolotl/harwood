"""IRAF 2.18 on the mkobsfile and mknobsfile cases: python3 run_iraf.py < /dev/null,
from a directory holding the .mag.1 and all.lis of ../apfile and imsets.mkimsets and
apercors.all of ../obsfile. Each case writes NAME, fNAME.dat, NAME.out and NAME.err."""
import sys
from pathlib import Path

from pyraf import iraf

iraf.noao(_doprint=0)
iraf.digiphot(_doprint=0)
iraf.photcal(_doprint=0)

Path("shifts.txt").write_text("stan0051cr.fits 1.5 -2.0\nstan0052cr.fits 0.5 0.5\n",
                              encoding="utf-8")
Path("obspars.txt").write_text("data0097cr.fits B 30.0 1.2 21:30:00.0\n", encoding="utf-8")
SETS = "imsets.mkimsets"
COMMON = dict(imsets=SETS, shifts="", apercors="", obsparams="", tolerance=8.0)
CASES = {
    "mA": (iraf.mkobsfile, dict(COMMON)),
    "mB": (iraf.mkobsfile, dict(COMMON, aperture=5)),
    "mC": (iraf.mkobsfile, dict(COMMON, allfilters="yes")),
    "mD": (iraf.mkobsfile, dict(COMMON, apercors="apercors.all")),
    "mE": (iraf.mkobsfile, dict(COMMON, shifts="shifts.txt")),
    "mF": (iraf.mkobsfile, dict(COMMON, minmagerr=0.02)),
    "mG": (iraf.mkobsfile, dict(COMMON, tolerance=0.0)),
    "mH": (iraf.mkobsfile, dict(COMMON, wrap="no")),
    "mI": (iraf.mkobsfile, dict(COMMON, obsparams="obspars.txt")),
    "mJ": (iraf.mkobsfile, dict(COMMON, verbose="no")),
    "nA": (iraf.mknobsfile, dict(COMMON)),
    "nB": (iraf.mknobsfile, dict(COMMON, aperture=11, allfilters="yes")),
    "nI": (iraf.mknobsfile, dict(COMMON, obsparams="obspars.txt", apercors="apercors.all")),
}
for name, (task, kw) in CASES.items():
    if len(sys.argv) > 1 and name not in sys.argv[1:]:
        continue
    keywords = dict(kw)
    sets = keywords.pop("imsets")
    if task is iraf.mknobsfile:
        task("@all.lis", "B,V", sets, name, Stdout=f"{name}.out", Stderr=f"{name}.err",
             **keywords)
    else:
        task("@all.lis", "B,V", name, imsets=sets, Stdout=f"{name}.out", Stderr=f"{name}.err",
             **keywords)
