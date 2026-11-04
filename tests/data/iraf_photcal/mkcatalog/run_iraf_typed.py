"""IRAF 2.18's mkcatalog, typed at the terminal: python3 run_iraf_typed.py CASE
< /dev/null, from a directory with CASE.ans (what is typed, the task's standard
input) and bin/ (the stand-in vi of ../mkimsets) first in the PATH; run C1 first,
C3 before C4. C1 makes catC1 (the columns: the end of the input that ends them
ends the records too) and C2 catC2 with verify on, which is asked after that end;
C3 adds records to catC1 retitled catC3, C4 reviews catC3's retitled catC4, C5 adds
to catC1 retitled catC5 with verify on, C6 adds to catC1 retitled catC6 and is
edited at the end; NT is a catalog whose title does not match its name and NM a
file not made by mkcatalog."""
import json
import os
import sys
from pathlib import Path

from pyraf import iraf

case = sys.argv[1]
iraf.noao(_doprint=0)
iraf.digiphot(_doprint=0)
iraf.photcal(_doprint=0)
iraf.chdir(os.getcwd())
plan = Path(f"{case}.plan")
plan.write_text(json.dumps([[["replace", "S2  ", "S2x "]]]))
os.environ["FAKE_EDIT_PLAN"] = str(plan.resolve())
kw = {"C1": dict(catalog="catC1"), "C2": dict(catalog="catC2", verify=True),
      "C3": dict(catalog="catC3"), "C4": dict(catalog="catC4", review=True),
      "C5": dict(catalog="catC5", verify=True), "C6": dict(catalog="catC6", edit=True),
      "NT": dict(catalog="catNT"), "NM": dict(catalog="catNM")}[case]
base = {"C3": "catC1", "C4": "catC3", "C5": "catC1", "C6": "catC1"}.get(case)
if base:
    Path(kw["catalog"]).write_text(Path(base).read_text().replace(
        f"CATALOG: {base}", f"CATALOG: {kw['catalog']}"))
if case == "NT":
    Path("catNT").write_text(Path("catC1").read_text().replace("CATALOG: catC1",
                                                               "CATALOG: other"))
if case == "NM":
    Path("catNM").write_text("PG2331 12.5 0.02\n")
typed = {"Stdin": f"{case}.ans"} if Path(f"{case}.ans").exists() else {}
cat = kw.pop("catalog")
iraf.mkcatalog(cat, **{"edit": False, "review": False, "verify": False, **kw},
               Stdout=f"{case}.out", Stderr=f"{case}.err", **typed)
