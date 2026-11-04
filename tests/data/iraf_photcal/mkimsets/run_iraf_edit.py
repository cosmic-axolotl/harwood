"""IRAF 2.18's mkimsets with its editor sessions and its questions: python3
run_iraf_edit.py CASE < /dev/null > CASE.term, from a directory with the .mag.1 files
and all.lis of ../apfile, plans.json and, for the cases that rename, CASE.ans, with bin/
(a stand-in vi, which makes the edits plans.json gives and keeps what it was shown as
CASE.plan.seen<n>, kept here as CASE.seen<n>) first in the PATH. The rename questions go
to the terminal, CASE.term, not to the task's Stdout."""
import json
import os
import sys
from pathlib import Path

from pyraf import iraf

case = sys.argv[1]
plans = json.loads(Path("plans.json").read_text())
Path(f"{case}.plan").write_text(json.dumps(plans[case]))
os.environ["FAKE_EDIT_PLAN"] = str(Path(f"{case}.plan").resolve())
iraf.noao(_doprint=0)
iraf.digiphot(_doprint=0)
iraf.photcal(_doprint=0)
iraf.chdir(os.getcwd())
kw = {"R1": dict(edit=False, rename=True, review=False),
      "E1": dict(edit=True, rename=False, review=False, imobsparams="obsE1"),
      "V1": dict(edit=False, rename=False, review=True),
      "EV": dict(edit=True, rename=True, review=True, imobsparams="obsEV")}[case]
typed = {"Stdin": f"{case}.ans"} if Path(f"{case}.ans").exists() else {}
iraf.mkimsets("@all.lis", "B,V", f"sets{case}", Stdout=f"sets{case}.out",
              Stderr=f"sets{case}.err", **typed, **kw)
