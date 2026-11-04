"""IRAF 2.18's mkconfig with sections typed at the terminal:
    python3 run_iraf_typed.py CASE
from a directory with CASE.ans, fc1.dat, tmine.dat and fobsA.dat (../obsfile).

The task reads a pseudo-terminal, not a file, so that each <EOF> line of CASE.ans
is a Ctrl-D, which ends one list and lets the next be typed -- with a file as the
standard input the first end of the input would end them all. What the task
prints goes to CASE.out and CASE.err, the configuration to CASE.cfg.
"""
import os
import sys
import time
from pathlib import Path

CASES = {
    # case: (catalog, observations, transform, verify)
    "T1": ("STDIN", "obsA", "mine", False),
    "T2": ("STDIN", "obsA", "mine", True),
    "T3": ("c1", "STDIN", "mine", True),
    "T4": ("c1", "obsA", "STDIN", False),
    "T5": ("c1", "obsA", "STDIN", True),
    "T6": ("STDIN", "STDIN", "STDIN", False),
    "T7": ("c1", "obsA", "STDIN", False),
    "T8": ("c1", "obsA", "STDIN", False),
}

case = sys.argv[1]
if len(sys.argv) == 2:
    import pexpect

    child = pexpect.spawn(sys.executable, [__file__, case, "child"], echo=False,
                          encoding="utf-8", timeout=60)
    child.expect("READY")
    time.sleep(2)
    for line in Path(f"{case}.ans").read_text(encoding="utf-8").splitlines():
        if line == "<EOF>":
            child.sendeof()
        else:
            child.sendline(line)
        time.sleep(0.4)
    child.expect(pexpect.EOF)
    sys.exit(0)

from pyraf import iraf  # noqa: E402

iraf.noao(_doprint=0)
iraf.digiphot(_doprint=0)
iraf.photcal(_doprint=0)
iraf.chdir(os.getcwd())
catalog, observations, transform, verify = CASES[case]
if os.path.exists(f"{case}.cfg"):
    os.remove(f"{case}.cfg")
print("READY", flush=True)
iraf.mkconfig(f"{case}.cfg", catalog, observations, transform, template="",
              verify=verify, edit=False, check=True, verbose=False,
              Stdout=f"{case}.out", Stderr=f"{case}.err")
