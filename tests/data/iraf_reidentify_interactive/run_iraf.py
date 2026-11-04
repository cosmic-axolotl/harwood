"""IRAF 2.18's reidentify, interactive: python3 run_iraf.py NAME MODE < /dev/null, from a
directory with NAME.cur, arc_lines.dat and arc3.fits, arc3b.fits (multispec_night()'s arcs,
written as multispec images by harwood's apall --format multispec writer). autoidentify
identifies arc3 first; then reidentify carries it to arc3's other apertures and to arc3b,
identify's keys read from NAME.cur, MODE the interactive parameter (yes or YES). PyRAF
asks a query parameter on the process's standard input, so with /dev/null every "Fit
dispersion function interactively?" takes its value, yes. What it prints is in NAME.out and
NAME.err, its log in NAME.log, the records in NAME.db and NAME.dbb. The cases ran as
yes yes and YES YES."""
import shutil
import sys
from pathlib import Path

from pyraf import iraf

name, mode = sys.argv[1], sys.argv[2]
shutil.rmtree("database", ignore_errors=True)
iraf.noao(_doprint=0)
iraf.onedspec(_doprint=0)
for task in ("autoidentify", "reidentify", "identify", "aidpars"):
    iraf.unlearn(task)
iraf.autoidentify("arc3", 4160.0, 0.29, coordlist="arc_lines.dat", interactive="no",
                  verbose="no", logfile="", plotfile="", Stdout="/dev/null")
iraf.reidentify("arc3", "arc3,arc3b", interactive=mode, verbose="no", nlost=40,
                coordlist="arc_lines.dat", logfiles="log", plotfile="", graphics="stdvdm",
                cursor=f"{name}.cur", Stdout=f"{name}.out", Stderr=f"{name}.err")
log = Path("log").read_text().splitlines()[2:]
Path(f"{name}.log").write_text("\n".join(log) + "\n")
shutil.copy("database/idarc3", f"{name}.db")
if Path("database/idarc3b").exists():
    shutil.copy("database/idarc3b", f"{name}.dbb")
