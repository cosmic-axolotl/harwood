"""IRAF 2.18 drawing sensfunc's graphs: python3 run_graphs.py NAME, from a directory
with std_mix, kpnoextinct.dat and cursor/NAME copied as NAME. A line `0 0 1 :text` is
what harwood reads as the answer to w or e, and IRAF from the terminal, so it goes to
Stdin. Metacode into NAME.vdm, for ../iraf_photcal/decode_gki.py --everything."""
import shutil
import sys
from pathlib import Path

from pyraf import iraf

name = sys.argv[1]
vdm = Path.home() / ".iraf/uparm/vdm"
vdm.unlink(missing_ok=True)
keys, answers = [], []
for line in Path(name).read_text().splitlines():
    words = line.split(None, 3)
    if words[3].startswith(":") and words[0] == "0":
        answers.append(words[3][1:])
    else:
        keys.append(line)
Path(f"{name}.cur").write_text("\n".join(keys) + "\n")
Path(f"{name}.ans").write_text("\n".join(answers) + "\n")
iraf.noao(_doprint=0)
iraf.onedspec(_doprint=0)
iraf.sensfunc("std_mix", f"i_{name}", apertures="1", extinction="kpnoextinct.dat",
              interactive="yes", answer="YES", device="stdvdm", cursor=f"{name}.cur",
              logfile=f"log_{name}", newextinction=f"ext_{name}", Stdin=f"{name}.ans",
              Stdout=f"{name}.out", Stderr=f"{name}.err")
shutil.move(vdm, f"{name}.vdm")
