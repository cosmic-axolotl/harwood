"""IRAF 2.18 on one interactive fitparams case, in a directory holding the files
of .. and of this folder: python3 run_iraf.py NAME < /dev/null. stdvdm takes the
graphics, so no screen is needed; the prompts read NAME.ans, and what the keys
print is in NAME.out and NAME.err. A blank answer after a bad command is `first`,
so an answers file that ends there runs for ever; stdvdm appends to uparm$vdm, so it
goes first and is moved aside after, and ../decode_gki.py makes NAME.json of it."""
import shutil
import sys
from pathlib import Path

import yaml
from pyraf import iraf

name = sys.argv[1]
kw = yaml.safe_load(Path("CASES.yaml").read_text(encoding="utf-8"))[name]
cfg = kw.pop("config", "configuration.mkconfig")
observations = kw.pop("observations", "observations.obsfile")
vdm = Path.home() / ".iraf/uparm/vdm"
vdm.unlink(missing_ok=True)
iraf.noao(_doprint=0)
iraf.digiphot(_doprint=0)
iraf.photcal(_doprint=0)
for key, value in kw.items():
    if isinstance(value, bool):
        kw[key] = "yes" if value else "no"
iraf.fitparams(observations, "catalogo.mkcatalog", cfg, f"{name}.fit",
               interactive="yes", logfile=f"{name}.log", log_fit="yes", log_results="yes",
               graphics="stdvdm", cursor=f"{name}.cur", Stdin=f"{name}.ans",
               Stdout=f"{name}.out", Stderr=f"{name}.err", **kw)
shutil.move(vdm, f"{name}.vdm")
