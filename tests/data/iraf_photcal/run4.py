"""IRAF 2.18 on the evalfit cases: python3 run4.py < /dev/null, in a copy of
this directory. Each case writes evNAME.txt and what it said in evNAME.err."""
from pathlib import Path

from pyraf import iraf

iraf.noao(_doprint=0)
iraf.digiphot(_doprint=0)
iraf.photcal(_doprint=0)
iraf.fitparams("observations.obsfile", "catalogo.mkcatalog", "config3.cfg", "fit3.txt",
               weighting="uniform", interactive="no", logfile="fit3.log", log_fit="yes",
               log_results="yes", Stdout="/dev/null")

# The parameters file with a record gone, and with one that did not converge.
text = Path("fit3.txt").read_text()
start, end = text.index("begin\tBFIT"), text.index("begin\tCFIT")
Path("fit3noB.txt").write_text(text[:start] + text[end:])
Path("fit3stat.txt").write_text(
    text[:start] + text[start:end].replace("status\t0\t(Solution converged)",
                                           "status\t1\t(Singular matrix)") + text[end:])

OBS, CAT = "observations.obsfile", "catalogo.mkcatalog"
F = "%-10s" + " %14.7f" * 9 + "\\n"
F2 = "%-10s" + " %14.7f" * 6 + "\\n"
CASES = {
    "1": (OBS, "config3.cfg", "fit3.txt", dict(catalogs=CAT)),
    "P": (OBS, "config3.cfg", "fit3.txt", dict(catalogs=CAT, format=F)),
    "Pe": (OBS, "config3.cfg", "fit3.txt", dict(catalogs=CAT, errors="equations", format=F)),
    "Pu": (OBS, "config3.cfg", "fit3.txt", dict(catalogs=CAT, errors="undefined",
                                                format=F2)),
    "prog": (OBS, "config3.cfg", "fit3.txt", dict(catalogs=CAT, objects="program",
                                                  format=F2)),
    "std": (OBS, "config3.cfg", "fit3.txt", dict(catalogs=CAT, objects="standards",
                                                 format=F)),
    "nocat": (OBS, "config3.cfg", "fit3.txt", dict(format=F2)),
    "print": (OBS, "config3.cfg", "fit3.txt",
              dict(catalogs=CAT, objects="standards", errors="undefined",
                   print="CI,XV,V,nosuch,VFIT,v1,error(mV)", format="")),
    "G": ("obs2.txt", "config3.cfg", "fit3.txt", dict(catalogs=CAT, format=F)),
    "Ge": ("obs2.txt", "config3.cfg", "fit3.txt", dict(catalogs=CAT, errors="equations",
                                                       format=F)),
    "auto": (OBS, "configuration.mkconfig", "fit1.txt",
             dict(catalogs=CAT, format="%-10s" + " %14.7f" * 6 + "\\n")),
    "autoprog": (OBS, "configuration.mkconfig", "fit1.txt",
                 dict(catalogs=CAT, objects="program")),
    "autonocat": (OBS, "configuration.mkconfig", "fit1.txt", dict()),
    "E": (OBS, "config2.cfg", "fitE.txt", dict(catalogs=CAT, errors="equations",
                                               format="%-10s" + " %14.7f" * 6 + "\\n")),
    "Eo": (OBS, "config2.cfg", "fitE.txt", dict(catalogs=CAT, print="BV", objects="standards",
                                                format="%-10s" + " %14.7f" * 7 + "\\n")),
    "noid": ("obs4.txt", "config4.cfg", "fit3.txt", dict(format="%14.7f %14.7f %14.7f\\n")),
    "dom": (OBS, "config5.cfg", "fit5.txt", dict(format="%-10s %14.7f %14.7f\\n")),
    "domg": (OBS, "config5.cfg", "fit5.txt", dict()),
    "badfmt": (OBS, "config3.cfg", "fit3.txt", dict(catalogs=CAT, format="%-10s %7.3f\\n")),
    "missing": (OBS, "config3.cfg", "fit3noB.txt", dict(catalogs=CAT)),
    "stat": (OBS, "config3.cfg", "fit3stat.txt", dict(catalogs=CAT)),
    "nofile": (OBS, "config3.cfg", "nosuchfile.txt", dict(catalogs=CAT)),
    "noobs": ("nosuchobs.txt," + OBS, "config3.cfg", "fit3.txt", dict(catalogs=CAT)),
}
for name, (obs, config, parameters, kw) in CASES.items():
    try:
        iraf.evalfit(obs, config, parameters, f"ev{name}.txt", Stderr=f"ev{name}.err", **kw)
    except iraf.IrafError as error:
        with open(f"ev{name}.err", "a") as handle:
            handle.write(f"IRAF stopped: {error}\n")
