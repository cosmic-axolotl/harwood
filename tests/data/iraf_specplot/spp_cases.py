"""The cases of IRAF's specplot gabarito: the spectra, the task's parameters, the
cursor file (IRAF's format) and what the prompts read, a line each. The spectra
are the splot gabarito's (../iraf_splot/sp_inputs.py)."""

SPACE = "\\040"


def keys(*lines: str) -> str:
    return "".join(line + "\n" for line in lines)


CASES = {
    "default": ("ms", {}, keys(":show", ":vshow", "0 0 1 q"), ""),
    "two_images": ("ms,one", {}, keys(":show", "0 0 1 r", "0 0 1 q"), ""),
    "layout": ("ms,one", {}, keys("0 0 1 x", ":show", "0 0 1 y", ":show", "0 0 1 z",
                                  ":show", ":step 900", ":show", ":fraction",
                                  ":fraction 0.5", "0 0 1 z", ":step", "0 0 1 q"), ""),
    "no_autolayout": ("ms,one", {"autolayout": "no", "step": 500.0},
                      keys(":show", "0 0 1 q"), ""),
    "offsets": ("ms,one", {"autoscale": "no"}, keys(":show", "0 0 1 q"), ""),
    "colon": ("ms", {}, keys(":offset", ":offset[2] 100", ":scale[1] 2", ":scale",
                             ":w0[1] 6300", ":w0", ":wpc[2] 1.3", ":wpc",
                             ":xlpos[1] 0.5", ":xlpos", ":ylpos 0.1", ":ylpos",
                             ":labels", ":labels index", ":labels", ":labels imname",
                             ":labels bogus", ":ulabel[1] hello there", ":ulabel",
                             ":ptype[2] 2", ":ptype", ":color[1] 3", ":color",
                             ":show", ":nonsense", ":offset[7] 3", "0 0 1 q"), ""),
    "velocity": ("ms", {}, keys(":velocity", ":velocity[2] 500", ":velocity",
                                ":redshift", ":redshift[3] 0.01", ":redshift",
                                ":vshow", "6563 0 1 v", ":velocity", ":redshift",
                                ":vshow", "0 0 1 q"), ""),
    "units": ("ms", {}, keys(":units nm", ":show", ":units[2] microns", ":show",
                             "0 0 1 q"), ""),
    "move": ("ms,one", {"labels": "index"},
             keys(":move[1] 6", ":show", ":shift[2] 1", ":show", ":shift -1", ":show",
                  "0 0 1 o", ":show", "0 0 1 q"), ""),
    "delete": ("ms,one", {}, keys("6563 3500 1 d", ":show", "6563 3500 1 e", ":show",
                                  "6400 1000 1 d", "6400 1000 1 d", ":show", "0 0 1 q"),
               ""),
    "add": ("ms", {}, keys("6563 1000 1 a", ":show", "6563 1000 1 i", ":show",
                           "6563 1000 1 a", "0 0 1 q"), "one\nslit\n\n"),
    "keys": ("ms", {}, keys("6563 3500 1 f", "0 0 1 r", "6563 3500 1 f",
                            "6563 3000 1 u", ":show", "6600 3000 1 t", "6700 3000 1 t",
                            ":show", "6563 3000 1 l", "6650 4000 1 l",
                            "6563 3000 1 p", "6450 3800 1 p", "6563 3000 1 p",
                            "0 0 1 q", "0 0 1 b", "0 0 1 q"),
             "6565\n6610\n6705\nH alpha\n"),
    "shift": ("ms", {}, keys("6563 3000 1 s", "6600 3300 1 s", "6620 3400 1 t",
                             "6640 3400 1 x", "6640 3600 1 y", "6650 3700 1 z",
                             "0 0 1 r", "0 0 1 q", ":show", "0 0 1 q"), ""),
    "window": ("ms,one", {}, keys("6500 1000 1 w", "6500 1000 1 e", "6650 5000 1 e",
                                  "0 0 1 w", "0 0 1 a", "0 0 1 q"), ""),
    "params": ("ms,one", {"apertures": "1,3", "scale": "2.", "offset": "100",
                          "labels": "imtitle", "sysid": "no", "title": "My spectra",
                          "xlabel": "lambda", "ylabel": "counts", "xlpos": 0.9,
                          "ylpos": 0.2, "ptype": "-2"},
               keys(":show", "0 0 1 q"), ""),
    "log": ("ms", {"transform": "log"}, keys(":vshow", "0 0 1 q"), ""),
    "limits": ("ms", {"xmin": 6450.0, "xmax": 6700.0, "ymin": 0.0},
               keys("0 0 1 r", "0 0 1 q"), ""),
    "ulabels": ("ms", {"ulabels": "labels.txt", "logfile": "sp.log"},
                keys(":ulabel", "0 0 1 q"), ""),
    "nospectra": ("ms", {"apertures": "9"}, keys("0 0 1 r", "0 0 1 d", "0 0 1 q"), ""),
}
