"""The cursor and answer files of the interactive fitparams cases, written here:
python make_cursors.py, from a directory holding the configuration, observations
and catalog of .. (and this folder's cfgplot.cfg). A cursor on a point is the
point's own coordinates, worked out by harwood's non-interactive arithmetic,
already held to IRAF's. CASES.yaml says how each is run."""
import sys
from pathlib import Path

import yaml

from harwood.core.photcal import config, fitsession

HERE = Path(__file__).parent
CFG, OBS, CAT = "configuration.mkconfig", "observations.obsfile", "catalogo.mkcatalog"


def session(cfg=CFG):
    return fitsession.FitSession(config.parse(cfg), [OBS], [CAT], "/dev/null")


s = session()
graphs = {}
for key in "hijk":
    s.graphkey = key
    graphs[key] = (s._axis(0).tolist(), s._axis(1).tolist())


def at(key, i, command):
    """The command on point i of the graph `key` draws."""
    xs, ys = graphs[key]
    return (xs[i], ys[i], command)


def on(command):
    return (0, 0, command)


ALL = ["h", "i", "j", "k", "l"]
GO = ["yes", "next", "yes", "next", "yes"]  # save, on; save, on; leave
CASES = {
    "keys": ([on("c")] + [on(k) for k in ALL]
             + [on(f": {c}") for c in ("show", "variables", "errors", "results", "low_reject",
                                       "high_reject", "nreject", "grow", "tol", "maxiter")],
             GO, {}),
    "data": ([on(f": {c}") for c in ("data", "vshow", "bogus", "v", "s", "e", "page", "h")]
             + [on("x"), on("r"), on("t"), on("t"), on("o"), on("r")], GO, {}),
    "files": ([on(": show keep.txt"), on(": errors keep.txt"), on(": results keep.txt"),
               on(": variables keep.txt"), on(": data keep.txt"), on(": vshow keep.txt"),
               on(": page keep.txt")], GO, {}),
    "del": ([at("i", 4, "d"), on(": errors"), on("f"), on(": errors"), at("i", 4, "c")],
            GO, {}),
    "delq": ([at("i", 4, "d"), at("i", 9, "d")], GO, {}),
    "undel": ([at("i", 4, "d"), at("i", 9, "d"), at("i", 4, "u"), on("f"), on(": errors")],
              GO, {}),
    "delh": ([on("h"), at("h", 2, "d"), on("f"), on("j"), at("j", 7, "d"), on("f"),
              on("k"), at("k", 11, "d"), on("f"), on(": results")], GO, {}),
    "near": ([at("i", 3, "c"), on("h"), at("h", 3, "c"), on("j"), at("j", 3, "c"),
              on("k"), at("k", 3, "c"), on("t"), at("k", 5, "c")], GO, {}),
    "few": ([at("i", i, "d") for i in range(13)] + [on("f"), on(": errors"), on(": results")],
            GO, {}),
    "photo": ([at("i", 4, "d"), on("f"), on(": errors"), on(": results"), on(": vshow")], GO,
              {"weighting": "photometric"}),
    "photons": ([at("i", 4, "d"), on("f"), on(": errors")], GO,
                {"weighting": "photometric", "addscatter": False}),
    "equations": ([at("i", 4, "d"), on("f"), on(": errors")], GO, {"weighting": "equations"}),
    "const": ([on(": const v2"), on(": errors"), on("f"), on(": errors"), on(": const 3 0.2"),
               on("f"), on(": errors"), on(": fit v2"), on(": fit v2 0.1"), on("f"),
               on(": errors"), on(": const"), on(": const nosuch"), on(": const 99"),
               on(": const 0"), on(": const v1 abc"), on(": fit 3"), on("f"), on(": errors")],
              GO, {}),
    "reject": ([on(": nreject 2"), on(": low_reject 1.5"), on(": high_reject 1.5"),
                on(": show"), on(": errors"), on("f"), on(": errors"), on(": results"),
                on("h"), on("r"), on("t")], GO, {}),
    "rejstart": ([on(": errors"), on(": results"), at("i", 4, "d"), on("f"), on(": errors")],
                 GO,
                 {"nreject": 2, "low_reject": 1.5, "high_reject": 1.5}),
    "grow": ([on(": nreject 1"), on(": low_reject 1.2"), on(": high_reject 1.2"),
              on(": grow 1.5"), on("f"), on(": errors")], GO, {}),
    "tol": ([on(": tol 0.1"), on(": tol"), on(": maxiter 4"), on(": maxiter"), on("f"),
             on(": errors"), on(": tol abc"), on(": nreject abc")], GO, {}),
    "graphkeys": ([on("g"), on("h"), at("i", 3, "c"), on("g"), on("h"), on("c"),
                   on("g"), on("g"), on("g"), on("g"), on("g"), on("g"), on("g"),
                   on("r"), on("c")],
                  ["h", "var 11, residuals", "h", "XV,fit", "z", "", "h", "foo,bar", "h",
                   "function", "h", "var 99,fit", "h", "var x,fit", "h", "nonlinear,res",
                   *GO], {}),
    "graphuser": ([on("g"), on("h"), on("r"), on("c"), on("d"), on("g"), on("c")],
                  ["h", "user 1,user 2", "h", "rat,nonlinear", *GO], {}),
    "plot": ([on("c"), on("r"), on("i"), on("l"), on("c"), on("g"), on("r")],
             ["l", "user 2,user 1", *GO], {"config": "cfgplot.cfg"}),
    "nosave": ([on(": errors")], ["no", "next", "no", "next", "yes"], {}),
    "prev": ([on(": show"), on(": show"), on(": show")],
             ["yes", "prev", "next", "yes", "prev", "yes", "next", "yes", "next", "yes"], {}),
    "again": ([on(": errors"), on("q"), on(": errors"), at("i", 4, "d"), on("f"),
               on(": errors"), on("q"), on(": errors")],
              ["yes", "again", "no", "again", "yes", "next", "yes", "next", "yes"], {}),
    "firstlast": ([on(": show"), on(": show"), on(": show")],
                  ["yes", "last", "yes", "first", "yes", "next", "yes", "next", "yes"], {}),
    "go": ([on(": errors")], ["yes", "go"], {}),
    "goall": ([], ["no", "go"], {}),
    "quit": ([on(": show"), on(": show")], ["yes", "quit", "no", "no", "quit", "yes"], {}),
    "endno": ([on(": show"), on(": show"), on(": show")],
              ["yes", "next", "yes", "next", "no", "first", "yes", "quit", "yes"], {}),
    "blank": ([on(": show")], ["", "", "", "bogus", "", "no", "quit", "y"], {}),
    "abbrev": ([on(": show"), on(": show")],
               ["YES", "n", "NO", "p", "y", "q", "y"], {}),
    "eof": ([on(": show")], [], {}),
    "nocursor": ([], [], {}),
    "wkey": ([on("w"), on("q"), on(": show")], GO, {}),
    # An axis with no width: the airmass of one image, at 1, under 1 and at 0.
    **{name: ([on("g"), on("h"), on("g")],
              ["h", "XV,fit", "h", "residuals,XB", *GO], {"observations": f"{name}.obsfile"})
       for name in ("flat", "flathalf", "flatzero")},
}
for name, (cursor, answers, _) in CASES.items():
    Path(HERE / f"{name}.cur").write_text(
        "".join(f"{x:.7g} {y:.7g} 1 {k}\n" for x, y, k in cursor) + "0 0 1 q\n"
        if cursor or name != "nocursor" else "")
    Path(HERE / f"{name}.ans").write_text("".join(a + "\n" for a in answers))
(HERE / "CASES.yaml").write_text(yaml.safe_dump({k: v[2] for k, v in CASES.items()}))
print(len(CASES), file=sys.stderr)
