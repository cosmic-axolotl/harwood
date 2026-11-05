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


def across(key, i, command):
    """The command on point i under :/transpose, the data's y across the screen."""
    xs, ys = graphs[key]
    return (ys[i], xs[i], command)


def w(key, where=None):
    """w and the key after it, both at point `where` of graph i or both at 0 0."""
    point = at("i", where, "") if where is not None else on("")
    return [point[:2] + ("w",), point[:2] + (key,)]


GT = [": /" + c for c in (
    "xwindow", "ywindow", "xwindow -6 -2", "xwindow", "ywindow -0.1 0.1", "xflip",
    "xflip yes", "yflip yes")]


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
    # w and each key after it; e, then the other corner; a bad key, and ?.
    "window": (w("z", 4) + [at("i", 4, "c")] + w("a") + w("x", 4) + w("y", 4) + w("p", 4)
               + w("c", 4) + w("l") + w("r") + w("u") + w("d") + w("j", 2) + w("k", 9)
               + w("b", 2) + w("t", 9) + w("m") + w("n") + w("f") + [at("i", 4, "c")]
               + w("g") + w("q") + w("?") + [at("i", 2, "w"), at("i", 2, "e"), at("i", 9, "e"),
                                             at("i", 4, "c")]
               + [on("w"), on("e"), on("q"), on(": errors")], GO, {}),
    # On a flipped axis gt_setr swaps min and max: j is the screen's left edge.
    "windowflip": (w("f") + w("j", 2) + w("k", 9) + w("a") + w("j", 2) + w("f") + w("k", 9)
                   + w("a") + w("f") + w("g") + w("c", 4) + w("x", 4) + w("b", 2)
                   + [on(": /xwindow"), on(": /ywindow"), on(": /xwindow -8 -7"), on("r"),
                      on(": /xwindow"), on(": /shift -7.2 0.01"),
                      on(": /expand -9 -6 -0.1 0.1"), on(": /ywindow")]
                   + w("u") + w("l") + [on(": errors")], GO, {}),
    # Another graph key lets the x limits go and keeps the y's, as ing_fit does.
    "windowkey": (w("z", 4) + [on("h"), on("i")] + w("f") + w("g") + [on("j"), on(": show")],
                  GO, {}),
    # The window goes with the equation; the flip and the marks stay for the next.
    "windoweq": (w("z", 4) + [on(": /xflip yes"), on(": /mark box"), on("q"), at("i", 4, "c"),
                              on(": /xflip"), on(": /mark")], GO, {}),
    "gtcolon": ([on(c) for c in GT] + [on(": /transpose"), on("r"), at("i", 4, "c"),
                                        on(": /transpose")]
                + [on(": /" + c) for c in (
                    "mark box", "mark", "type", "subtitle a subtitle",
                    "comments some comments", "title new title", "xlabel new x", "xunits mag",
                    "xbuf", "xbuf 0.2", "ybuf",
                    "clip", "xsize", "sysid", "drawtitle", "drawxlabels", "line", "color",
                    "xtransform", "redraw", "bogus", "mark nosuch", "xtransform nosuch",
                    "type nosuch", "xview", "yview", "xview 0.2 0.8")] + [on("r")], GO, {}),
    # :/shift with nothing to read takes the last real read; :/window is not there.
    "gtcolon2": ([on(": /" + c) for c in (
                     "expand -7 -1 -0.1 0.1", "shift -3 0", "shift", "window 0 0 1 a",
                     "drawtitle no", "drawxlabels no", "drawylabels no")]
                 + [on("r")] + [on(": /" + c) for c in (
                     "drawtitle yes", "drawxlabels yes", "drawylabels yes", "sysid no",
                     "drawgrid yes yes", "transpose")]
                 + [across("i", 4, "d"), on("f"), on(": errors"), on(": /help")], GO, {}),
    "transposed": ([on(": /transpose"), on("r"), across("i", 4, "c"), across("i", 7, "d"),
                    on("f"), across("i", 7, "u"), on("f"), across("i", 4, "w"),
                    across("i", 4, "x"), across("i", 4, "c"), across("i", 2, "w"),
                    across("i", 2, "e"), across("i", 9, "e"), on(": /transpose"), on("r"),
                    on(": errors")], GO, {}),
    "marks": ([on(": /" + c) for c in ("mark diamond", "redraw")] + [at("i", 4, "c")]
              + [on(": /" + c) for c in ("type line", "type", "redraw", "type line3", "type",
                                         "type mark", "type", "xsize 4", "ysize", "redraw",
                                         "mark cross", "redraw", "mark plus")]
              + [at("i", 4, "d"), on("f"), on(": errors")], GO, {}),
    # A logarithmic axis is drawn out to whole decades of elog; nearest is on it.
    "logaxis": ([on("k"), on(": /xtransform log"), on(": /xtransform"), on("r"),
                 at("k", 5, "c"), at("k", 7, "c"), at("k", 7, "d"), on("f"), at("k", 3, "w"),
                 at("k", 3, "e"), at("k", 11, "e"), at("k", 5, "c"),
                 on(": /ytransform logarithmic"), on("r"), at("k", 5, "c"),
                 on(": /xtransform linear"), on(": /ytransform lin"), on("r"), at("k", 5, "c"),
                 on(": errors")], GO, {}),
    # Cursor mode's commands come to the task from a file, which knows none.
    "cursormode": ([on(": .gflush"), on(": .bogus"), on(": .help"), on(": show")], GO, {}),
    # An axis with no width: the airmass of one image, at 1, under 1 and at 0.
    **{name: ([on("g"), on("h"), on("g")],
              ["h", "XV,fit", "h", "residuals,XB", *GO], {"observations": f"{name}.obsfile"})
       for name in ("flat", "flathalf", "flatzero")},
}
for name, (cursor, answers, _) in CASES.items():
    Path(HERE / f"{name}.cur").write_text(
        "".join(f"{x:.7g} {y:.7g} 1 {k}\n" for x, y, k in cursor) + "0 0 1 q\n"
        if cursor or name != "nocursor" else "", encoding="utf-8")
    Path(HERE / f"{name}.ans").write_text("".join(a + "\n" for a in answers), encoding="utf-8")
(HERE / "CASES.yaml").write_text(yaml.safe_dump({k: v[2] for k, v in CASES.items()}),
                                 encoding="utf-8")
print(len(CASES), file=sys.stderr)
