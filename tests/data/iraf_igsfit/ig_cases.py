"""The sessions of IRAF's fitcoords graph (igsfit) gabarito, on the inputs of
tests/data/iraf_transform/tr_inputs.py: name -> (images, parameters, cursor, answers).

The cursor is IRAF's file of reads, 'x y wcs key' or ':command'; the answers are
what the prompts read from STDIN ('Fit <name> interactively', then 'Write
coordinate map to the database'). On arc the graph opens as Y (pixels) against
Residuals: lines 1-301, residuals within +-0.006."""

YES2 = "yes\nyes\n"

CASES = {
    # c, the colon commands and their refusals; the orders changed and not refitted
    "colon": ("arc", {}, "135.8 0.0058 1 c\n:show\n:corners\n:function\n:xorder\n"
              ":yorder\n:xorder 1\n:yorder 0\n:function foo\n:fun leg\n:bogus\n:xorder 3\n"
              ":show\n0 0 1 q\n", YES2),
    # a new fit after the orders and the function change
    "refit": ("arc", {}, ":xorder 3\n:yorder 4\n:function legendre\n0 0 1 f\n"
              "150 0 1 c\n0 0 1 q\n", YES2),
    # d p, then fit; the deletions list written
    "delete_point": ("arc", {}, "135.8 0.0010 1 d\n0 0 1 p\n0 0 1 f\n0 0 1 r\n0 0 1 q\n",
                     YES2),
    # d x: a whole column of the slit
    "delete_x": ("arc", {}, "105.7 0.0019 1 d\n0 0 1 x\n0 0 1 f\n0 0 1 q\n", YES2),
    # d y and d z: one line position, one feature
    "delete_yz": ("arc", {}, "79.6 -0.0015 1 d\n0 0 1 y\n224.1 -0.0015 1 d\n0 0 1 z\n"
                  "0 0 1 f\n0 0 1 q\n", YES2),
    # u brings back what d took, and what the deletion list held (weight 1)
    "undelete": ("arc", {"deletions": "dels.db"},
                 "135.8 0.0010 1 d\n0 0 1 z\n135.8 0.0010 1 u\n0 0 1 p\n"
                 "51.0 0.0034 1 u\n0 0 1 z\n0 0 1 f\n0 0 1 r\n0 0 1 q\n", YES2),
    # u with nothing deleted, d with everything gone near: nothing asked
    "nothing_to_undelete": ("arc", {}, "100 0 1 u\n0 0 1 q\n", YES2),
    # the axes: x and y chosen, nothing redrawn until r
    "axes": ("arc", {}, "0 0 1 x\n0 0 1 x\n0 0 1 y\n0 0 1 z\n0 0 1 r\n30 6450 1 c\n"
             "0 0 1 x\n0 0 1 s\n0 0 1 y\n0 0 1 r\n0 0 1 r\n6450 0 1 c\n"
             "0 0 1 x\n0 0 1 y\n0 0 1 y\n0 0 1 y\n0 0 1 r\n0 0 1 q\n", YES2),
    # zoom on a column (constant x), next and last, a point there, unzoom
    "zoom_x": ("arc", {}, "135.8 0.0010 1 z\n0 0 1 x\n0 0 1 n\n0 0 1 n\n150 0.002 1 c\n"
               "0 0 1 l\n0 0 1 l\n0 0 1 l\n150 0.002 1 d\n0 0 1 p\n0 0 1 z\n0 0 1 p\n"
               "0 0 1 n\n0 0 1 q\n", YES2),
    # zoom on a feature (constant z), and the last one past the end
    "zoom_z": ("arc", {}, "299 0.0015 1 z\n0 0 1 z\n0 0 1 n\n0 0 1 l\n0 0 1 l\n"
               "300 0 1 c\n300 0 1 d\n0 0 1 x\n0 0 1 f\n0 0 1 q\n", YES2),
    # zoom on y, with x and y changed under it
    "zoom_y": ("arc", {}, "0 0 1 x\n0 0 1 x\n0 0 1 r\n31 0.001 1 z\n0 0 1 y\n0 0 1 n\n"
               "0 0 1 x\n0 0 1 z\n0 0 1 r\n0 0 1 p\n0 0 1 q\n", YES2),
    # the window: w with gtools' keys, e's two corners, :/ commands
    "window": ("arc", {}, "100 0 1 w\n100 0 1 j\n200 0.003 1 w\n200 0.003 1 k\n"
               "150 0 1 c\n0 0 1 w\n50 -0.004 1 e\n250 0.004 1 e\n0 0 1 w\n0 0 1 f\n"
               ":/ywindow -0.01 0.01\n0 0 1 w\n0 0 1 a\n0 0 1 w\n0 0 1 q\n0 0 1 q\n", YES2),
    # transpose: the window swapped, the marks not
    "transpose": ("arc", {}, ":/transpose\n0 0 1 r\n0.001 135.8 1 c\n0.001 135.8 1 d\n"
                  "0 0 1 p\n0 0 1 q\n", YES2),
    # what rings the bell
    "bells": ("arc", {}, "0 0 1 k\n0 0 1 x\n0 0 1 q\n0 0 1 y\n0 0 1 w\n0 0 1 d\n"
              "100 0 1 d\n0 0 1 k\n100 0 1 z\n0 0 1 k\n100 0 1 u\n0 0 1 n\n0 0 1 q\n", YES2),
    # the map not written: the log still says it, the deletions are kept
    "no_write": ("arc", {}, "135.8 0.0010 1 d\n0 0 1 p\n0 0 1 f\n0 0 1 q\n", "yes\nno\n"),
    # not by hand after all
    "answer_no": ("arc", {}, "0 0 1 q\n", "no\n"),
    # two images, one file of reads and of answers
    "two": ("arc,arcl", {}, "135.8 0.0010 1 d\n0 0 1 p\n0 0 1 f\n0 0 1 q\n"
            ":xorder 4\n0 0 1 f\n0 0 1 q\n", "yes\nyes\nyes\nyes\n"),
    # one feature: z = y + f(x), x against residuals
    "single": ("one", {"xorder": 3}, "31 0 1 c\n0 0 1 f\n0 0 1 x\n0 0 1 y\n0 0 1 r\n"
               "0 0 1 q\n", YES2),
    # the dispersion along the lines: x against residuals
    "axis1": ("arct", {"xorder": 4, "yorder": 5}, "135.8 0.001 1 d\n0 0 1 y\n0 0 1 f\n"
              "0 0 1 r\n135.8 0 1 c\n0 0 1 q\n", YES2),
    # combine, and a log axis of gtools
    "combine": ("arc,arcl", {"combine": "yes", "fitname": "both"},
                ":/xtransform log\n0 0 1 r\n200 0 1 c\n0 0 1 q\n", YES2),
}
