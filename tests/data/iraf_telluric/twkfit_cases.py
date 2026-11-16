"""The sessions of IRAF's telluric and skytweak by hand (twk_fit), on the spectra of
twk_inputs.py: name -> (task, parameters, cursor, answers).

The cursor is IRAF's file of reads, 'x y wcs key' or 'x y wcs :command', read on from
one spectrum to the next; wcs 1 is the candidates' graph, 2 the one below. The answers
are what 'Search interactively?' reads, a line for each spectrum asked. On s the three
candidates lie about 0, 1 and 2 (the offset is 1) over 6800-7100 A; on o (skytweak)
about 500 less, at and more the offset."""

Q = "0 0 1 q\n"
T = dict(input="s", output="c", cal="t")
K = dict(input="o", output="c", cal="k")
YES = "yes\nyes\nyes\nyes\n"

CASES = {
    # y over the top candidate, then the middle (the step halved); x changes the
    # graph, x over the lowest, x over the middle
    "choose": ("telluric", T, "6900 2.0 1 y\n6900 1.0 1 y\n6900 1.0 1 x\n6900 0.0 1 x\n"
               "6900 1.0 1 x\n6900 1.9 1 y\n" + Q + "6900 0.1 1 y\n" + Q, YES),
    # a after the shift and scale were set by hand
    "auto": ("telluric", T, "0 0 1 :shift 0.2\n0 0 1 :scale 0.8\n0 0 1 a\n" + Q + Q, YES),
    # e doubles the step shown, the scale's up to 0.99
    "expand": ("telluric", T, "0 0 1 e\n0 0 1 e\n0 0 1 e\n6900 1 1 x\n0 0 1 e\n0 0 1 e\n"
               + Q + Q, YES),
    # the graph below: calibration, data, none
    "below": ("telluric", T, "0 0 1 c\n0 0 1 c\n0 0 1 d\n0 0 1 d\n0 0 1 d\n0 0 1 c\n"
              "0 0 1 r\n" + Q + "0 0 1 c\n" + Q, YES),
    # the colon commands, shown and set, and what they refuse
    "colon": ("telluric", T, "0 0 1 :shift\n0 0 1 :scale\n0 0 1 :dshift\n0 0 1 :dscale\n"
              "0 0 1 :offset\n0 0 1 :smooth\n0 0 1 :sample\n0 0 1 :dshift 0.5\n"
              "0 0 1 :dscale 0.3\n0 0 1 :dscale 1.5\n0 0 1 :dscale -1\n0 0 1 :offset 0.5\n"
              "0 0 1 :offset 0.5\n0 0 1 :smooth 5\n0 0 1 :smooth 5\n0 0 1 :smooth 0\n"
              "0 0 1 :bogus\n0 0 1 :s 3\n0 0 1 :sh 0.3\n0 0 1 :sc\n0 0 1 :dshift\n"
              "0 0 1 :shift abc\n" + Q + "0 0 1 :smooth\n0 0 1 :offset\n" + Q, YES),
    # sample regions: added to '*', appended, begun anew, a key that is neither
    "sample": ("telluric", T, "6850 1 1 s\n6900 1 1 s\n7000 1 1 s\n7040 1 1 s\n"
               "0 0 1 :sample\n6940 1 1 s\n6960 1 1 n\n0 0 1 :sample\n6850 1 1 s\n"
               "6900 1 1 k\n0 0 1 :sample 6850:6900,7000:7040\n0 0 1 a\n0 0 1 :sample\n"
               "0 0 1 :sample *\n0 0 1 :sample\n7090 1 1 s\n"
               "6990 1 1 n\n" + Q + "0 0 1 :sample\n" + Q, YES),
    # the window keys over each graph, and what they share
    "window": ("telluric", T, "6850 1 1 w\n6850 1 1 j\n7000 1 1 w\n7000 1 1 k\n"
               "6900 1 1 w\n6900 1 1 u\n0 0 1 w\n0 0 1 n\n6900 0.8 2 w\n6900 0.8 2 t\n"
               "6900 0.8 2 w\n6860 0.5 2 e\n6960 0.9 2 e\n0 0 2 w\n0 0 2 a\n"
               "6900 1 1 w\n6900 1 1 x\n0 0 1 w\n0 0 1 q\n" + Q + Q, YES),
    # gtools' colon commands on one graph and the other
    "gtools": ("telluric", T, "0 0 1 :/xwindow 6850 6950\n0 0 1 :/ywindow 0.5 1.5\n"
               "0 0 2 :/ywindow 0.2 1.2\n0 0 2 :/xwindow 6900 7000\n0 0 1 :/xwindow\n"
               "0 0 1 :/title Mine\n0 0 1 r\n0 0 2 :/xlabel Lambda\n0 0 2 r\n"
               "0 0 1 :/comments nothing\n0 0 1 r\n0 0 1 a\n" + Q + Q, YES),
    # an x axis that runs the other way: the sample's ticks are not drawn
    "flip": ("telluric", dict(T, sample="6850:6900,7000:7040"),
             "0 0 1 w\n0 0 1 f\n6900 2 1 y\n0 0 2 w\n0 0 2 f\n0 0 2 w\n0 0 2 g\n" + Q + Q,
             YES),
    "transpose": ("telluric", T, "0 0 1 :/transpose\n0 0 1 r\n0 0 2 :/transpose\n"
                  "0 0 2 r\n" + Q + Q, YES),
    "log": ("telluric", T, "0 0 1 :/ytransform log\n0 0 1 r\n0 0 2 :/ytransform log\n"
            "0 0 2 r\n" + Q + Q, YES),
    # no step in the scale: the graph opens on the shifts; y then shows three alike
    "dscale0": ("telluric", dict(T, dscale=0.0), "6900 2 1 x\n6900 1 1 y\n0 0 1 e\n"
                "6900 2 1 y\n" + Q + Q, YES),
    # no step in the shift: x cannot be chosen
    "dshift0": ("telluric", dict(T, dshift=0.0), "6900 1 1 x\n0 0 1 e\n6900 2 1 x\n"
                + Q + Q, YES),
    # nothing found before the graph; the offset and the boxcar given
    "given": ("telluric", dict(T, xcorr=False, tweakrms=False, shift=0.5, scale=1.2,
                               offset=0.5, smooth=7),
              "6900 1.5 1 y\n0 0 1 a\n" + Q + Q, YES),
    # the keys that ring, and the cursor ending before any q
    "bells": ("telluric", T, "0 0 1 k\n0 0 1 z\n0 0 1 w\n0 0 1 h\n", YES),
    # the answers: yes, no, NO; then nothing more is asked
    "answers": ("telluric", dict(input="s,s3", output="c1,c2", cal="t"),
                "6900 2 1 y\n" + Q, "yes\nno\nNO\n"),
    "answers_all": ("telluric", dict(input="s,s3", output="c1,c2", cal="t"),
                    Q + "6900 2 1 y\n" + Q + Q + Q, "no\nYES\n"),
    # a calibration at or below zero: the error of the first graph
    "zero": ("telluric", dict(input="s", output="c", cal="tz", xcorr=False,
                              tweakrms=False), Q + Q, YES),
    "threshold": ("telluric", dict(input="s", output="c", cal="tz", threshold=0.05),
                  "6900 2 1 y\n" + Q + Q, YES),
    # skytweak: the candidates 500 apart
    "sky": ("skytweak", dict(K, offset=500.0), "6950 1050 1 y\n6950 520 1 y\n"
            "6950 520 1 x\n6950 20 1 x\n0 0 1 d\n0 0 1 :offset 200\n"
            "6940 500 1 s\n6960 500 1 s\n0 0 1 a\n" + Q + "0 0 1 :smooth 3\n" + Q, YES),
    "sky_default": ("skytweak", K, "0 0 1 :shift\n0 0 1 :scale\n0 0 1 e\n" + Q + Q, YES),
}
