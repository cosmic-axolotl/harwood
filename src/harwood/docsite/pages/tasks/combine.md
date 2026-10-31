ccdred's `combine`: [imcombine](task:imcombine) with ccdred's scaling and defaults
(`average`, no rejection), `--ccdtype` to pick the frames of a type and `--subsets` to
combine each filter apart. Unlike `zerocombine`, `darkcombine` and `flatcombine` it
has `--weight` and `--grow`; its grow goes only along the line, in whole pixels.

It is checked against IRAF 2.18's `combine`, over every rejection, scaling, weight
and grow; `--ccdtype` and `--subsets` are the ones the other presets test.
