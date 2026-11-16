"""The sessions of IRAF's illumination by hand, on the images of ls_inputs.py: name ->
(images, parameters, cursor, answers).

The cursor is IRAF's file of reads, 'x y wcs key' or 'x y wcs :command': first the
bins' graph (il_gsetbins: the image averaged across the slit, lines 1-120 of sky2
against about 480-720), up to its q, then each bin's ICFIT graph up to its q. The
answers are what xt_answer reads: 'Determine illumination interactively for <image>',
then '... at bin <n>' while the answer is no or yes."""

Q = "0 0 1 q\n"
FITS = Q * 8

CASES = {
    # s twice: bins added to the default ones; show, a key that rings
    "add": ("sky2", {}, "20 500 1 s\n40 500 1 s\n0 0 1 :show\n60.6 500 1 s\n80.4 500 1 s\n"
            "0 0 1 :bins\n0 0 1 k\n" + Q + FITS, "yes\nYES\n"),
    # i, then bins marked afresh; the second end before the first, and beyond the data
    "fresh": ("sky2", {}, "0 0 1 i\n0 0 1 :show\n50 500 1 s\n10 500 1 s\n70 500 1 s\n"
              "140 500 1 s\n0 0 1 :show\n" + Q + FITS, "yes\nYES\n"),
    # i alone: one bin of everything
    "all": ("sky2", {}, "0 0 1 i\n" + Q + FITS, "yes\nYES\n"),
    # :bins sets them, and :b shows; gtools' window, after which every colon redraws
    "colon": ("sky2", {}, "0 0 1 :bins 5:25 90:110 40:60\n0 0 1 :show\n"
              "0 0 1 :/xwindow 1 100\n0 0 1 :b\n0 0 1 :bins 5:25 40:60 70:95\n"
              "0 0 1 :bogus\n" + Q + FITS, "yes\nYES\n"),
    # the bins given, one marked more; the fits: yes, no, yes with a new order, NO
    "answers": ("sky2", dict(bins="1:30 40:70 100:120"), "80 500 1 s\n95 500 1 s\n" + Q
                + Q + "0 0 1 :order 3\n0 0 1 f\n" + Q, "yes\nyes\nno\nyes\nNO\n"),
    # no to the image: no bins' graph, but each bin is asked (no, no, yes, NO); the
    # next image is asked again, and yes brings its bins' graph
    "image_no": ("sky2,sky2", dict(output="o,o2"), Q + "0 0 1 i\n30 500 1 s\n90 500 1 s\n"
                 + Q + Q, "no\nno\nno\nyes\nNO\nyes\nYES\n"),
    # the other way up: the dispersion along the columns
    "axis1": ("sky1", {}, "20 500 1 s\n40 500 1 s\n" + Q + FITS, "yes\nYES\n"),
    # s and then q: no bin added, the graph ends
    "s_then_q": ("sky2", {}, "20 500 1 s\n" + Q + FITS, "yes\nYES\n"),
    # an x axis run backwards: the ticks are not drawn
    "flip": ("sky2", {}, "0 0 1 :/xflip yes\n20 500 1 s\n40 500 1 s\n" + Q + FITS,
             "yes\nYES\n"),
}
