IRAF's `plot.graph`: lists of numbers and image sections, up to twenty curves on
one pair of axes. A text file is a list -- a line with x and y, or one with y alone,
numbered from 1, whichever its first line with numbers is -- and anything else is
an image, its section averaged onto one axis (`--axis`). With no input, `-` or
`STDIN` the list comes on standard input, as in IRAF's `txdump ... | graph`.

```console
$ harwood graph magerr.txt --pointmode
$ txdump.py m92010.mag.1 mag,merr | harwood graph --pointmode --marker cross
$ harwood graph spec.fits,flat.fits[*,100] --wcs world --ltypes 1,2
$ harwood graph frame.fits[5:20,3:15] --axis 2 --logy --device column.svg
```

The window is the data's, through all the curves, unless `--wx1`/`--wx2` (or
`--wy1`/`--wy2`) differ; with the box drawn it is IRAF's glabax window: an axis of
no width is widened by 1%, a log axis that reaches zero or under becomes IRAF's
elog (logarithmic beyond ±10, linear between), and with `--round` -- always on a
log axis -- it goes out to the next major tick (`--majrx`, `--majry`; fewer than
three: no ticks and no rounding). An H or M format multiplies a window given by 15.

* `--pointmode` marks the points: `--marker` point, box, plus, cross, circle,
  diamond, hline, vline, hebar, vebar, sized `--szmarker` (NDC); 0 or less reads a
  third column of sizes (a negative one scales them, and in world units).
* Without it, `--marker lhist` and `bhist` draw the curve as steps, bhist down to
  the bottom of the window.
* `--ltypes` and `--colors` are lists of line types (1 solid, 2 dashed, 3 dotted,
  4 dot-dashed) and colours (1-9): each curve takes the next, the last one when
  the list is over.
* `--lintran` maps x from p1..p2 onto q1..q2 (p1 = p2 = 0: pixels 1 to n),
  `--transpose` swaps x and y.
* `--wcs logical|physical|world`: an image's coordinates, and its label and format.

The plot goes to the window when there is a screen, and to an SVG in the run's
directory when there is not; `--device name.svg` writes there.

As IRAF reads them: a list's lines with no number are reported ("no args"), so is
one number in an x,y list ("only 1 arg"); a comment must start in the first column;
a comma ends a number (`10,7` is 10); `1d1` is sexagesimal, 1:01. A name that is
not a text file is an image, so a list that is not there is "Cannot open image".

Not ported: `--overplot` and `--append`, which draw on IRAF's graphics device
(give every curve to one graph); the viewport and `--fill` are the device's. On 77
cases -- lists, standard input, images projected onto each axis, every mark, the
histograms, windows, log axes, styles -- what graph says, the window and every
point it draws are IRAF 2.18's metacode.
