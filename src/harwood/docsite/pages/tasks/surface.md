IRAF's `plot.surface`: an image seen in perspective as a surface, with the lines a
ridge hides taken out.

```console
$ harwood surface s011.fits
$ harwood surface wpix.fits --angh 40 --angv 60 --label
$ harwood surface m51.fits[200:320,200:320] --floor 200 --ceiling 900 --device m51.svg
```

The image is reduced to about `--xres` by `--yres` points (64 by 64) by block
averaging, or `--subsample` (`--xres 0 --yres 0`: every pixel), and clipped to
`--floor` and `--ceiling` -- with IRAF's messages when they leave nothing ("Entire
image is at or below specified floor"), when they are the same ("degenerate
values and will be ignored"), or when the image is constant.

The view is NCAR's: the grid twice as wide as the data's range is high, the eye ten
such ranges away, `--angh` degrees around (the default -33) and `--angv` up (25),
and TRN32S's perspective, the picture made square. `--label` draws the two axes
along the base on the sides toward the eye, with their ticks, X-AXIS and Y-AXIS and
the first and last column and line, as srf_perimeter does.

What is IRAF's: on 18 cases the messages and the title are IRAF 2.18's, and nearly
every vertex IRAF draws is within one unit of NCAR's 1024-wide frame of a point
harwood projects. The hidden lines are taken out by painting the grid's cells from
the farthest to the nearest, where NCAR keeps a floating horizon, so where a line
disappears behind a ridge is not IRAF's to the point; the faithful port of NCAR is
in harwood's queue. Not yet: `--append`. IRAF's `surface ... >G meta` saves the
metacode; here `--device name.svg` saves the picture.
