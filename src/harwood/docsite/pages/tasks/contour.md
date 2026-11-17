IRAF's `plot.contour`: a contour map of an image or a section of one, with its
labelled perimeter.

```console
$ harwood contour s011.fits
$ harwood contour wpix.fits --no-perimeter
$ harwood contour m51.fits[200:320,200:320] --ncontours 8 --nhi 0 --device m51.svg
```

The image is first reduced to about `--xres` by `--yres` points (64 by 64) by block
averaging, or with `--subsample` by taking every few pixels, keeping its aspect
ratio with `--preserve` -- and says so, as IRAF does: "Image will be block averaged
by 8 in x and 8 in y". `--xres 0 --yres 0` keeps every pixel.

The levels are NCAR's: from `--floor` (the minimum) to `--ceiling` (the maximum)
by `--interval`, or by a round step that gives about `--ncontours` of them (16
when neither is given); `--zero` is taken off the data first. Every fourth level is
a major one and carries its value as a label (`--no-label` drops them), the labels
scaled -- and the scale said -- when the values are under 0.1 or from 1e5 up.
Levels below zero are dashed, in `--dashpat`'s pattern (528: a short dash, a
long gap). `--nhi 0` marks the local highs and lows (H, L and their value), `--nhi
1` writes the value at a grid of points.

The curves are NCAR's, as IRAF draws them: traced across the grid (CONREC),
smoothed by a spline under tension (DASHSMTH), and on a major level broken where
its label is written, along the line.

The line under the perimeter is IRAF's: "contoured from 38.375 to 2838.375,
interval = 100." `--title` replaces the image's name and title above it.

What is IRAF's and what is not: on 19 cases the grid and its message, the levels,
the title and that line are IRAF 2.18's, and so is the drawing -- every polyline
and every text NCAR puts into IRAF's metacode, point for point. `wcslab --append`
draws a grid of RA and Dec over the map (with `--no-perimeter`, as the exercises
do); contour's own `--append`, a map over another graph, is not here.
