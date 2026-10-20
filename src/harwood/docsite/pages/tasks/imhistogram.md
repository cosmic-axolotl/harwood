IRAF's `imhistogram`: the histogram of the pixel values between `z1` and
`z2` (the image's own minimum and maximum unless given), in `nbins` bins --
plotted, as IRAF does by default, or listed with `--listout`, bin centre and
count, as IRAF lists them. Integer images get IRAF's treatment -- limits
rounded to integers and, with autoscale, a bin width that is a whole number
of values -- decided by the image's pixel type, not by its values. The
lists are checked against IRAF 2.18's.

```example
title: Where the pixels of a frame lie
after: reduced
run: harwood imhistogram red/obj001.fits --nbins 5 --listout
output: |
  nbins: 5
```

```example
title: The same, as a plot in a file
after: reduced
run: harwood imhistogram red/obj001.fits --device obj001.hist.svg
output: |
  -> obj001.hist.svg
```

The plot goes where IRAF's `stdgraph` would: the window, when there is a
screen (`q` closes it), and otherwise an SVG in the run's directory, whose
path the command prints -- a plot made over ssh is still there afterwards.
`--device name.svg` writes it there in any case. Its title is IRAF's: the
kind of histogram, the image and its title, and `From z1= to z2=, nbins=,
width=`.

| | |
|---|---|
| `--plot-type` | `line` (the default) joins the bin centres; `box` draws each bin as a step |
| `--no-logy` | counts on a linear axis; the default is logarithmic, as in IRAF |
| `--hist-type` | `normal`, `cumulative`, `difference`, `second_difference` |

Most pixels of a frame of sky fall in the first bin; a long tail of
brighter bins is the stars. Narrow the range (`--z1`, `--z2`) to look at the
sky itself.
