A background fitted along the lines (or columns) and subtracted, as IRAF's
`imred.generic.background` does it: [fit1d](task:fit1d) with the difference. On a
rectified long-slit frame, the fit across the slit on the sky's columns takes the
sky's lines out and leaves the object.

```console
$ harwood background ngc660_transform.fits ngc660_bg_sub.fits --sample "10:100 160:250" --order 3 --low-reject 3 --high-reject 3
$ harwood background ngc660_transform.fits ngc660_bg_sub.fits --order 3 --low-reject 3 --high-reject 3 --interactive --lines 275
```

`--sample` names the sky's columns; `--function` (chebyshev), `--order`, the
rejection and `--axis` are fit1d's. With `--interactive`, as the course runs it,
the line asked for (`--lines`, or the terminal's 'Fit line =') opens in the icfit
window: `s` twice on each side of the object marks the sky, `f` fits, `t` starts
the sample over, `q` goes back to ask for another line, and an empty answer
fits the image with what was set. `--cursor` reads the same keys from a file.

On the 7.2 frame (her ngc660_master, transformed) and on synthetic frames --
both axes, every function, rejection, growing, averaging, and by hand from cursor
files -- the output is IRAF 2.18's pixel for pixel.
