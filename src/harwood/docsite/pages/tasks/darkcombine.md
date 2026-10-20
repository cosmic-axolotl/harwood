Combines dark frames into one master dark. It is
[imcombine](task:imcombine) with IRAF's settings for darks, from ccdred's
`darkcombine.cl`: average after dropping the highest value at each pixel
(`reject=minmax`, `nlow=0`, `nhigh=1`), of the frames of type `dark`, each
scaled by its exposure time first (`scale=exposure`), so that darks of
different lengths can be combined.

[ccdproc](task:ccdproc) `--darkcor` then subtracts the master scaled by the
ratio of each frame's exposure to the dark's. Both are read from
`--exposure-key`, `EXPTIME` by default; IRAF reads `DARKTIME`, so give
`--exposure-key DARKTIME` for IRAF's behaviour where the two differ. As
with the bias, process the darks first -- overscan, trim, and the bias --
so that the master is the dark current alone:

```console
$ harwood ccdproc @dark.lis --overscan --trim --zerocor --zero Zero.fits --output dark
$ harwood darkcombine 'dark/*.fits' Dark.fits
```

The sample night has no dark frames, so there is no example to run here.
