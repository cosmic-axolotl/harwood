Drizzle (Fruchter & Hook 2002, PASP 114, 144): dithered frames on a finer grid, getting back
resolution an undersampled camera loses to its pixels. Each input pixel, shrunk to a drop of `--pixfrac`
of its side, lands on the output where the dither put it, and adds to what is under it by the area
they share. Nothing is interpolated, and the flux is kept. IRAF has [imshift](task:imshift),
[imcombine](task:imcombine) and `geotran`, which each interpolate, and smooth what they interpolate.

```console
$ harwood drizzle 'f*.fits' d.fits --coords stars.coo --scale 0.5 --pixfrac 0.8
```

**Where the frames are.** `--coords`, stars seen in all, within 3 pixels of the same place, measures
it. `--shifts` says it, a table with a row for each frame, `dx` and `dy` against the first or its
first two columns. `--use-wcs` takes it from each frame's own WCS, onto the first's, which allows a
rotation and a distortion. The output's WCS is the first frame's on the finer pixels, and it is read
by the same tools.

**What to choose.** `--scale` is an output pixel in input pixels, and `--pixfrac` the side of the drop.
Smaller is sharper and leaves holes: an output pixel on which nothing landed is zero, and with more than
0.1% of those it warns. The grid wants dithers by fractions of a pixel, four frames or more for a
scale of 0.5. Frames all at one fraction sample the same phases, and it warns of that too.

**The level.** The image is in counts per output pixel, so that a star has the flux it has in the first
frame, and the sky scale^2 of it. `--sky match`, the default, brings each frame's level to the first's,
`surface` takes each one's sky off by [skymodel](task:skymodel)'s mesh, and `none` leaves them. Frames of
different depth come together by `--zero`, or their stars with `--coords`, and each pixel counts by
its inverse variance, from the frame's `VAR` or the noise of its sky. A pixel flagged in the DQ, or above
`--datamax`, lands with no weight. A cosmic ray that is not flagged lands too: run
[cosmicrays](task:cosmicrays) first. `--weight` writes the weight of each output pixel.

**The noise.** A drop can fall on several output pixels, so neighbours share noise, and what the pixels'
scatter shows is below the truth. The header's `DRIZNCOR` is the factor, Casertano et al. (2000)'s, and
each run says so.

Not IRAF's, so there is no IRAF to compare with. It is held to overlaps done by counting, to flux and
centroid at several scales, to the width a star should have, to the noise of blocks against the frames',
to the formula for its factor within 15%, and to `reproject`'s exact overlap on a rotated frame. On the
two 60 s V frames of M92 a star keeps its flux to within 2%. It has not been compared with STScI's
`drizzle` package, which is not installed here.
