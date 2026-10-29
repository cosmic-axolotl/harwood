Works out, for each image, how much light falls outside a small aperture:
the aperture correction that turns a magnitude measured in a small aperture
-- where the noise of the sky is small -- into the magnitude of the whole
star. It reads [phot](task:phot)'s files measured in many apertures,
harwood's or IRAF's `.mag`.

```console
$ harwood mkapfile stan0051cr.fits.mag.1 20 stan0051cr.apercors --largeap 11 --magfile stan0051cr.mkapfile --logfile stan0051cr.mkapfile.log
```

**The curve of growth.** How a star's magnitude changes from one aperture
to the next depends on the seeing, not on how bright the star is. So the
differences between successive apertures are fitted, not the magnitudes,
by Stetson's DAOGROW model: a Moffat wing, a Gaussian core and an
exponential, in proportions (`--swings`, `--pwings`, `--pgauss`,
`--rgescale`, `--xwings`) shared by every image, and one seeing radius per
image. `--nparams` says how many of the five are fitted, in that order;
the rest stay at their starting values. Give several photometry files at
once and they are fitted together: a frame with three stars borrows the
shape of the profile from the frames with more.

**Which apertures count.** A star is used out to the last aperture before
one whose error is above `--maglim` (0.10 mag) or undefined; a faint star's
outer apertures are mostly sky, and would only add noise.

**What is adopted.** Each image's curve is the weighted mean of its stars'
differences, pulled towards the model where there are few stars or they
are noisy, and the stars far from the rest lose weight progressively -- a
neighbour in a star's outer apertures does not drag the curve. The
correction, written to `apercors` with its error, is that curve summed from
`--smallap` to `--largeap` (0 means the last aperture read).

**Best magnitudes.** `--magfile` writes, for every star, the magnitude
corrected out to `largeap` from whichever aperture gives the smallest
error, with the image's filter, exposure time, airmass and time -- the file
`obsfile` reads to match the stars between filters. `--logfile` writes
IRAF's log: the model's parameters, each image's seeing radius, and each
image's curve star by star. `--names` adds each star's name from the
photometry as an eleventh column, `Obj`, so [obsfile](task:obsfile) can
write the stars under the names the standards catalog uses
(`incolumns 1,6,7,2,3,4,5,8,9,11`); IRAF's file has ten.

**How far to trust a correction.** With a few stars per image, less than
the error beside it says. On the Skinakas night -- three stars per image --
IRAF's `.mag` file keeps the errors to 0.001 mag, and in the small
apertures the brightest star's are 0.0011 to 0.0015: rounding them changes
its weight up to twofold, and one image's correction from 1 to 12 pixels
moves by 0.12 mag (its stated error is 0.16). From 6 pixels the stated
error falls to 0.006 mag, but the same rounding still moves a correction by
0.06: the fit itself, not only the small aperture, is that sensitive with
three stars. harwood's phot keeps full precision, so its numbers are not
rounded; rounded as IRAF rounds them, they give IRAF's corrections to the
last digit. Fitting the night's images together, which share the model,
gives each more stars than it has: the twelve fitted at once, the same
rounding moves a correction by 0.03 mag at most, and by 0.011 from 6
pixels.

**Identical to IRAF.** The arithmetic is IRAF's as its Fortran is compiled,
single precision where IRAF's is, and the corrections, the best magnitudes
and the log are byte for byte IRAF's on the twelve Skinakas images, alone
and together. One habit of IRAF's is kept: the first fit's convergence test
starts from a value left by the previous fit in the same IRAF process, so
IRAF run twice in one session gives a slightly different second answer.
harwood always gives the first, which is what a fresh IRAF gives. The
interactive graphs and the `obsparams` file (airmasses from a separate
table) are not ported; set the airmass in the headers with
[setairmass](task:setairmass) before phot.
