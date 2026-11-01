Photometry with an effective PSF (Anderson & King 2000, PASP 112, 1360). IRAF's daophot fits an
analytic function and a table of residuals; an effective PSF is the image of a star as the pixels
themselves saw it, built from the stars of the frame with no function assumed. IRAF has no task
for it.

```console
$ harwood psfphot obj001.fits obj001.coo --psfout obj001.epsf.fits --residual obj001.res.fits
```

**The PSF.** With no `--psf` one is built from `--nstars` stars: the brightest with nothing near
them and no pixel above `--datamax`, or the list in `--psfstars`. `--size` is its side, and its
wings beyond go into the zero point, a constant you measure against bright stars with an
aperture. `--oversampling` is grid points to a pixel: 2 is right for a few dozen stars, and on a
crowded field known beforehand 4 was three times worse until there are some hundreds.

**The fit.** Each star's flux and position, on the pixels within `--radius`, the sky off, from
[skymodel](task:skymodel)'s mesh or `--sky`, and every other star taken out with the fits so far.
It goes round the list `--passes` times, the brightest first. `--residual` writes what is left, the
best check there is: stars that remain are stars the list has missed.

**The table** has `x`, `y`, `flux`, `mag` and `merr`, `msky`, `chi` and `ier`: 1 is off the image and
2 too few pixels. Errors come from the image's VAR, or with none from the scatter about the fit, and
the table says which.

Not IRAF's, so it has no IRAF to compare with. It is held to 300 stars on a crowded field known
beforehand, where the bright ones are right to 0.015 mag, against 0.008 with the true PSF; to the
true PSF itself; and to an aperture on the same stars, which it beats where they are faint.
