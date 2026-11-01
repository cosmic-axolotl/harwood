Image subtraction by ZOGY (Zackay, Ofek & Gal-Yam 2016, ApJ 830, 27), the statistic behind ZTF's
and most of today's searches for what has changed. IRAF's way is psfmatch and imarith: blur the
sharper image to the other, losing what it had, and subtract. Here neither is blurred, the noise of
the difference is white and flat, and it is proved optimal for a point source.

```console
$ harwood imalign new.fits ref.fits bright.coo --output aligned
$ harwood psfphot aligned/new.fits new.coo --psfout new.epsf.fits
$ harwood psfphot ref.fits ref.coo --psfout ref.epsf.fits
$ harwood zogy aligned/new.fits ref.fits diff.fits --psf-new new.epsf.fits --psf-reference ref.epsf.fits --coords bright.coo --scorr score.fits --candidates new.ecsv
```

The images have to be on the same pixels, [imalign](task:imalign) first, and with `--coords` what
is left of the alignment is measured on those stars and taken out, by a fraction of a pixel in
Fourier space, which leaves the noise white; `--no-align` leaves it. Their scatter is the
astrometric noise, unless `--astrometric-x` and `--astrometric-y` say. The sky of each comes off
by [skymodel](task:skymodel)'s mesh, which also gives the noise of each. Each PSF is a file from
[psfphot](task:psfphot), or `--fwhm-new` and `--fwhm-reference` for a Gaussian. Of a file an
elliptical Moffat is fitted and tapered, as an effective PSF has the noise of its stars in it and
a sharp end, and dividing by that in Fourier space leaves rings around bright stars;
`--psf-model epsf` uses it as it is.

**What it writes.** The output is D, the difference, of noise near 1 everywhere, and it warns when it
is not: a wrong PSF or zero point, or images not on the same pixels. `--scorr` is the score to
threshold, in sigma at each pixel: it has the noise of the sources in it, from the image's VAR or
`--gain`, so that a bright star that has not changed is not a candidate; without either, they are,
and it warns. `--flux` is the flux of a
point source at each pixel, in the units where the zero points are 1. `--candidates` lists the peaks
above `--threshold`, with their score, flux and its error, but not within `--margin` of the edge,
where the transforms wrap round, a PSF's width by default, or near a pixel above `--datamax`, as a
saturated star does not subtract. Look at the score of those by eye.

**The zero points.** Only their ratio matters to what is subtracted: `--zero-new` and
`--zero-reference`, or `--coords` with stars seen in both finds it from their light.

Not IRAF's, so it has no IRAF to compare with. It is held to transients put there beforehand, at
different seeing and zero points: the noise of the difference is 1 and white, what has not changed
is gone, the transient is at its pixel, and its flux is what was put there within its error. And
to two real exposures of M92 with nothing new in them, where nothing above 6 sigma comes out, or
next to nothing: that took the alignment to a fraction of a pixel and the smooth PSF, as with
neither 94 candidates did. The noise of the difference is 1.2 there, as a crowded field keeps the
light of its faintest stars in the noise; the score, which is what is thresholded, has 0.95.
