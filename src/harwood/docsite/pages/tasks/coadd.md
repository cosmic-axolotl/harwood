Proper coaddition (Zackay & Ofek 2017, ApJ 836, 187): frames of different seeing and depth added so
that nothing they say about a source is lost. IRAF's way is [imcombine](task:imcombine), a weighted
mean of the pixels, where a frame counts as a whole: its sharp detail cannot count for more than its
blur does. Here each frame counts at each spatial frequency for what it knows there, and the noise
of the result comes out white, and flat.

```console
$ harwood daofind aligned/a.fits stars.coo --fwhmpsf 3 --sigma 8 --threshold 10
$ harwood coadd 'aligned/*.fits' sum.fits --coords stars.coo --score score.fits
```

The frames have to be on the same pixels, [imalign](task:imalign) first. With `--coords`, what is
left of the alignment is measured on those stars and taken out, by a fraction of a pixel in Fourier
space, which leaves the noise white; `--no-align` leaves it. The same stars give the zero points,
from their light within 8 pixels against the first frame's, good to a few per cent when the seeing
differs, and an effective PSF for each frame. `--zero`, `--psf`,
from [psfphot](task:psfphot) `--psfout`, or `--fwhm` for Gaussians, say those instead, with commas,
one for each frame. Of an effective PSF an elliptical Moffat is fitted and tapered, as it has the
noise of its stars in it and a sharp end; `--psf-model epsf` uses it as it is. The sky of each frame
comes off by [skymodel](task:skymodel)'s mesh, which also gives its noise.

**What it writes.** The output reads as one exposure with the first frame's zero point: a star has
the flux it would have there. Its noise is white, and it says what it should be and what it
measured. Its PSF is its own, sharper than the mean of the frames, and `--psfout` writes it for
[psfphot](task:psfphot) `--psf`. `--score` is the filter matched to a point source, in sigma of the
background at each pixel, which is the one to threshold to find what is faint. `--flux` is the flux
of a point source at each pixel, in the first frame's units.

**What it cannot do.** A transform has no gaps, so a pixel flagged in a frame's DQ, or above
`--datamax`, takes the median of the other frames there, at that frame's zero point, and the sky if
it is bad in all of them, which it says. A cosmic ray that is not flagged goes in, so run
[cosmicrays](task:cosmicrays) first. The noise is taken as the sky's, the same all over: the score
near a bright star does not have the star's own noise in it.

Not IRAF's, so it has no IRAF to compare with. It is held to stars put there beforehand, at seeing
from 2 to 5 pixels and different depths and zero points: the noise is white and what the equations
say, each star's score is the root of the sum of the squares of what each frame gives it, above the
best frame and above a weighted mean, and its flux is what was put there within its error. One frame
comes back as itself. On the two 60 s V frames of M92, stars come out with the sum of what each frame
gives them.
