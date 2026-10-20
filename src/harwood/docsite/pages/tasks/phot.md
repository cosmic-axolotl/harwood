IRAF's `phot`: for each star of a coordinate list it recentres the star,
fits the sky in an annulus around the new centre, and sums the light in
one or more apertures. The arithmetic is a port of IRAF 2.18's apphot, held
by the test suite to what IRAF prints for the same pixels, so these are
IRAF's magnitudes.

```example
title: Aperture photometry of every star daofind found
after: found
run: harwood phot red/obj001.fits obj001.coo obj001.mag --apertures 4,6 --annulus 10 --dannulus 5 --epadu 2 --zmag 25 --exposure EXPTIME
output: |
  measured: 24
```

## The parameters, IRAF's names

| | |
|---|---|
| `--apertures 4,6` | aperture radii, pixels; several at once |
| `--annulus`, `--dannulus` | the sky annulus: inner radius and width |
| `--epadu` | the gain, electrons per ADU: it sets the photon noise in the errors |
| `--readnoise` | electrons |
| `--zmag` | the zero point of the instrumental magnitudes, arbitrary until standards tie it down |
| `--exposure EXPTIME` | the header keyword with the exposure time, so magnitudes are per second |
| `--datamax` | the saturation level: a star with a pixel above it is flagged |
| `--calgorithm`, `--cbox` | how stars are recentred ([center](task:center)) |
| `--salgorithm` | how the sky is estimated ([fitsky](task:fitsky)) |

The radius trades light against noise: too small, and a change of seeing
moves the answer; too large, and the noise of the sky under the extra area
swamps the star. An aperture that reaches into the sky annulus is refused,
since the sky would then be measured over the star.

## The table

IRAF's columns: the starting and final positions, the shift and its error,
the sky (`msky`, `stdev`, `sskew`, `nsky`, `nsrej`), and for each aperture
the sum, area, flux, `mag` and `merr` -- with `cier`, `sier` and `pier`
saying what, if anything, went wrong at the centring, the sky and the
aperture. The error has three parts: the photon noise of the star, the
noise of the sky over the aperture, and the uncertainty in the mean sky
itself, which in a crowded field is the largest of the three and is the one
a simple implementation leaves out. A star whose flux comes out negative has
an undefined magnitude, as in IRAF.

The table is ECSV, with the 1-based convention recorded in it; it opens in
the window's [Files](page:app/files) panel, in astropy, or in a spreadsheet.
