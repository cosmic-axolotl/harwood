The illumination along the slit of a long-slit sky flat, as IRAF's
`longslit.illumination` makes it: the image averaged along the dispersion in
bins, the profile across the slit of each bin fitted with ICFIT, normalized,
and interpolated between the bins over the whole image. Divided into a frame,
it flattens the sky along the slit, which a dome flat does not light as the
sky does.

```console
$ harwood illumination SkyFlat.fits Illum.fits --dispaxis 2 --function chebyshev --order 2 --high-reject 1 --grow 2 --niterate 2
$ harwood illumination flat_response.fits flat_illum.fits --bins "1:222 109:220 330:440" --interactive
$ harwood imarith Resp.fits '*' Illum.fits SuperFlat.fits
```

The bins are `--bins`, ranges along the dispersion, or `--nbins` equal ones
(five by default) -- cut as IRAF cuts them, npts/nbins pixels each in
integers, so the pixels left over at the end are in none. Ranges are clipped to
the image and sorted, and may overlap. Each bin is fitted in turn, with the
fit's parameters carrying on to the next as IRAF's do; `--interactive` opens
the icfit window for each, `--cursor` reads the keys from a file. In a cursor
file, the keys before the first `q` are the bins graph's: `:bins 1:200
400:600` sets them, `:bins` and `:show` print them, `i` makes them `*`. The
graph's `s` key, which marks a bin with the cursor, needs the window, which
harwood does not have yet: give those bins with `--bins`.

As IRAF does it, each profile is normalized at its pixel nx/2 - 1, not the
middle of the slit; a bin sits at (first + last)/2, in integers; between two
bins the image is mapped onto the interpolator's grid so that a bin's centre
and the pixel after it land on the same place. The interpolator is
`--interpolator`, poly3 by default, falling back to linear with fewer than four
bins (poly3, spline3) and to poly3 with fewer than six (poly5). With one bin,
its profile is copied over the whole image.

The output carries CCDMEAN = 1 and MKILLUM, which [ccdproc](task:ccdproc)'s
`--illumcor` requires; ccdproc computes its mean again, as IRAF's does.

On the course's 6.1 frames the illumination of SkyFlat with the text's fit is
IRAF's Illum.fits pixel for pixel, and Resp times Illum its SuperFlat; on 7.1,
the three bins she marked and the orders 8, 8 and 6 she chose give
flat_illum.fits, and ccdproc's flat and illumination correction with both
gives the galaxy and the lamp IRAF made. Every case compared on synthetic sky
flats -- one to seven bins, bins in any order or overlapping, the five
interpolators and their fallbacks, both dispersion axes, sections -- gives
IRAF 2.18's file pixel for pixel.
