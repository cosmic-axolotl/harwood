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
| `--epadu` | the gain, electrons per ADU: it sets the photon noise in the errors; `--gain GAIN` reads it from the header |
| `--readnoise` | electrons; `--ccdread RDNOISE` reads it from the header |
| `--zmag` | the zero point of the instrumental magnitudes, arbitrary until standards tie it down |
| `--exposure EXPTIME` | the header keyword with the exposure time, so magnitudes are per second |
| `--filter`, `--airmass`, `--obstime` | the header keywords with the filter, airmass and time of the exposure, written with every star (`--ifilter`, `--xairmass`, `--otime` when there are none) |
| `--datamax` | the saturation level: a star with a pixel above it is flagged |
| `--calgorithm`, `--cbox` | how stars are recentred ([center](task:center)) |
| `--salgorithm` | how the sky is estimated ([fitsky](task:fitsky)) |

The radius trades light against noise: too small, and a change of seeing
moves the answer; too large, and the noise of the sky under the extra area
swamps the star. An aperture that reaches into the sky annulus is refused,
since the sky would then be measured over the star; one that ends where the
annulus starts is fine.

## The table

IRAF's columns: the starting and final positions, the shift and its error,
the sky (`msky`, `stdev`, `sskew`, `nsky`, `nsrej`), and for each aperture
the sum, area, flux, `mag` and `merr` -- with `cier`, `sier` and `pier`
saying what, if anything, went wrong at the centring, the sky and the
aperture; and the exposure's `itime`, `ifilter`, `xairmass` and `otime`,
which [mkimsets](task:mkimsets) and [mkapfile](task:mkapfile) read to take
the night on to standard magnitudes. The star's `id` is the coordinate
list's, so names given there (`1s`, `PG2331+055A`) travel with it. The error has three parts: the photon noise of the star, the
noise of the sky over the aperture, and the uncertainty in the mean sky
itself, which in a crowded field is the largest of the three and is the one
a simple implementation leaves out. A star whose flux comes out negative has
an undefined magnitude, as in IRAF.

**When the image carries a variance** -- as it does once
[ccdproc](task:ccdproc) knew the gain -- `merr` comes from it instead: the
variance of every pixel in the aperture, weighted as the sum is, plus the
uncertainty of the mean sky as before. That knows what the noise model
cannot: a flat-fielded pixel in a vignetted corner stands for fewer photons
than its counts suggest, and there IRAF's error, built from `--epadu` and
the sky alone, comes out short -- by √0.6 on a test field whose corner the
flat cuts to 0.6. IRAF's error is kept beside it as `merr_model`, for
comparing with an IRAF reduction, `ferr` holds the flux error, and the
table's metadata says which `merr` is which. With the image's DQ, `pdq`
holds the bits of the pixels each aperture took -- saturated, a cosmic ray
-- and the run names the stars that have any.

The table is ECSV, with the 1-based convention recorded in it; it opens in
the window's [Files](page:app/files) panel, in astropy, or in a spreadsheet.

## The sky from a file

`--salgorithm file --skyfile skies` takes each star's sky from a text file
instead of measuring it, as IRAF does: one `x y msky stdev sskew nsky
nsrej` line per star, in the order of the coordinates -- the columns a
[fitsky](task:fitsky) table has. A line that is not complete flags the star
`211`; a star with no line left, `210`. `histplot` and `radplot` work here
as in fitsky, with `--gcommands`.

## With the cursor

`--interactive` hands the window's keys to the task, as IRAF's
`interactive+` does, and `--icommands file` reads them from a file of
`x y 1 key` lines instead -- IRAF's icommands, `\040` for the space bar --
which is how they were checked: on IRAF 2.18's own runs, the same rows,
numbered the same way.

| key | |
|---|---|
| space | measure at the cursor and write the row |
| `f` | measure at the cursor without writing |
| `c` | centre at the cursor |
| `t` / `s` | the sky at the cursor / around the current centre (phot) |
| `a` | the sky averaged over several positions: `t` at each, `q` to finish (phot) |
| `p` / `o` | magnitudes with the sky as it is; `o` writes them (phot) |
| `m` / `n` | move to / measure the next star of `--coords` |
| `l` | measure every star left in the list |
| `r` | back to the start of the list |
| `e` | the last error messages |
| `q` | done |

`:m 5` and `:n 5` move to or measure star 5; `:annulus 12`, `:cbox 7`,
`:apertures 3,5` and the other parameters are set by name for what follows.
A cursor within a FWHM of the star just fetched from the list measures that
star, as in IRAF. Each row carries `id`, its place in the output, and `lid`,
its place in the list -- 0 for one the cursor chose.
