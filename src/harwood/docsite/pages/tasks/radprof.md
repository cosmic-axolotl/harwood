The radial profile of each star in a list, with its FWHM: the star's centre, sky and
magnitudes as [phot](task:phot) measures them, and then every pixel within `radius`
of the centre as a point of intensity against distance. It answers the question
imexamine's `r` answers, for a whole list and on the record.

```console
$ harwood radprof obj001.fits 12 0.5 --coords obj001.coo --epadu 2 --apertures 3 5
```

**The profile.** A cubic spline of `--order` pieces is fitted through the pixels,
less the sky. Pixels beyond `--kreject` times the rms of the fit are taken out and it
is fitted again, up to `--nreject` times. The table's `intensity` is the curve at
every `step` from the centre out to `radius`, divided by its value at the centre,
`inorm`. `tintensity` is the light inside each radius, divided by the light inside
`radius`, `tinorm`. The radii are in the file's `harwood_radius`.

**The FWHM**, `pfwhm`, is twice where the curve crosses half, in a straight line
between the steps on either side of it; a step well under the FWHM is what makes it
worth reading. Too many pieces and the spline follows the noise, too few and it
misses the core: the column of `tintensity` should rise smoothly and settle near 1.

**The rest** is phot's: `--calgorithm`, `--cbox`, `--salgorithm`, `--annulus`,
`--dannulus`, `--epadu` and the others mean what they mean there, and `cier`, `sier`,
`pier` and `rier` say what went wrong at each stage. An aperture larger than
`radius` is summed only as far as the profile's box and is short of light, which
IRAF does not mention and harwood warns of.

**Identical to IRAF**, to the digits it writes, with its error codes, on the
synthetic field with the spline's order from 1 to 8, with and without rejection, and
with stars at the edges and off them. One habit is kept: when a fit fails, `rier`
stays set for the stars after it, because IRAF never clears it. The table is
harwood's own and not IRAF's `.prf` text, which no other task reads. The
window's keys and the plots are not ported: imexamine's `r` is the interactive way to
the same number.
