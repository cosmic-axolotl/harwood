Finds where an image is on the sky and writes its WCS: TAN, and SIP distortion (Shupe et al. 2005) if
asked. IRAF's `ccmap` fits a plate from stars you paired by hand. Here the pairs are found: every
rotation, mirror image and scale in the range is tried, with all pairs of stars voting for an offset,
and the right one makes a peak nothing else makes.

```console
$ harwood daofind obj.fits obj.coo --fwhmpsf 3 --sigma 8 --threshold 5
$ harwood platesolve obj.fits obj.coo --gaia --scale 0.86 --output solved.fits --matched pairs.ecsv
```

**What it needs.** Stars of the frame, the second argument, brightest first by their `mag` column. A catalogue:
`--gaia` asks ESA's service for Gaia DR3, and Heidelberg's when that does not answer, and only the place
and the radius leave the machine; or `--catalog`, a table with `ra` and `dec` in degrees, and `mag`,
`pmra` and `pmdec` if it has them. About where it is: `--ra` and `--dec`, or the header's RA and DEC at
its `EQUINOX` or `EPOCH`, or its WCS. About its scale: `--scale`, in arcsec a pixel, within
`--scale-error`, or what its CD says. A rotation is not needed, but `--rotation` and `--flip` make it
quicker. The catalogue is cut to about as dense on the sky as the frame's `--nstars` brightest are on
the frame, since more only adds to what agrees by chance.

**What it does then.** From the vote, pairs within a radius that closes to `--tolerance`, a plane fitted
to them, what stands off left out, round again. Proper motions bring the catalogue from
`--catalog-epoch` to the frame's year, from `--epoch` or `DATE-OBS`. `--sip` fits a distortion of that
order, forward and inverse, as any FITS tool reads it.

**It can say no.** A plate fits anything a little, and a list at random has pairs too. Their number is
held to what chance gives at that density, in sigma of chance: the search takes the best of many, so
that `--significance` is 20, and a wrong solution reaches 15. Pairs at random also fill the radius,
where true ones sit in its middle, so that an rms above 0.55 of `--tolerance` fails too. Either way no
WCS is written.

**What it writes.** `--output` is a copy of the file, its old WCS out and the new one in, with
`PLATERMS`, the rms in arcsec, and `PLATEN`, the pairs. `--matched` lists the pairs with what is left of
each. Without `--output`, it only reports.

Not IRAF's, so there is no IRAF to compare with. It is held to fields put there beforehand, at any
rotation and mirror, with stars missing from either list and a distortion, where the WCS found gives
the true positions through `astropy.wcs`, and to lists at random, which it refuses. **It has not solved a
real frame.** On the tutorial's M92 frames it finds nothing above chance, with Gaia DR3 around the
header's pointing and over a wide range of scales: they say neither scale nor orientation, the pointing
of 1987 may be arcminutes off, and the cluster's density gives chance a great deal to work with. It
refuses them, which is what a test of them holds it to. Try it on a frame with a known WCS before
trusting it on your own.
