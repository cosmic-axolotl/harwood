[phot](task:phot) with its choices made: the box, the sky annulus and the apertures
on the command line, in pixels, and everything else fixed. For a first look at a
night, or for bright stars where the choices do not matter.

```console
$ harwood qphot obj001.fits 5 10 10 3 5 8 --coords obj001.coo --exposure EXPTIME --airmass AIRMASS --filter FILTER
```

The stars are centred by the marginal centroid in a box `cbox` pixels wide, and the
sky is the centroid of the smoothed histogram of the annulus from `annulus` to
`annulus + dannulus`; the apertures are all that are left on the line. `--zmag`,
`--epadu` and the header keywords mean what they mean in phot, and so does the
table it writes, or IRAF's text file with `--format iraf`. `--interactive` and
`--icommands` take phot's keys. Anything else, a different centring or sky, a noise
term from the read noise, is phot's business: phot takes all of it.

**Identical to IRAF.** The `.mag` file is byte for byte IRAF's but for the lines
that say who wrote it and when. Like IRAF's, its `TASK` line says `phot`.
