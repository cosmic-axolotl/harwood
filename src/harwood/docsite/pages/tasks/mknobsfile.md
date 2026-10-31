Makes the observations file directly from the photometry: one aperture's magnitude
of each star in [phot](task:phot)'s files, matched between the filters of each image
set, ready for [fitparams](task:fitparams) and [invertfit](task:invertfit). It is
what [mkapfile](task:mkapfile) and [obsfile](task:obsfile) do together, without the
curve of growth, when one aperture will do.

```console
$ harwood mknobsfile @all.lis B,V imsets.mkimsets observations --aperture 11 --tolerance 8
```

**The sets** come from [mkimsets](task:mkimsets), a line per field,
`PG2331+055 : stan0051cr.fits stan0052cr.fits`. **The aperture** is `--aperture`,
counted from 1 in the order phot measured them; `--apercors` adds mkapfile's
corrections, which are for one small aperture and one large one, so the aperture
here has to be the small one. `--shifts`, `--obsparams`, `--tolerance`,
`--minmagerr`, `--allfilters` and `--wrap` mean what they mean in
[obsfile](task:obsfile), which does this work once the fields are taken out, except
that here a star measured in only some of the filters is written by default, as in
IRAF. An output that exists already is refused, not overwritten, also as in IRAF.

**Identical to IRAF.** The observations file, the format file and what it prints
agree byte for byte with IRAF's on the twelve Skinakas images. IRAF's file names no
star, so the stars are numbered in each set (`PG2331+055-1`); to write them under
the names the standards catalog uses, go through mkapfile's `--names` and obsfile.
