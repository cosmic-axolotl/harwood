Puts together, for each field, the magnitudes measured in each filter: the
stars of the B image are matched to the stars of the V image by position,
and each star is written with all its filters, as the observations file that
[fitparams](task:fitparams) and [invertfit](task:invertfit) read.

```console
$ harwood obsfile @mkapfile.lista 1,6,7,2,3,4,5,8,9,0 B,V imsets.mkimsets observations.obsfile --tolerance 8
```

**Image sets.** The sets file says which images are one field seen in each
filter, a set per line -- `PG2331+055 : stan0051cr.fits stan0052cr.fits` --
with the images in the order of `idfilters`. Stars are only matched within a
set. A set missing an image, or with one too many, is warned about and
used as far as it goes; an image in the photometry that no set names is
skipped, with a warning.

**Columns.** The photometry is read from text files by column number:
`incolumns` gives the column of the image, x, y, filter, exposure time,
airmass, time, magnitude, error and the star's name, `0` for one the files
do not have. mkapfile's best magnitudes are `1,6,7,2,3,4,5,8,9,0`. With a
name column each star is written under its own name -- the name
fitparams looks up in the standards catalog; without one, under the set's
name and a number (`PG2331+055-2`).

**Matching.** Each star of the first filter's image takes the nearest star
of each other filter's image within `--tolerance` pixels, each star taken
once. If the telescope moved between the filters by more than that, give
`--shifts`, a file of image, x shift and y shift. `--tolerance 0` pairs the
stars by their order in the files instead. Only stars found in every filter
are written, unless `--no-allfilters`.

**Corrections.** `--apercors` adds the aperture corrections
[mkapfile](task:mkapfile) wrote, to magnitudes measured in its small
aperture (its best magnitudes are corrected already: do not add them
twice). `--normtime` corrects to an exposure time of 1. Errors smaller than
`--minmagerr` are raised to it. `--obsparams` replaces the filter,
exposure time, airmass or time the photometry carries, image by image.

**The output.** A line per star and filter, the later ones starting with
`*` -- or one long line each with `--no-wrap` -- and beside it
`f<observations>.dat`, the configuration's observations section for these
columns, to start [fitparams](task:fitparams)' configuration from.

**Identical to IRAF.** The file, the format file, what is printed and every
warning are byte for byte IRAF's, on the Skinakas night and on cases that
take every path: names or none, by position or by order, any filter or
all, corrections, shifts, obsparams, and every kind of bad line in a sets
file. IRAF's quirks are kept where they decide the result -- the matching
walks the stars in y order by IRAF's own quicksort, and a set written as
` : image image`, without a name, is dropped without a word, as IRAF drops
it. Any of the sets, the observing parameters, the shifts and the corrections
can be given as `STDIN` and typed at the terminal instead, image by image, with
IRAF's prompts -- in the window, each is a question -- and `--verify` asks again
for each answer; `--answers` reads what would be typed from a file. Those
sessions, too, say and write what IRAF's do.
