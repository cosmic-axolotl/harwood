IRAF's `imextensions`: the image extensions of FITS files, named so that any task
can read them. `--output file`, the default, gives one per line, for an @file;
`list` puts them on one line; `none` only counts them.

`--index` is a range (`1-`, `2,4`, `1-9x2`; 0 is the primary header), `--extname`
patterns for the whole of EXTNAME, `--extver` a range. `--lname` and `--lver` put the
name and version in each entry, and `--no-lindex` leaves the index out when there is
something else to name it by.

Line by line it is what IRAF 2.18 prints, but: an ASCII table is not listed (IRAF
lists it as an image); `mef.fits[NOPE]` lists nothing (IRAF lists every extension);
`mef.fits[SCI,2]` is read (IRAF dies); a name without `.fits` comes back with it; and
there is no limit of 255 characters on `list`.
