Groups the night's images into image sets -- one field seen once in each
filter -- and writes the sets file [obsfile](task:obsfile) reads to know
which B image goes with which V image.

```console
$ harwood mkimsets @imsets.lista B,V imsets.mkimsets
```

**Where the filters come from.** By default from the photometry files
(phot's `.mag`, harwood's or IRAF's), which carry each image's filter,
exposure time, airmass and time -- with harwood's phot, name the header
keywords with `--filter`, `--airmass` and `--obstime` there; with `--input images`, from the image headers, the filter from
`--filter` (`FILTER`).

**How the sets are made.** The images are listed with their filters,
sorted by name -- usually the order they were taken -- or on `--sort`
(`image`, the filter, or one of `--fields`, numerically), and walked in
order: an image of each filter in `idfilters` makes a set. A filter seen
again before the set is complete closes it, its missing filters written
`INDEF`, and the new image starts the next one; an image whose filter is
not in the list is left out. Each of these is reported, so a night where
one V exposure was lost does not silently pair every later B with the
wrong V.

**Names.** IRAF calls the sets OBS1, OBS2 ... and asks for better names
one by one. Here `--names PG2331+055,LSI61235` names them in order, and
`--name-keyword OBJECT` names each after its first image's header, so the
names match the standards catalog without editing the file by hand.
`--imobsparams` also writes each image's filter, exposure time, airmass
and time, a file obsfile's `--obsparams` reads -- the place to correct a
wrong airmass for the whole calibration.

**The editor and the names, as IRAF asks.** `--edit` opens the image list in the
editor first, below IRAF's explanation of what to do with it: put the images of a
set together, correct a filter, drop an image, write INDEF for a missing one.
`--rename` shows each set and asks for its name (`--answers` reads them from a
file). `--review` opens the sets in the editor at the end, below the warnings
grouping gave, to be corrected; the warnings and the explanation are taken out
afterwards, as IRAF takes them out -- leaving the two blank lines at the top of the
file that IRAF leaves. In the window the editor is a dialog of its own; at the
terminal it is `$EDITOR`, as IRAF's edit uses its editor. IRAF has all three on;
harwood has them off, so that a run can be repeated.

**Identical to IRAF.** The sets file, the image list and every message are
IRAF's, byte for byte, with the editor sessions off and with each of them on --
the same edits made by IRAF's editor and harwood's, what each was shown is the
same, and so are the names asked for and the files left.
