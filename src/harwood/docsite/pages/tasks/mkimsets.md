Groups the night's images into image sets -- one field seen once in each
filter -- and writes the sets file [obsfile](task:obsfile) reads to know
which B image goes with which V image.

```console
$ harwood mkimsets @imsets.lista B,V imsets.mkimsets
```

**Where the filters come from.** By default from the photometry files
(phot's `.mag`), which carry each image's filter, exposure time, airmass
and time; with `--input images`, from the image headers, the filter from
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

**Identical to IRAF.** The sets file, the image list and every message are
IRAF's, byte for byte, with the editor sessions (edit, rename, review) off.
Editing the list in an editor is not ported: sort, rename or correct with
the options above, or edit the sets file afterwards -- it is plain text.
