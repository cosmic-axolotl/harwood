Spells out the image names of onedspec's record format: a root and record
numbers, `sp.0001`, `sp.0002`..., the way IRAF's old spectra were named, one
to a file.

```console
$ harwood names sp 5,1-3,2
sp.0001
sp.0002
sp.0003
sp.0005
$ harwood names sp,arc 10-20x4 --append [1:500]
```

The records are whole numbers, from 0, written as `1-5`, `7` or `10-20x4`,
and they come out in increasing order, each once, whatever order they were
given in, as onedspec's own copy of IRAF's `decode_ranges` has them; `7-4` is
4 to 7. More than a thousand is an error, and so is a range left open, `998-`.
No records gives the roots themselves. `--append` puts text after each name,
a section say; `--check` keeps only the images that exist.

Of a root's extension IRAF takes off only one of three letters ending in `h`
(`.imh`) or of two ending in `l` (`.pl`), and so does harwood: `sp.fits` with
record 1 is `sp.fits.0001`. Thirteen sessions IRAF 2.18 ran print the same,
line for line.
