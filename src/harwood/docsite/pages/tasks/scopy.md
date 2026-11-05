Copies spectra: some apertures, some bands, a wavelength region, into one
spectrum or one per aperture. It is [sarith](task:sarith) with `copy`, as
IRAF's scopy script is.

```console
$ harwood scopy in o --apertures 1,3 --renumber --offset 10 --verbose
in[*,1](1)  -->  o[*,1](11)
in[*,3](3)  -->  o[*,2](12)
$ harwood scopy night.ms.fits blue.fits --w1 4000 --w2 4500
$ harwood scopy night.ms.fits ap --format onedspec
```

`--w1` and `--w2` cut a region, resampled onto as many pixels as it covered or
with `--no-rebin` at the nearest whole pixels; `--apertures`, `--beams` and
`--bands` choose, a `!` before taking the others; `--renumber`, `--offset`,
`--format onedspec`, `--clobber` and `--merge` are sarith's. It is held to
IRAF's with sarith's sessions.
