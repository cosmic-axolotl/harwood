Gives spectra their wavelengths from a list, one a pixel -- a model's or an
atlas's spectrum whose wavelengths came separately.

```console
$ harwood scoords model.fits waves.dat
SCOORDS: Setting coordinates for model from coordinate file waves.dat.
$ harwood scoords 'atlas/*.fits' @waves.lis
```

Each line of the file that starts with a number is a pixel's wavelength,
whatever follows it on the line; the others are skipped. There must be as many
as the spectrum has pixels, and they must rise or fall all the way -- from the
third on, as IRAF looks: a first step the wrong way goes unnoticed. One file
serves every spectrum, or there is one for each. The spectrum, of one
aperture, is changed in place; its wavelengths are the list's, as the lookup
table IRAF writes into the world coordinates gives them back.

Six sessions IRAF 2.18 ran -- a good list, one with words around its numbers,
one short and one long, one that turns back, and one whose first step does --
say and set the same. `--label` and `--units` are IRAF's, and change nothing
here: harwood's wavelength column has no label to set (IRAF's own writes the
label where the units go).
