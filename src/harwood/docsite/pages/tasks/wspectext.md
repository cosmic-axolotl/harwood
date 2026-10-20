Writes an extracted spectrum out as text, for a spreadsheet, a plotting
program, or a colleague without harwood: one row per point, one column per
quantity, with a comment at the top saying what each column is.

```console
$ harwood wspectext sp0019.ms.fits sp0019.txt
  points: 1024
  columns: pixel, flux, error, raw
  -> sp0019.txt
$ head -8 sp0019.txt
# aperture 1 at 53.95, -2.9 to +2.9 (5.8 px wide) -- DHCep 6707
# from sp0019.ms.fits
# pixel: position along the dispersion axis, 1-based
# flux: extracted flux, background subtracted
# error: uncertainty on the flux (IRAF's sigma band; variance weights only)
# raw: unweighted, uncleaned aperture sum, background subtracted (IRAF's raw band)
pixel,flux,error,raw
1,1383.79,15.8535,1390.56
```

`--format csv` (the default), `tsv` for `cut` and `awk`, or `fixed` for
aligned columns; `--columns` picks some; `--no-header` leaves the comment
out; `--aperture` picks one of several. A spectrum with wavelengths gets a
`wavelength` column, and `--lines` lists the catalogued lines that fall in
it.
