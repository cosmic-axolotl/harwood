Makes the correction for a neutral density filter from its transmission curve:
a spectrum, or a frame, to divide what was observed through the filter by.

```console
$ harwood ndprep nd1m.250mag.dat nd250 3500 5 300
$ harwood ndprep nd4m.l250mag.dat nd.log 3600 4 250 --logarithm
$ harwood ndprep nd1m.100mag.dat ndframe 4000 10 80 --nspace 5 --dispaxis 1
$ harwood ndprep ? x 1 1 2
```

The curve, from `--directory` (`onedstds$ctio/`, which harwood carries), is
interpolated onto `nw` wavelengths from `w0` in steps of `dw` -- sinterp's
`curves`, as IRAF's ndprep script calls it. `--logarithm` resamples it onto
wavelengths even in log10 between the same ends, as dispcor would (`--no-flux`
without conserving flux); `--nspace` makes a frame of that many lines
(`--dispaxis 1`) or columns (`2`, the default) of it. `?` as the curve lists
the directory's filters.

Five sessions IRAF 2.18 ran -- linear, log with and without the flux
conserved, a frame of rows and one of columns -- make the same: the linear
ones IRAF's to the bit, the log ones within a unit of the last place, their
wavelengths IRAF's world coordinates.
