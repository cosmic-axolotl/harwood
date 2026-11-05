Measures spectra in bandpasses: a band's flux, and with one or two bands of
continuum beside it, an index and an equivalent width -- Lick indices, say.

```console
$ cat bands.dat
band1 4100 20 none
lick 4150 10 none blue 4130 10 none red 4170 10 none
filtered 4200 30 filt.dat
$ harwood sbands in bands.out bands.dat
$ tail -3 bands.out
                in[*,1](1)   band1     1043.16
                in[*,1](1)    lick     1087.71    cont     993.446   1.09488 -0.948836
                in[*,1](1) filtere     951.659
```

A line of the band file is a band -- id, centre, width and filter, a file of
wavelength and response or `none` for a flat one -- and up to two continuum
bands after it, each with its own id. The flux is the sum of the pixels the
band covers, the pixels at its ends by the fraction of them inside, each
weighed by the filter's response; `--normalize` (the default) divides by the
weights' sum. With two continua the continuum is their line at the band's
centre, with one it is that one; the index is flux over continuum and the
equivalent width (1 - index) times the band's width. `--mag` gives magnitudes,
`--magzero` minus 2.5 log10. The measurements are added to the output file,
after a header unless `--no-verbose`; `STDOUT` says them instead.

As IRAF's, the band file is read a word after another: a continuum written
without its id shifts the rest of the line along, the following number taken
as its centre, and the band that results is IRAF's. A band that starts before
the spectrum is measured from its edge, as the world coordinates' inverse
stops half a pixel past it; one that ends past the far end is not measured.
Each spectrum is named by its line of the multispec image IRAF would have
written, `in[*,1](1)`. Four sessions IRAF 2.18 ran write the same, line for
line.
