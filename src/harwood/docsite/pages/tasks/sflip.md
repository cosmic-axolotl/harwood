Turns spectra round: their data, their coordinates, or both.

```console
$ harwood sflip blue.fits blue.r.fits --coord-flip
$ harwood sflip 'night2/*.ms.fits' --coord-flip --data-flip
```

Both switches, and every flux keeps its wavelength while the pixels run the
other way: a spectrograph that puts the red end on the left made to read like
the rest of the night. The data alone (`--data-flip`, the default) puts the
fluxes against the coordinates the other way round; the coordinates alone
(`--coord-flip --no-data-flip`) the same from the other side -- those are for a
spectrum whose data, or whose dispersion solution, was read backwards. The
coordinates are the wavelength and the pixel columns, the position along the
dispersion axis being IRAF's physical coordinate, which its coord_flip turns
round with the wavelengths; the data are the flux, error, sky, raw and DQ
bands.

Five sessions IRAF 2.18 ran -- each switch, both, neither, and one in place --
are held to listpixels' world and physical coordinates of every line, and to
its pixels. Where IRAF flips in place when no output is given, harwood writes
to the run's directory; give the input's own name as the output to flip it in
place.
