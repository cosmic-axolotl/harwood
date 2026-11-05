Changes what the apertures of spectra are: their beam numbers, their
dispersion (type, first wavelength, step, redshift), their limits across the
slit, their titles -- from a table, or from defaults for all of them.

```console
$ harwood sapertures in --apertures 1,3 --beam 7 --verbose
in:
  Aperture 1:
    beam 1 --> 7
  Aperture 3:
    beam 5 --> 7
$ harwood sapertures 'fibres/*.ms.fits' --apidtable objects.dat
$ harwood sapertures arc.fits --wcsreset
```

A line of `--apidtable` is aperture, beam, dtype, w1, dw, z, aplow, aphigh and
title; a line with fewer than those nine is read as aperture, beam and title.
INDEF leaves a value as it is, and so does a default left out. A negative beam
is ignored (and said, with `--verbose`). An aperture not in the table takes
the defaults; `--apertures` says which to change at all. `--wcsreset` puts
every aperture back in pixels, the table and the defaults ignored.

As IRAF's, w1 and dw are wavelengths whatever the dispersion: `--dtype 1
--w1 3.6 --dw 0.0001` makes a spectrum that starts at 3.6 Å, its steps even in
log10, not one that starts at 10^3.6. The limits are where the aperture is on
the slit, not about its centre. harwood's wavelengths are a column, the
dispersion's over 1+z, so an aperture given `--dtype -1` loses them, where
IRAF keeps w1 and dw in its world coordinates for another sapertures to
bring back.

Ten sessions IRAF 2.18 ran -- no change, beams, a linear dispersion, a log
one, a redshift, limits, a title, pixels, a reset and a table with its short
lines and a negative beam -- say the same on both streams, to the last figure
of their doubles, and leave each aperture as IRAF's.
