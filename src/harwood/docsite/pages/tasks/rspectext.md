Reads a text spectrum -- lines of wavelength and flux -- into a spectrum, the
other way round from [wspectext](task:wspectext).

```console
$ harwood rspectext atlas.txt atlas.fits --dtype nonlinear --title "solar atlas"
$ harwood rspectext model.txt model.fits --dtype linear --crval1 4000 --cdelt1 2.5
```

Each line that is two numbers is a pixel, the wavelength first; the others --
comments, a line of one number -- are skipped, and a third number on a line is
ignored. A FITS header in front of the numbers, ending with `END`, as IRAF's
wspectext writes with `header=yes`, says what the spectrum is: its title (the
last `OBJECT` card, as IRAF's rtextimage takes it) and its dispersion
(`CRVAL1`, `CDELT1`, `CRPIX1`, `DC-FLAG`). Without one, `--title` and `--flux`
(calibrated: `CA-FLAG` and `EX-FLAG` 0) say the rest, and `--dtype` the
dispersion:

- `linear` (the default) and `log`: from `--crval1` and `--cdelt1` (in log10
  for `log`); the first column is not read.
- `nonlinear`: the first column is each pixel's wavelength.
- `interp`: the same, resampled onto an even grid from the first wavelength to
  the last, as many pixels, the fluxes not conserved -- dispcor on IRAF's
  lookup table, a straight line between the wavelengths of whole pixels.
- `none`: pixels only.

As in IRAF, both columns go through seven figures (`%g`) on their way, the
fluxes then kept in real and the wavelengths in double. Seven sessions IRAF
2.18 ran -- each dispersion type, a calibrated flux, and a header IRAF's own
wspectext wrote -- give the same wavelengths and pixels, as far as listpixels'
seven figures say. An output that exists stops the task, as IRAF's does: its
script looks for the name without `.fits` and rtextimage then refuses to write
over the file.
