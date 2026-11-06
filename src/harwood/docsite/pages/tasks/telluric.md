Divides the telluric bands out of spectra with the spectrum of a star that has nothing
else in it, shifted and raised to the ratio of the airmasses -- the shift and the power
found for each spectrum by cross-correlation and the least RMS.

```console
$ harwood telluric s c t
ok telluric in 1.13s
TELLURIC:
  Output: c - spectrum s
  Input: s[*,1] - star one
  Calibration: t[*,1] - tell one
  Tweak: shift = 0.70, scale = 0.988, normalization = 0.9737
  WARNING: 1 pixels outside of calibration limits
  Input: s[*,2] - star two
  Calibration: t[*,2] - tell two
  Tweak: shift = 0.70, scale = 1.012, normalization = 0.9735
  WARNING: 1 pixels outside of calibration limits
  -> c.fits
$ harwood telluric @night @night hd84937 --sample 6860:6960,7150:7350
```

Each spectrum is divided by the calibration -- the aperture of the same number, or
the first with `--ignoreaps` or when there is none such -- interpolated by `--interp`
at its wavelengths, shifted by the shift, and raised to the scale times the ratio of
their airmasses (Beer's law: AIRMASS in the headers, or `--airmass`). Values of the
calibration below `--threshold` are taken to be it. The output is multiplied by the
mean of the correction, so that its level stays where it was. Only the first band is
corrected; the others are copied.

The shift starts at `--shift` and the scale at `--scale`. With `--xcorr` the two are
cross-correlated over `--lag` pixels, their continua taken out and their ends tapered,
for the shift; with `--tweakrms` the shift and then the scale are moved, by steps from
`--dshift` and `--dscale` halved until they are below 0.01, to where the RMS of the
corrected spectrum over `--sample` is least -- the RMS of each pixel less the mean of
those three either side, which is what a residual band leaves. What is found for one
spectrum is where the next one starts. The log gives the shift, the scale and the
normalization, and warns of pixels beyond the calibration's last, which take its last
value.

IRAF 2.18's telluric and skytweak are held to the bit: twenty-seven sessions -- the
shift and scale found by cross-correlation and the least RMS, each alone or given; the
lag, the sample, the steps; the calibration's apertures by number, ignored or missing;
thresholds, and a calibration that goes to zero; the airmass from the header or given;
a calibration that covers part of the spectrum; in place; four bands; two inputs with
one calibration or one each -- say the same, line for line, and leave the same pixels.
IRAF's interactive tweaking, by eye with its keys, is not harwood's yet: `--interactive`
says so.
