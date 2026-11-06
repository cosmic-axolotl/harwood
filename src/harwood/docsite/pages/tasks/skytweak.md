Subtracts the sky from spectra with a sky spectrum shifted and scaled -- the shift and
the scale found for each spectrum by cross-correlation and the least RMS.

```console
$ harwood skytweak o d k --sample 6900:6960
ok skytweak in 0.61s
SKYTWEAK:
  Output: d - spectrum o
  Input: o[*,1] - object one
  Calibration: k[*,1] - sky one
  Tweak: shift = -0.62, scale = 1.155
```

skytweak is [telluric](task:telluric)'s twin: the same search for the shift and scale,
the same parameters, but the calibration -- the sky -- times the scale is subtracted
from each spectrum rather than divided into it, and there is no airmass or threshold.
Its steps start smaller, `--dshift 0.1` and `--dscale 0.1`, sky lines being sharp. Held
to IRAF 2.18's to the bit with telluric's sessions.
