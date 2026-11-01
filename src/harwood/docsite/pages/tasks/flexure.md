Measures how far a spectrograph has bent, from the sky's own lines against a frame whose wavelengths
are known. The arc from the start of the night is not where the sky is now, and IRAF's way is
`fxcor` and `specshift`, by hand. Here the two are one step.

```console
$ harwood flexure obj.fits sky_reference.fits --dispaxis 2 --output moved.fits
```

Each frame's sky is taken as one spectrum, the median across `--columns`, or across the slit without
an object, and a spectrum of one line is itself. A running median of `--width` comes off both, to
leave the lines, and the shift is the peak of their correlation within `--reach`, found to a fraction
of a pixel. Pieces of the spectrum give its error. It warns when the frames agree under 0.5 at the
best shift: that is no shift to trust.

`--output` is a copy of the frame with `CRPIX` of the dispersion axis moved by the shift, and `FLEXURE`
saying so. A positive shift means the frame's lines are at higher pixels than the reference's.

Not IRAF's. It is held to shifts put there beforehand, in sharp lines and with noise. On the tutorial's
real twilights it finds 0.005 and 0.002 pixels between frames of one twilight, with a correlation of 0.997
and 0.998, and a shift put into a real one comes back to within 0.08 of a pixel. Real flexure, between
frames an hour apart with real sky lines, is not shown: the tutorial's objects have hardly any sky.
