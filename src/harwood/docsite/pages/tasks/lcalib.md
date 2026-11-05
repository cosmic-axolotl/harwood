Lists the calibration tables [standard](task:standard), sensfunc and
[calibrate](task:calibrate) read: a star's bands, magnitudes and flux, a
site's extinction, the stars of a directory.

```console
$ harwood lcalib mags --star-name feige24 --caldir redcal
3350.0       11.701
3400.0       11.725
$ harwood lcalib flam --star-name V --caldir blackbody --mag 10.3 --magband V --teff A0V
Blackbody: V = 10.30, Teff = 9480
3000.0   5.7248E-13
$ harwood lcalib ext --extinction onedstds\$kpnoextinct.dat
3200.0        1.017
```

`bands` gives each band's centre and width, `mags` its magnitude, `fnu` and
`flam` its flux, F-nu = fnuzero 10^(-0.4 m) (`--fnuzero`, 3.68e-20 erg/cm²/s/Hz
by default) and F-lambda = F-nu c / λ²; `ext` the extinction table's
wavelengths and magnitudes per airmass, and `stars` the list of a
directory's stars, its `standards.men`. A blackbody calibration (`--caldir
blackbody`, a band's name as the star) takes `--mag`, `--magband` and
`--teff`, a temperature or a spectral type, as standard does. `--caldir` is a
name of the directories harwood carries (`redcal`, `spec50cal`, ...), or
IRAF's `onedstds$redcal/`, or a directory of your own.

The numbers are computed and printed in real, as IRAF's: seven sessions IRAF
2.18 ran -- each option, another fnuzero and a blackbody -- print the same on
both of its streams, line for line.
