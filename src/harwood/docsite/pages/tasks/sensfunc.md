Fits the instrument's response -- the sensitivity function -- from every
standard measured into the `std` file by [standard](task:standard): the
ratio of the counts each band gave to the flux the star is known to have,
as a smooth function of wavelength, in magnitudes.

```console
$ harwood sensfunc std --sensitivity sens --extinction onedstds$kpnoextinct.dat
```

**The extinction is taken out band by band**, at the airmass of the
exposure that band came from, before the fit. That is what makes stars
observed at different airmasses comparable in one fit; if the atmosphere
and the instrument were not separated properly, the error would show as a
slope from blue to red, largest in the blue. The test suite closes that loop
with a second star at another airmass -- and checks that without the
extinction, the slope does appear.

The fit is IRAF's icfit -- `--function spline3` of `--order` pieces by
default -- with no rejection unless asked (`--niterate`), as in IRAF;
`--interactive` opens it in the icfit panel, where the order can be changed
and points deleted with IRAF's keys.

The solution goes to `database/sens<name>` and a sensitivity spectrum
`<name>.fits`, which [calibrate](task:calibrate) reads. The rms of the fit is
also given in per cent of flux, since magnitudes hide size: it is the
systematic floor of every spectrum calibrated with this solution, and
observing the object for longer does not lower it.
