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

**One fit per aperture.** Each aperture's stars get their own
sensitivity function, `<name>.0001.fits` for aperture 1 and so on, which
[calibrate](task:calibrate) finds by itself; `--ignoreaps` fits them all
together into `<name>.fits`, and `--apertures 1,3` chooses. The fit is
IRAF's curfit -- `--function spline3` of `--order` pieces by default, no
rejection -- in IRAF's own single precision, so the spectrum written is
IRAF's to the last bit. Each fit's statistics (rms, and every star's mean
deviation in each third of the spectrum; then, per wavelength, the mean,
its error and the extinction) are appended to `--logfile`, as IRAF writes
them.

**Working on the stars.** `--interactive` shows IRAF's graphs --
sensitivity and residuals against wavelength (`--graphs sra` adds the
residuals against airmass; `e` is the extinction, dashed where a residual has
changed it, and `c` the composite points, each bar as tall as the error, so half of
it either way, as IRAF draws them) -- and its keys, on the graph under the mouse.
Every frame of fifteen cursor files is held to IRAF's graphics metacode. IRAF's `i`
and `l`, the standards' spectra flux calibrated, are not ported, and their plots
say so; nor are `:images`, `:skys`, `:fluxlimits`, `:marks` or `:colors`:

- `d` then `p`, `s` or `w` deletes the point, the whole star or the whole
  wavelength nearest the cursor; `u` brings them back; `w` gives them a
  new weight; `m` moves them to the cursor's height; `a` adds a point.
- `s` shifts every star to a common mean -- the grey shift for a star seen
  through cloud or partly off the slit, which changes its level but not its
  colour; `s` again takes it back.
- `c` combines the stars, weighted, into one composite point per
  wavelength (`c` again goes back, with any composite point deleted taken
  out of every star).
- `e` fits the residual extinction: at each wavelength the slope of the
  residuals against airmass, a curve through those slopes, and its
  significance; answered yes, it is taken out of the stars and the revised
  extinction curve is written to `--newextinction`.
- `f` refits, `g` refits and redraws, `o` goes back to the original data,
  `:order 4`, `:function legendre`, `:stats`, `:vstats`; `q` writes the
  fit -- the fit as it stands, as in IRAF, so a deletion after the last `f`
  is not in it.

`--cursor file` reads the same keys from a file (`x y graph key`); the
weight `w` asks for and the answer `e` asks for are the next line's `:`
text (`:0.25`, `:yes`), where IRAF reads them from the terminal. On 17 IRAF
runs, batch and from cursor files, every spectrum, logfile and statistics
file is identical to IRAF's.

The rms of the fit is also given in per cent of flux, since magnitudes hide
size: it is the systematic floor of every spectrum calibrated with this
solution, and observing the object for longer does not lower it.
