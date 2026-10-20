IRAF's `autoidentify`: identifies the lines of an arc without anyone
marking them, from an approximate wavelength (`--crval`, at `--crpix`, the
middle by default) and dispersion (`--cdelt`). The spacing ratios of every
three neighbouring lines in the arc are matched against those in the line
list; patterns of up to `--npattern` lines vote for which peak is which
line; straight dispersions are drawn through the best-voted pairs, and each
is tried -- fitted, more lines placed from the list, scored on its rms and on
how much of both the arc and the list it accounts for. The best is kept if
it is good enough (`--rms`, `--fmatch`), and written to
`database/id<image>` as [identify](task:identify) would.

```console
$ harwood autoidentify arc.fits 6300 2.5 --coordlist 'linelists$henear.dat'
```

The approximate wavelength and dispersion are required, as they are in
IRAF, and either may be `INDEF` to search for it -- `harwood autoidentify
arc.fits INDEF INDEF` searches every dispersion the list allows, which on a
dense list takes tens of minutes and often ends with no answer; `--plan`
says so before it starts. `--crsearch` and `--cdsearch` say how far around a
given value to look. The search parameters are IRAF's: `ntarget`,
`npattern`, `nneighbors`, `nbins`, `ndmax`, `aidord`, `maxnl`, `nfound`,
`aidsigma`, `minratio`.

The approximation has to be good to about one part in a hundred. The arc's
`OBJECT` is usually the grating setting, not the wavelengths it covers, and
a spectrograph at "6707" need not be centred on 6707 Å: check against the
observing log or a previous night's solution.

`identify` with `--crval` and `--cdelt` runs the same search; this task is
there because it is the name IRAF users reach for, with IRAF's parameters
in front.
