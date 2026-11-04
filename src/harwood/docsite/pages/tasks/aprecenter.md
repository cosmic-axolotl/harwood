Moves stored apertures onto where this frame's profile peaks, and writes
them to the frame's own `database/ap<image>`. It is [apall](task:apall) with
only the recentering on, as `aprecenter` is in IRAF, with its `--nsum 1`.
Usually the apertures come from another frame:

```console
$ harwood aprecenter B.fits --references A
ok aprecenter in 0.28s
  apertures: 2
  centres: 35.87, 70.53
  limits: -3.73:+3.78, -5.10:+3.07
  recenter: RECENTER  - 2 apertures shifted by 1.27 for B.fits.
  -> database/apB
```

By default every aperture moves by the median of their moves (`--shift`), so
the apertures keep their spacing; `--no-shift` moves each on its own peak,
`--npeaks` uses only the brightest and `--aprecenter` only the apertures it
names. The limits and the trace stay the reference's.
