Follows each aperture of a frame along the dispersion and stores the fitted
curve in `database/ap<image>`, without extracting. It is
[apall](task:apall) with only tracing on, as `aptrace` is in IRAF: the
profile is centred every `--t-step` lines, on `--t-nsum` lines summed, and
the centres are fitted with `--t-function` of `--t-order`. Apertures that
are not stored yet are found first.

```console
$ harwood aptrace obj.fits --nfind 2
ok aptrace in 0.28s
  apertures: 2
  centres: 34.59, 69.26
  limits: -5.00:+5.00, -5.00:+5.00
  -> database/apobj
```

`--explain` says what the trace came to: how many points, the rms of the
fit and how far the spectrum drifts from one end to the other.
`--fittrace` opens the points to fit them by hand. The trace is what
[apsum](task:apsum) extracts along, and what another frame takes with
`--references`.
