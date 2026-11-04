Finds the spectra across the slit and writes an aperture for each to
`database/ap<image>`, and does nothing else. It is [apall](task:apall) with
only its first step on, as in IRAF, where `apfind` is apall too: the same
finder, with apfind's own defaults. The one that matters is `--nsum 1`: the
profile is a single dispersion line, not the ten apall adds up, so a faint
spectrum is harder to find and the centres are those of that one line.

```console
$ harwood apfind obj.fits --nfind 2
ok apfind in 0.28s
  apertures: 2
  centres: 34.60, 69.80
  limits: -5.00:+5.00, -5.00:+5.00
  -> database/apobj
```

The apertures have the default limits, `--lower` and `--upper`, until
[apresize](task:apresize) sizes them, and no trace until
[aptrace](task:aptrace) follows them. A frame that already has apertures in
the database keeps them: nothing is found again, as in IRAF.
