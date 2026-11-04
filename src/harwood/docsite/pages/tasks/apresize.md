Resizes the apertures of a frame to its profile, and writes them back to
`database/ap<image>`. It is [apall](task:apall) with only the resizing on,
as `apresize` is in IRAF, and with apresize's `--nsum 1`: the profile is one
dispersion line. Each aperture is set where the profile, above a straight
background between the minima either side, falls to `--ylevel` of its peak;
`--llimit` and `--ulimit` bound it, and `--avglimits` gives them all the
average.

```console
$ harwood apfind obj.fits --nfind 2
$ harwood apresize obj.fits
ok apresize in 0.11s
  apertures: 2
  centres: 34.60, 69.80
  limits: -3.80:+3.79, -0.72:+1.11
  -> database/apobj
```

The second aperture comes out a pixel and a half wide, because on a single
line the fainter spectrum is mostly noise; IRAF sizes it the same. Sum more
lines (`--nsum 10`) for a profile worth sizing on. With no apertures stored,
it finds them first.
