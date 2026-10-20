Converts a moment between the systems astronomers count time in: UT, Julian
date, modified Julian date, Besselian or Julian epoch, and the local mean
sidereal time at the site -- what is overhead, which is why observing plans
are made in it.

```example
title: When obj001 was taken, every way
run: harwood asttimes obj001.fits
output: |
  jd: 2461296.587153
  mjd: 61296.087153
```

The moment comes from the frame's header (`DATE-OBS`, `UT`, the middle of
the exposure) or from `--date` and `--ut`; the site from `--observatory` or
`OBSERVAT`.

The sidereal time takes UT as UT1, as IRAF does, rather than asking the
internet for the Earth's rotation: the difference is under a second, and an
observatory's computer is often offline.
