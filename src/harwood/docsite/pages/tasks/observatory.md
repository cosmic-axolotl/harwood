Lists the observatories harwood knows, or looks one up: name, latitude,
longitude (positive west, as IRAF stores it), height and time zone. The
table is IRAF's, so the codes are the ones IRAF headers write in
`OBSERVAT` -- `kpno`, `ctio`, `lapalma` -- and a few common aliases find the
same site.

```example
title: Kitt Peak
run: harwood observatory kpno
output: |
  Kitt Peak National Observatory
```

Every task that needs a site -- [setairmass](task:setairmass),
[setjd](task:setjd), [airmass](task:airmass), [asttimes](task:asttimes) --
takes one of these codes, and reads `OBSERVAT` when given none.
