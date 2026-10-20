Two things under one name.

**IRAF's airmass:** the airmass at an elevation above the horizon, by
Allen's formula with the atmosphere's scale height (`--scale`, 750), and
nothing else needed.

```example
title: The airmass thirty degrees above the horizon
run: harwood airmass --elevation 30
output: |
  airmass 1.996 at an elevation of 30 degrees
```

`--radians` takes the elevation in radians.

**The airmass of an exposure:** from a frame's header, or from `--ra`,
`--dec` (with `--epoch`), `--date`, `--ut` and `--exptime`, at a site --
with the same arithmetic as [setairmass](task:setairmass), IRAF's, so the
two agree: the hour angle from the header's `ST` (or worked out from the
UT), the coordinates precessed to the date, and the airmass at the start,
middle and end combined into the effective one.

```example
title: The airmass of a frame, from its header
run: harwood airmass obj001.fits
output: |
  airmass: 1.0343
```

The site is `--observatory` (or `--latitude`, `--longitude` positive west,
`--altitude`), the header's `OBSERVAT`, or the reduction's default. To
write the airmass into many headers, use [setairmass](task:setairmass).
Above two airmasses harwood also warns about differential refraction: the
image is spread into a short spectrum along the vertical, and a slit not at
the parallactic angle loses part of it.
