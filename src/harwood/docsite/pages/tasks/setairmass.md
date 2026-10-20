Computes, for each frame, the airmass the light actually came through and
the time at the middle of the exposure, and writes them into the header, as
IRAF's `setairmass` does.

The airmass a header carries is usually the one at the moment the shutter
opened. For a short exposure that is fine; for a long one it is not. The
course frames of 25 January 1992 say `AIRMASS = 1.52` for a twenty-minute
exposure of DH Cep: true when the shutter opened, but by the time it closed
the star had sunk to 1.61, and the light came through a mean of 1.57. A
flux calibration that takes the extinction out with 1.52 removes too
little of it, by about three per cent in the blue. harwood reproduces
IRAF's values for those frames to 0.01.

```example
title: The effective airmass of every object frame
run: harwood setairmass @obj.lis --observatory kpno
output: |
  obj001.fits │ 02:05:30.0 │ 1.0343
```

For each frame it writes:

| keyword | |
|---|---|
| `AIRMASS` | the effective airmass: Simpson's rule over the start, middle and end, (start + 4 middle + end) / 6, as IRAF |
| `AMSTART`, `AMEND` | the airmass at the start and at the end |
| `UTMIDDLE` | the UT at mid-exposure, which [refspectra](task:refspectra) sorts on |
| `JD` | the Julian date at mid-exposure |

The old `AIRMASS` goes to HISTORY, whatever it was. The site comes from
`--observatory` (see [observatory](task:observatory)) or the header's
`OBSERVAT`, or from `--latitude`, `--longitude` and `--altitude` for a site
not in the table -- longitude positive **west**, as IRAF stores it: Kitt
Peak is +111.6. The airmass is corrected for refraction; below three degrees
of altitude no formula describes it and none is given. `--no-update`
reports without writing.
