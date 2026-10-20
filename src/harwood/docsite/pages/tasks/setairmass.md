IRAF's `setairmass`, ported and checked against IRAF's output: for each
frame, the airmass at the beginning, middle and end of the exposure,
combined by Simpson's rule into the effective airmass the light came
through, and the UT of the middle -- written into the header as IRAF writes
them.

The airmass a header carries is usually the one at the moment the shutter
opened. For a short exposure that is fine; for a long one it is not. The
course frames of 25 January 1992 say `AIRMASS = 1.52` for a twenty-minute
exposure of DH Cep: true when the shutter opened, but by the time it closed
the star had sunk to 1.61, and the light came through a mean of 1.5645 --
IRAF's number, and harwood's to the fourth decimal. A flux calibration that
takes the extinction out with 1.52 removes too little of it, by about three
per cent in the blue.

```example
title: The effective airmass of every object frame
run: harwood setairmass @obj.lis --observatory kpno
output: |
  obj001.fits │ 02:05:30.0 │ 1.0343
```

## How

As IRAF: the coordinates (`--ra`, `--dec`) are precessed from their
equinox (`--equinox`) to the date; the hour angle is the header's sidereal
time (`--st`) minus the right ascension; and the airmass is Allen's formula
-- the secant of the zenith distance, rounded off near the horizon by an
atmosphere `--scale` (750) times thinner than the Earth's radius.

Unlike IRAF, a header with no `ST` is not refused: the sidereal time is
worked out from the UT and the site's longitude, by IRAF's own formula, and
harwood says it did so. The sample night above has no `ST`.

## What it writes

| keyword | |
|---|---|
| `AIRMASS` (`--airmass`) | the airmass `--outtype` says -- `effective` (the default), `beginning`, `middle` or `end` -- in single precision, as IRAF |
| `UTMIDDLE` (`--utmiddle`) | the UT of mid-exposure as a date and time, `1992-01-25T02:18:22.00`, as IRAF |
| `AMSTART`, `AMEND` | harwood's own: the two ends of the range |
| `JD` | the Julian date of mid-exposure |

`--intype` says what the header's time marks: the `beginning` of the
exposure (the default), its `middle` or its `end`. The old `AIRMASS` goes
to HISTORY, whatever it was; `--no-override` keeps one already there;
`--no-update` reports without writing.

The site comes from `--observatory`, the header's `OBSERVAT`, or the
reduction's default ([observatory](task:observatory)), or from `--latitude`,
`--longitude` and `--altitude` -- longitude positive **west**, as IRAF
stores it: Kitt Peak is +111.6.
