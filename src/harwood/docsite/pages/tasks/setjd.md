IRAF's `setjd`, ported and checked against IRAF's output: for each frame,
the Julian date of the middle of the exposure, the heliocentric Julian date
beside it, and the local Julian day of the night, written into the header
in double precision as IRAF writes them.

The Earth's orbit is about 1000 light-seconds across, so the same photon
can arrive up to 16.6 minutes earlier or later depending on where the Earth
is. For anything periodic -- an eclipsing binary, a Cepheid, a transit --
that is the difference between a light curve and a smear, which is why
periods are quoted against HJD.

```example
title: Julian dates for the object frames
run: harwood setjd @obj.lis --observatory kpno
output: |
  2461296.58715
```

| keyword | |
|---|---|
| `JD` | the Julian date at mid-exposure |
| `HJD` | heliocentric, by IRAF's own approximate position of the Sun -- good to about a second, and IRAF's number to the last digit |
| `LJD` | the local Julian day: `int(JD - timezone/24)`, one number for a whole night, changing at local noon |

The keywords it reads are IRAF's and can be renamed: `--date` (which may
carry the time, as an ISO date does), `--time`, `--exposure` (`-EXPTIME`
moves the time back instead of forward; empty, not at all), `--ra`,
`--dec`, `--epoch` (`B1950` and `J2000` are read). `--utdate` and
`--uttime` say whether the date and time are UT or local, using the site's
time zone. An empty output keyword (`--hjd ""`) is not computed or written;
`--listonly` writes nothing.

harwood adds `--barycentric`: the barycentric date from astropy's
ephemeris instead of IRAF's Sun, better by up to four seconds and what
pulsar timing uses. The site's time zone comes from `--observatory`, the
header's `OBSERVAT`, or the reduction's default (see
[observatory](task:observatory)).
