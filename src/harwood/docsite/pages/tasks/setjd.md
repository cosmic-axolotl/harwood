Writes the Julian date of the middle of each exposure into its header, and
the heliocentric Julian date beside it, as IRAF's `setjd` does.

The Earth's orbit is about 1000 light-seconds across, so the same photon
can arrive up to 16.6 minutes earlier or later depending on where the Earth
is. For anything periodic -- an eclipsing binary, a Cepheid, a transit --
that is the difference between a light curve and a smear, which is why
periods are quoted against HJD.

```example
title: Julian dates for the object frames
run: harwood setjd @obj.lis --observatory kpno
output: |
  2461296.587153
```

| keyword | |
|---|---|
| `JD` | Julian date at mid-exposure |
| `HJD` | heliocentric Julian date; with `--barycentric`, barycentric instead (better by up to 4 s, and what pulsar timing uses) |
| `LJD` | the local Julian day the night belongs to, so that one night is one number |

The keywords can be renamed (`--jd`, `--hjd`, `--ljd`), and the UT and
exposure are read from `--ut` and `--exposure`. The correction needs the
star's position, from `RA`, `DEC` and `EPOCH` in the header.
