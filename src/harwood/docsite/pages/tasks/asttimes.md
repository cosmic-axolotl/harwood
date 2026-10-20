Converts a moment between the ways astronomers count time: local zone
time, UT, Julian epoch, Julian date, and the local mean sidereal time at
the site -- what is overhead, which is why observing plans are made in it.

**IRAF's form**, ported and checked against IRAF's output: a date and the
local zone time at a site, or a file of them, one `year month day time` a
line, printed as IRAF prints them, with the day of the week.

```example
title: A night at Kitt Peak, as IRAF's asttimes prints it
run: harwood asttimes --year 1992 --month 1 --day 24 --time 19 --observatory kpno
output: |
  1992-01-24 │ FRI │ 19:00:00.0 │ 02:00:00.0 │ 1992.06457 │ 2448646.5833 │ 02:48:10.5
```

**From a frame**, harwood's: the date and the middle of the exposure from
its header (or `--date` and `--ut`).

```example
title: When obj001 was taken, every way
run: harwood asttimes obj001.fits
output: |
  jd: 2461296.587153
  mjd: 61296.087153
```

**The other way round**, also harwood's: `--jd` or `--epoch` back to a date
and time, and `--lmst` with `--date` for the UT at which a sidereal time
comes round that night -- when a field crosses the meridian, say.

```example
title: When is it 18:00 sidereal at Kitt Peak on 13 September 2026?
run: harwood asttimes --date 2026-09-13 --lmst 18:00 --observatory kpno
output: |
  18:00:00.0
```

The sidereal time is IRAF's formula, taking UT as UT1, rather than asking
the internet for the Earth's rotation: the difference is under a second,
and an observatory's computer is often offline.
