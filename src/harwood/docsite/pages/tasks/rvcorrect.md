IRAF's `astutil.rvcorrect`: what to add to an observed radial velocity to have it
heliocentric or relative to the local standard of rest, and the heliocentric
Julian date of the observation. The observations come from text files, from the
images' headers, or from the parameters.

```console
$ harwood rvcorrect --files obs.txt --observatory ctio --input
$ harwood rvcorrect --images "spec*.fits" --imupdate
$ harwood rvcorrect --year 1985 --month 12 --day 19 --ut 3.35 --ra 1:37:41 --dec 13:08:44 \
      --epoch 1950 --vobs 780 --observatory lna --precise
```

A line of a file is `year month day ut ra dec [epoch [vobs]]` (a line with fewer
than six numbers is skipped). An image gives its DATE-OBS (yyyy-mm-dd, dd/mm/yy, or
with its time, which then wins over UT), UT, RA, DEC, EPOCH and VOBS; its OBSERVAT
card wins over `--observatory`. `--imupdate` writes HJD, VHELIO, VLSR and VSUN back.

The table is IRAF's: the HJD, the observed velocity, the heliocentric and LSR ones,
and the four components -- the Earth's rotation (VDIURNAL), the Earth about the
Earth-Moon barycentre (VLUNAR), that barycentre about the Sun (VANNUAL), and the
Sun toward the solar apex (VSOLAR: `--vsun` 20 km/s toward `--ra-vsun` 18h,
`--dec-vsun` +30, `--epoch-vsun` 1900 by default).

`--precise` is harwood's: astropy's correction to the solar system's barycentre
(a modern ephemeris and the Earth's orientation, good to about a cm/s) on a line
of its own beside the table, whose low-precision theories are IRAF's (about
10 m/s). On 15 cases -- lists, several observatories, another solar motion, the
parameters, images with each form of date -- the table and the warnings are IRAF
2.18's, and the velocities agree with the doubles it records to 5e-9 km/s.
