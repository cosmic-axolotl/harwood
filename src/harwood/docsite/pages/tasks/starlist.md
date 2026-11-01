IRAF's `starlist`: positions and magnitudes of artificial stars, one star per line, ready for
`mkobjects`. For a seed the lines are IRAF's own.

```example
title: A hundred stars with a Salpeter luminosity function
run: harwood starlist stars.txt 100 --luminosity salpeter --minmag 5 --maxmag 20
output: |
  -> stars.txt
```

`--spatial` is `uniform`, `hubble` around `--xcenter` and `--ycenter` with `--core-radius` and
`--base`, or `file`: `--sfile` has a radius and a relative density a line, around the same
centre. `--luminosity` is `uniform`, `salpeter`, `bands`, `powlaw`, or `file`: `--lfile` has a
magnitude and a relative probability a line. Each has the parameters of the same names as
IRAF's. A list is added to the end of a file that is there, and its head records the
parameters, with the time on top.

IRAF's interactive plots of the two distributions are not here.
