IRAF's `starlist`: positions and magnitudes of artificial stars, one star per line, ready for
`mkobjects`. For a seed the lines are IRAF's own.

```example
title: A hundred stars with a Salpeter luminosity function
run: harwood starlist stars.txt 100 --luminosity salpeter --minmag 5 --maxmag 20
output: |
  -> stars.txt
```

`--spatial` is `uniform`, or `hubble` around `--xcenter` and `--ycenter` with `--core-radius`
and `--base`. `--luminosity` is `uniform`, `salpeter`, `bands` or `powlaw`, each with the
parameters of the same names as IRAF's. A list is added to the end of a file that is there,
and its head records the parameters, with the time on top.

The functions IRAF can read from `sfile` and `lfile`, and its interactive plots of the two
distributions, are not here.
