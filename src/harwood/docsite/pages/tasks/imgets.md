Gets one parameter from one header and prints it, alone, for a script to
read -- what IRAF's `imgets` leaves in `imgets.value`:

```example
title: The exposure time of a frame
run: harwood imgets obj001.fits EXPTIME
output: |
  60.0
```

```console
$ exptime=$(harwood imgets obj001.fits EXPTIME)
$ harwood imgets obj001.fits naxis1
$ harwood imgets 'obj001.fits[1]' OBJECT
```

The value is printed as the card holds it -- `15`, `5.581000E3`, `T` -- and
a string without its trailing blanks, as IRAF prints them; the keyword can
be given in any case. IRAF's standard fields work too, with or without `i_`
in front: `naxis`, `naxis1` to `naxis7` (1 beyond the image's axes),
`pixtype` (IRAF's type codes: 3 short, 4 int, 6 real, 7 double, 11 unsigned
short), `title` (the OBJECT keyword) and `pixfile`.

When the parameter is not there, the answer is `--default`, IRAF's `0`,
and a warning says so -- on stderr, so `$(...)` still gets just the value.
From Python, `harwood.run("imgets", image="obj001.fits", param="EXPTIME")`
gives it in `result.data["value"]`, with `result.data["found"]`.

Not ported: IRAF's `ctime`, `mtime` and `limtime`, its own file times in
seconds since 1980, local time.
