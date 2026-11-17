Sets, changes or deletes a header keyword by ccdred's name for it, as IRAF's `ccdhedit`:
the instrument's translation says which keyword that is -- `exptime` is `INTEG` in the
demonstration's headers, `imagetyp` is `CCDTYPE` -- and, for an image type, how this
instrument writes it.

```example
title: Mark frames as flats, the instrument's way
run: harwood ccdhedit 'flat00*.fits' imagetyp flat
output: |
  -> flat001.fits
```

`--type` is `string` (the default), `real` or `integer`; an empty value deletes the
keyword. Without `--instrument` the translation is the one `harwood setinstrument` put
in the project file; with none, the parameter's own name is the keyword.

For `imagetyp` the value is ccdred's type -- object, zero, dark, flat, illum, fringe,
comp, other -- and what goes into the header is the instrument's word for it, the
translation read backwards: with `kpno/camera`, `zero` is written `BIAS (5)`. The
`artobs` and `subsection` tasks of [ccdtest](task:artobs) write their headers through
this; their sessions against IRAF 2.18 hold it.
